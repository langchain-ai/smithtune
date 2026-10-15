"""Process-boundary checks use a local fake trainer; no provider calls."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smithtune import cli, controller
from smithtune.artifacts import _json_dump, output_lock
from smithtune.providers.base import PipelineError


def wait_for(check):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if value := check():
            return value
        time.sleep(0.05)
    raise AssertionError("controller did not reach the expected state")


@pytest.fixture
def local_trainer(tmp_path):
    injection = tmp_path / "fake-provider"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text('''
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from smithtune import cli

real_provider = cli.get_provider
class Provider:
    requires_active_client = True
    def __init__(self, name):
        self.name = name
    def settings_from_options(self, options):
        return real_provider(self.name).settings_from_options(options)
    def plan(self, *args, **kwargs):
        return {"execution": {"requires_active_client": True}}
    def train(self, data_dir, run_dir, run_id, settings, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "invocation.json").write_text(json.dumps({
            "data_dir": str(data_dir), "run_id": run_id,
            "settings": asdict(settings), **kwargs,
        }))
        (run_dir / "epochs.json").write_text('[{"epoch": 1, "resume_checkpoint": "saved-checkpoint"}]')
        print("local trainer ready", flush=True)
        try:
            while not (run_dir / "release").exists():
                time.sleep(.02)
            if os.environ.get("FAKE_TRAIN_FAILURE"):
                raise RuntimeError("local training failure")
            (run_dir / "result.json").write_text('{"best": {"resume_checkpoint": "saved-checkpoint"}}')
            return {"status": "completed"}
        finally:
            (run_dir / "cleanup-attempted").touch()

cli.get_provider = lambda name: Provider(name)
''')
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(injection), str(Path(cli.__file__).parents[1])])}
    runs = []

    def start(*extra, provider="baseten", fail=False):
        run = tmp_path / f"run {len(runs)}"
        launched = subprocess.run(
            [sys.executable, "-m", "smithtune", "train", "--provider", provider,
             "--run-dir", str(run), "--data-dir", str(tmp_path / "prepared data"),
             "--detach", "--confirm", *extra],
            env={**environment, **({"FAKE_TRAIN_FAILURE": "1"} if fail else {})},
            capture_output=True, text=True, timeout=30,
        )
        assert launched.returncode == 0, launched.stderr
        receipt = json.loads(launched.stdout)
        runs.append((run, receipt))
        wait_for(lambda: (run / "invocation.json").exists())
        return run, receipt

    yield start
    for run, _ in runs:
        (run / "release").touch()
        wait_for(lambda run=run: not controller.status(run)["controller_active"])


@pytest.mark.parametrize("provider", ["baseten", "fireworks"])
def test_detached_training_survives_launcher_exit_and_preserves_settings(local_trainer, provider):
    run, launched = local_trainer(
        "--run-id", "example run; literal $(text)", "--max-epochs", "3", "--batch-size", "2",
        "--evaluate", "--max-points-per-trajectory", "1", provider=provider,
    )
    observed = controller.status(run)
    assert observed["status"] == "running"
    assert observed["controller_active"] is True
    assert os.getsid(launched["pid"]) == launched["pid"]
    assert observed["provider_resources"] == "not_checked"
    invocation = json.loads((run / "invocation.json").read_text())
    assert invocation["run_id"] == "example run; literal $(text)"
    assert invocation["settings"]["max_epochs"] == 3
    assert invocation["settings"]["batch_size"] == 2
    assert invocation["replay"]["max_points_per_trajectory"] == 1
    assert Path(invocation["data_dir"]).is_absolute()
    request = json.loads((controller.directory(run) / "request.json").read_text())
    assert "--detach" not in request["arguments"]
    assert "environment" not in request
    (run / "release").touch()
    wait_for(lambda: controller.status(run)["status"] == "completed")
    assert (run / "result.json").exists()
    assert (run / "cleanup-attempted").exists()


def test_failed_training_retains_checkpoint_and_reports_manual_recovery(local_trainer):
    run, _ = local_trainer(fail=True)
    (run / "release").touch()
    observed = wait_for(lambda: (state if (state := controller.status(run))["status"] == "failed" else None))
    assert observed["exit_code"] != 0
    assert "epochs.json" in observed["artifacts"]
    assert observed["recovery"]
    assert (run / "cleanup-attempted").exists()
    assert "local training failure" in Path(observed["log_path"]).read_text()


def test_killed_controller_is_interrupted_and_never_restarted(local_trainer):
    run, launched = local_trainer()
    os.kill(launched["pid"], signal.SIGKILL)
    wait_for(lambda: not controller.status(run)["controller_active"])
    observed = controller.status(run)
    assert observed["status"] == "interrupted"
    assert observed["automatic_restart"] is False
    assert (run / "epochs.json").exists()
    assert not (run / "cleanup-attempted").exists()
    assert not (run / "result.json").exists()


def test_sigterm_allows_training_cleanup(local_trainer):
    run, launched = local_trainer()
    os.kill(launched["pid"], signal.SIGTERM)
    wait_for(lambda: not controller.status(run)["controller_active"])
    observed = controller.status(run)
    assert observed["status"] == "interrupted"
    assert observed["exit_code"] == 143
    assert (run / "cleanup-attempted").exists()


def test_duplicate_launch_is_refused_before_another_worker_starts(tmp_path, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    run = tmp_path / "run"
    controller.directory(run).mkdir()
    with pytest.raises(PipelineError, match="already recorded"):
        controller.launch(["train", "--confirm"], run, "run", "baseten")
    spawn.assert_not_called()


def test_ambiguous_startup_is_unknown_without_claiming_cleanup(tmp_path):
    run = tmp_path / "run"
    _json_dump(controller.directory(run) / "controller.json", {
        "schema_version": 1, "status": "starting", "run_dir": str(run.resolve()),
    })
    assert controller.status(run)["status"] == "unknown"
    assert controller.status(run)["provider_resources"] == "not_checked"
    with output_lock(controller.directory(run)):
        assert controller.status(run)["controller_active"] is True


def test_spawn_failure_is_recorded_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.subprocess, "Popen", Mock(side_effect=OSError("cannot spawn")))
    run = tmp_path / "run"
    with pytest.raises(PipelineError, match="could not start"):
        controller.launch(["train", "--confirm"], run, "run", "baseten")
    assert controller.status(run)["status"] == "launch_failed"
    with pytest.raises(PipelineError, match="already recorded"):
        controller.launch(["train", "--confirm"], run, "run", "baseten")


@pytest.mark.parametrize("arguments,match", [
    (["train", "--detach"], "requires --confirm"),
    (["train", "--detach", "--confirm"], "does not require"),
])
def test_detach_gates_launch(arguments, match, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_provider", lambda _: SimpleNamespace(requires_active_client=False, settings_from_options=lambda options: options))
    launch = Mock()
    monkeypatch.setattr(controller, "launch", launch)
    with pytest.raises(SystemExit):
        cli.main(arguments)
    assert match in capsys.readouterr().err
    launch.assert_not_called()


def test_status_and_logs_need_no_data_rights_acknowledgment(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    control = controller.directory(run)
    _json_dump(control / "controller.json", {
        "schema_version": 1, "status": "completed", "run_dir": str(run.resolve()),
    })
    (control / "controller.log").write_text("first\nsecond\nthird\n")
    monkeypatch.setattr(cli.data_rights, "require_acknowledgment", Mock(side_effect=AssertionError("observation needs no acknowledgment")))
    cli.main(["status", "--run-dir", str(run)])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    cli.main(["logs", "--run-dir", str(run), "--tail", "2", "--follow"])
    assert capsys.readouterr().out == "second\nthird\n"
    cli.main(["logs", "--run-dir", str(run), "--tail", "0"])
    assert capsys.readouterr().out == ""
    with pytest.raises(PipelineError, match="non-negative"):
        controller.logs(run, tail=-1)


def test_detach_requires_data_rights_acknowledgment(monkeypatch):
    monkeypatch.setattr(cli.data_rights, "require_acknowledgment", Mock(side_effect=PipelineError("acknowledgment required")))
    launch = Mock()
    monkeypatch.setattr(controller, "launch", launch)
    with pytest.raises(SystemExit):
        cli.main(["train", "--detach", "--confirm"])
    launch.assert_not_called()


def test_dash_prefixed_values_reach_the_worker(local_trainer):
    run, _ = local_trainer("--run-id=--example")
    assert json.loads((run / "invocation.json").read_text())["run_id"] == "--example"


def test_baseten_termination_is_an_interruption(tmp_path, monkeypatch):
    from smithtune.providers.baseten import TerminationRequested

    run = tmp_path / "run"
    control = controller.directory(run)
    _json_dump(control / "controller.json", {"status": "starting"})
    _json_dump(control / "request.json", {"cwd": str(tmp_path), "arguments": []})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "main", Mock(side_effect=TerminationRequested("SIGTERM")))
    assert controller.run(control) == 143
    assert json.loads((control / "controller.json").read_text())["status"] == "interrupted"


def test_observer_cannot_prevent_worker_startup(tmp_path):
    control = controller.directory(tmp_path / "run")
    _json_dump(control / "controller.json", {"status": "starting"})
    _json_dump(control / "request.json", {"cwd": str(tmp_path), "arguments": []})
    script = '''
import sys
from pathlib import Path
from smithtune import cli, controller
cli.main = lambda arguments: None
print("ready", flush=True)
raise SystemExit(controller.run(Path(sys.argv[1])))
'''
    with output_lock(control):
        worker = subprocess.Popen([sys.executable, "-c", script, str(control)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert worker.stdout.readline() == "ready\n"
        with pytest.raises(subprocess.TimeoutExpired):
            worker.wait(timeout=0.2)
    _, errors = worker.communicate(timeout=20)
    assert worker.returncode == 0, errors


def test_completion_during_observation_uses_final_receipt(tmp_path, monkeypatch):
    run = tmp_path / "run"
    receipt_path = controller.directory(run) / "controller.json"
    receipt = {"schema_version": 1, "status": "running", "run_dir": str(run.resolve())}
    _json_dump(receipt_path, receipt)

    @contextmanager
    def completes_before_lock(*args, **kwargs):
        _json_dump(receipt_path, {**receipt, "status": "completed"})
        with output_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(controller, "output_lock", completes_before_lock)
    assert controller.status(run)["status"] == "completed"


def test_follow_waits_for_startup_and_drains_final_output(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    control = controller.directory(run)
    control.mkdir()
    log_path = control / "controller.log"
    log_path.touch()
    observations = iter([{"status": "unknown", "controller_active": False}, {"status": "completed", "controller_active": False}])

    def observe(_run):
        result = next(observations)
        if result["status"] == "completed":
            log_path.write_text("training completed\n")
        return result

    monkeypatch.setattr(controller, "status", observe)
    controller.logs(run, follow=True)
    assert capsys.readouterr().out == "training completed\n"

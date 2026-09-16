"""Provider dispatch and saved Baseten endpoint wiring, with no provider calls."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from smithtune import cli
from smithtune.inference import BasetenEndpoint


@pytest.fixture(autouse=True)
def validate_evaluation_model(monkeypatch):
    validate = Mock()
    monkeypatch.setattr(cli.baseten_deployment, "validate_evaluation_model", validate)
    return validate


def test_baseten_deploy_dispatches_explicit_hardware_and_defaults(monkeypatch, capsys):
    deploy = Mock(return_value={"status": "ready"})
    monkeypatch.setattr(cli.baseten_deployment, "deploy", deploy)
    cli.main(["deploy", "--provider", "baseten", "--run-dir", "runs/training",
              "--accelerator", "H200:1", "--max-seq-len", "32768", "--confirm"])
    deploy.assert_called_once_with(
        Path("runs/training"), accelerator="H200:1", max_seq_len=32768,
        timeout=1800, confirm=True,
    )
    assert json.loads(capsys.readouterr().out) == {"status": "ready"}


def test_baseten_deploy_forwards_custom_settings_and_confirmation(monkeypatch):
    deploy = Mock(return_value={"status": "preview"})
    monkeypatch.setattr(cli.baseten_deployment, "deploy", deploy)
    cli.main(["deploy", "--provider", "baseten", "--run-dir", "run",
              "--accelerator", "H200:2", "--max-seq-len", "8192",
              "--deployment-timeout", "90"])
    deploy.assert_called_once_with(
        Path("run"), accelerator="H200:2", max_seq_len=8192,
        timeout=90, confirm=False,
    )


@pytest.mark.parametrize("extra, expected", [
    ([], "requires --accelerator and --max-seq-len"),
    (["--accelerator", "H200:1"], "requires --accelerator and --max-seq-len"),
    (["--max-seq-len", "32768"], "requires --accelerator and --max-seq-len"),
    (["--account-id", "account"], "does not accept"),
    (["--output-model-id", "model"], "does not accept"),
    (["--deployment-id", "deployment"], "does not accept"),
    (["--deployment-shape", "shape"], "does not accept"),
])
def test_baseten_deploy_rejects_missing_and_foreign_options(monkeypatch, capsys, extra, expected):
    deploy = Mock()
    monkeypatch.setattr(cli.baseten_deployment, "deploy", deploy)
    with pytest.raises(SystemExit) as error:
        cli.main(["deploy", "--provider", "baseten", "--run-dir", "run", *extra])
    assert error.value.code == 2
    assert expected in capsys.readouterr().err
    deploy.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--accelerator", "H200:1"], ["--max-seq-len", "32768"],
    ["--deployment-timeout", "1800"],
])
def test_fireworks_rejects_baseten_deployment_options(capsys, extra):
    with pytest.raises(SystemExit):
        cli.main(["deploy", "--run-dir", "run", *extra])
    assert "require --provider baseten" in capsys.readouterr().err


def test_fireworks_deploy_and_undeploy_defaults_unchanged(monkeypatch):
    provider = Mock()
    provider.deploy.return_value = {"status": "ready"}
    monkeypatch.setattr(cli, "FireworksProvider", lambda: provider)
    cli.main(["deploy", "--run-dir", "run", "--account-id", "account",
              "--output-model-id", "model", "--deployment-id", "deployment",
              "--deployment-shape", "shape", "--confirm"])
    provider.deploy.assert_called_once_with(
        Path("run"), "account", "model", "deployment", "shape", confirm=True,
    )
    cli.main(["undeploy", "--account-id", "account", "--deployment-id", "deployment", "--confirm"])
    provider.undeploy.assert_called_once_with("account", "deployment", confirm=True)


def test_baseten_undeploy_uses_receipt(monkeypatch, capsys):
    undeploy = Mock(return_value={"status": "deactivated"})
    monkeypatch.setattr(cli.baseten_deployment, "undeploy", undeploy)
    cli.main(["undeploy", "--provider", "baseten", "--run-dir", "run", "--confirm"])
    undeploy.assert_called_once_with(Path("run"), confirm=True)
    assert json.loads(capsys.readouterr().out)["status"] == "deactivated"


@pytest.mark.parametrize("args, expected", [
    (["undeploy", "--provider", "baseten"], "requires --run-dir"),
    (["undeploy", "--provider", "baseten", "--run-dir", "run", "--account-id", "account"], "omit --account-id"),
    (["undeploy", "--provider", "baseten", "--run-dir", "run", "--deployment-id", "deployment"], "omit --account-id"),
    (["undeploy", "--run-dir", "run"], "requires --provider baseten"),
    (["undeploy"], "requires --account-id and --deployment-id"),
    (["deploy", "--run-dir", "run"], "Fireworks deploy requires"),
])
def test_undeploy_provider_options_and_fireworks_required_flags(capsys, args, expected):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
def test_baseten_replay_uses_endpoint_and_checkpoint_from_receipt(tmp_path, monkeypatch, command):
    endpoint = BasetenEndpoint("model123", "deploy123", 32768)
    load = Mock(return_value=(endpoint, "checkpoint-name"))
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", load)
    plan = Mock(return_value={"status": "planned"})
    evaluate = Mock(return_value={"status": "complete"})
    monkeypatch.setattr(cli.replay_evaluation, "prepare_replay_evaluation", plan)
    monkeypatch.setattr(cli.replay_evaluation, "run_replay_evaluation", evaluate)
    cli.main([command, "--provider", "baseten", "--serving-mode", "existing", "--run-dir", "run",
              "--data-dir", "data", "--output-dir", str(tmp_path / "replay")])
    load.assert_called_once_with(Path("run"))
    called = plan if command == "eval-plan" else evaluate
    assert called.call_args.kwargs["baseten_endpoint"] == endpoint
    if command == "evaluate":
        assert called.call_args.args[2] == "checkpoint-name"
        assert "replay_sampler" not in called.call_args.kwargs


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
@pytest.mark.parametrize("flag, value", [
    ("--model-id", "model123"), ("--deployment-id", "deploy123"),
    ("--max-seq-len", "32768"), ("--tuned-model", "checkpoint-name"),
])
def test_baseten_receipt_rejects_manual_endpoint_overrides(tmp_path, monkeypatch, capsys, command, flag, value):
    load = Mock()
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", load)
    with pytest.raises(SystemExit):
        cli.main([command, "--provider", "baseten", "--serving-mode", "existing", "--run-dir", "run",
                  "--output-dir", str(tmp_path / "replay"), flag, value])
    assert "uses the saved Baseten endpoint" in capsys.readouterr().err
    load.assert_not_called()


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
def test_fireworks_replay_requires_prepared_data_for_run_dir(tmp_path, capsys, command):
    with pytest.raises(SystemExit):
        cli.main([command, "--run-dir", "run", "--output-dir", str(tmp_path)])
    assert "manifest.json" in capsys.readouterr().err


def test_fireworks_evaluation_requires_training_run(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--output-dir", str(tmp_path)])
    assert "evaluate requires --run-dir" in capsys.readouterr().err


@pytest.fixture
def temporary_baseten(monkeypatch):
    from contextlib import contextmanager

    endpoint = BasetenEndpoint("model123", "deploy123", 32768)
    events = []
    settings = {"accelerator": "H200:1", "max_seq_len": 32768}
    plan = Mock(return_value={"settings": settings, "cleanup": "deactivate", "checkpoint": {"base_model": "test-base"}})
    monkeypatch.setattr(cli.baseten_deployment, "plan", plan)

    @contextmanager
    def temporary(*args, **kwargs):
        events.append(("enter", args, kwargs))
        try:
            yield endpoint, "checkpoint-name"
        finally:
            events.append(("exit",))

    monkeypatch.setattr(cli.baseten_deployment, "temporary", temporary)
    monkeypatch.setenv("BASETEN_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    prepare = Mock(return_value={"status": "planned", "training_base_model": "test-base"})
    evaluate = Mock(return_value={"status": "complete"})
    monkeypatch.setattr(cli.replay_evaluation, "prepare_replay_evaluation", prepare)
    monkeypatch.setattr(cli.replay_evaluation, "run_replay_evaluation", evaluate)
    return endpoint, events, plan, prepare, evaluate


def temporary_args(tmp_path, command="evaluate"):
    return [command, "--provider", "baseten", "--serving-mode", "temporary",
            "--run-dir", str(tmp_path / "training"), "--output-dir", str(tmp_path / "replay")]


def test_temporary_plan_uses_context_without_invented_endpoint(tmp_path, temporary_baseten, capsys):
    _, events, plan, prepare, evaluate = temporary_baseten
    cli.main([*temporary_args(tmp_path, "eval-plan"), "--accelerator", "H200:1", "--max-seq-len", "32768"])
    plan.assert_called_once_with(tmp_path / "training", accelerator="H200:1", max_seq_len=32768,
                                 timeout=600)
    assert prepare.call_args.kwargs == {"baseten_endpoint": None, "baseten_context_limit": 32768}
    assert json.loads(capsys.readouterr().out)["deployment"]["cleanup"] == "deactivate"
    assert json.loads((tmp_path / "replay/plan.json").read_text())["deployment"]["settings"]["max_seq_len"] == 32768
    assert events == []
    evaluate.assert_not_called()


def test_temporary_evaluation_preflights_separately_and_exits(tmp_path, temporary_baseten, capsys):
    endpoint, events, _, prepare, evaluate = temporary_baseten
    output = tmp_path / "replay"
    output.mkdir()
    (output / "plan.json").write_text('{"existing": true}')
    cli.main([*temporary_args(tmp_path), "--confirm"])
    assert prepare.call_args.kwargs == {"baseten_context_limit": 32768}
    assert prepare.call_args.args[1] != output
    assert not prepare.call_args.args[1].exists()
    assert (output / "plan.json").read_text() == '{"existing": true}'
    assert events[0] == ("enter", (tmp_path / "training",), {
        "accelerator": None, "max_seq_len": None, "timeout": 600, "confirm": True,
    })
    assert events[-1] == ("exit",)
    assert evaluate.call_args.args[2] == "checkpoint-name"
    assert evaluate.call_args.kwargs["baseten_endpoint"] == endpoint
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_temporary_evaluation_forwards_requested_settings(tmp_path, temporary_baseten):
    _, events, _, _, _ = temporary_baseten
    cli.main([*temporary_args(tmp_path), "--confirm", "--accelerator", "H200:2",
              "--max-seq-len", "32768", "--deployment-timeout", "1800"])
    assert events[0][2] == {"accelerator": "H200:2", "max_seq_len": 32768,
                           "timeout": 1800, "confirm": True}


@pytest.mark.parametrize("failure", [cli.PipelineError("evaluation failed"), KeyboardInterrupt()])
def test_temporary_evaluation_exits_when_evaluation_fails(tmp_path, temporary_baseten, failure):
    _, events, _, _, evaluate = temporary_baseten
    evaluate.side_effect = failure
    with pytest.raises(SystemExit if isinstance(failure, cli.PipelineError) else KeyboardInterrupt):
        cli.main([*temporary_args(tmp_path), "--confirm"])
    assert events[-1] == ("exit",)


def test_temporary_evaluation_does_not_provision_when_data_preflight_fails(tmp_path, temporary_baseten):
    _, events, _, prepare, evaluate = temporary_baseten
    prepare.side_effect = cli.PipelineError("prepared data mismatch")
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm"])
    assert events == []
    evaluate.assert_not_called()


def test_temporary_evaluation_requires_confirmation_before_preflight(tmp_path, temporary_baseten, capsys):
    _, events, plan, prepare, _ = temporary_baseten
    with pytest.raises(SystemExit):
        cli.main(temporary_args(tmp_path))
    assert "require --confirm" in capsys.readouterr().err
    plan.assert_not_called()
    prepare.assert_not_called()
    assert events == []


@pytest.mark.parametrize("flag, value, message", [
    ("--model-id", "model", "omit --model-id"),
    ("--deployment-id", "deployment", "omit --model-id"),
    ("--tuned-model", "checkpoint", "omit --model-id"),
    ("--account-id", "account", "unrecognized arguments"),
    ("--deployment-shape", "shape", "unrecognized arguments"),
    ("--base-model", "base", "does not support --base-model"),
    ("--concurrency", "0", "concurrency must be positive"),
    ("--max-output-tokens", "0", "--max-output-tokens must be positive"),
])
def test_temporary_evaluation_rejects_invalid_options_before_provisioning(tmp_path, temporary_baseten, capsys, flag, value, message):
    _, events, plan, _, _ = temporary_baseten
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm", flag, value])
    assert message in capsys.readouterr().err
    plan.assert_not_called()
    assert events == []


@pytest.mark.parametrize("missing", ["BASETEN_API_KEY", "ANTHROPIC_API_KEY"])
def test_temporary_evaluation_requires_credentials_before_provisioning(tmp_path, temporary_baseten, monkeypatch, capsys, missing):
    _, events, plan, _, _ = temporary_baseten
    monkeypatch.delenv(missing)
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm"])
    assert missing in capsys.readouterr().err
    plan.assert_not_called()
    assert events == []


def test_temporary_evaluation_checks_selected_judge_credential(tmp_path, temporary_baseten, monkeypatch, capsys):
    _, events, plan, _, _ = temporary_baseten
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm", "--judge-model", "accounts/test/models/judge"])
    assert "FIREWORKS_API_KEY is not set for the judge" in capsys.readouterr().err
    plan.assert_not_called()
    assert events == []


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
@pytest.mark.parametrize("provider", ["baseten", "fireworks"])
def test_temporary_mode_rejects_missing_run_or_wrong_provider(tmp_path, capsys, command, provider):
    with pytest.raises(SystemExit):
        cli.main([command, "--provider", provider, "--serving-mode", "temporary", "--output-dir", str(tmp_path)])
    assert ("requires --run-dir" if provider == "baseten" else "uses the serverless sampler") in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["existing", "temporary"])
def test_baseten_evaluation_requires_separate_training_and_output_directories(tmp_path, capsys, mode):
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--provider", "baseten", "--serving-mode", mode,
                  "--run-dir", str(tmp_path), "--output-dir", str(tmp_path / ".")])
    assert "must be different directories" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["baseten", "fireworks"])
def test_existing_mode_rejects_temporary_baseten_settings(tmp_path, capsys, provider):
    with pytest.raises(SystemExit):
        cli.main(["eval-plan", "--provider", provider, "--serving-mode", "existing", "--output-dir", str(tmp_path), "--accelerator", "H200:1"])
    assert ("requires --provider baseten --serving-mode temporary" if provider == "baseten" else "uses the serverless sampler") in capsys.readouterr().err


@pytest.mark.parametrize("filename", ["evaluation-config.json", "results.jsonl"])
def test_first_temporary_deployment_refuses_existing_evaluation_output(tmp_path, temporary_baseten, capsys, filename):
    _, events, _, _, evaluate = temporary_baseten
    output = tmp_path / "replay"
    output.mkdir()
    (output / filename).write_text("{}")
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm"])
    assert "without a saved Baseten endpoint" in capsys.readouterr().err
    assert events == []
    evaluate.assert_not_called()


def test_temporary_preflight_rejects_training_model_mismatch(tmp_path, temporary_baseten, capsys):
    _, events, _, prepare, evaluate = temporary_baseten
    prepare.return_value["training_base_model"] = "another-base"
    with pytest.raises(SystemExit):
        cli.main([*temporary_args(tmp_path), "--confirm"])
    assert "base model differs" in capsys.readouterr().err
    assert events == []
    evaluate.assert_not_called()


@pytest.mark.parametrize("state", ["completed", "mismatched", "pending"])
def test_saved_temporary_endpoint_is_activated_only_for_pending_evaluation(tmp_path, temporary_baseten, monkeypatch, state):
    endpoint, events, _, _, evaluate = temporary_baseten
    run_dir = tmp_path / "training"
    run_dir.mkdir()
    (run_dir / "endpoint.json").write_text("{}")
    load = Mock(return_value=(endpoint, "checkpoint-name"))
    cleanup = Mock(return_value={"status": "inactive"})
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", load)
    monkeypatch.setattr(cli.baseten_deployment, "undeploy", cleanup)

    def evaluate_lazily(*args, **kwargs):
        assert events == []
        if state == "mismatched":
            raise cli.PipelineError("evaluation configuration changed")
        if state == "pending":
            with kwargs["baseten_lifecycle"] as route:
                assert route == "checkpoint-name"
                assert events[0][0] == "enter"
        else:
            kwargs["baseten_cleanup"]()
        return {"status": "complete"}

    evaluate.side_effect = evaluate_lazily
    if state == "mismatched":
        with pytest.raises(SystemExit):
            cli.main([*temporary_args(tmp_path), "--confirm"])
    else:
        cli.main([*temporary_args(tmp_path), "--confirm"])
    load.assert_called_once_with(run_dir, require_ready=False)
    if state == "pending":
        assert events[-1] == ("exit",)
    else:
        assert events == []
    if state == "completed":
        cleanup.assert_called_once_with(run_dir, confirm=True)
    else:
        cleanup.assert_not_called()


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
def test_receipt_evaluation_checks_training_model_before_writing_or_running(tmp_path, monkeypatch, validate_evaluation_model, command):
    validate_evaluation_model.side_effect = cli.PipelineError("prepared data base model differs")
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", Mock(return_value=(BasetenEndpoint("model123", "deploy123", 32768), "checkpoint-name")))
    run = Mock()
    prepare = Mock()
    monkeypatch.setattr(cli.replay_evaluation, "run_replay_evaluation", run)
    monkeypatch.setattr(cli.replay_evaluation, "prepare_replay_evaluation", prepare)
    with pytest.raises(SystemExit):
        cli.main([command, "--provider", "baseten", "--serving-mode", "existing", "--run-dir", str(tmp_path / "training"),
                  "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "replay")])
    validate_evaluation_model.assert_called_once_with(tmp_path / "training", tmp_path / "data")
    run.assert_not_called()
    prepare.assert_not_called()
    assert not (tmp_path / "replay").exists()


def test_temporary_retry_uses_corrected_context_before_resume_validation(tmp_path, temporary_baseten, monkeypatch):
    endpoint, events, _, _, evaluate = temporary_baseten
    run_dir = tmp_path / "training"
    run_dir.mkdir()
    (run_dir / "endpoint.json").write_text("{}")
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", Mock(return_value=(
        BasetenEndpoint(endpoint.model_id, endpoint.deployment_id, 65536), "checkpoint-name")))

    def evaluate_lazily(*args, **kwargs):
        assert kwargs["baseten_endpoint"] == endpoint
        assert events == []
        with kwargs["baseten_lifecycle"]:
            assert events[0][0] == "enter"
        return {"status": "complete"}

    evaluate.side_effect = evaluate_lazily
    cli.main([*temporary_args(tmp_path), "--max-seq-len", "32768", "--confirm"])
    assert events[-1] == ("exit",)

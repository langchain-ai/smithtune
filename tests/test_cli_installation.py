"""Customer-facing CLI checks, also run against wheels outside the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

import pytest

from smithtune import artifacts, cli, curation, doctor
from smithtune.providers.base import PipelineError


def test_module_entrypoint_and_version_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "smithtune", "--version"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == f"smithtune {version('smithtune')}"
    assert not list(tmp_path.iterdir())


def test_help_does_not_import_training_sdks_or_access_network(tmp_path):
    script = '''
import builtins
import socket
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'training', 'torch', 'transformers', 'fireworks', 'baseten'}:
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
def no_network(*args, **kwargs):
    raise AssertionError('network access')
builtins.__import__ = guarded_import
socket.create_connection = no_network
from smithtune.cli import main
main(['--help'])
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "usage: smithtune" in result.stdout


def test_default_and_explicit_data_paths_follow_invocation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for command in ("prepare", "plan", "train", "eval-plan", "evaluate"):
        extras = {
            "prepare": ["--workspace-id", "w", "--dataset-id", "d"],
            "plan": [],
            "train": ["--run-dir", "runs/test", "--run-id", "test"],
            "eval-plan": ["--output-dir", "replay"],
            "evaluate": ["--output-dir", "replay", "--tuned-model", "test"],
        }[command]
        assert cli._parser().parse_args([command, *extras]).data_dir == tmp_path / "data"
        assert cli._parser().parse_args([command, *extras, "--data-dir", "custom"]).data_dir == Path("custom")
    assert curation.DEFAULT_SELECTION_DIR.resolve() == tmp_path / "data/selections"


def test_default_selection_is_written_in_working_directory(tmp_path, monkeypatch):
    from test_curation import API, root, uid

    monkeypatch.chdir(tmp_path)
    # Use the same service boundary fake as the behavioral curation tests.
    api = API([[root(1, "a")]])
    result = curation.create_dataset(
        workspace_id=uid(100), project_id=uid(101), name="local-output",
        start_time="2026-09-01T00:00:00Z", end_time="2026-09-08T00:00:00Z",
        runner=api,
    )
    assert Path(result["selection"]).resolve().is_relative_to(tmp_path / "data/selections")
    assert Path(result["selection"]).is_file()


def test_doctor_redacts_configuration_and_is_offline(monkeypatch, capsys):
    marker = "test-only-placeholder-must-not-appear"
    monkeypatch.setenv("FIREWORKS_API_KEY", marker)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    def unexpected(*args, **kwargs):
        raise AssertionError("doctor must not invoke external commands")
    monkeypatch.setattr(subprocess, "run", unexpected)
    cli.main(["doctor"])
    output = capsys.readouterr().out
    assert marker not in output
    report = json.loads(output)
    assert report["credentials"]["FIREWORKS_API_KEY"] == "set"
    assert report["credentials"]["BASETEN_API_KEY"] == "unset"
    assert report["packages"]["smithtune"] == version("smithtune")
    assert report["tools"]["firectl"]["required_for"] == "deploy, undeploy"
    assert "https://" in report["tools"]["langsmith"]["help"]


@pytest.mark.parametrize("tool", ["langsmith", "firectl"])
def test_missing_companion_has_actionable_error(tool, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(tool)
    monkeypatch.setattr(artifacts.subprocess, "run", missing)
    with pytest.raises(PipelineError, match=f"cannot run {tool}.*Install.*smithtune doctor"):
        artifacts._run([tool, "test"])


def test_missing_companion_uses_cli_error_path(tmp_path):
    environment = {**os.environ, "PATH": ""}
    result = subprocess.run(
        [sys.executable, "-I", "-m", "smithtune", "undeploy", "--account-id", "test", "--deployment-id", "test", "--confirm"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Install firectl" in result.stderr
    assert "Traceback" not in result.stderr

"""Customer-facing CLI checks, also run against wheels outside the checkout."""

import json
import os
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version

import pytest

from smithtune import artifacts, cli, doctor
from smithtune.dataset_artifacts import new_run_directory
from smithtune.providers.base import PipelineError


def test_module_entrypoint_and_version_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "smithtune", "--version"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == f"smithtune {version('smithtune')}"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("argv", [
    ["--help"], ["models", "list", "--help"],
    ["dataset", "triage", "--help"], ["dataset", "publish-splits", "--help"],
    ["models", "list"], ["models", "list", "--provider", "baseten"],
    ["models", "list", "--provider", "fireworks"],
])
def test_discovery_does_not_import_training_sdks_or_access_network(tmp_path, argv):
    script = '''
import builtins
import socket
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'training', 'torch', 'transformers', 'fireworks', 'baseten', 'deepagents', 'langchain_openai', 'pydantic_monty', 'openai', 'anthropic'}:
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
def no_network(*args, **kwargs):
    raise AssertionError('network access')
builtins.__import__ = guarded_import
socket.create_connection = no_network
from smithtune.cli import main
'''
    script += f"main({argv!r})\n"
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    if "--help" in argv:
        assert "usage: smithtune" in result.stdout
    else:
        from smithtune.providers import get_provider
        from smithtune.providers.base import ModelOptions

        report = json.loads(result.stdout)
        assert report["source"] == "smithtune_support_registry"
        assert report["live_availability_checked"] is False
        expected_providers = {argv[-1]} if "--provider" in argv else {"baseten", "fireworks"}
        assert {model["provider"] for model in report["models"]} == expected_providers
        for model in report["models"]:
            adapter = get_provider(model["provider"])
            by_alias = adapter.model_from_options(ModelOptions(model=model["alias"]))
            by_id = adapter.model_from_options(ModelOptions(model=model["model_id"]))
            assert by_alias == by_id
            assert model["training_context_limit"] == by_alias.training_context_limit
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("argv", [
    ["models"], ["models", "unknown"], ["models", "list", "--provider", "unknown"],
])
def test_invalid_models_command_fails_during_argument_parsing(argv, capsys):
    with pytest.raises(SystemExit) as failure:
        cli.main(argv)
    assert failure.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_default_and_explicit_data_paths_follow_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_version", lambda: "0.1.0")
    monkeypatch.chdir(tmp_path)
    for command in ("prepare", "plan", "train", "eval-plan", "evaluate"):
        extras = {
            "prepare": ["--workspace-id", "w", "--dataset-id", "d", "--model", "qwen3p8-27b"],
            "plan": [],
            "train": ["--run-dir", "runs/test", "--run-id", "test"],
            "eval-plan": ["--output-dir", "replay"],
            "evaluate": ["--output-dir", "replay", "--tuned-model", "test"],
        }[command]
        assert cli._parser().parse_args([command, *extras]).data_dir == tmp_path / "data"
        assert cli._parser().parse_args([command, *extras, "--data-dir", "custom"]).data_dir == Path("custom")
    assert new_run_directory().resolve().parent == tmp_path / "data/datasets"


def test_dataset_publish_splits_dispatches_without_preparation(tmp_path, monkeypatch, capsys):
    calls = []

    def publish(data_dir):
        calls.append(data_dir)
        return {"status": "complete", "data_dir": str(data_dir)}

    monkeypatch.setattr(cli.reporting, "publish_prepared_splits", publish)

    cli.main(["dataset", "publish-splits", "--data-dir", str(tmp_path)])

    assert calls == [tmp_path]
    assert json.loads(capsys.readouterr().out) == {
        "status": "complete",
        "data_dir": str(tmp_path),
    }


def test_default_pull_directory_is_in_working_directory(tmp_path, monkeypatch):
    from smithtune import dataset_workflow
    from test_triage import API, source

    monkeypatch.chdir(tmp_path)
    result = dataset_workflow.run("pull", runner=API(),
                                  **{k: v for k, v in source().items() if k != "seed"})
    directory = Path(result["run_dir"]).resolve()
    assert directory.is_relative_to((tmp_path / "data/datasets").resolve())
    assert (directory / "snapshot.json").is_file()


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


def test_invalid_local_json_artifacts_fail_with_pipeline_errors(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(PipelineError, match="cannot read valid JSON from"):
        artifacts._load_json(broken)
    with pytest.raises(PipelineError, match="cannot read valid JSON from"):
        artifacts._load_json(tmp_path / "missing.json")

    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"a": 1}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(PipelineError, match="cannot read valid JSONL from"):
        artifacts._load_jsonl(rows)
    rows.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert artifacts._load_jsonl(rows) == [{"a": 1}, {"b": 2}]


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


@pytest.mark.parametrize("argv", [["skill", "export"], ["capture-contract"], ["promote"]])
def test_removed_commands_are_rejected(argv, capsys):
    from smithtune import cli

    with pytest.raises(SystemExit) as failure:
        cli.main(argv)
    assert failure.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

import subprocess

import pytest

from smithtune import artifacts, curation, dataset
from smithtune.cli import main
from smithtune.providers.base import PipelineError


@pytest.mark.parametrize("stderr", ["Error: HTTP 429\n", "request timed out\n", "", None])
def test_cli_bubbles_up_stderr_without_command_or_stdout(monkeypatch, capsys, tmp_path, stderr):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr=stderr, output="private response")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        main(["capture-contract", "--workspace-id", "workspace-123", "--run-id", "private-run",
              "--output", str(tmp_path / "contract.json")])
    assert error.value.code == 2
    output = capsys.readouterr().err
    expected = stderr.strip() if stderr else "langsmith exited with status 1"
    assert output.endswith(f"error: {expected}\n")
    assert "--body" not in output
    assert "private" not in output


def test_contract_failure_keeps_example_context(monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Error: HTTP 429")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    example = {"id": "example-123", "source_thread_id": "thread-123",
               "metadata": {"source_project_id": "project-123"}}
    with pytest.raises(PipelineError) as error:
        dataset.capture_example_contracts("workspace-123", [example])
    assert str(error.value) == "example example-123: cannot collect tools: Error: HTTP 429"


def test_curation_retains_existing_error_handling(monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="request timed out; private message")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError) as error:
        curation._api("workspace-123", "POST", "/api/v1/datasets", {"name": "test"})
    assert "outcome may be unknown" in str(error.value)
    assert "private" not in str(error.value)

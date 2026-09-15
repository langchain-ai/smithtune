import subprocess

import pytest

from smithtune import artifacts, curation, dataset
from smithtune.cli import main
from smithtune.providers.base import PipelineError


@pytest.mark.parametrize(("stderr", "stdout", "expected"), [
    ("Error: HTTP 429\n", "ignored response", "Error: HTTP 429"),
    ("request timed out\n", None, "request timed out"),
    ("", '{"detail":["Invalid run type"]}\nHTTP 422\n', '{"detail":["Invalid run type"]}\nHTTP 422'),
    (" \n", "request failed\n", "request failed"),
    ("", "", "langsmith exited with status 1"),
    (None, None, "langsmith exited with status 1"),
])
def test_cli_bubbles_up_diagnostic_without_command(monkeypatch, capsys, tmp_path, stderr, stdout, expected):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr=stderr, output=stdout)

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        main(["capture-contract", "--workspace-id", "workspace-123", "--run-id", "private-run",
              "--output", str(tmp_path / "contract.json")])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert output.endswith(f"error: {expected}\n")
    assert "--body" not in output
    assert "private" not in output
    assert "ignored response" not in output


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

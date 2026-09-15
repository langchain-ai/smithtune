import json
import subprocess

import pytest

from smithtune import artifacts, curation, dataset
from smithtune.cli import main
from smithtune.providers.base import PipelineError


REQUEST_TIMEOUT = ('HTTP POST /api/v1/runs/query: Post "https://api.smith.langchain.com/api/v1/runs/query": '
                   'context deadline exceeded (Client.Timeout exceeded while awaiting headers)')


@pytest.fixture
def retry_sleeps(monkeypatch):
    sleeps = []
    monkeypatch.setattr(dataset.time, "sleep", sleeps.append)
    monkeypatch.setattr(dataset.random, "uniform", lambda low, high: 0.5)
    return sleeps


@pytest.mark.parametrize(("stderr", "stdout", "expected"), [
    ("Error: HTTP 429\n", "ignored response", "Error: HTTP 429"),
    ("request timed out\n", None, "request timed out"),
    ("", '{"detail":["Invalid run type"]}\nHTTP 422\n', '{"detail":["Invalid run type"]}\nHTTP 422'),
    (" \n", "request failed\n", "request failed"),
    ("", "", "langsmith exited with status 1"),
    (None, None, "langsmith exited with status 1"),
])
def test_cli_bubbles_up_diagnostic_without_command(monkeypatch, capsys, tmp_path, stderr, stdout, expected, retry_sleeps):
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


def test_contract_failure_keeps_example_context(monkeypatch, retry_sleeps):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Error: HTTP 429")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    example = {"id": "example-123", "metadata": {"source_scope": "thread", "source_scope_id": "thread-123",
                                                 "source_project_id": "project-123"}}
    with pytest.raises(PipelineError) as error:
        dataset.capture_example_contracts("workspace-123", [example])
    assert str(error.value) == "example example-123: cannot collect tools: Error: HTTP 429"


def test_curation_retains_existing_error_handling(monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="request timed out; private message")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError) as error:
        curation._api("workspace-123", "POST", "/api/v1/datasets", {"name": "test"})
    assert str(error.value) == "LangSmith POST /api/v1/datasets: request failed"


@pytest.mark.parametrize(("diagnostic", "reason"), [
    ('{"detail":"Rate limit exceeded."}\nHTTP 429', "rate limit reached"),
    (REQUEST_TIMEOUT, "request timed out"),
])
def test_transient_error_retries_only_current_page(monkeypatch, retry_sleeps, capsys, diagnostic, reason):
    cursors = []

    def run(argv, **kwargs):
        body = json.loads(argv[argv.index("--body") + 1])
        cursor = body.get("cursor")
        cursors.append(cursor)
        if len(cursors) in (2, 3):
            raise subprocess.CalledProcessError(1, argv, output=diagnostic)
        response = ({"runs": [{"id": "first"}], "cursors": {"next": "page-2"}} if cursor is None
                    else {"runs": [{"id": "second"}], "cursors": {}})
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(response))

    monkeypatch.setattr(artifacts.subprocess, "run", run)
    runs = dataset._query_contract_runs("workspace-123", {}, runner=dataset._run_langsmith)
    assert [run["id"] for run in runs] == ["first", "second"]
    assert cursors == [None, "page-2", "page-2", "page-2"]
    assert retry_sleeps == [5.5, 10.5]
    assert f"LangSmith {reason}; retrying in 5.5s (1/5)" in capsys.readouterr().err


@pytest.mark.parametrize(("diagnostic", "attempts"), [
    ('{"detail":"Rate limit exceeded."}\nHTTP 429', 6),
    ("HTTP 403", 1),
    ("HTTP 500", 1),
    ("context canceled", 1),
    ("request timed out", 6),
    (REQUEST_TIMEOUT, 6),
    ("Client.Timeout exceeded while awaiting headers", 6),
])
def test_query_retry_limit_preserves_original_error(monkeypatch, retry_sleeps, diagnostic, attempts):
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv, stderr=diagnostic)

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError) as error:
        dataset._query_contract_runs("workspace-123", {}, runner=dataset._run_langsmith)
    assert str(error.value) == diagnostic
    assert len(calls) == attempts
    assert retry_sleeps == ([5.5, 10.5, 20.5, 40.5, 60] if attempts == 6 else [])


@pytest.mark.parametrize("diagnostic", ["HTTP 429", REQUEST_TIMEOUT])
def test_transient_error_outside_trace_queries_is_not_retried(monkeypatch, retry_sleeps, diagnostic):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr=diagnostic)

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError):
        curation._api("workspace-123", "POST", "/api/v1/datasets", {"name": "test"})
    assert retry_sleeps == []

from __future__ import annotations

import subprocess

import pytest

from smithtune import artifacts, curation, dataset
from smithtune.providers.base import PipelineError


@pytest.mark.parametrize(("stderr", "stdout", "expected"), [
    ('Error: HTTP 401', '{"detail":"private response"}', "HTTP 401 Unauthorized; check your LangSmith credentials"),
    ('Error: HTTP 403', None, "check the API key's access"),
    ('Error: HTTP 404', None, "check the resource ID and workspace"),
    ('Error: HTTP 429', None, "HTTP 429 Too Many Requests; rate limit reached"),
    ('Error: HTTP 503', None, "HTTP 503 Service Unavailable"),
    ('Error: HTTP 422', None, "HTTP 422 Unprocessable Entity"),
    ('Error: HTTP 599', None, "HTTP 599 request failed"),
    ('request failed (status code: 502)', None, "HTTP 502 Bad Gateway"),
    ('HTTP/2 500', None, "HTTP 500 Internal Server Error"),
    (None, '{"status_code":429,"detail":"private response"}', "HTTP 429"),
    ('Post "https://private-host": context deadline exceeded', None, "request timed out"),
    ('request timed out; private response', None, "request timed out"),
    ('dial tcp: lookup private-host: no such host', None, "connection failed"),
    ('tls: failed to verify certificate: x509: certificate signed by unknown authority', None, "TLS connection failed"),
    ('API key is not configured', None, "authentication is not configured"),
    ('private response', None, "run the operation directly with langsmith"),
    (None, None, "request failed"),
    ('x' * 100000, None, "request failed"),
    (b'Error: HTTP 429\xff', None, "HTTP 429"),
])
def test_langsmith_failure_is_actionable_without_echoing_data(monkeypatch, stderr, stdout, expected):
    command = ["langsmith", "api", "runs/query", "--body", '{"messages":"private request"}',
               "--header", "Authorization: Bearer private-credential"]
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv, output=stdout, stderr=stderr)

    monkeypatch.setattr(artifacts.subprocess, "run", run)
    with pytest.raises(PipelineError) as error:
        artifacts._run(command, capture=True)
    message = str(error.value)
    assert message.startswith("LangSmith runs/query failed (exit 1):")
    assert expected in message
    assert "private" not in message
    assert "--body" not in message
    assert len(message) < 500
    assert calls == [command]


def test_contract_failure_keeps_example_context(monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Error: HTTP 429")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    example = {"id": "example-123", "source_thread_id": "thread-123",
               "metadata": {"source_project_id": "project-123"}}
    with pytest.raises(PipelineError, match="example example-123: cannot collect tools: LangSmith runs/query.*HTTP 429"):
        dataset.capture_example_contracts("workspace-123", [example])


def test_export_failure_captures_stderr(monkeypatch, capsys):
    def fail(argv, **kwargs):
        assert kwargs["stderr"] == subprocess.PIPE
        assert not kwargs["capture_output"]
        raise subprocess.CalledProcessError(1, argv, stderr="HTTP 403: private response")

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError, match="LangSmith dataset export.*HTTP 403"):
        artifacts._run(["langsmith", "dataset", "export", "dataset-123", "export.json"])
    assert capsys.readouterr().err == ""


def test_successful_export_keeps_stdout_and_warnings(monkeypatch, capsys):
    def run(argv, **kwargs):
        assert "stdout" not in kwargs and not kwargs["capture_output"]
        print('{"status":"exported"}')
        return subprocess.CompletedProcess(argv, 0, stderr="export warning\n")

    monkeypatch.setattr(artifacts.subprocess, "run", run)
    artifacts._run(["langsmith", "dataset", "export", "dataset-123", "export.json"])
    captured = capsys.readouterr()
    assert captured.out == '{"status":"exported"}\n'
    assert captured.err == "export warning\n"


@pytest.mark.parametrize(("stderr", "expected"), [
    ("Error: HTTP 409", "dataset name already exists"),
    ("timeout; private response", "request timed out"),
])
def test_curation_real_runner_keeps_mutation_context(monkeypatch, stderr, expected):
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv, stderr=stderr)

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError) as error:
        curation._api("workspace-123", "POST", "/api/v1/datasets", {"name": "private request"})
    assert expected in str(error.value)
    assert "outcome may be unknown" in str(error.value)
    assert "private" not in str(error.value)
    assert len(calls) == 1


def test_operation_omits_query_parameters():
    command = ["langsmith", "api", "/api/v1/examples?dataset=private-id", "--method", "GET"]
    error = artifacts._langsmith_error(command, subprocess.CalledProcessError(1, command, stderr="HTTP 403"))
    assert "LangSmith /api/v1/examples" in str(error)
    assert "private" not in str(error)
    assert "outcome may be unknown" not in str(error)


def test_non_langsmith_errors_remain_unchanged(monkeypatch):
    error = subprocess.CalledProcessError(1, ["firectl", "get"], stderr="provider diagnostic")

    def fail(argv, **kwargs):
        assert "stderr" not in kwargs
        raise error

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        artifacts._run(["firectl", "get"])
    assert caught.value is error


def test_cli_prints_safe_langsmith_diagnostic(monkeypatch, capsys, tmp_path):
    from smithtune.cli import main

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr="Error: HTTP 429", output='{"detail":"private response"}')

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    output = tmp_path / "contract.json"
    with pytest.raises(SystemExit) as error:
        main(["capture-contract", "--workspace-id", "workspace-123", "--run-id", "private-run",
              "--output", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "LangSmith runs/query failed (exit 1): HTTP 429 Too Many Requests" in captured.err
    assert "--body" not in captured.err
    assert "private" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("operation", ["get", "export"])
@pytest.mark.parametrize(("status", "hint"), [
    ("401 Unauthorized", "check your LangSmith credentials"),
    ("403 Forbidden", "check the API key's access"),
    ("404 Not Found", "check the resource ID and workspace"),
    ("429 Too Many Requests", "rate limit reached"),
])
def test_dataset_sdk_http_errors(monkeypatch, operation, status, hint):
    def fail(argv, **kwargs):
        stderr = f'Error: getting dataset: GET "https://private-host/private-id": {status} {{"detail":"private response with HTTP 500"}}'
        raise subprocess.CalledProcessError(1, argv, stderr=stderr)

    monkeypatch.setattr(artifacts.subprocess, "run", fail)
    with pytest.raises(PipelineError) as error:
        artifacts._run(["langsmith", "dataset", operation, "private-id"], capture=operation == "get")
    assert f"LangSmith dataset {operation} failed (exit 1): HTTP {status}" in str(error.value)
    assert hint in str(error.value)
    assert "private" not in str(error.value)

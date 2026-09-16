"""Exercise SDK retries and error propagation at the HTTP boundary, offline."""

import json
from datetime import UTC, datetime
from email.utils import format_datetime
from uuid import UUID

import pytest
import requests
from langsmith import client as sdk

from smithtune.evaluation import langsmith as reporting
from smithtune.providers.base import PipelineError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "unit-test-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://localhost:9999")
    # Keep the actual SDK retry logic, without waiting in offline tests.
    monkeypatch.setattr(sdk.time, "sleep", lambda _: None)
    value = reporting.make_client(str(UUID(int=900)))
    yield value
    value.close()


def response(status, body, headers=None):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps(body).encode()
    value.headers.update(headers or {})
    value.url = "http://localhost:9999/runs/batch?api_key=unit-test-secret"
    return value


def test_rate_limit_exposes_safe_diagnostic(client, monkeypatch):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(
        429, {"detail": "Monthly trace usage limit exceeded."}, {"Retry-After": "3600"},
    ))
    with pytest.raises(PipelineError, match="POST /runs/batch.*429.*Monthly trace usage limit exceeded.*3600") as error:
        client.request_with_retries("POST", "/runs/batch?api_key=unit-test-secret")
    assert "unit-test-secret" not in str(error.value)


@pytest.mark.parametrize("failure", [429, 500, "connection"])
def test_native_sdk_retries_are_preserved(client, monkeypatch, failure):
    calls, sleeps = [], []

    def send(method, url, **kwargs):
        calls.append((method, url, kwargs.get("data")))
        if len(calls) < 3:
            if failure == "connection":
                raise requests.ConnectionError("private transport diagnostic")
            return response(failure, {"detail": "Rate limit exceeded."}, {"Retry-After": "2"})
        return response(200, {})

    monkeypatch.setattr(client.session, "request", send)
    monkeypatch.setattr(sdk.time, "sleep", sleeps.append)
    result = client.request_with_retries("POST", "/runs/batch", stop_after_attempt=3, request_kwargs={"data": b'{}'})
    assert result.status_code == 200
    assert calls[0] == calls[1] == calls[2]
    assert len(sleeps) == 2
    if failure == 429:
        assert sleeps[0] >= 2


def test_native_http_adapter_retries_are_preserved(client):
    for adapter in client.session.adapters.values():
        retry = adapter.max_retries
        assert retry.total > 0
        assert retry.respect_retry_after_header
        assert retry.is_retry("POST", 429, has_retry_after=True)
        assert retry.is_retry("POST", 503)


@pytest.mark.parametrize("retry_after", ["2", format_datetime(datetime(2026, 9, 17, tzinfo=UTC)), "unit-test-secret"])
def test_retry_after_diagnostic_accepts_seconds_or_date_only(client, monkeypatch, retry_after):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(429, {}, {"Retry-After": retry_after}))
    with pytest.raises(PipelineError) as error:
        client.request_with_retries("POST", "/runs/batch")
    message = str(error.value)
    assert "unit-test-secret" not in message
    if retry_after != "unit-test-secret":
        assert "Retry-After:" in message


@pytest.mark.parametrize("body", [
    {"detail": "Rate limit exceeded: private trajectory text; unit-test-secret"},
    {"error": {"message": "a@b.com"}},
    {"detail": "Failed input: 123-45-6789"},
    {"detail": ["private message"]},
])
def test_arbitrary_response_bodies_are_not_logged(client, monkeypatch, body, caplog):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(403, body))
    with pytest.raises(PipelineError) as error:
        client.create_feedback(run_id=UUID(int=10), key="score", score=1, feedback_id=UUID(int=20),
                               session_id=UUID(int=30), trace_id=UUID(int=10))
    output = str(error.value) + caplog.text
    assert "server detail omitted" in output
    for private in ("private trajectory", "private message", "unit-test-secret", "a@b.com", "123-45-6789"):
        assert private not in output


def test_real_sdk_batch_exhaustion_is_raised_instead_of_swallowed(client, monkeypatch, caplog):
    calls = []

    def send(method, url, **kwargs):
        if url.endswith('/info'):
            return response(200, {"version": "0.16.0", "instance_flags": {}})
        assert url.endswith('/runs/batch')
        calls.append(1)
        return response(429, {"detail": "Monthly trace usage limit exceeded."}, {"Retry-After": "2"})

    monkeypatch.setattr(client.session, "request", send)
    run_id = str(UUID(int=10))
    with pytest.raises(PipelineError, match="POST /runs/batch.*429"):
        client.batch_ingest_runs(create=[{
            "id": run_id, "name": "replay", "run_type": "llm", "inputs": {"messages": ["private message"]},
            "trace_id": run_id, "dotted_order": f"20260916T120000000000Z{run_id}",
        }])
    assert len(calls) == 3
    assert "unit-test-secret" not in caplog.text
    assert "private message" not in caplog.text


def test_non_lookup_404_surfaces_safe_endpoint(client, monkeypatch, caplog):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(404, {"detail": "private message"}))
    with pytest.raises(PipelineError, match="POST /feedback.*404"):
        client.create_feedback(run_id=UUID(int=10), key="score", score=1, feedback_id=UUID(int=20),
                               session_id=UUID(int=30), trace_id=UUID(int=10))
    assert "private message" not in caplog.text


def test_native_retry_warning_does_not_log_malformed_header(client, monkeypatch, caplog):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(
        429, {"detail": "Rate limit exceeded."}, {"Retry-After": "unit-test-secret"},
    ))
    with pytest.raises(PipelineError):
        client.request_with_retries("POST", "/runs/batch", stop_after_attempt=3)
    assert "unit-test-secret" not in caplog.text
    assert "using the SDK default delay" in caplog.text

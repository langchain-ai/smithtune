"""Exercise the real SDK HTTP error path without network access."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from uuid import UUID

import pytest
import requests

from smithtune.evaluation import langsmith as reporting
from smithtune.evaluation import langsmith_client as transport
from smithtune.providers.base import PipelineError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "unit-test-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://localhost:9999")
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


def test_rate_limit_exposes_safe_diagnostic_without_sdk_exception_body(client, monkeypatch):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(
        429, {"detail": "Monthly trace usage limit exceeded."}, {"Retry-After": "3600"},
    ))
    with pytest.raises(PipelineError, match="POST /runs/batch.*429.*Monthly trace usage limit exceeded.*3600") as error:
        client.request_with_retries("POST", "/runs/batch?api_key=unit-test-secret")
    assert "unit-test-secret" not in str(error.value)


@pytest.mark.parametrize("header, expected", [("2", 2), ("0", 0), ("invalid", 5.5), ("NaN", 5.5), ("-2", 0)])
def test_429_retries_same_request_then_succeeds(client, monkeypatch, header, expected, capsys):
    calls, sleeps = [], []

    def send(method, url, **kwargs):
        calls.append((method, url, kwargs.get("data")))
        return response(429, {"detail": "Rate limit exceeded."}, {"Retry-After": header}) if len(calls) == 1 else response(200, {})

    monkeypatch.setattr(client.session, "request", send)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    monkeypatch.setattr(transport.random, "uniform", lambda *_: .5)
    assert client.request_with_retries("POST", "/runs/batch", request_kwargs={"data": b'{}'}).status_code == 200
    assert calls[0] == calls[1]
    assert sleeps == [expected]
    assert "retrying" in capsys.readouterr().err


def test_retry_after_http_date():
    now = datetime.now(UTC)
    delay = transport._retry_after(response(429, {}, {"Retry-After": format_datetime(now + timedelta(seconds=30))}))
    assert 28 <= delay <= 30


def test_sdk_http_adapter_cannot_retry_or_sleep_outside_budget(client):
    for adapter in client.session.adapters.values():
        assert adapter.max_retries.total == 0
        assert not adapter.max_retries.respect_retry_after_header
        assert not adapter.max_retries.is_retry("POST", 429, has_retry_after=True)


@pytest.mark.parametrize("status, attempts", [(429, 4), (401, 1), (403, 1), (500, 1)])
def test_retry_budget_overrides_sdk_attempt_count(client, monkeypatch, status, attempts):
    calls, sleeps = [], []

    def send(*args, **kwargs):
        calls.append(1)
        return response(status, {"detail": "Request rejected."}, {"Retry-After": "1"})

    monkeypatch.setattr(client.session, "request", send)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    with pytest.raises(PipelineError, match=f"HTTP {status}"):
        client.request_with_retries("POST", "/feedback", stop_after_attempt=10)
    assert len(calls) == attempts
    assert sleeps == [1] * (attempts - 1)


def test_diagnostic_redacts_echoed_headers_and_trajectory(client, monkeypatch, capsys):
    private = "Private trajectory text about a customer"
    body = {"post": [{"inputs": {"messages": [{"content": private}]}}]}
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(429, {
        "detail": f"Rate limit exceeded: {private}; unit-test-secret; private-gateway-token",
        "inputs": private,
    }, {"Retry-After": "3600"}))
    with pytest.raises(PipelineError) as error:
        client.request_with_retries("POST", "/runs/batch?token=unit-test-secret", request_kwargs={
            "data": json.dumps(body).encode(), "headers": {"Authorization": "private-gateway-token"},
        })
    output = str(error.value) + capsys.readouterr().err
    assert "Rate limit exceeded" in output
    assert "[redacted]" in output
    for value in (private, "unit-test-secret", "private-gateway-token", "?token"):
        assert value not in output


def test_real_sdk_batch_failure_is_raised_instead_of_swallowed(client, monkeypatch, caplog):
    def send(method, url, **kwargs):
        if url.endswith('/info'):
            return response(200, {"version": "0.16.0", "instance_flags": {}})
        assert url.endswith('/runs/batch')
        return response(429, {"detail": "Monthly trace usage limit exceeded."}, {"Retry-After": "3600"})

    monkeypatch.setattr(client.session, "request", send)
    run_id = str(UUID(int=10))
    with pytest.raises(PipelineError, match="POST /runs/batch.*429"):
        client.batch_ingest_runs(create=[{
            "id": run_id, "name": "replay", "run_type": "llm", "inputs": {"messages": ["private message"]},
            "trace_id": run_id, "dotted_order": f"20260916T120000000000Z{run_id}",
        }])
    assert "unit-test-secret" not in caplog.text
    assert "private message" not in caplog.text


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("echo", ["My private email is a@b.com", "a@b.com", "123-45-6789"])
def test_partial_input_echo_is_omitted_for_both_sdk_argument_styles(client, monkeypatch, direct, echo):
    private = 'My private email is a@b.com and identifier is 123-45-6789 and here is more conversation'
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(429, {
        "detail": f"Failed input: {echo}",
    }, {"Retry-After": "3600"}))
    args = {"data": json.dumps({"inputs": {"content": private}}).encode()}
    with pytest.raises(PipelineError) as error:
        client.request_with_retries("POST", "/sessions", **(args if direct else {"request_kwargs": args}))
    assert "echoes request data" in str(error.value)
    assert "private email" not in str(error.value)
    assert "a@b.com" not in str(error.value)
    assert "123-45-6789" not in str(error.value)


def test_non_lookup_404_surfaces_safe_endpoint(client, monkeypatch, caplog):
    monkeypatch.setattr(client.session, "request", lambda *a, **k: response(404, {
        "detail": "Endpoint not found: unit-test-secret",
    }))
    with pytest.raises(PipelineError, match="POST /feedback.*404"):
        client.create_feedback(run_id=UUID(int=10), key="score", score=1, feedback_id=UUID(int=20),
                               session_id=UUID(int=30), trace_id=UUID(int=10))
    assert "unit-test-secret" not in caplog.text

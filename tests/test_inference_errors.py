"""Failure summaries retain actionable status without provider-controlled text."""

import http.client
import io
import json
import urllib.error

import pytest

from smithtune.inference import _post_json, safe_inference_error
from smithtune.providers.base import PipelineError


@pytest.mark.parametrize(("body", "status", "expected"), [
    ({"error": {"code": "insufficient_quota"}}, 429, "provider quota or credit exhausted (HTTP 429)"),
    ({"error": {"code": "rate_limit_exceeded"}}, 429, "provider rate limit exceeded (HTTP 429)"),
    ({"error": {"message": "private data"}}, 429, "provider rate or quota limit exceeded (HTTP 429)"),
    ({"error": {"code": ["private data"]}}, 402, "payment required (HTTP 402)"),
    ({}, 401, "authentication failed (HTTP 401)"),
    ({}, 403, "access denied (HTTP 403)"),
    ({}, 503, "provider request failed (HTTP 503)"),
    (["private data"], 500, "provider request failed (HTTP 500)"),
])
def test_chained_transport_error_summary(body, status, expected):
    response = io.BytesIO(json.dumps(body).encode())

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.invalid/private", status, "private data", {}, response)

    with pytest.raises(PipelineError) as failure:
        _post_json(urllib.request.Request("https://example.invalid"), "judge", opener=reject)
    summary = safe_inference_error(failure.value)
    assert summary == "HTTPError: " + expected
    assert "private" not in summary
    assert response.closed


@pytest.mark.parametrize("payload", [b"not JSON: private data", b'"private data"', b"x" * 9000])
def test_unrecognized_http_body_is_not_printed(payload):
    error = urllib.error.HTTPError("https://example.invalid/private", 500, "private data", {}, io.BytesIO(payload))
    assert safe_inference_error(error) == "HTTPError: provider request failed (HTTP 500)"


def test_sdk_status_and_structured_code_are_preserved():
    class SDKError(Exception):
        def __init__(self):
            super().__init__("private data")
            self.status_code = 429
            self.body = {"code": "insufficient_quota", "message": "private data"}

    assert safe_inference_error(SDKError()) == (
        "SDKError: provider quota or credit exhausted (HTTP 429)"
    )


def test_interrupted_error_body_retains_http_status():
    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"private data")

    error = urllib.error.HTTPError("https://example.invalid/private", 503, "private data", {}, BrokenResponse())
    assert safe_inference_error(error) == "HTTPError: provider request failed (HTTP 503)"


@pytest.mark.parametrize(("error", "expected"), [
    (TimeoutError("private data"), "TimeoutError: provider request timed out"),
    (urllib.error.URLError(TimeoutError("private data")), "TimeoutError: provider request timed out"),
    (urllib.error.URLError("private data"), "URLError: provider connection failed"),
    (ValueError("private data"), "ValueError"),
])
def test_other_failures_do_not_echo_arbitrary_messages(error, expected):
    outer = PipelineError("private data")
    outer.__cause__ = error
    summary = safe_inference_error(outer)
    assert summary == expected
    assert "private" not in summary


def test_exception_cycle_does_not_block_reporting():
    error = RuntimeError("private data")
    error.__cause__ = error
    assert safe_inference_error(error) == "RuntimeError"

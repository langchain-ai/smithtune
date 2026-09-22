import io
import json
import subprocess
import sys
import traceback
import urllib.request
from email.message import Message
from urllib.response import addinfourl

import pytest

from smithtune import dataset, inference
from smithtune.inference_contract import ContractError, InferenceContract
from smithtune.providers import fireworks
from smithtune.providers.base import PipelineError


PRIVATE = "private-trajectory-marker"


def contract_for(schema):
    return InferenceContract(
        tools=({"type": "function", "function": {"name": "lookup", "parameters": schema}},),
        tools_sha256="", system_prompt=None, system_prompt_sha256=None,
        provenance={}, inference_settings={}, contract_sha256="",
    )


@pytest.mark.parametrize(("schema", "arguments", "keyword"), [
    ({"type": "string"}, {PRIVATE: PRIVATE}, "type"),
    ({"enum": ["allowed"]}, PRIVATE, "enum"),
    ({"type": "string", "pattern": "^allowed$"}, PRIVATE, "pattern"),
    ({"type": "object", "additionalProperties": False}, {PRIVATE: PRIVATE}, "additionalProperties"),
    ({"type": "object", "additionalProperties": {"type": "integer"}}, {PRIVATE: PRIVATE}, "type"),
    ({"type": "object", "required": [PRIVATE]}, {}, "required"),
    (False, PRIVATE, "validation"),
])
def test_contract_errors_never_echo_values_or_dynamic_keys(schema, arguments, keyword):
    contract = contract_for(schema)
    messages = [{"role": "assistant", "tool_calls": [{"function": {
        "name": "lookup", "arguments": json.dumps(arguments),
    }}]}]
    with pytest.raises(PipelineError) as error:
        inference._chat_request_body("model", messages, 16, request_contract=contract)
    assert str(error.value) == (
        f"cannot build inference request: arguments for tool at index 0 do not match its JSON Schema ({keyword})"
    )
    assert PRIVATE not in "".join(traceback.format_exception(error.value))


def test_schema_resolution_error_hides_reference_and_cause():
    contract = contract_for({"$ref": f"https://example.invalid/{PRIVATE}"})
    with pytest.raises(ContractError) as error:
        contract.validate_tool_arguments("lookup", {})
    assert "external retrieval is disabled" in str(error.value)
    assert PRIVATE not in "".join(traceback.format_exception(error.value))


def test_validator_failure_hides_exception(monkeypatch):
    class BrokenValidator:
        def __init__(self, *args, **kwargs):
            pass

        def iter_errors(self, arguments):
            raise ValueError(arguments)

    monkeypatch.setattr("smithtune.inference_contract.validator_for", lambda _: BrokenValidator)
    with pytest.raises(ContractError) as error:
        contract_for({}).validate_tool_arguments("lookup", PRIVATE)
    assert str(error.value) == "cannot validate arguments for tool at index 0"
    assert PRIVATE not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_subprocess_output_is_captured_before_sanitizing(capture, stream, capfd):
    command = [sys.executable, "-c", f"import sys; print({PRIVATE!r}, file=sys.{stream}); sys.exit(1)"]
    with pytest.raises(PipelineError) as error:
        dataset._run_langsmith(command, capture=capture)
    assert str(error.value) == "langsmith failed: exited with status 1"
    assert PRIVATE not in "".join(traceback.format_exception(error.value))
    output = capfd.readouterr()
    assert not output.out and not output.err


@pytest.mark.parametrize("capture", [False, True])
def test_successful_subprocess_capture_contract(capture, capfd):
    command = [sys.executable, "-c", f"print({PRIVATE!r})"]
    result = dataset._run_langsmith(command, capture=capture)
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
    assert result.stdout == (PRIVATE + "\n" if capture else None)
    assert result.stderr == ("" if capture else None)
    output = capfd.readouterr()
    assert not output.out and not output.err


def test_subprocess_preserves_stdin_without_echoing_it(capfd):
    command = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"]
    result = dataset._run_langsmith(command, capture=True, input=PRIVATE)
    assert result.stdout == PRIVATE
    output = capfd.readouterr()
    assert not output.out and not output.err


def test_trajectory_tool_reads_sanitize_both_error_streams(monkeypatch, capsys):
    monkeypatch.setattr("smithtune.triage_source.time.sleep", lambda _: None)
    monkeypatch.setattr("smithtune.curation._sleep", lambda _: None)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr=PRIVATE,
                                           output=f"HTTP 504: Client.Timeout exceeded; {PRIVATE}")
    example = {"id": "example-123", "metadata": {"source_scope": "thread", "source_scope_id": "thread-123",
                                                 "source_project_id": "project-123"}}
    with pytest.raises(PipelineError) as error:
        dataset.capture_example_bindings([example], "workspace-123", runner=fail)
    assert "HTTP 504; request timed out" in str(error.value)
    assert "example example-123" in str(error.value)
    assert PRIVATE not in "".join(traceback.format_exception(error.value))
    output = capsys.readouterr()
    assert PRIVATE not in output.out + output.err


@pytest.fixture
def fireworks_transport(monkeypatch):
    requests = []
    response_status = [200]
    location = ["https://redirect.example.invalid/collect"]
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-only-key")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "test-only-session")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})

    def open_request(handler, request):
        requests.append(request)
        headers = Message()
        headers["Location"] = location[0]
        body = {"choices": [{"message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}]}
        response = addinfourl(io.BytesIO(json.dumps(body).encode()), headers, request.full_url, response_status[0])
        response.msg = "test response"
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", open_request)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", open_request)
    return requests, response_status, location


def call_fireworks(entrypoint):
    if entrypoint == "smoke":
        return fireworks._inference_smoke_test("model")
    return inference._fireworks_chat_completion("model", [{"role": "user", "content": "hello"}], 16)


@pytest.mark.parametrize("entrypoint", ["smoke", "inference"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("target", ["https://redirect.example.invalid/collect", "http://redirect.example.invalid/collect"])
def test_fireworks_refuses_redirects_without_forwarding_credentials(fireworks_transport, entrypoint, status, target):
    requests, response_status, location = fireworks_transport
    response_status[0] = status
    location[0] = target
    with pytest.raises(PipelineError, match=f"failed with HTTP {status}"):
        call_fireworks(entrypoint)
    assert len(requests) == 1
    assert requests[0].full_url == fireworks.INFERENCE_URL
    assert requests[0].get_header("Authorization") == "Bearer test-only-key"


@pytest.mark.parametrize("entrypoint", ["smoke", "inference"])
def test_fireworks_non_redirect_response_still_works(fireworks_transport, entrypoint):
    requests, _, _ = fireworks_transport
    result = call_fireworks(entrypoint)
    assert len(requests) == 1
    assert requests[0].get_method() == "POST"
    assert result == ({"http_status": 200, "finish_reason": "stop"} if entrypoint == "smoke"
                      else {"role": "assistant", "content": "ready"})

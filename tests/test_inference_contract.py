from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from smithtune import inference_contract


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
TOOLS_SHA256 = "e80fa6320e88f14fe6aece5340b20bd3445f7cb6c36cac78b8bee20cfc5702ed"
SYSTEM_PROMPT_SHA256 = "823412d1eacb67956220e532959f0104603057c88704863ca38e7cd188fda812"


def contract_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "main_model_inference_contract",
        "tools": copy.deepcopy(TOOLS),
        "tools_sha256": TOOLS_SHA256,
        "system_prompt": {
            "role": "system",
            "content": "policy",
            "sha256": SYSTEM_PROMPT_SHA256,
        },
        "provenance": {
            "source_workspace_id": "workspace-id",
            "source_run_id": "run-id",
            "source_trace_id": "trace-id",
        },
        "inference_settings": {
            "temperature": 0.2,
            "top_p": 0.9,
            "parallel_tool_calls": True,
        },
    }


def write_contract(path: Path, payload: dict | None = None) -> None:
    path.write_text(json.dumps(payload or contract_payload()), encoding="utf-8")


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "find x"}],
    [{"role": "system", "content": "different recorded policy"}, {"role": "user", "content": "find x"}],
    [{"role": "system", "content": [{"type": "text", "text": "recorded policy"}]},
     {"role": "system", "content": "additional instructions"}, {"role": "user", "content": "find x"}],
])
@pytest.mark.parametrize("legacy", [False, True])
def test_contract_builds_tool_aware_request_without_mutating_messages(tmp_path: Path, messages, legacy):
    path = tmp_path / "inference_contract.json"
    payload = contract_payload()
    if not legacy:
        del payload["system_prompt"]
    write_contract(path, payload)
    contract = inference_contract.load_inference_contract(path)
    original = copy.deepcopy(messages)

    request = contract.build_fireworks_request(
        model="accounts/fireworks/models/model",
        messages=messages,
        max_tokens=512,
    )

    assert request == {
        "model": "accounts/fireworks/models/model",
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.2,
        "top_p": 0.9,
        "parallel_tool_calls": True,
        "max_tokens": 512,
    }
    assert messages == original
    assert request["messages"] == original
    assert request["messages"] is not messages
    assert contract.tools_sha256 == TOOLS_SHA256
    assert contract.system_prompt_sha256 == (SYSTEM_PROMPT_SHA256 if legacy else None)
    assert len(contract.contract_sha256) == 64

    base_request = contract.build_fireworks_request(
        model="accounts/fireworks/models/base",
        messages=messages,
        max_tokens=512,
    )
    tuned_request = contract.build_fireworks_request(
        model="accounts/example/deployments/tuned",
        messages=messages,
        max_tokens=512,
    )
    assert {key: value for key, value in base_request.items() if key != "model"} == {
        key: value for key, value in tuned_request.items() if key != "model"
    }


def test_contract_rejects_tool_hash_mismatch(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    payload = contract_payload()
    payload["tools_sha256"] = "0" * 64
    write_contract(path, payload)

    with pytest.raises(inference_contract.ContractError, match="tools_sha256"):
        inference_contract.load_inference_contract(path)


@pytest.mark.parametrize("legacy", [False, True])
def test_contract_round_trip_preserves_declared_hash(tmp_path: Path, legacy):
    payload = contract_payload()
    semantic = {"schema_version": 1, "tools_sha256": TOOLS_SHA256,
                "inference_settings": payload["inference_settings"]}
    if legacy:
        semantic["system_prompt_sha256"] = SYSTEM_PROMPT_SHA256
    else:
        del payload["system_prompt"]
    payload["contract_sha256"] = inference_contract.json_sha256(semantic)
    path = tmp_path / "contract.json"
    write_contract(path, payload)
    contract = inference_contract.load_inference_contract(path)
    assert contract.contract_sha256 == payload["contract_sha256"]
    assert contract.to_dict() == payload
    assert ("system_prompt_sha256" in contract.manifest_summary()) is legacy
    write_contract(path, contract.to_dict())
    assert inference_contract.load_inference_contract(path) == contract

    payload["contract_sha256"] = "0" * 64
    write_contract(path, payload)
    with pytest.raises(inference_contract.ContractError, match="contract_sha256"):
        inference_contract.load_inference_contract(path)


def test_legacy_prompt_metadata_still_checks_its_own_integrity(tmp_path: Path):
    payload = contract_payload()
    payload["system_prompt"]["content"] = "edited without updating its hash"
    path = tmp_path / "contract.json"
    write_contract(path, payload)
    with pytest.raises(inference_contract.ContractError, match="system_prompt.sha256"):
        inference_contract.load_inference_contract(path)


def test_contract_rejects_provider_specific_or_secret_inference_settings(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    payload = contract_payload()
    payload["inference_settings"]["api_key"] = "must-not-be-copied"
    write_contract(path, payload)

    with pytest.raises(inference_contract.ContractError, match="unsupported inference setting"):
        inference_contract.load_inference_contract(path)


def test_contract_rejects_invalid_tool_parameter_schema(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    payload = contract_payload()
    payload["tools"][0]["function"]["parameters"] = {
        "type": "definitely-not-a-json-schema-type"
    }
    payload["tools_sha256"] = inference_contract.json_sha256(payload["tools"])
    write_contract(path, payload)

    with pytest.raises(inference_contract.ContractError, match="invalid JSON Schema"):
        inference_contract.load_inference_contract(path)


def test_contract_validates_recorded_tool_names_and_arguments(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    write_contract(path)
    contract = inference_contract.load_inference_contract(path)

    contract.validate_messages(
        [
            {"role": "system", "content": "policy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"x"}',
                        },
                    }
                ],
            },
        ]
    )

    unknown = [
        {"role": "system", "content": "policy"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "missing", "arguments": "{}"},
                }
            ],
        },
    ]
    with pytest.raises(inference_contract.ContractError, match="unknown tool missing"):
        contract.validate_messages(unknown)

    invalid_arguments = copy.deepcopy(unknown)
    invalid_arguments[1]["tool_calls"][0]["function"] = {
        "name": "lookup",
        "arguments": '{"query":1}',
    }
    with pytest.raises(inference_contract.ContractError, match="do not match its JSON Schema"):
        contract.validate_messages(invalid_arguments)


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize("reference", ["https://schemas.example.invalid/query.json", "query.json", "file:///schema.json"])
def test_tool_schema_never_fetches_external_references(tmp_path, monkeypatch, keyword, reference):
    fetch = Mock(side_effect=AssertionError("external schema retrieval attempted"))
    monkeypatch.setattr("urllib.request.urlopen", fetch)
    payload = contract_payload()
    payload["tools"][0]["function"]["parameters"] = {
        "$id": "https://schemas.example.invalid/tool.json",
        "properties": {"query": {keyword: reference}},
    }
    payload["tools_sha256"] = inference_contract.json_sha256(payload["tools"])
    path = tmp_path / "contract.json"
    write_contract(path, payload)
    contract = inference_contract.load_inference_contract(path)
    with pytest.raises(inference_contract.ContractError, match="tool lookup.*external retrieval is disabled"):
        contract.validate_tool_arguments("lookup", {"query": "x"})
    fetch.assert_not_called()


def test_tool_schema_resolves_saved_definitions_offline(tmp_path, monkeypatch):
    fetch = Mock(side_effect=AssertionError("external schema retrieval attempted"))
    monkeypatch.setattr("urllib.request.urlopen", fetch)
    payload = contract_payload()
    payload["tools"][0]["function"]["parameters"] = {
        "$id": "https://schemas.example.invalid/tool.json",
        "$defs": {"query": {"type": "string"}},
        "properties": {"query": {"$ref": "#/$defs/query"}},
    }
    payload["tools_sha256"] = inference_contract.json_sha256(payload["tools"])
    path = tmp_path / "contract.json"
    write_contract(path, payload)
    contract = inference_contract.load_inference_contract(path)
    contract.validate_tool_arguments("lookup", {"query": "x"})
    with pytest.raises(inference_contract.ContractError, match="do not match its JSON Schema"):
        contract.validate_tool_arguments("lookup", {"query": 1})
    fetch.assert_not_called()


@pytest.mark.parametrize("inputs", [None,
    {"messages": [{"role": "human", "content": "find x"}]},
    {"messages": [[{"role": "system", "content": "recorded prompt must not be captured"}]]},
])
def test_contract_capture_does_not_require_or_copy_message_inputs(tmp_path: Path, inputs):
    run = {
        "id": "run-id",
        "trace_id": "trace-id",
        "session_id": "project-id",
        "name": "ChatModel",
        "run_type": "llm",
        "start_time": "2026-09-03T12:00:00Z",
        "extra": {
            "metadata": {"ls_provider": "provider", "ls_model_name": "model"},
            "invocation_params": {
                "model": "model",
                "temperature": 0.2,
                "top_p": 0.9,
                "api_key": "must-not-be-copied",
                "tools": copy.deepcopy(TOOLS),
            },
        },
    }

    if inputs is not None:
        run["inputs"] = inputs
    original = copy.deepcopy(run)
    payload = inference_contract.contract_from_run(run, workspace_id="workspace-id")
    path = tmp_path / "captured.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = inference_contract.load_inference_contract(path)

    assert contract.tools == tuple(TOOLS)
    assert run == original
    assert contract.system_prompt is None
    assert contract.system_prompt_sha256 is None
    assert "system_prompt" not in payload
    assert "recorded prompt" not in json.dumps(payload)
    assert contract.to_dict() == payload
    assert contract.inference_settings == {"temperature": 0.2, "top_p": 0.9}
    assert "api_key" not in json.dumps(payload)
    assert payload["provenance"] == {
        "source_workspace_id": "workspace-id",
        "source_project_id": "project-id",
        "source_run_id": "run-id",
        "source_trace_id": "trace-id",
        "source_run_name": "ChatModel",
        "source_run_start_time": "2026-09-03T12:00:00Z",
        "model_provider": "provider",
        "model_name": "model",
    }


def test_contract_capture_requires_an_llm_run_with_tools():
    with pytest.raises(inference_contract.ContractError, match="LLM run"):
        inference_contract.contract_from_run(
            {"run_type": "chain", "extra": {"invocation_params": {"tools": TOOLS}}},
            workspace_id="workspace-id",
        )

    with pytest.raises(inference_contract.ContractError, match="invocation_params.tools"):
        inference_contract.contract_from_run(
            {
                "run_type": "llm",
                "inputs": {"messages": [[{"role": "system", "content": "policy"}]]},
                "extra": {"invocation_params": {}},
            },
            workspace_id="workspace-id",
        )

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import inference_contract


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


def test_contract_builds_tool_aware_request_without_mutating_system_message(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    write_contract(path)
    contract = inference_contract.load_inference_contract(path)
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "find x"},
    ]

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
    assert request["messages"][0]["content"] == "policy"
    assert contract.tools_sha256 == TOOLS_SHA256
    assert contract.system_prompt_sha256 == SYSTEM_PROMPT_SHA256
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


def test_contract_rejects_hash_mismatch_and_system_prompt_drift(tmp_path: Path):
    path = tmp_path / "inference_contract.json"
    payload = contract_payload()
    payload["tools_sha256"] = "0" * 64
    write_contract(path, payload)

    with pytest.raises(inference_contract.ContractError, match="tools_sha256"):
        inference_contract.load_inference_contract(path)

    write_contract(path)
    contract = inference_contract.load_inference_contract(path)
    with pytest.raises(inference_contract.ContractError, match="system prompt"):
        contract.build_fireworks_request(
            model="accounts/fireworks/models/model",
            messages=[{"role": "system", "content": "different"}],
            max_tokens=512,
        )


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


def test_contract_can_be_created_from_an_approved_llm_run(tmp_path: Path):
    run = {
        "id": "run-id",
        "trace_id": "trace-id",
        "session_id": "project-id",
        "name": "ChatModel",
        "run_type": "llm",
        "start_time": "2026-09-03T12:00:00Z",
        "inputs": {
            "messages": [
                [
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "schema", "messages", "SystemMessage"],
                        "kwargs": {"content": "policy", "type": "system"},
                    },
                    {
                        "lc": 1,
                        "type": "constructor",
                        "id": ["langchain", "schema", "messages", "HumanMessage"],
                        "kwargs": {"content": "find x", "type": "human"},
                    },
                ]
            ]
        },
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

    payload = inference_contract.contract_from_run(run, workspace_id="workspace-id")
    path = tmp_path / "captured.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = inference_contract.load_inference_contract(path)

    assert contract.tools == tuple(TOOLS)
    assert contract.system_prompt["content"] == "policy"
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

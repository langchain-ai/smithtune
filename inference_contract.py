"""Validated inference contracts and provider request construction."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for


class ContractError(ValueError):
    """The inference contract or a request built from it is invalid."""


ALLOWED_INFERENCE_SETTINGS = {
    "frequency_penalty",
    "parallel_tool_calls",
    "presence_penalty",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "top_p",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def content_sha256(content: Any) -> str:
    if isinstance(content, str):
        encoded = content.encode()
    else:
        encoded = canonical_json(content).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_tools(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ContractError("tools must be an array")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, source in enumerate(value):
        if not isinstance(source, Mapping) or source.get("type") != "function":
            raise ContractError(f"tool {index} must be an OpenAI-style function tool")
        function = source.get("function")
        if not isinstance(function, Mapping):
            raise ContractError(f"tool {index} has no function object")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ContractError(f"tool {index} has no function name")
        if name in names:
            raise ContractError(f"duplicate tool name: {name}")
        if not isinstance(parameters, Mapping):
            raise ContractError(f"tool {name} has no JSON Schema parameters object")
        try:
            validator_for(parameters).check_schema(parameters)
        except SchemaError as exc:
            raise ContractError(f"tool {name} has an invalid JSON Schema: {exc.message}") from exc
        names.add(name)
        tools.append(copy.deepcopy(dict(source)))
    return tuple(tools)


def _canonicalize_captured_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError("LLM run has no extra.invocation_params.tools")
    canonical: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ContractError("captured tool definitions must be objects")
        declarations = entry.get("function_declarations")
        if isinstance(declarations, list):
            canonical.extend(_canonicalize_captured_tools(declarations))
            continue
        if entry.get("type") == "function" and isinstance(entry.get("function"), Mapping):
            canonical.append(copy.deepcopy(dict(entry)))
            continue
        name = entry.get("name")
        parameters = entry.get("parameters", entry.get("input_schema"))
        if isinstance(name, str) and isinstance(parameters, Mapping):
            function = {"name": name, "parameters": copy.deepcopy(dict(parameters))}
            if isinstance(entry.get("description"), str):
                function["description"] = entry["description"]
            canonical.append({"type": "function", "function": function})
            continue
        raise ContractError("captured tool definition cannot be normalized")
    _validate_tools(canonical)
    return canonical


def _validate_settings(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError("inference_settings must be an object")
    unsupported = sorted(set(value) - ALLOWED_INFERENCE_SETTINGS)
    if unsupported:
        raise ContractError(f"unsupported inference setting: {unsupported[0]}")
    return {key: copy.deepcopy(item) for key, item in value.items() if item is not None}


@dataclass(frozen=True)
class InferenceContract:
    tools: tuple[dict[str, Any], ...]
    tools_sha256: str
    # Legacy prompt metadata only; retained for artifact and hash compatibility.
    system_prompt: dict[str, Any] | None
    system_prompt_sha256: str | None
    provenance: dict[str, Any]
    inference_settings: dict[str, Any]
    contract_sha256: str

    def validate_tool_arguments(self, name: str, arguments: Any) -> None:
        tool = next(
            (item for item in self.tools if item["function"]["name"] == name),
            None,
        )
        if tool is None:
            raise ContractError(f"unknown tool {name}")
        schema = tool["function"]["parameters"]
        validator = validator_for(schema)(schema)
        try:
            errors = sorted(
                validator.iter_errors(arguments),
                key=lambda error: tuple(str(part) for part in error.path),
            )
        except Exception as exc:
            raise ContractError(f"cannot validate arguments for tool {name}: {exc}") from exc
        if errors:
            raise ContractError(
                f"arguments for tool {name} do not match its JSON Schema: {errors[0].message}"
            )

    def validate_messages(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Validate recorded tool calls without imposing a system prompt."""
        for message in messages:
            tool_calls = message.get("tool_calls")
            if tool_calls is None:
                continue
            if not isinstance(tool_calls, list):
                raise ContractError("message tool_calls must be an array")
            for call in tool_calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping):
                    raise ContractError("tool call has no function object")
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name:
                    raise ContractError("tool call has no function name")
                if not isinstance(arguments, str):
                    raise ContractError(f"arguments for tool {name} must be JSON text")
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"arguments for tool {name} are not valid JSON") from exc
                self.validate_tool_arguments(name, parsed)

    def build_fireworks_request(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        self.validate_messages(messages)
        if not isinstance(model, str) or not model:
            raise ContractError("model must be a non-empty string")
        if max_tokens < 1:
            raise ContractError("max_tokens must be positive")
        request = copy.deepcopy(self.inference_settings)
        request.update(
            {
                "model": model,
                "messages": copy.deepcopy(list(messages)),
                "tools": copy.deepcopy(list(self.tools)),
                "max_tokens": max_tokens,
            }
        )
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        return request

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "format": "main_model_inference_contract",
            "tools": copy.deepcopy(list(self.tools)),
            "tools_sha256": self.tools_sha256,
            "provenance": copy.deepcopy(self.provenance),
            "inference_settings": copy.deepcopy(self.inference_settings),
            "contract_sha256": self.contract_sha256,
        }
        if self.system_prompt is not None:
            payload["system_prompt"] = copy.deepcopy(self.system_prompt)
        return payload

    def manifest_summary(self) -> dict[str, Any]:
        summary = {
            "schema_version": 1,
            "contract_sha256": self.contract_sha256,
            "tools_sha256": self.tools_sha256,
            "tool_count": len(self.tools),
            "provenance": copy.deepcopy(self.provenance),
            "inference_settings": copy.deepcopy(self.inference_settings),
        }
        if self.system_prompt_sha256 is not None:
            summary["system_prompt_sha256"] = self.system_prompt_sha256
        return summary


def load_inference_contract(path: Path) -> InferenceContract:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid inference contract JSON from {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ContractError("inference contract must be an object")
    if payload.get("schema_version") != 1:
        raise ContractError("inference contract schema_version must be 1")
    if payload.get("format") != "main_model_inference_contract":
        raise ContractError("inference contract format is invalid")

    tools = _validate_tools(payload.get("tools"))
    declared_tools_hash = _require_sha256(payload.get("tools_sha256"), "tools_sha256")
    calculated_tools_hash = json_sha256(list(tools))
    if declared_tools_hash != calculated_tools_hash:
        raise ContractError("tools_sha256 does not match the canonical tool schemas")

    system_prompt = payload.get("system_prompt")
    prompt_hash = None
    if system_prompt is not None:
        if not isinstance(system_prompt, Mapping) or system_prompt.get("role") != "system":
            raise ContractError("system_prompt must be a system message")
        if "content" not in system_prompt:
            raise ContractError("system_prompt must contain content")
        prompt_hash = _require_sha256(system_prompt.get("sha256"), "system_prompt.sha256")
        if prompt_hash != content_sha256(system_prompt["content"]):
            raise ContractError("system_prompt.sha256 does not match system_prompt.content")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ContractError("provenance must be an object")
    settings = _validate_settings(payload.get("inference_settings"))
    semantic_contract = {
        "schema_version": 1,
        "tools_sha256": declared_tools_hash,
        "inference_settings": settings,
    }
    if prompt_hash is not None:
        semantic_contract["system_prompt_sha256"] = prompt_hash
    contract_hash = json_sha256(semantic_contract)
    declared_contract_hash = payload.get("contract_sha256")
    if declared_contract_hash is not None and declared_contract_hash != contract_hash:
        raise ContractError("contract_sha256 does not match the semantic inference contract")
    return InferenceContract(
        tools=tools,
        tools_sha256=declared_tools_hash,
        system_prompt=copy.deepcopy(dict(system_prompt)) if system_prompt is not None else None,
        system_prompt_sha256=prompt_hash,
        provenance=copy.deepcopy(dict(provenance)),
        inference_settings=settings,
        contract_sha256=contract_hash,
    )


def contract_from_run(run: Mapping[str, Any], *, workspace_id: str) -> dict[str, Any]:
    """Capture tool schemas and inference settings from a LangSmith LLM run."""
    if run.get("run_type") != "llm":
        raise ContractError("contract capture requires an LLM run")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise ContractError("source workspace ID is required")
    extra = run.get("extra")
    if not isinstance(extra, Mapping):
        raise ContractError("LLM run has no extra.invocation_params.tools")
    invocation = extra.get("invocation_params")
    if not isinstance(invocation, Mapping):
        raise ContractError("LLM run has no extra.invocation_params.tools")
    tools = _canonicalize_captured_tools(invocation.get("tools"))
    tool_hash = json_sha256(tools)
    settings = {
        key: copy.deepcopy(invocation[key])
        for key in sorted(ALLOWED_INFERENCE_SETTINGS)
        if invocation.get(key) is not None
    }
    metadata = extra.get("metadata") if isinstance(extra.get("metadata"), Mapping) else {}
    provenance_values = {
        "source_workspace_id": workspace_id,
        "source_project_id": run.get("session_id"),
        "source_run_id": run.get("id"),
        "source_trace_id": run.get("trace_id"),
        "source_run_name": run.get("name"),
        "source_run_start_time": run.get("start_time"),
        "model_provider": metadata.get("ls_provider"),
        "model_name": metadata.get("ls_model_name") or invocation.get("model_name") or invocation.get("model"),
    }
    provenance = {key: value for key, value in provenance_values.items() if value is not None}
    semantic_contract = {
        "schema_version": 1,
        "tools_sha256": tool_hash,
        "inference_settings": settings,
    }
    return {
        "schema_version": 1,
        "format": "main_model_inference_contract",
        "tools": tools,
        "tools_sha256": tool_hash,
        "provenance": provenance,
        "inference_settings": settings,
        "contract_sha256": json_sha256(semantic_contract),
    }

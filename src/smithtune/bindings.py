"""Portable per-assistant tool evidence from LangSmith trajectory items."""

from __future__ import annotations

from smithtune.inference_contract import (
    ContractError, _canonicalize_captured_tools, json_sha256, parse_inference_contract,
)
from smithtune.providers.base import PipelineError


SCHEMA_VERSION = 1


def tool_contract(tools, **provenance):
    return parse_inference_contract({"schema_version": 1, "format": "main_model_inference_contract",
                                     "tools": tools, "tools_sha256": json_sha256(tools), "provenance": provenance})


def trajectory_bindings(items):
    """Normalize the availability recorded on each assistant's trajectory item."""
    bindings = []
    for index, item in enumerate(items):
        message = item["message"]
        if message.get("role") not in ("ai", "assistant"):
            continue
        context = f"assistant message {index}"
        metadata = item.get("metadata")
        if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(key), str) or not metadata[key] for key in ("run_id", "trace_id")
        ):
            raise PipelineError(f"{context} has no run/trace provenance in the trajectory response")
        context += f" run {metadata['run_id']}"
        if not isinstance(message.get("available_tools"), list):
            raise PipelineError(f"{context} has unknown tool availability; /v1/trajectory must record available_tools (including [] for no tools)")
        try:
            tools = _canonicalize_captured_tools(message["available_tools"]) if message["available_tools"] else []
            tool_contract(tools)
        except ContractError as exc:
            raise PipelineError(f"{context}: unsupported available_tools: {exc}") from exc
        bindings.append({"message_index": index, "run_id": metadata["run_id"],
                         "trace_id": metadata["trace_id"], "tools": tools})
    return {"schema_version": SCHEMA_VERSION, "assistant_runs": bindings}


def read_bindings(metadata, messages, *, positions=None):
    """Validate coverage and return original-position bindings, without fallback."""
    source = metadata.get("smithtune_source") if isinstance(metadata, dict) else None
    if not isinstance(source, dict) or source.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError("missing per-assistant tool provenance; pull again using trajectory available_tools")
    values = source.get("assistant_runs")
    if not isinstance(values, list):
        raise PipelineError("assistant_runs must be an array")
    result = {}
    for binding in values:
        if not isinstance(binding, dict):
            raise PipelineError("invalid assistant binding")
        index = binding.get("message_index")
        if type(index) is not int or index < 0 or index in result:
            raise PipelineError("duplicate or invalid assistant message position")
        if any(not isinstance(binding.get(key), str) or not binding[key] for key in ("run_id", "trace_id")):
            raise PipelineError(f"assistant message {index} has no producing run/trace identity")
        try:
            tool_contract(binding.get("tools"))
        except ContractError as exc:
            raise PipelineError(f"assistant message {index}: {exc}") from exc
        result[index] = binding
    original = list(range(len(messages))) if positions is None else positions
    if len(original) != len(messages) or any(type(i) is not int for i in original) or original != sorted(set(original)):
        raise PipelineError("invalid original message positions")
    expected = {original[i] for i, m in enumerate(messages) if m.get("role") in ("ai", "assistant")}
    if not expected <= result.keys() or (positions is None and expected != result.keys()):
        raise PipelineError("assistant bindings do not cover exactly the recorded assistant messages")
    if any(original[i] in result for i, m in enumerate(messages) if m.get("role") not in ("ai", "assistant")):
        raise PipelineError("binding points to a non-assistant message")
    return result


def row_bindings(row):
    source = row["_source"]
    return read_bindings(source.get("metadata"), row["messages"], positions=source.get("message_positions"))


def validate_bound_messages(example):
    from smithtune.dataset import validate_import_messages

    messages = validate_import_messages(example)
    bindings = read_bindings(example.get("metadata"), example["inputs"]["messages"])
    for index, binding in bindings.items():
        try:
            tool_contract(binding["tools"]).validate_messages([messages[index]])
        except ContractError as exc:
            raise PipelineError(f"message {index} run {binding['run_id']}: {exc}") from exc
    return bindings


def evidence_hash(example):
    return json_sha256({"messages": example["inputs"]["messages"], "source": example["metadata"]["smithtune_source"]})


def training_targets(row):
    """Expand only in memory at the rendering boundary; keep parent split rows."""
    from smithtune.dataset import _has_message_content

    source = row["_source"]
    override = row.get("tool_policy") == "global_override"
    bindings = None if override else row_bindings(row)
    positions = source["message_positions"]
    for i, message in enumerate(row["messages"]):
        if message.get("role") != "assistant" or not _has_message_content(message):
            continue
        binding = bindings[positions[i]] if bindings is not None else None
        yield {"messages": row["messages"][:i + 1], "tools": row["tools"] if override else binding["tools"],
               "message_index": positions[i], "run_id": binding["run_id"] if binding else None}

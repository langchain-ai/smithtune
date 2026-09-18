"""Verified assistant-output identities and portable per-message tool evidence.

Trajectory IDs are message IDs, never run IDs. Only identities found in an LLM
run's outputs establish production; copies in inputs (including supplied history)
do not. UI trajectory metadata alone is insufficient: it also attributes input
history to the run that consumed it.
"""

from __future__ import annotations

import copy
import json

from smithtune.inference_contract import (
    ContractError, _canonicalize_captured_tools, json_sha256, parse_inference_contract,
)
from smithtune.providers.base import PipelineError


SCHEMA_VERSION = 1


def tool_contract(tools, **provenance):
    return parse_inference_contract({"schema_version": 1, "format": "main_model_inference_contract",
                                     "tools": tools, "tools_sha256": json_sha256(tools), "provenance": provenance})


def _output_messages(outputs):
    """Read documented StandardMessage and LangChain ChatGeneration envelopes.

    Deliberately do not recursively search arbitrary output JSON: nested tool
    results and quoted messages are not evidence of an LLM output identity.
    """
    if not isinstance(outputs, dict):
        return []
    if isinstance(outputs.get("messages"), list):
        values = outputs["messages"]
    elif isinstance(outputs.get("generations"), list):
        values = [item.get("message") for batch in outputs["generations"]
                  for item in (batch if isinstance(batch, list) else [batch])
                  if isinstance(item, dict)]
    elif isinstance(outputs.get("choices"), list):
        values = [item.get("message") for item in outputs["choices"] if isinstance(item, dict)]
    else:
        values = [outputs]
    result = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if value.get("type") == "constructor" and value.get("lc") == 1:
            if (value.get("id") or [])[-1:] != ["AIMessage"]:
                continue
            value = {"type": "ai", **(value.get("kwargs") or {})}
        elif isinstance(value.get("data"), dict):
            value = {"type": value.get("type"), **value["data"]}
        if value.get("role", value.get("type")) in {"ai", "assistant"}:
            result.append(value)
    return result


def _action(message):
    """Compare visible output across StandardMessage, LangChain and provider formats."""
    raw = message.get("content")
    if raw is None:
        raw = []
    elif isinstance(raw, str):
        raw = [{"type": "text", "text": raw}] if raw else []
    if not isinstance(raw, list):
        raise PipelineError("output content cannot be verified")
    calls = message.get("tool_calls") or (message.get("additional_kwargs") or {}).get("tool_calls")
    content = []
    for block in raw:
        if not isinstance(block, dict):
            raise PipelineError("output content block cannot be verified")
        kind = block.get("type")
        if kind in {"text", "output_text"}:
            if block.get("text"):
                content.append({"type": "text", "text": block["text"]})
        elif kind in {"tool_call", "tool_use", "function_call"}:
            if calls:
                continue  # Parsed calls supersede partial native blocks, as in the trajectory normalizer.
            args = block.get("args") if kind == "tool_call" else block.get("input")
            call_id = block.get("id")
            if kind == "function_call":
                args = json.loads(block["arguments"])
                call_id = block.get("call_id") or call_id
            content.append({"type": "tool_call", "id": call_id, "name": block.get("name"), "args": args})
        elif kind in {"reasoning", "thinking"}:
            summary = block.get("summary")
            text = "\n".join(item["text"] for item in summary if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"]) if isinstance(summary, list) else ""
            if kind == "reasoning":
                text = text or block.get("reasoning") or block.get("text") or ""
            else:
                text = block.get("thinking") or ""
            content.append({"type": "reasoning", "reasoning": text})
        else:
            content.append(copy.deepcopy(block))
    if calls:
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            raise PipelineError("output tool calls cannot be verified")
        for call in calls:
            if "function" in call:
                fn = call["function"]
                if not isinstance(fn, dict):
                    raise PipelineError("output tool call function cannot be verified")
                args = fn.get("arguments")
                args = json.loads(args) if isinstance(args, str) else args
                value = {"type": "tool_call", "id": call.get("id"), "name": fn.get("name"), "args": args}
            else:
                value = {"type": "tool_call", "id": call.get("id"), "name": call.get("name"), "args": call.get("args")}
            content.append(value)
    return content


class BindingCapture:
    """Keep only fingerprints and offered tools for relevant producing outputs."""

    def __init__(self, messages):
        self.messages = messages
        self.wanted = {m.get("id") for m in messages if m.get("role") in {"ai", "assistant"} and isinstance(m.get("id"), str) and m["id"]}
        self.outputs, self.seen_runs = {}, {}

    def add(self, runs):
        for run in runs:
            if str(run.get("run_type", "")).lower() != "llm":
                continue
            rid = run.get("id")
            if not isinstance(rid, str) or not rid or not run.get("trace_id"):
                raise PipelineError("LLM evidence is missing run/trace identity")
            fingerprint = json_sha256(run)
            if rid in self.seen_runs:
                if self.seen_runs[rid] != fingerprint:
                    raise PipelineError(f"run {rid} changed during capture")
                continue
            self.seen_runs[rid] = fingerprint
            for message in _output_messages(run.get("outputs")):
                mid = message.get("id")
                if not isinstance(mid, str) or mid not in self.wanted:
                    continue
                candidate = {"run_id": rid, "trace_id": run["trace_id"]}
                try:
                    candidate["output_sha256"] = json_sha256(_action(message))
                    extra = run.get("extra")
                    params = extra.get("invocation_params") if isinstance(extra, dict) else None
                    if not isinstance(params, dict) or not isinstance(params.get("tools"), list):
                        candidate["unknown_tools"] = True
                    else:
                        tools = _canonicalize_captured_tools(params["tools"]) if params["tools"] else []
                        tool_contract(tools)
                        candidate["tools"] = tools
                except (ContractError, ValueError, TypeError, KeyError, PipelineError) as exc:
                    candidate["error"] = str(exc)
                self.outputs.setdefault(mid, []).append(candidate)

    def finish(self):
        bindings, occurrences = [], set()
        for index, message in enumerate(self.messages):
            if message.get("role") not in {"ai", "assistant"}:
                continue
            mid = message.get("id")
            matches = self.outputs.get(mid, []) if isinstance(mid, str) else []
            candidates = [candidate["run_id"] for candidate in matches]
            context = f"message {index} ({mid!r}); candidate runs {candidates}"
            if len(matches) != 1 or mid in occurrences:
                raise PipelineError(f"missing or ambiguous producing-run provenance for {context}; supply unique stable output-message IDs and the producing LLM runs")
            occurrences.add(mid)
            candidate = matches[0]
            try:
                if candidate.get("error"):
                    raise ContractError(candidate["error"])
                if json_sha256(_action(message)) != candidate["output_sha256"]:
                    raise PipelineError(f"recorded message differs from producing output for {context}")
                if candidate.get("unknown_tools"):
                    raise PipelineError(f"unknown tool availability for {context}; record invocation_params.tools (including [] for no tools)")
            except (ContractError, ValueError, TypeError, KeyError) as exc:
                raise PipelineError(f"unsupported output/tool evidence for {context}: {exc}") from exc
            bindings.append({"message_index": index, "run_id": candidate["run_id"],
                             "trace_id": candidate["trace_id"], "tools": candidate["tools"]})
        return {"schema_version": SCHEMA_VERSION, "assistant_runs": bindings}


def capture_bindings(messages, runs):
    """Require one stable output identity for every assistant occurrence."""
    capture = BindingCapture(messages)
    capture.add(runs)
    return capture.finish()


def read_bindings(metadata, messages, *, positions=None):
    """Validate coverage and return original-position bindings, without fallback."""
    source = metadata.get("smithtune_source") if isinstance(metadata, dict) else None
    if not isinstance(source, dict) or source.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError("missing per-assistant tool provenance; pull again or capture producing LLM output identities")
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
    expected = {original[i] for i, m in enumerate(messages) if m.get("role") in {"ai", "assistant"}}
    if not expected <= result.keys() or (positions is None and expected != result.keys()):
        raise PipelineError("assistant bindings do not cover exactly the recorded assistant messages")
    if any(original[i] in result for i, m in enumerate(messages) if m.get("role") not in {"ai", "assistant"}):
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

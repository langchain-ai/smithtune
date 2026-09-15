"""Capture and prepare LangSmith trajectory datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _run, _utc_now
from smithtune.inference_contract import (
    ContractError,
    InferenceContract,
    TOOL_MERGE_POLICY,
    contract_from_runs,
    load_inference_contract,
    parse_inference_contract,
    json_sha256,
)
from smithtune.providers.base import ModelSpec, PipelineError, ReasoningPolicy
from smithtune.rendering import validate_model_context, validate_reasoning_support


SPLIT_SEED = 42


DEFAULT_VALIDATION_FRACTION = 0.1


DEFAULT_TEST_FRACTION = 0.1


LANGSMITH_PAGE_SIZE = 100


ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


@dataclass(frozen=True)
class Audit:
    examples: int
    messages: int
    roles: dict[str, int]
    tool_calls: int
    tool_results: int
    duplicate_message_warnings: list[dict[str, Any]]
    rendered_datums: int = 0
    context_tokens: int = 0
    target_tokens: int = 0
    max_context_tokens: int = 0
    reasoning_blocks: int = 0
    readable_reasoning_blocks: int = 0


def _run_langsmith(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return _run(command, capture=capture)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or f"langsmith exited with status {exc.returncode}"
        raise PipelineError(detail) from None


def _langsmith_dataset_command(workspace_id: str, dataset_id: str) -> list[str]:
    return [
        "langsmith", "dataset", "get", dataset_id,
        "--workspace", workspace_id, "--format", "json",
    ]


def _langsmith_export_command(
    workspace_id: str,
    dataset_id: str,
    output: Path,
    count: int,
) -> list[str]:
    return [
        "langsmith", "dataset", "export", dataset_id, str(output),
        "--limit", str(count), "--workspace", workspace_id,
    ]


def _langsmith_page_command(
    workspace_id: str,
    dataset_id: str,
    offset: int,
    limit: int,
) -> list[str]:
    return [
        "langsmith", "api",
        f"/api/v1/examples?dataset={dataset_id}&limit={limit}&offset={offset}",
        "--workspace", workspace_id, "--method", "GET",
    ]


def _contract_read(command: list[str], *, runner: Callable[..., Any]) -> Any:
    for attempt in range(6):
        try:
            result = runner(command, capture=True)
            break
        except PipelineError as exc:
            retryable = re.search(r"\b(HTTP 429|context deadline exceeded|Client\.Timeout exceeded|request timed out|Query timeout exceeded)\b", str(exc), re.I)
            if attempt == 5 or not retryable:
                raise
            delay = min(5 * 2**attempt + random.uniform(0, 1), 60)
            reason = "rate limit reached" if retryable[0].upper() == "HTTP 429" else "request timed out"
            print(f"LangSmith {reason}; retrying in {delay:.1f}s ({attempt + 1}/5)", file=sys.stderr)
            time.sleep(delay)
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise PipelineError("LangSmith run query returned invalid JSON") from exc


def _read_contract_run(workspace_id: str, run_id: str, *, runner: Callable[..., Any]) -> dict[str, Any]:
    # V2 queries require a project. A direct lookup resolves it when only a run ID is known.
    run = _contract_read([
        "langsmith", "api", f"/api/v1/runs/{quote(run_id, safe='')}",
        "--workspace", workspace_id, "--method", "GET",
    ], runner=runner)
    if not isinstance(run, dict) or run.get("id") != run_id:
        raise PipelineError(f"LangSmith did not return the requested run {run_id}")
    return run


def _project_start_time(workspace_id: str, project_id: str, *, runner: Callable[..., Any]) -> str:
    project = _contract_read([
        "langsmith", "api", f"/api/v1/sessions/{quote(project_id, safe='')}",
        "--workspace", workspace_id, "--method", "GET",
    ], runner=runner)
    if not isinstance(project, dict) or project.get("id") != project_id:
        raise PipelineError(f"LangSmith did not return the requested project {project_id}")
    start = project.get("start_time")
    try:
        parsed = datetime.fromisoformat(start)
        if parsed.tzinfo is None:
            raise ValueError("missing timezone")
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"source project {project_id} has no valid start time") from exc
    return start


def _query_contract_runs(
    workspace_id: str, query: dict[str, Any], *, runner: Callable[..., Any],
) -> list[dict[str, Any]]:
    body = {
        "page_size": LANGSMITH_PAGE_SIZE,
        "selects": ["ID", "TRACE_ID", "PROJECT_ID", "NAME", "RUN_TYPE", "START_TIME", "EXTRA", "METADATA"],
        "max_start_time": _utc_now(),
        **query,
    }
    runs: dict[str, dict[str, Any]] = {}
    # V2 defaults to one day and caps each query at 401 days. Walk the full
    # project history in bounded windows, deduplicating boundary runs by ID.
    start = datetime.fromisoformat(body["min_start_time"])
    end = datetime.fromisoformat(body["max_start_time"])
    while start <= end:
        window_end = min(start + timedelta(days=400), end)
        body.update(min_start_time=start.isoformat(), max_start_time=window_end.isoformat())
        body.pop("cursor", None)
        cursors: set[str] = set()
        while True:
            response = _contract_read([
                "langsmith", "api", "/api/v2/runs/query", "--workspace", workspace_id,
                "--method", "POST", "--body", _canonical(body),
            ], runner=runner)
            if not isinstance(response, dict) or not isinstance(response.get("items"), list):
                raise PipelineError("LangSmith returned an invalid run query page")
            for item in response["items"]:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                    raise PipelineError("LangSmith returned an invalid run")
                if item.get("project_id") not in body["project_ids"]:
                    raise PipelineError("LangSmith returned a run from a different project")
                # Keep the contract's existing representation independent of API wire names.
                run = {**item, "session_id": item["project_id"]}
                run.pop("project_id")
                if isinstance(run.get("run_type"), str):
                    run["run_type"] = run["run_type"].lower()
                metadata = run.pop("metadata", None)
                if metadata is not None:
                    run["extra"] = {**(run.get("extra") or {}), "metadata": metadata}
                if run["id"] in runs and runs[run["id"]] != run:
                    raise PipelineError(f"LangSmith returned conflicting records for run {run['id']}; capture again")
                runs[run["id"]] = run
            cursor = response.get("next_cursor")
            if cursor is None:
                break
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise PipelineError("LangSmith returned an invalid or repeated query cursor")
            cursors.add(cursor)
            body["cursor"] = cursor
        if window_end == end:
            break
        start = window_end
    return list(runs.values())


def _query_thread_llm_runs(
    workspace_id: str, project_id: str, thread_id: str, *, start_time: str, runner: Callable[..., Any],
) -> list[dict[str, Any]]:
    # Child calls need not repeat their thread ID. Resolve roots first to
    # avoid an expensive trace_filter join when retrieving their LLM calls.
    trace_ids: set[str] = set()
    for key in ("thread_id", "conversation_id", "session_id"):
        thread_filter = f"and(eq(metadata_key,{json.dumps(key)}),eq(metadata_value,{json.dumps(thread_id)}))"
        roots = _query_contract_runs(workspace_id, {
            "project_ids": [project_id], "is_root": True, "filter": thread_filter,
            "selects": ["ID", "PROJECT_ID"], "min_start_time": start_time,
        }, runner=runner)
        trace_ids.update(root["id"] for root in roots)
    ordered_traces = sorted(trace_ids)
    thread_runs: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(ordered_traces), LANGSMITH_PAGE_SIZE):
        batch = ordered_traces[offset : offset + LANGSMITH_PAGE_SIZE]
        for run in _query_contract_runs(workspace_id, {
            "project_ids": [project_id], "run_type": "LLM",
            "filter": f"in(trace_id, {json.dumps(batch)})", "min_start_time": start_time,
        }, runner=runner):
            if run.get("trace_id") not in batch:
                raise PipelineError("LangSmith returned a run from a different trace")
            if run["id"] in thread_runs and thread_runs[run["id"]] != run:
                raise PipelineError(f"run {run['id']} changed during the thread scan; capture again")
            thread_runs[run["id"]] = run
    return list(thread_runs.values())


def capture_inference_contract(
    workspace_id: str,
    run_id: str,
    output: Path,
    *,
    runner: Callable[..., Any] = _run_langsmith,
) -> dict[str, Any]:
    """Collect all function tools in the selected LLM run's conversation thread."""
    source = _read_contract_run(workspace_id, run_id, runner=runner)
    if source.get("run_type") != "llm":
        raise PipelineError("contract capture requires an LLM run ID inside the sample conversation")
    project_id, trace_id = source.get("session_id"), source.get("trace_id")
    if not isinstance(project_id, str) or not project_id or not isinstance(trace_id, str) or not trace_id:
        raise PipelineError("source LLM run has no project or trace ID")
    start_time = _project_start_time(workspace_id, project_id, runner=runner)
    roots = _query_contract_runs(
        workspace_id, {"project_ids": [project_id], "ids": [trace_id], "page_size": 1, "min_start_time": start_time}, runner=runner,
    )
    if len(roots) != 1 or roots[0]["id"] != trace_id:
        raise PipelineError(f"LangSmith did not return the source trace root {trace_id}")
    metadata = (roots[0].get("extra") or {}).get("metadata") or {}
    thread_keys = ("thread_id", "conversation_id", "session_id")
    thread_id = next((metadata[key] for key in thread_keys if isinstance(metadata.get(key), str) and metadata[key]), None)
    if thread_id is None:
        raise PipelineError("source trace has no thread ID; choose an LLM run from a conversation thread")
    runs = _query_thread_llm_runs(workspace_id, project_id, thread_id, start_time=start_time, runner=runner)
    try:
        payload = contract_from_runs(runs, workspace_id=workspace_id, source_run_id=run_id, thread_id=thread_id)
    except ContractError as exc:
        raise PipelineError(f"cannot capture inference contract: {exc}") from exc
    # Only publish a usable contract after every page and every tool passes.
    _json_dump(output, payload)
    return {
        "output": str(output),
        "source_run_id": run_id,
        "source_thread_id": thread_id,
        "llm_run_count": len(runs),
        "trace_count": len(payload["provenance"]["source_trace_ids"]),
        "contract_sha256": payload["contract_sha256"],
        "tools_sha256": payload["tools_sha256"],
        "tool_count": len(payload["tools"]),
    }


def download_dataset(
    workspace_id: str,
    dataset_id: str,
    raw_dir: Path,
    runner: Callable[..., Any] = _run_langsmith,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download every example and preserve the raw LangSmith data."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_result = runner(
        _langsmith_dataset_command(workspace_id, dataset_id),
        capture=True,
    )
    try:
        dataset = json.loads(dataset_result.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError("langsmith dataset get returned invalid JSON") from exc
    count = dataset.get("example_count")
    if dataset.get("id") != dataset_id or not isinstance(count, int) or count < 1:
        raise PipelineError("LangSmith returned an invalid dataset identity or count")
    _json_dump(raw_dir / "dataset.json", dataset)
    runner(
        _langsmith_export_command(
            workspace_id,
            dataset_id,
            raw_dir / "dataset-export.json",
            count,
        )
    )

    examples: list[dict[str, Any]] = []
    for offset in range(0, count, LANGSMITH_PAGE_SIZE):
        limit = min(LANGSMITH_PAGE_SIZE, count - offset)
        page_result = runner(
            _langsmith_page_command(workspace_id, dataset_id, offset, limit),
            capture=True,
        )
        try:
            page = json.loads(page_result.stdout)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"LangSmith example page at offset {offset} is invalid") from exc
        if not isinstance(page, list) or len(page) != limit:
            raise PipelineError(
                f"LangSmith returned {len(page) if isinstance(page, list) else 'invalid'} "
                f"examples at offset {offset}; expected {limit}"
            )
        examples.extend(page)
    if len({example.get("id") for example in examples}) != count:
        raise PipelineError("LangSmith pagination returned duplicate example IDs")
    _json_dump(raw_dir / "examples.json", examples)
    return dataset, examples


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_parts(content: list[dict[str, Any]]) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []
    for part in content:
        if set(part) != {"type", "text"} or part.get("type") != "text":
            raise PipelineError(f"invalid text content block: {part!r}")
        if not isinstance(part["text"], str):
            raise PipelineError("text content must be a string")
        parts.append({"type": "text", "text": part["text"]})
    return parts


def _validate_source_message(message: dict[str, Any]) -> None:
    """Validate native block structure independently of the target renderer."""
    if not isinstance(message, dict):
        raise PipelineError("message must be an object")
    role = message.get("role")
    if role not in ROLE_MAP:
        raise PipelineError(f"unsupported message role: {role!r}")
    message_id = message.get("id")
    if message_id is not None and (not isinstance(message_id, str) or not message_id):
        raise PipelineError("message id must be a non-empty string when present")
    content = message.get("content")
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise PipelineError("message content must be a string or a list")
    for part in content:
        if not isinstance(part, dict):
            raise PipelineError("content blocks must be objects")
        kind = part.get("type")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                raise PipelineError("text content must be a string")
        elif kind == "reasoning":
            if role != "ai":
                raise PipelineError("reasoning blocks are valid only in ai messages")
            if not isinstance(part.get("reasoning", ""), str):
                raise PipelineError("readable reasoning must be a string")
        elif kind == "tool_call":
            if role != "ai":
                raise PipelineError("tool_call blocks are valid only in ai messages")
            if not all(isinstance(part.get(key), str) and part[key] for key in ("id", "name")):
                raise PipelineError("tool_call id and name must be non-empty strings")
            if not isinstance(part.get("args", {}), (dict, str)):
                raise PipelineError("tool_call args must be an object or string")
        else:
            raise PipelineError(f"unsupported native content block type: {kind!r}")


def _validate_reasoning_policy(reasoning_policy: ReasoningPolicy) -> None:
    if reasoning_policy not in ("omit", "preserve"):
        raise PipelineError("reasoning_policy must be 'omit' or 'preserve'")


def convert_message(
    message: dict[str, Any],
    *,
    reasoning_policy: ReasoningPolicy = "omit",
    model: ModelSpec | None = None,
) -> dict[str, Any]:
    """Map native blocks to cookbook fields; never copy opaque reasoning state."""
    _validate_source_message(message)
    _validate_reasoning_policy(reasoning_policy)
    role = message["role"]
    message_id = message.get("id")

    content = message.get("content")
    converted: dict[str, Any] = {"role": ROLE_MAP[role], "content": content}
    if message_id is not None:
        converted["id"] = message_id
    for key in ("name", "tool_call_id"):
        if key in message:
            converted[key] = message[key]

    if isinstance(content, str):
        pass
    elif isinstance(content, list):
        reasoning = [part for part in content if part["type"] == "reasoning"]
        if reasoning and reasoning_policy == "preserve":
            if model is None:
                raise PipelineError("preserving reasoning requires a target ModelSpec")
            validate_reasoning_support(model)
            # Only readable text is trainable. Signatures and encrypted state
            # are provider replay metadata and are never rendered as targets.
            readable = "".join(part.get("reasoning", "") for part in reasoning)
            if readable:
                seen_visible = False
                for part in content:
                    if part["type"] != "reasoning":
                        seen_visible = True
                    elif seen_visible and part.get("reasoning"):
                        raise PipelineError("reasoning after visible content cannot be preserved without reordering")
                converted["reasoning_content"] = readable
        content = [part for part in content if part["type"] != "reasoning"]
        text = [part for part in content if isinstance(part, dict) and part.get("type") == "text"]
        calls = [part for part in content if isinstance(part, dict) and part.get("type") == "tool_call"]
        if len(text) + len(calls) != len(content):
            raise PipelineError("only text and tool_call content blocks are supported")
        seen_tool_call = False
        for part in content:
            if part["type"] == "tool_call":
                seen_tool_call = True
            elif seen_tool_call:
                raise PipelineError("text after tool_call blocks cannot be converted without reordering")
        if calls:
            if role != "ai":
                raise PipelineError("tool_call blocks are valid only in ai messages")
            converted["content"] = _text_parts(text) if text else ""
            converted["tool_calls"] = []
            for call in calls:
                args = call.get("args", {})
                if not isinstance(args, dict):
                    raise PipelineError("tool_call args must be an object")
                if not all(isinstance(call.get(key), str) and call[key] for key in ("id", "name")):
                    raise PipelineError("tool_call id and name must be non-empty strings")
                converted["tool_calls"].append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                        },
                    }
                )
        else:
            converted["content"] = _text_parts(text)
    else:
        raise PipelineError("message content must be a string or a list")
    return converted


def _validate_tool_pairs(
    messages: list[dict[str, Any]], example_id: str, *, native: bool = False,
) -> tuple[int, int]:
    calls: set[str] = set()
    results: set[str] = set()
    for message in messages:
        content = message.get("content")
        message_calls = (
            [part for part in content if part["type"] == "tool_call"]
            if native and isinstance(content, list)
            else message.get("tool_calls", [])
        )
        for call in message_calls:
            call_id = call["id"]
            if call_id in calls:
                raise PipelineError(f"example {example_id} repeats tool call id {call_id}")
            calls.add(call_id)
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise PipelineError(f"example {example_id} has a tool result without tool_call_id")
            if call_id not in calls or call_id in results:
                raise PipelineError(f"example {example_id} has unmatched tool calls or results")
            results.add(call_id)
    if calls != results:
        raise PipelineError(f"example {example_id} has unmatched tool calls or results")
    return len(calls), len(results)


def _has_message_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    has_text = bool(content) if isinstance(content, str) else any(
        part.get("text") for part in (content or [])
    )
    return bool(has_text or message.get("tool_calls") or message.get("reasoning_content"))


SOURCE_SCOPES = ("thread", "trace")


def _source_identity(example: dict[str, Any]) -> dict[str, str]:
    """Read the conversation scope and its ID from example metadata."""
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    scope, scope_id = metadata.get("source_scope"), metadata.get("source_scope_id")
    if scope not in SOURCE_SCOPES:
        raise PipelineError("example metadata source_scope must be thread or trace")
    if not isinstance(scope_id, str) or not scope_id:
        raise PipelineError("example metadata source_scope_id must be a non-empty string")
    return {"source_scope": scope, "source_scope_id": scope_id}


def validate_trajectories(
    examples: list[dict[str, Any]],
    expected_count: int,
) -> Audit:
    """Validate LangSmith message trajectories without changing them."""
    if len(examples) != expected_count:
        raise PipelineError(f"expected {expected_count} examples, got {len(examples)}")

    seen_examples: set[str] = set()
    role_counts: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []
    message_count = tool_calls = tool_results = 0
    reasoning_blocks = readable_reasoning_blocks = 0

    for source_index, example in enumerate(examples):
        example_id = example.get("id")
        metadata = example.get("metadata")
        if not isinstance(example_id, str) or not example_id:
            raise PipelineError(f"example {source_index} has no id")
        if example_id in seen_examples:
            raise PipelineError(f"duplicate example id: {example_id}")
        seen_examples.add(example_id)
        if not isinstance(metadata, dict):
            raise PipelineError(f"example {example_id} has no metadata object")
        try:
            _source_identity(example)
        except PipelineError as exc:
            raise PipelineError(f"example {example_id}: {exc}") from exc
        if metadata.get("trajectory_format") != "messages":
            raise PipelineError(f"example {example_id} is not in trajectory messages format")
        if metadata.get("conversation_scope") != "root":
            raise PipelineError(f"example {example_id} is not a root conversation")
        if example.get("outputs") not in (None, {}):
            raise PipelineError(f"example {example_id} has unexpected outputs")

        inputs = example.get("inputs")
        messages = inputs.get("messages") if isinstance(inputs, dict) else None
        if not isinstance(messages, list) or not messages:
            raise PipelineError(f"example {example_id} has no messages")
        saved_triage = metadata.get("smithtune_triage")
        if saved_triage is not None and (
            not isinstance(saved_triage, dict)
            or saved_triage.get("messages_sha256") != json_sha256(messages)
        ):
            raise PipelineError(f"example {example_id}: triaged example messages changed after judging")
        for position, message in enumerate(messages):
            try:
                _validate_source_message(message)
            except PipelineError as exc:
                raise PipelineError(f"example {example_id} message {position}: {exc}") from exc
        if not any(message["role"] == "ai" for message in messages):
            raise PipelineError(f"example {example_id} has no assistant training target")
        call_count, result_count = _validate_tool_pairs(messages, example_id, native=True)
        tool_calls += call_count
        tool_results += result_count
        seen_message_values: dict[tuple[str, str], int] = {}
        for position, source in enumerate(messages):
            role_counts[source["role"]] += 1
            message_count += 1
            message_id = source.get("id")
            if message_id is not None:
                duplicate_key = (message_id, _canonical(source["content"]))
                if duplicate_key in seen_message_values:
                    warnings.append(
                        {
                            "code": "duplicate_message",
                            "example_id": example_id,
                            "first_position": seen_message_values[duplicate_key],
                            "repeat_position": position,
                            "message_id": message_id,
                        }
                    )
                else:
                    seen_message_values[duplicate_key] = position
            if isinstance(source["content"], list):
                for part in source["content"]:
                    if part["type"] == "reasoning":
                        reasoning_blocks += 1
                        readable_reasoning_blocks += bool(part.get("reasoning"))

    return Audit(
        examples=len(examples),
        messages=message_count,
        roles=dict(sorted(role_counts.items())),
        tool_calls=tool_calls,
        tool_results=tool_results,
        duplicate_message_warnings=warnings,
        reasoning_blocks=reasoning_blocks,
        readable_reasoning_blocks=readable_reasoning_blocks,
    )


def _source_workspace(
    example: dict[str, Any], workspace_id: str, source_workspace_id: str | None,
) -> str:
    metadata = example.get("metadata") or {}
    value = metadata.get("source_workspace_id", source_workspace_id if source_workspace_id is not None else workspace_id)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PipelineError(f"example {example['id']}: source workspace ID must be a non-empty string without surrounding whitespace")
    return value


def capture_example_contracts(
    workspace_id: str, examples: list[dict[str, Any]], *,
    source_workspace_id: str | None = None, runner: Callable[..., Any] = _run_langsmith,
    checkpoint_path: Path | None = None,
) -> dict[str, InferenceContract]:
    """Collect a separate union of function tools for each source trajectory."""
    contracts: dict[str, InferenceContract] = {}
    checkpoint_identity = json_sha256({
        "schema_version": 1, "workspace_id": workspace_id,
        "source_workspace_id": source_workspace_id, "examples": examples,
        "tool_merge_policy": TOOL_MERGE_POLICY,
    })
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = _load_json(checkpoint_path)
        if isinstance(checkpoint, dict) and checkpoint.get("identity") == checkpoint_identity:
            payload = checkpoint.get("contracts")
            if checkpoint.get("contracts_sha256") != json_sha256(payload):
                raise PipelineError(f"capture checkpoint hash mismatch; remove {checkpoint_path} and retry")
            contracts = _parse_example_contracts(payload)
            if not set(contracts).issubset(example["id"] for example in examples):
                raise PipelineError(f"capture checkpoint contains unknown examples; remove {checkpoint_path} and retry")
            print(f"Resuming tool capture: {len(contracts)}/{len(examples)} examples already saved", file=sys.stderr)
    sources: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    project_starts: dict[tuple[str, str], str] = {}
    for example in examples:
        example_id = example["id"]
        if example_id in contracts:
            continue
        try:
            saved_triage = (example.get("metadata") or {}).get("smithtune_triage")
            if saved_triage is not None:
                if not isinstance(saved_triage, dict) or saved_triage.get("messages_sha256") != json_sha256(example["inputs"]["messages"]):
                    raise PipelineError("triaged example messages changed after judging")
                payload = parse_inference_contract(saved_triage.get("contract")).to_dict()
                payload["provenance"].update(source_example_id=example_id,
                                             source_project_id=example["metadata"].get("source_project_id"))
                contracts[example_id] = parse_inference_contract(payload)
                continue
            source_workspace = _source_workspace(example, workspace_id, source_workspace_id)
            identity = _source_identity(example)
            scope, scope_id = identity["source_scope"], identity["source_scope_id"]
            metadata = example.get("metadata") or {}
            project_id = example.get("source_session_id") or metadata.get("source_project_id")
            if example.get("source_session_id") and metadata.get("source_project_id") and example["source_session_id"] != metadata["source_project_id"]:
                raise PipelineError("conflicting source project IDs")
            if not isinstance(project_id, str) or not project_id:
                raise PipelineError("missing source project ID; add metadata.source_project_id to the example (or supply source_session_id)")
            project_key = (source_workspace, project_id)
            if project_key not in project_starts:
                project_starts[project_key] = _project_start_time(source_workspace, project_id, runner=runner)
            start_time = project_starts[project_key]
            key = (source_workspace, project_id, scope, scope_id)
            if key not in sources:
                if scope == "thread":
                    runs = _query_thread_llm_runs(source_workspace, project_id, scope_id, start_time=start_time, runner=runner)
                else:
                    runs = _query_contract_runs(source_workspace, {
                        "project_ids": [project_id], "run_type": "LLM",
                        "trace_id": scope_id, "min_start_time": start_time,
                    }, runner=runner)
                    if any(run.get("trace_id") != scope_id for run in runs):
                        raise PipelineError("LangSmith returned a run from a different trace")
                sources[key] = contract_from_runs(
                    runs, workspace_id=source_workspace, thread_id=scope_id if scope == "thread" else None,
                )
            payload = copy.deepcopy(sources[key])
            payload["provenance"].update(source_example_id=example_id, source_project_id=project_id)
            contracts[example_id] = parse_inference_contract(payload)
            if checkpoint_path is not None:
                saved = {key: contract.to_dict() for key, contract in contracts.items()}
                _json_dump(checkpoint_path, {"identity": checkpoint_identity, "contracts": saved,
                                            "contracts_sha256": json_sha256(saved)})
        except (ContractError, PipelineError) as exc:
            raise PipelineError(f"example {example_id}: cannot collect tools: {exc}") from exc
    return contracts


def _parse_example_contracts(payload: Any) -> dict[str, InferenceContract]:
    if not isinstance(payload, dict) or not payload:
        raise PipelineError("example inference contracts must be a non-empty object")
    try:
        contracts = {key: parse_inference_contract(value) for key, value in payload.items()}
    except ContractError as exc:
        raise PipelineError(f"invalid example inference contract: {exc}") from exc
    for key, contract in contracts.items():
        if contract.provenance.get("source_example_id") != key:
            raise PipelineError(f"example inference contract {key} has different source provenance")
    return contracts


def _example_contract_snapshot(
    workspace_id: str, dataset_id: str, examples: list[dict[str, Any]],
    source_sha: str, raw_dir: Path, *, fetch: bool, source_workspace_id: str | None = None,
) -> dict[str, InferenceContract]:
    path = raw_dir / "example_contracts.json"
    identity = {"schema_version": 1, "workspace_id": workspace_id,
                "dataset_id": dataset_id, "source_examples_sha256": source_sha,
                "tool_merge_policy": TOOL_MERGE_POLICY}
    if fetch:
        checkpoint_path = raw_dir / "example_contracts.partial.json"
        contracts = capture_example_contracts(workspace_id, examples, source_workspace_id=source_workspace_id,
                                             checkpoint_path=checkpoint_path)
        payload = {key: contract.to_dict() for key, contract in contracts.items()}
        _json_dump(path, {**identity, "contracts": payload, "contracts_sha256": json_sha256(payload)})
        checkpoint_path.unlink(missing_ok=True)
    if not path.exists():
        raise PipelineError("no cached example inference contracts; run prepare without --no-fetch or supply --inference-contract")
    snapshot = _load_json(path)
    if not isinstance(snapshot, dict) or any(snapshot.get(key) != value for key, value in identity.items()):
        raise PipelineError("cached example inference contracts do not match this dataset export; prepare again without --no-fetch")
    payload = snapshot.get("contracts")
    if snapshot.get("contracts_sha256") != json_sha256(payload):
        raise PipelineError("cached example inference contracts hash mismatch; prepare again without --no-fetch")
    contracts = _parse_example_contracts(payload)
    if set(contracts) != {example["id"] for example in examples}:
        raise PipelineError("cached inference contracts do not cover every example")
    for example in examples:
        expected_workspace = _source_workspace(example, workspace_id, source_workspace_id)
        if contracts[example["id"]].provenance.get("source_workspace_id") != expected_workspace:
            raise PipelineError(
                f"cached inference contract for example {example['id']} has a different source workspace; "
                "prepare again without --no-fetch"
            )
    return contracts


def prepare_sft_rows(
    examples: list[dict[str, Any]],
    contract: InferenceContract | None = None,
    *,
    example_contracts: dict[str, InferenceContract] | None = None,
    reasoning_policy: ReasoningPolicy = "omit",
    model: ModelSpec | None = None,
) -> list[dict[str, Any]]:
    """Convert validated LangSmith trajectories to provider-neutral SFT rows."""
    _validate_reasoning_policy(reasoning_policy)
    rows = []
    for source_index, example in enumerate(examples):
        if example_contracts is not None:
            if example["id"] not in example_contracts:
                raise PipelineError(f"example {example['id']} has no captured tool schemas")
            contract = example_contracts[example["id"]]
        identity = _source_identity(example)
        messages = []
        for position, source in enumerate(example["inputs"]["messages"]):
            try:
                converted = convert_message(source, reasoning_policy=reasoning_policy, model=model)
            except PipelineError as exc:
                raise PipelineError(f"example {example['id']} message {position}: {exc}") from exc
            had_reasoning = isinstance(source["content"], list) and any(
                part["type"] == "reasoning" for part in source["content"]
            )
            if had_reasoning and not _has_message_content(converted):
                continue
            messages.append(converted)
        _validate_tool_pairs(messages, example["id"])
        if not any(message["role"] == "assistant" and _has_message_content(message) for message in messages):
            raise PipelineError(f"example {example['id']} has no assistant training target after reasoning conversion")
        row = {
            "messages": messages,
            "_source": {
                "example_id": example["id"],
                **identity,
                "source_index": source_index,
                "metadata": copy.deepcopy(example["metadata"]),
            },
        }
        if contract is not None:
            try:
                contract.validate_messages(messages)
            except ContractError as exc:
                raise PipelineError(f"example {example['id']} violates inference contract: {exc}") from exc
            row["tools"] = copy.deepcopy(list(contract.tools))
            row["_source"]["contract_sha256"] = contract.contract_sha256
        rows.append(row)
    return rows


def _split_rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{row['_source']['source_scope_id']}".encode()).hexdigest()


def split_rows(
    rows: list[dict[str, Any]],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign each conversation to a partition by hashing its source scope ID.

    Every example is one whole conversation, so rows are split independently;
    the hash keeps assignments stable across runs and row order.
    """
    if not 0 <= validation_fraction <= 1 or not 0 <= test_fraction <= 1:
        raise PipelineError("validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction > 1:
        raise PipelineError("validation and test fractions cannot total more than one")

    ranked = sorted(rows, key=_split_rank)
    train_fraction = 1 - validation_fraction - test_fraction
    if math.isclose(train_fraction, 0, abs_tol=1e-9):
        # Fractions that sum to one can leave a floating-point remainder that
        # would otherwise force a one-row training partition.
        train_fraction = 0.0
    required_partitions = sum(
        fraction > 0
        for fraction in (train_fraction, validation_fraction, test_fraction)
    )
    if len(ranked) < required_partitions:
        raise PipelineError("not enough conversations for the requested non-zero fractions")

    validation_count = 0
    if validation_fraction > 0:
        validation_count = max(1, round(len(ranked) * validation_fraction))
        validation_count = min(
            validation_count,
            len(ranked) - int(train_fraction > 0) - int(test_fraction > 0),
        )
    test_count = 0
    if test_fraction > 0:
        test_count = max(1, round(len(ranked) * test_fraction))
        test_count = min(test_count, len(ranked) - validation_count - int(train_fraction > 0))
    validation = ranked[:validation_count]
    test = ranked[validation_count : validation_count + test_count]
    held_out = {id(row) for row in validation + test}
    train = [row for row in rows if id(row) not in held_out]
    return train, validation, test


def _trajectory_content_hash(row: dict[str, Any]) -> str:
    normalized = []
    for message in row["messages"]:
        item = {
            key: copy.deepcopy(message[key])
            for key in ("role", "content", "name")
            if key in message
        }
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                calls.append(
                    {
                        "type": call.get("type"),
                        "function": {
                            "name": function.get("name"),
                            "arguments": arguments,
                        },
                    }
                )
            item["tool_calls"] = calls
        normalized.append(item)
    return hashlib.sha256(_canonical(normalized).encode()).hexdigest()


def _validate_split_isolation(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    """Reject identical conversations recorded under different IDs across partitions."""
    partitions = {"train": train, "validation": validation, "test": test}
    for left_index, (left_name, left_rows) in enumerate(partitions.items()):
        left_content = {_trajectory_content_hash(row) for row in left_rows}
        for right_name, right_rows in list(partitions.items())[left_index + 1 :]:
            right_content = {_trajectory_content_hash(row) for row in right_rows}
            if left_content & right_content:
                raise PipelineError(f"content hash overlap between {left_name} and {right_name}")


def _load_dataset_source(
    workspace_id: str,
    dataset_id: str,
    raw_dir: Path,
    *,
    fetch: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if fetch:
        download_dataset(workspace_id, dataset_id, raw_dir)
    examples = _load_json(raw_dir / "examples.json")
    export = _load_json(raw_dir / "dataset-export.json")
    dataset = _load_json(raw_dir / "dataset.json")
    if not isinstance(examples, list) or not isinstance(export, list):
        raise PipelineError("LangSmith exports must be JSON arrays")
    expected_count = dataset.get("example_count")
    if not isinstance(expected_count, int) or len(examples) != expected_count or len(export) != expected_count:
        raise PipelineError(
            f"dataset export has {len(export)} rows and {len(examples)} examples, expected {expected_count}"
        )
    if dataset.get("id") != dataset_id:
        raise PipelineError("LangSmith dataset identity or example count changed")
    source_sha = hashlib.sha256((raw_dir / "examples.json").read_bytes()).hexdigest()
    return dataset, examples, source_sha


def prepare_dataset(
    workspace_id: str,
    dataset_id: str,
    model: ModelSpec,
    data_dir: Path,
    *,
    inference_contract: InferenceContract | None = None,
    source_workspace_id: str | None = None,
    reasoning_policy: ReasoningPolicy = "omit",
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    fetch: bool = True,
    check_render: bool = True,
) -> dict[str, Any]:
    """Prepare one trajectory dataset with a deterministic thread-level split."""
    model.validate()
    _validate_reasoning_policy(reasoning_policy)
    dataset, examples, source_sha = _load_dataset_source(
        workspace_id,
        dataset_id,
        data_dir / "raw",
        fetch=fetch,
    )
    expected_count = len(examples)
    audit = validate_trajectories(examples, expected_count)
    example_contracts = None
    if inference_contract is None:
        example_contracts = _example_contract_snapshot(
            workspace_id, dataset_id, examples, source_sha, data_dir / "raw", fetch=fetch,
            source_workspace_id=source_workspace_id,
        )
    description_replacements = []
    captured_contracts = example_contracts or ({"global": inference_contract} if inference_contract else {})
    for example_id, contract in captured_contracts.items():
        for replacement in contract.provenance.get("tool_description_replacements", []):
            description_replacements.append({
                **replacement,
                **({"example_id": example_id} if example_contracts is not None else {}),
                "source_workspace_id": contract.provenance.get("source_workspace_id"),
                "source_thread_id": contract.provenance.get("source_thread_id"),
            })
    rows = prepare_sft_rows(examples, inference_contract, example_contracts=example_contracts,
                            reasoning_policy=reasoning_policy, model=model)
    messages_removed = audit.messages - sum(len(row["messages"]) for row in rows)
    reasoning_preserved = audit.readable_reasoning_blocks if reasoning_policy == "preserve" else 0
    rejected: list[dict[str, Any]] = []
    if check_render:
        rows, rejected, rendered = validate_model_context(rows, model)
    else:
        rendered = {}
    train, validation, test = split_rows(rows, validation_fraction, test_fraction)
    _validate_split_isolation(train, validation, test)
    audit_value = {
        **asdict(audit), **rendered,
        "tool_description_replacements": len(description_replacements),
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "langsmith": {
            "workspace_id": workspace_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset.get("name"),
            "examples": expected_count,
        },
        "split": {
            "method": "sha256(seed:source_scope_id)",
            "seed": SPLIT_SEED,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "prepared": {"accepted": len(rows), "rejected": len(rejected)},
        "conversion": {
            "roles": ROLE_MAP,
            "tool_calls": "StandardMessage tool_call blocks to OpenAI tool_calls",
            "tool_schemas_added": inference_contract is not None or example_contracts is not None,
            "system_prompts_added": False,
            "messages_filtered": messages_removed > 0,
            "messages_removed": messages_removed,
            "reasoning_policy": reasoning_policy,
            "reasoning_blocks": {
                "source": audit.reasoning_blocks,
                "preserved": reasoning_preserved,
                "omitted": audit.reasoning_blocks - reasoning_preserved,
            },
        },
        "model": asdict(model),
        "provider": {
            "name": model.provider,
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        },
        "source_examples_sha256": source_sha,
        "audit": audit_value,
    }
    if inference_contract is not None:
        manifest["inference_contract"] = inference_contract.manifest_summary()
    if example_contracts is not None:
        payload = {key: contract.to_dict() for key, contract in example_contracts.items()}
        manifest["example_contracts"] = {"count": len(payload), "sha256": json_sha256(payload)}
        _json_dump(data_dir / "prepared" / "example_contracts.json", payload)
    _jsonl_dump(data_dir / "prepared" / "train.jsonl", train)
    _jsonl_dump(data_dir / "prepared" / "validation.jsonl", validation)
    _jsonl_dump(data_dir / "prepared" / "test.jsonl", test)
    _json_dump(data_dir / "prepared" / "manifest.json", manifest)
    _json_dump(data_dir / "prepared" / "warnings.json", audit.duplicate_message_warnings)
    _json_dump(data_dir / "prepared" / "tool_description_replacements.json", description_replacements)
    _json_dump(data_dir / "prepared" / "rejected.json", rejected)
    if inference_contract is not None:
        _json_dump(
            data_dir / "prepared" / "inference_contract.json",
            inference_contract.to_dict(),
        )
    return manifest


def _model_from_manifest(manifest: dict[str, Any]) -> ModelSpec:
    try:
        return ModelSpec(**manifest["model"])
    except (KeyError, TypeError) as exc:
        raise PipelineError("prepared manifest has no valid model profile") from exc


def _prepared_split(manifest: dict[str, Any]) -> dict[str, int]:
    """Return the prepared manifest's split row counts after validating them."""
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise PipelineError("prepared manifest has no valid split counts")
    counts: dict[str, int] = {}
    for partition in ("train", "validation", "test"):
        count = split.get(partition)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PipelineError(f"prepared manifest has an invalid {partition} row count")
        counts[partition] = count
    return counts


def _require_prepared_provider(
    manifest: dict[str, Any],
    expected_provider: str,
    *,
    allow_legacy: bool = False,
) -> ModelSpec:
    """Return the prepared model after verifying its provider identity."""
    model = _model_from_manifest(manifest)
    if "provider" not in manifest:
        if not allow_legacy or model.provider != expected_provider:
            raise PipelineError(
                f"prepared provider mismatch: expected {expected_provider}"
            )
        identity = {
            "name": expected_provider,
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        }
    else:
        identity = manifest["provider"]
    if not isinstance(identity, dict):
        raise PipelineError("prepared provider mismatch: manifest has no provider identity")
    if identity.get("name") != expected_provider or model.provider != expected_provider:
        raise PipelineError(
            f"prepared provider mismatch: expected {expected_provider}"
        )
    if (
        identity.get("renderer") != model.renderer
        or identity.get("tokenizer_revision") != model.tokenizer_revision
    ):
        raise PipelineError("prepared provider mismatch: renderer identity differs from model")
    model.validate()
    return model


def _prepared_inference_contract(
    data_dir: Path,
    manifest: dict[str, Any],
) -> InferenceContract | None:
    summary = manifest.get("inference_contract")
    if summary is None:
        return None
    if not isinstance(summary, dict):
        raise PipelineError("prepared manifest has an invalid inference contract summary")
    try:
        contract = load_inference_contract(data_dir / "prepared" / "inference_contract.json")
    except ContractError as exc:
        raise PipelineError(f"prepared inference contract is invalid: {exc}") from exc
    if summary.get("contract_sha256") != contract.contract_sha256:
        raise PipelineError("prepared inference contract hash differs from the manifest")
    return contract



def _prepared_example_contracts(
    data_dir: Path, manifest: dict[str, Any],
) -> dict[str, InferenceContract] | None:
    summary = manifest.get("example_contracts")
    if summary is None:
        return None
    payload = _load_json(data_dir / "prepared" / "example_contracts.json")
    if not isinstance(summary, dict) or summary.get("sha256") != json_sha256(payload):
        raise PipelineError("prepared example inference contracts hash differs from the manifest")
    contracts = _parse_example_contracts(payload)
    if summary.get("count") != len(contracts):
        raise PipelineError("prepared example inference contract count differs from the manifest")
    return contracts

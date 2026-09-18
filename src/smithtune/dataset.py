"""Capture and prepare LangSmith trajectory datasets."""

from __future__ import annotations

import copy
import hashlib
import json
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

from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _load_jsonl, _run, _utc_now, exclusive_output
from smithtune.inference_contract import (
    ContractError,
    InferenceContract,
    contract_from_runs,
    load_inference_contract,
    parse_inference_contract,
    json_sha256,
)
from smithtune.providers.base import ModelSpec, PipelineError, ReasoningPolicy
from smithtune.rendering import validate_model_context, validate_reasoning_support


SPLIT_SEED = 42
SPLIT_METHOD = "persistent-source-hash-v1"
SPLIT_NAMES = ("train", "validation", "test")


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


def _run_langsmith(command: list[str], *, capture: bool = False, input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        # Export commands must not bypass sanitization by inheriting stderr.
        result = _run(command, capture=True, **({"input": input} if input is not None else {}))
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or "").strip() or (exc.stdout or "").strip()
        status = re.search(r"\bHTTP [45]\d\d\b", diagnostic, re.I)
        timeout = re.search(
            r"\b(context deadline exceeded|Client\.Timeout exceeded|request timed out|Query timeout exceeded)\b",
            diagnostic, re.I,
        )
        details = [status.group().upper()] if status else []
        if timeout:
            details.append("request timed out")
        detail = "; ".join(details) or f"exited with status {exc.returncode}"
        raise PipelineError(f"langsmith failed: {detail}") from None
    if not capture:
        return subprocess.CompletedProcess(result.args, result.returncode)
    return result


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


def _contract_read(command: list[str], *, runner: Callable[..., Any], attempts: int = 6) -> Any:
    for attempt in range(attempts):
        try:
            result = runner(command, capture=True)
            break
        except PipelineError as exc:
            retryable = re.search(r"\b(HTTP 429|context deadline exceeded|Client\.Timeout exceeded|request timed out|Query timeout exceeded)\b", str(exc), re.I)
            if attempt + 1 == attempts or not retryable:
                raise
            delay = min(5 * 2**attempt + random.uniform(0, 1), 60)
            reason = "rate limit reached" if retryable[0].upper() == "HTTP 429" else "request timed out"
            print(f"LangSmith {reason}; retrying in {delay:.1f}s ({attempt + 1}/{attempts - 1})", file=sys.stderr)
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


def _project_start_time(workspace_id: str, project_id: str, *, runner: Callable[..., Any], read_attempts: int = 6) -> str:
    project = _contract_read([
        "langsmith", "api", f"/api/v1/sessions/{quote(project_id, safe='')}",
        "--workspace", workspace_id, "--method", "GET",
    ], runner=runner, attempts=read_attempts)
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
    workspace_id: str, query: dict[str, Any], *, runner: Callable[..., Any], read_attempts: int = 6,
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
            ], runner=runner, attempts=read_attempts)
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
    if not isinstance(role, str) or role not in ROLE_MAP:
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
        if saved_triage is not None and "smithtune_source" in metadata:
            from smithtune.bindings import evidence_hash
            if saved_triage.get("evidence_sha256") != evidence_hash(example):
                raise PipelineError(f"example {example_id}: triaged evidence changed after judging")
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


def validate_import_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate an import without rewriting messages or choosing a training model."""
    validate_trajectories([example], 1)
    messages = []
    for position, message in enumerate(example["inputs"]["messages"]):
        if position and message["role"] == "system":
            raise PipelineError(f"example {example['id']} has misplaced system message")
        try:
            messages.append(convert_message(message))
        except PipelineError as exc:
            raise PipelineError(f"example {example['id']} message {position}: {exc}") from exc
    return messages


def _malformed_trajectory_reason(example_id: str, error: PipelineError) -> str | None:
    detail = str(error)
    if detail.startswith(f"example {example_id} repeats tool call id "):
        return "repeated_tool_call_id"
    if detail == f"example {example_id} has a tool result without tool_call_id":
        return "tool_result_without_tool_call_id"
    if detail == f"example {example_id} has unmatched tool calls or results":
        return "unmatched_tool_calls_or_results"
    if detail.startswith(f"example {example_id} message "):
        if "unsupported native content block type" in detail:
            return "unsupported_native_content"
        return "invalid_message"
    if detail == f"example {example_id} has no messages":
        return "missing_messages"
    if detail == f"example {example_id} has no assistant training target":
        return "missing_assistant_training_target"
    if detail == f"example {example_id} has unexpected outputs":
        return "unexpected_outputs"
    if detail == f"example {example_id} has misplaced system message":
        return "misplaced_system_message"
    return None


def _recorded_tool_call_reason(error: ContractError) -> str:
    detail = str(error)
    if detail.startswith("unknown tool "):
        return "unknown_tool"
    if " do not match its JSON Schema" in detail:
        return "invalid_tool_arguments"
    if " are not valid JSON" in detail:
        return "invalid_tool_arguments_json"
    return "recorded_tool_call_incompatible"


def _exclude_malformed_trajectories(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Quarantine malformed whole trajectories while preserving every source message."""
    accepted, warnings = [], []
    for example in examples:
        example_id = example.get("id")
        try:
            validate_trajectories([example], 1)
        except PipelineError as exc:
            reason = _malformed_trajectory_reason(example_id, exc) if isinstance(example_id, str) else None
            if reason is None:
                raise
            identity = _source_identity(example)
            tool_error = reason in {
                "repeated_tool_call_id",
                "tool_result_without_tool_call_id",
                "unmatched_tool_calls_or_results",
            }
            warning = {
                "code": "malformed_tool_trajectory_excluded" if tool_error else "malformed_trajectory_excluded",
                "example_id": example_id,
                **identity,
                "reason": reason,
            }
            warnings.append(warning)
            print(
                f"Warning: excluding malformed{' tool' if tool_error else ''} trajectory {example_id} "
                f"({identity['source_scope']} {identity['source_scope_id']}): {reason}",
                file=sys.stderr,
            )
        else:
            accepted.append(example)
    return accepted, warnings


def _source_workspace(
    example: dict[str, Any], workspace_id: str, source_workspace_id: str | None,
) -> str:
    metadata = example.get("metadata") or {}
    value = metadata.get("source_workspace_id", source_workspace_id if source_workspace_id is not None else workspace_id)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PipelineError(f"example {example['id']}: source workspace ID must be a non-empty string without surrounding whitespace")
    return value


def _source_key(
    example: dict[str, Any], workspace_id: str, source_workspace_id: str | None,
) -> tuple[str, str, str, str]:
    workspace = _source_workspace(example, workspace_id, source_workspace_id)
    identity = _source_identity(example)
    metadata = example.get("metadata") or {}
    project = example.get("source_session_id") or metadata.get("source_project_id")
    if example.get("source_session_id") and metadata.get("source_project_id") and example["source_session_id"] != metadata["source_project_id"]:
        raise PipelineError("conflicting source project IDs")
    if not isinstance(project, str) or not project:
        raise PipelineError("missing source project ID; add metadata.source_project_id to the example (or supply source_session_id)")
    return workspace, project, identity["source_scope"], identity["source_scope_id"]


def _validate_unique_sources(
    examples: list[dict[str, Any]], workspace_id: str, source_workspace_id: str | None,
) -> None:
    seen = {}
    for example in examples:
        try:
            key = _source_key(example, workspace_id, source_workspace_id)
        except PipelineError as exc:
            raise PipelineError(f"example {example['id']}: {exc}") from exc
        if key in seen:
            workspace, project, scope, scope_id = key
            raise PipelineError(
                f"examples {seen[key]} and {example['id']} reference the same {scope} {scope_id} "
                f"in project {project}, workspace {workspace}; keep one complete trajectory per source conversation"
            )
        seen[key] = example["id"]


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


def prepare_sft_rows(
    examples: list[dict[str, Any]], contract: InferenceContract | None = None, *,
    example_contracts=None, reasoning_policy: ReasoningPolicy = "omit",
    model: ModelSpec | None = None, workspace_id: str | None = None,
    source_workspace_id: str | None = None, exclusion_warnings=None,
) -> list[dict[str, Any]]:
    """Keep canonical conversations; expand targets only at the rendering boundary."""
    from smithtune.bindings import read_bindings, tool_contract

    if example_contracts is not None:
        raise PipelineError("legacy conversation-wide contracts require per-message re-capture")
    _validate_reasoning_policy(reasoning_policy)
    rows = []
    for source_index, example in enumerate(examples):
        identity = _source_identity(example)
        try:
            bindings = read_bindings(example.get("metadata"), example["inputs"]["messages"]) if contract is None else None
            messages, positions = [], []
            for position, source in enumerate(example["inputs"]["messages"]):
                try:
                    converted = convert_message(source, reasoning_policy=reasoning_policy, model=model)
                except PipelineError as exc:
                    raise PipelineError(f"message {position}: {exc}") from exc
                if source["role"] == "ai":
                    current = contract or tool_contract(bindings[position]["tools"])
                    current.validate_messages([converted])
                if source["role"] == "ai" and not _has_message_content(converted):
                    continue
                messages.append(converted)
                positions.append(position)
            _validate_tool_pairs(messages, example["id"])
            if not any(m["role"] == "assistant" for m in messages):
                raise PipelineError("no assistant training target after reasoning conversion")
        except (PipelineError, ContractError) as exc:
            if exclusion_warnings is None:
                raise PipelineError(f"example {example['id']} {exc}") from exc
            exclusion_warnings.append({"code": "incompatible_inference_contract_excluded", "example_id": example["id"],
                                       **identity, "reason": str(exc)})
            continue
        row = {"messages": messages, "tool_policy": "global_override" if contract else "per_assistant",
               "_source": {"example_id": example["id"], **identity, "source_index": source_index,
                           "message_positions": positions, "metadata": copy.deepcopy(example["metadata"])}}
        if workspace_id is not None:
            row["_source"]["source_key"] = list(_source_key(example, workspace_id, source_workspace_id))
        if contract is not None:
            row["tools"] = copy.deepcopy(list(contract.tools))
            row["_source"]["contract_sha256"] = contract.contract_sha256
        rows.append(row)
    return rows


def capture_example_bindings(examples, workspace_id, *, runner=_run, source_workspace_id=None, checkpoint_path=None):
    """Attach trajectory-native tools only when recorded messages match exactly."""
    from smithtune.curation import _fetch_trajectory, _trajectory_page_too_large
    from smithtune.triage_source import _fetch

    def read(command, **kwargs):
        try:
            return _fetch(command, runner=runner, cache_dir=Path("."), use_cache=False, **kwargs)
        except subprocess.CalledProcessError as exc:
            if _trajectory_page_too_large(exc):
                raise
            detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or f"langsmith exited with status {exc.returncode}"
            raise PipelineError(detail) from None

    identity = json_sha256({"schema_version": 2, "workspace_id": workspace_id,
                           "source_workspace_id": source_workspace_id, "examples": examples})
    captured = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = _load_json(checkpoint_path)
        if saved.get("identity") == identity:
            captured = saved.get("bindings")
            if not isinstance(captured, dict) or saved.get("sha256") != json_sha256(captured):
                raise PipelineError("saved assistant bindings hash mismatch; remove the capture checkpoint and retry")
            if not set(captured) <= {example["id"] for example in examples}:
                raise PipelineError("saved assistant bindings contain unknown examples")
    enriched, failures = [], []
    for original in examples:
        example = copy.deepcopy(original)
        metadata = example["metadata"]
        if "smithtune_source" not in metadata and example["id"] in captured:
            metadata["smithtune_source"] = copy.deepcopy(captured[example["id"]])
        if "smithtune_source" not in metadata:
            if metadata.get("smithtune_triage"):
                failures.append({"code": "incompatible_inference_contract_excluded", "example_id": example["id"],
                                 **_source_identity(example), "reason": "legacy triage evidence requires fresh pull and judging"})
                continue
            workspace, project, scope, scope_id = _source_key(example, workspace_id, source_workspace_id)
            try:
                trajectory = _fetch_trajectory(workspace, project, {"key": scope + "_id", "id": scope_id}, runner=read)
            except PipelineError as exc:
                raise PipelineError(f"example {example['id']}: cannot read trajectory tools: {exc}") from exc
            try:
                if trajectory["training_error"]:
                    raise PipelineError(trajectory["training_error"])
                if json_sha256(trajectory["messages"]) != json_sha256(example["inputs"]["messages"]):
                    raise PipelineError("source trajectory messages differ from the saved dataset example; pull a fresh dataset or supply --inference-contract")
                if scope == "trace" and set(trajectory["trace_ids"]) != {scope_id}:
                    raise PipelineError("trajectory evidence belongs to another trace")
                metadata["smithtune_source"] = trajectory["source"]
                captured[example["id"]] = metadata["smithtune_source"]
                if checkpoint_path is not None:
                    _json_dump(checkpoint_path, {"identity": identity, "bindings": captured, "sha256": json_sha256(captured)})
            except PipelineError as exc:
                failures.append({"code": "incompatible_inference_contract_excluded", "example_id": example["id"],
                                 **_source_identity(example), "reason": str(exc)})
                continue
        enriched.append(example)
    return enriched, failures


def _split_key(row: dict[str, Any]) -> str:
    key = row["_source"].get("source_key")
    if (not isinstance(key, list) or len(key) != 4
            or not all(isinstance(value, str) and value for value in key)
            or key[2] not in SOURCE_SCOPES):
        raise PipelineError("split assignment requires a complete source workspace, project, scope and scope ID")
    return json.dumps(key, separators=(",", ":"))


def _split_settings(validation_fraction: float, test_fraction: float) -> dict:
    if not 0 <= validation_fraction <= 1 or not 0 <= test_fraction <= 1:
        raise PipelineError("validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction > 1:
        raise PipelineError("validation and test fractions cannot total more than one")
    return {"method": SPLIT_METHOD, "seed": SPLIT_SEED,
            "validation_fraction": validation_fraction, "test_fraction": test_fraction}


def _load_split_assignments(data_dir: Path, settings: dict, *, required: bool = False) -> dict[str, str]:
    prepared = data_dir / "prepared"
    path = prepared / "split_assignments.json"
    if path.exists():
        saved = _load_json(path)
        if not isinstance(saved, dict) or saved.get("schema_version") != 1:
            raise PipelineError(f"invalid split assignments in {path}")
        if any(saved.get(key) != value for key, value in settings.items()):
            raise PipelineError("saved split settings differ; reuse the original fractions and algorithm")
        assignments = saved.get("assignments")
        if not isinstance(assignments, dict) or saved.get("assignments_sha256") != json_sha256(assignments):
            raise PipelineError(f"invalid split assignments or checksum in {path}")
        for key, partition in assignments.items():
            try:
                canonical = _split_key({"_source": {"source_key": json.loads(key)}})
            except (ValueError, TypeError, PipelineError):
                raise PipelineError(f"invalid source identity in {path}") from None
            if canonical != key or partition not in SPLIT_NAMES:
                raise PipelineError(f"invalid split assignment in {path}")
        return assignments

    manifest_path = prepared / "manifest.json"
    if not manifest_path.exists():
        if required or any((prepared / f"{name}.jsonl").exists() for name in SPLIT_NAMES):
            raise PipelineError(f"no complete previous preparation in {data_dir}")
        return {}
    manifest = _load_json(manifest_path)
    split = manifest.get("split") if isinstance(manifest, dict) else None
    if not isinstance(split, dict) or split.get("method") != "sha256(seed:source_scope_id)":
        raise PipelineError(f"missing split_assignments.json in {prepared}; restore it before preparing again")
    if any(split.get(key) != settings[key] for key in ("seed", "validation_fraction", "test_fraction")):
        raise PipelineError("saved split settings differ; reuse the original fractions and algorithm")
    contracts = _prepared_example_contracts(data_dir, manifest) or {}
    assignments = {}
    counts = _prepared_split(manifest)
    for partition in SPLIT_NAMES:
        rows = _load_jsonl(prepared / f"{partition}.jsonl")
        if len(rows) != counts[partition]:
            raise PipelineError("previous split files do not match the manifest; restore them before preparing again")
        for row in rows:
            source = row.get("_source") if isinstance(row, dict) else None
            if not isinstance(source, dict) or not isinstance(source.get("metadata"), dict):
                raise PipelineError("previous split rows have no valid source metadata")
            metadata = copy.deepcopy(source["metadata"])
            contract = contracts.get(source.get("example_id"))
            provenance = contract.provenance if contract else {}
            # The old manifest did not record --source-workspace-id. Recover it
            # from the actual capture, never from this preparation's flags.
            for field in ("source_workspace_id", "source_project_id"):
                if not metadata.get(field):
                    metadata[field] = provenance.get(field)
            if not metadata.get("source_workspace_id"):
                raise PipelineError("cannot recover previous source workspace for split assignments; saved source metadata or capture provenance is required")
            example = {"id": source.get("example_id"), "metadata": metadata}
            key = json.dumps(_source_key(example, metadata["source_workspace_id"], None), separators=(",", ":"))
            if key in assignments:
                raise PipelineError("duplicate source identity in previous split files")
            assignments[key] = partition
    return assignments


def split_rows(
    rows: list[dict[str, Any]],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    *, assignments: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep saved assignments; use fixed hash ranges for unseen conversations."""
    _split_settings(validation_fraction, test_fraction)
    assignments = assignments if assignments is not None else {}
    partitions: dict[str, list] = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        key = _split_key(row)
        if key not in assignments:
            rank = int(hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).hexdigest(), 16)
            if rank < int(validation_fraction * 2**256):
                assignments[key] = "validation"
            elif rank < int((validation_fraction + test_fraction) * 2**256):
                assignments[key] = "test"
            else:
                assignments[key] = "train"
        partitions[assignments[key]].append(row)
    for name, fraction in zip(SPLIT_NAMES, (1 - validation_fraction - test_fraction, validation_fraction, test_fraction), strict=True):
        if fraction > 1e-9 and not partitions[name]:
            print(f"No conversations in the {name} split; add more data. Existing assignments remain fixed.", file=sys.stderr)
    return partitions["train"], partitions["validation"], partitions["test"]


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


@exclusive_output("data_dir")
def prepare_dataset(
    workspace_id: str,
    dataset_id: str,
    model: ModelSpec,
    data_dir: Path,
    *,
    inference_contract: InferenceContract | None = None,
    source_workspace_id: str | None = None,
    split_from: Path | None = None,
    reasoning_policy: ReasoningPolicy = "omit",
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    fetch: bool = True,
    check_render: bool = True,
    sync_splits: bool = True,
) -> dict[str, Any]:
    """Prepare one trajectory dataset with a deterministic thread-level split."""
    model.validate()
    _validate_reasoning_policy(reasoning_policy)
    split_settings = _split_settings(validation_fraction, test_fraction)
    assignments = _load_split_assignments(data_dir, split_settings)
    if split_from is not None:
        previous = _load_split_assignments(split_from, split_settings, required=True)
        if any(key in assignments and assignments[key] != value for key, value in previous.items()):
            raise PipelineError("--split-from conflicts with this directory's saved split assignments")
        assignments.update(previous)
    dataset, source_examples, source_sha = _load_dataset_source(
        workspace_id,
        dataset_id,
        data_dir / "raw",
        fetch=fetch,
    )
    expected_count = len(source_examples)
    examples, malformed_warnings = _exclude_malformed_trajectories(source_examples)
    _validate_unique_sources(examples, workspace_id, source_workspace_id)
    contract_warnings: list[dict[str, Any]] = []
    if inference_contract is None and any("smithtune_source" not in e["metadata"] for e in examples):
        captured_path = data_dir / "raw" / "bound_examples.json"
        if fetch:
            examples, contract_warnings = capture_example_bindings(examples, workspace_id, source_workspace_id=source_workspace_id,
                                                                  checkpoint_path=data_dir / "raw" / "bindings.partial.json")
            _json_dump(captured_path, {"workspace_id": workspace_id, "source_sha256": source_sha, "source_workspace_id": source_workspace_id, "examples": examples, "exclusions": contract_warnings,
                                       "sha256": json_sha256({"examples": examples, "exclusions": contract_warnings})})
            (data_dir / "raw" / "bindings.partial.json").unlink(missing_ok=True)
        elif captured_path.exists():
            saved = _load_json(captured_path)
            if saved.get("workspace_id") != workspace_id:
                raise PipelineError("cached bindings use a different workspace; prepare again without --no-fetch")
            if saved.get("source_sha256") != source_sha:
                raise PipelineError("saved per-message capture differs from the export; prepare again")
            if saved.get("source_workspace_id") != source_workspace_id:
                raise PipelineError("cached bindings use a different source workspace; prepare again")
            if saved.get("sha256") != json_sha256({"examples": saved.get("examples"), "exclusions": saved.get("exclusions")}):
                raise PipelineError("cached per-assistant bindings hash mismatch; prepare again without --no-fetch")
            examples, contract_warnings = saved["examples"], saved["exclusions"]
        else:
            raise PipelineError("export has no per-assistant bindings; prepare without --no-fetch to capture provenance")
    rows = prepare_sft_rows(examples, inference_contract, reasoning_policy=reasoning_policy, model=model,
                            workspace_id=workspace_id, source_workspace_id=source_workspace_id,
                            exclusion_warnings=contract_warnings)
    excluded_contract_ids = {warning["example_id"] for warning in contract_warnings}
    examples = [example for example in examples if example["id"] not in excluded_contract_ids]
    audit = validate_trajectories(examples, len(examples))
    messages_removed = audit.messages - sum(len(row["messages"]) for row in rows)
    reasoning_preserved = audit.readable_reasoning_blocks if reasoning_policy == "preserve" else 0
    context_rejected: list[dict[str, Any]] = []
    if check_render:
        rows, context_rejected, rendered = validate_model_context(rows, model)
    else:
        rendered = {}
    renderer_warnings = [
        warning for warning in context_rejected
        if warning.get("code") == "renderer_incompatible_trajectory_excluded"
    ]
    for warning in [*contract_warnings, *renderer_warnings]:
        print(
            f"Warning: excluding trajectory {warning['example_id']} "
            f"({warning['source_scope']} {warning['source_scope_id']}): {warning['reason']}",
            file=sys.stderr,
        )
    preparation_warnings = [*malformed_warnings, *contract_warnings, *renderer_warnings]
    rejected = [*malformed_warnings, *contract_warnings, *context_rejected]
    train, validation, test = split_rows(rows, validation_fraction, test_fraction, assignments=assignments)
    _validate_split_isolation(train, validation, test)
    audit_value = {
        **asdict(audit), **rendered,
        "malformed_trajectories": len(malformed_warnings),
        "malformed_tool_trajectories": sum(
            warning["code"] == "malformed_tool_trajectory_excluded" for warning in malformed_warnings
        ),
        "incompatible_inference_contracts": len(contract_warnings),
        "renderer_incompatible_trajectories": len(renderer_warnings),
    }
    manifest = {
        "schema_version": 2,
        "tool_policy": "global_override" if inference_contract else "per_assistant",
        "target_policy": "each_assistant_once",
        "created_at_utc": _utc_now(),
        "langsmith": {
            "workspace_id": workspace_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset.get("name"),
            "examples": expected_count,
            "split_sync": {"status": "pending"},
        },
        "split": {
            **split_settings,
            "assignments_sha256": json_sha256(assignments),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "prepared": {"accepted": len(rows), "rejected": len(rejected)},
        "conversion": {
            "roles": ROLE_MAP,
            "tool_calls": "StandardMessage tool_call blocks to OpenAI tool_calls",
            "tool_schemas_added": True,
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
    # Persist the ledger before replacing split rows so an interrupted write
    # cannot erase historical assignments during the next preparation.
    _json_dump(data_dir / "prepared" / "split_assignments.json", {
        "schema_version": 1, **split_settings, "assignments": assignments,
        "assignments_sha256": json_sha256(assignments),
    })
    if inference_contract is not None:
        manifest["inference_contract"] = inference_contract.manifest_summary()
    _jsonl_dump(data_dir / "prepared" / "train.jsonl", train)
    _jsonl_dump(data_dir / "prepared" / "validation.jsonl", validation)
    _jsonl_dump(data_dir / "prepared" / "test.jsonl", test)
    _json_dump(data_dir / "prepared" / "manifest.json", manifest)
    _json_dump(
        data_dir / "prepared" / "warnings.json",
        [*audit.duplicate_message_warnings, *preparation_warnings],
    )
    _json_dump(data_dir / "prepared" / "rejected.json", rejected)
    if inference_contract is not None:
        _json_dump(
            data_dir / "prepared" / "inference_contract.json",
            inference_contract.to_dict(),
        )
    from smithtune.evaluation.langsmith import synchronize_splits

    manifest["langsmith"]["split_sync"] = synchronize_splits(
        data_dir, manifest, source_examples, enabled=sync_splits,
    )
    _json_dump(data_dir / "prepared" / "manifest.json", manifest)
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


def require_current_preparation(manifest):
    if manifest.get("schema_version") != 2 or manifest.get("target_policy") != "each_assistant_once":
        raise PipelineError("prepared artifacts predate per-assistant tools and target-only loss; run prepare again")

"""Capture and prepare LangSmith trajectory datasets."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from artifacts import _json_dump, _jsonl_dump, _load_json, _run, _utc_now
from inference_contract import (
    ContractError,
    InferenceContract,
    contract_from_run,
    load_inference_contract,
)
from providers.base import ModelSpec, PipelineError, ReasoningPolicy
from rendering import validate_model_context, validate_reasoning_support


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


def _langsmith_run_contract_command(workspace_id: str, run_id: str) -> list[str]:
    body = {
        "id": [run_id],
        "limit": 1,
        "select": [
            "id",
            "trace_id",
            "session_id",
            "name",
            "run_type",
            "start_time",
            "extra",
        ],
    }
    return [
        "langsmith",
        "api",
        "runs/query",
        "--workspace",
        workspace_id,
        "--body",
        _canonical(body),
    ]


def capture_inference_contract(
    workspace_id: str,
    run_id: str,
    output: Path,
    *,
    runner: Callable[..., Any] = _run,
) -> dict[str, Any]:
    """Capture tool schemas and inference settings from one LangSmith LLM run."""
    result = runner(
        _langsmith_run_contract_command(workspace_id, run_id),
        capture=True,
    )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError("LangSmith run query returned invalid JSON") from exc
    runs = response.get("runs") if isinstance(response, dict) else response
    if not isinstance(runs, list) or len(runs) != 1 or runs[0].get("id") != run_id:
        raise PipelineError(f"LangSmith did not return the requested run {run_id}")
    try:
        payload = contract_from_run(runs[0], workspace_id=workspace_id)
    except ContractError as exc:
        raise PipelineError(f"cannot capture inference contract: {exc}") from exc
    _json_dump(output, payload)
    return {
        "output": str(output),
        "source_run_id": run_id,
        "contract_sha256": payload["contract_sha256"],
        "tools_sha256": payload["tools_sha256"],
        "tool_count": len(payload["tools"]),
    }


def download_dataset(
    workspace_id: str,
    dataset_id: str,
    raw_dir: Path,
    runner: Callable[..., Any] = _run,
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
        if text and calls:
            raise PipelineError("interleaved text and tool_call blocks cannot be converted without reordering")
        if calls:
            if role != "ai":
                raise PipelineError("tool_call blocks are valid only in ai messages")
            converted["content"] = ""
            converted["tool_calls"] = []
            for call in calls:
                if not isinstance(call.get("args"), dict):
                    raise PipelineError("tool_call args must be an object")
                if not all(isinstance(call.get(key), str) and call[key] for key in ("id", "name")):
                    raise PipelineError("tool_call id and name must be non-empty strings")
                converted["tool_calls"].append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False, separators=(",", ":")),
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


def _source_identity(example: dict[str, Any]) -> dict[str, str | None]:
    """Resolve native and legacy LangSmith source identities."""
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    native_thread_id = example.get("source_thread_id")
    legacy_thread_id = metadata.get("source_thread_id")
    trace_id = metadata.get("source_trace_id")
    for label, value in (
        ("source_thread_id", native_thread_id),
        ("metadata.source_thread_id", legacy_thread_id),
        ("metadata.source_trace_id", trace_id),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise PipelineError(f"example source identity {label} must be a non-empty string")
    if native_thread_id and legacy_thread_id and native_thread_id != legacy_thread_id:
        raise PipelineError("example has conflicting source_thread_id values")
    thread_id = native_thread_id or legacy_thread_id
    if not thread_id and not trace_id:
        raise PipelineError("example has no source thread or trace identity")
    return {"source_thread_id": thread_id, "source_trace_id": trace_id}


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
        for message in messages:
            _validate_source_message(message)
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


def prepare_sft_rows(
    examples: list[dict[str, Any]],
    contract: InferenceContract | None = None,
    *,
    reasoning_policy: ReasoningPolicy = "omit",
    model: ModelSpec | None = None,
) -> list[dict[str, Any]]:
    """Convert validated LangSmith trajectories to Fireworks SFT rows."""
    _validate_reasoning_policy(reasoning_policy)
    rows = []
    for source_index, example in enumerate(examples):
        identity = _source_identity(example)
        messages = []
        for source in example["inputs"]["messages"]:
            converted = convert_message(source, reasoning_policy=reasoning_policy, model=model)
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


def _source_group(row: dict[str, Any]) -> tuple[str, str]:
    source = row["_source"]
    if source.get("source_thread_id"):
        return ("thread", source["source_thread_id"])
    return ("trace", source["source_trace_id"])


def split_rows(
    rows: list[dict[str, Any]],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= validation_fraction <= 1 or not 0 <= test_fraction <= 1:
        raise PipelineError("validation and test fractions must be between zero and one")
    if validation_fraction + test_fraction > 1:
        raise PipelineError("validation and test fractions cannot total more than one")

    ranked_groups = sorted(
        {_source_group(row) for row in rows},
        key=lambda group: hashlib.sha256(
            (
                f"{SPLIT_SEED}:{group[1]}"
                if group[0] == "thread"
                else f"{SPLIT_SEED}:trace:{group[1]}"
            ).encode()
        ).hexdigest(),
    )
    train_fraction = 1 - validation_fraction - test_fraction
    required_partitions = sum(
        fraction > 0
        for fraction in (train_fraction, validation_fraction, test_fraction)
    )
    if len(ranked_groups) < required_partitions:
        raise PipelineError("not enough source thread or trace groups for the requested non-zero fractions")

    validation_count = 0
    if validation_fraction > 0:
        validation_count = max(1, round(len(ranked_groups) * validation_fraction))
        validation_count = min(
            validation_count,
            len(ranked_groups) - int(train_fraction > 0) - int(test_fraction > 0),
        )
    test_count = 0
    if test_fraction > 0:
        test_count = max(1, round(len(ranked_groups) * test_fraction))
        test_count = min(
            test_count,
            len(ranked_groups) - validation_count - int(train_fraction > 0),
        )
    validation_groups = set(ranked_groups[:validation_count])
    test_groups = set(ranked_groups[validation_count : validation_count + test_count])
    held_out_groups = validation_groups | test_groups
    train = [row for row in rows if _source_group(row) not in held_out_groups]
    validation = [row for row in rows if _source_group(row) in validation_groups]
    test = [row for row in rows if _source_group(row) in test_groups]
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
    partitions = {"train": train, "validation": validation, "test": test}
    for left_index, (left_name, left_rows) in enumerate(partitions.items()):
        left_groups = {_source_group(row) for row in left_rows}
        left_content = {_trajectory_content_hash(row) for row in left_rows}
        for right_name, right_rows in list(partitions.items())[left_index + 1 :]:
            right_groups = {_source_group(row) for row in right_rows}
            if overlap := left_groups & right_groups:
                raise PipelineError(
                    f"source group overlap between {left_name} and {right_name}: {sorted(overlap)[0]}"
                )
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
    if audit.tool_calls and inference_contract is None:
        raise PipelineError("tool trajectories require an inference contract with canonical tool schemas")
    rows = prepare_sft_rows(examples, inference_contract, reasoning_policy=reasoning_policy, model=model)
    messages_removed = audit.messages - sum(len(row["messages"]) for row in rows)
    reasoning_preserved = audit.readable_reasoning_blocks if reasoning_policy == "preserve" else 0
    rejected: list[dict[str, Any]] = []
    if check_render:
        rows, rejected, rendered = validate_model_context(rows, model)
    else:
        rendered = {}
    train, validation, test = split_rows(rows, validation_fraction, test_fraction)
    _validate_split_isolation(train, validation, test)
    audit_value = {**asdict(audit), **rendered}
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
            "method": (
                "sha256(seed:source_thread_id)"
                if all(row["_source"].get("source_thread_id") for row in rows)
                else "sha256(seed:source_thread_id; seed:trace:source_trace_id fallback)"
            ),
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
            "tool_schemas_added": inference_contract is not None,
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
    _jsonl_dump(data_dir / "prepared" / "train.jsonl", train)
    _jsonl_dump(data_dir / "prepared" / "validation.jsonl", validation)
    _jsonl_dump(data_dir / "prepared" / "test.jsonl", test)
    _json_dump(data_dir / "prepared" / "manifest.json", manifest)
    _json_dump(data_dir / "prepared" / "warnings.json", audit.duplicate_message_warnings)
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

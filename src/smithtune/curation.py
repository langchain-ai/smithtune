"""Select conversations through root trace filters and import their trajectories into LangSmith datasets."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import UUID

from smithtune import checkpoint as storage
from smithtune.artifacts import _json_dump, _load_json, _run, _utc_now, output_lock
from smithtune.dataset_artifacts import new_run_directory
from smithtune.providers.base import PipelineError


SCHEMA_VERSION = 2
# The trajectory endpoint scopes one conversation by either key.
TRAJECTORY_KEYS = ("thread_id", "trace_id")
PREVIEW_ROOTS = 20
# Each conversation is fetched and written one at a time; keep runs bounded.
MAX_LIMIT = 2000
# Conversations fetched and written at once; bounded to stay gentle on the API.
MAX_CONCURRENCY = 4
DEFAULT_CONCURRENCY = 4
DEFAULT_SOURCE_WINDOW = timedelta(days=1)
# Trajectory reads are idempotent, so transient failures are retried; example writes are not.
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 1.0
_sleep = time.sleep


class _TrajectoryPageTooLarge(PipelineError):
    """The trajectory read must be retried with a smaller page, not unchanged."""


def _trajectory_page_too_large(exc: subprocess.CalledProcessError) -> bool:
    error = (exc.stdout or "") + (exc.stderr or "")
    return (bool(re.search(r"\bHTTP 400\b", error))
            and "trajectory view exceeded response data size limit" in error
            and "Narrow the requested trajectory page" in error)


def _api(workspace_id, method, path, body=None, *, runner=_run):
    command = ["langsmith", "api", path, "--workspace", workspace_id, "--method", method]
    if body is not None:
        command += ["--input", "-"]
    try:
        result = runner(command, capture=True, input=json.dumps(body) if body is not None else None)
    except subprocess.CalledProcessError as exc:
        # API errors can echo message contents; only expose the HTTP status.
        status = re.search(r"\bHTTP [45]\d\d\b", (exc.stderr or "") + (exc.stdout or ""))
        detail = status.group() if status else "request failed"
        if path == "/v1/trajectory" and _trajectory_page_too_large(exc):
            raise _TrajectoryPageTooLarge(
                f"LangSmith {method} {path}: HTTP 400; trajectory page exceeds response size limit"
            ) from exc
        if path == "/api/v1/datasets" and detail == "HTTP 409":
            detail += "; dataset name already exists"
        raise PipelineError(f"LangSmith {method} {path}: {detail}") from exc
    except OSError as exc:
        raise PipelineError("cannot run langsmith; check that the CLI is installed and on PATH") from exc
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise PipelineError(f"LangSmith {method} {path} returned invalid JSON") from exc


def _uuid(value: Any, field: str) -> str:
    try:
        if not isinstance(value, str):
            raise ValueError
        return str(UUID(value))
    except ValueError as exc:
        raise PipelineError(f"{field} must be a UUID") from exc


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{field} must be a nonempty string")
    return value


def _destination(name, dataset_id):
    if (name is None) == (dataset_id is None):
        raise PipelineError("use --name for a new dataset or --dataset-id for an existing dataset")
    return (_text(name, "name") if name is not None else None,
            _uuid(dataset_id, "dataset ID") if dataset_id is not None else None)


def _time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
        return parsed.astimezone(UTC).isoformat()
    except (ValueError, AttributeError) as exc:
        raise PipelineError("timestamps must be ISO 8601 with a timezone") from exc


def resolve_time_window(start_time: str | None, end_time: str | None) -> tuple[str, str]:
    """Resolve an optional source window, defaulting to the 24 hours ending now."""
    resolved_end = _time(end_time if end_time is not None else _utc_now())
    resolved_start = _time(start_time) if start_time is not None else (
        datetime.fromisoformat(resolved_end) - DEFAULT_SOURCE_WINDOW
    ).isoformat()
    if datetime.fromisoformat(resolved_start) >= datetime.fromisoformat(resolved_end):
        raise PipelineError("start_time must precede end_time")
    return resolved_start, resolved_end


def _write_new(path: Path, value: dict) -> None:
    # CLI callers hold the run directory lock across selection and import.
    try:
        if path.exists():
            raise FileExistsError(path)
        _json_dump(path, value)
    except OSError as exc:
        raise PipelineError(f"cannot create {path}; use a new writable path") from exc


def _matches(values: Any) -> dict[str, dict]:
    """Validate root traces and key them by trace ID, preserving order."""
    if not isinstance(values, list):
        raise PipelineError("matches must be a list of root traces")
    matches = {}
    for item in values:
        if not isinstance(item, dict):
            raise PipelineError("each match must be a root trace object")
        trace_id = _uuid(item.get("trace_id"), "trace_id")
        thread_id = item.get("thread_id")
        if thread_id is not None:
            thread_id = _text(thread_id, "thread_id")
        match = {"trace_id": trace_id, "thread_id": thread_id, "start_time": _time(item.get("start_time"))}
        if trace_id in matches and matches[trace_id] != match:
            raise PipelineError(f"conflicting query results for trace {trace_id}; select again")
        matches[trace_id] = match
    return matches


def _conversation(match: dict) -> dict[str, str]:
    """A root selects its whole thread when it has one, otherwise its single trace."""
    if match["thread_id"] is not None:
        return {"key": "thread_id", "id": match["thread_id"]}
    return {"key": "trace_id", "id": match["trace_id"]}


def _select_dataset(
    *, workspace_id: str, project_id: str, start_time: str, end_time: str,
    output: Path, limit: int, filter: str | None = None,
    runner: Callable[..., Any] = _run,
) -> dict:
    workspace_id = _uuid(workspace_id, "workspace_id")
    project_id = _uuid(project_id, "project_id")
    start_time, end_time = _time(start_time), _time(end_time)
    if datetime.fromisoformat(start_time) >= datetime.fromisoformat(end_time):
        raise PipelineError("start_time must precede end_time")
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise PipelineError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    if filter is not None:
        _text(filter, "filter")
    if output.exists():
        raise PipelineError(f"selection already exists: {output}; use a new path")
    bound = f"lt(start_time, {json.dumps(end_time)})"
    body = {
        "project_ids": [project_id], "is_root": True,
        "min_start_time": start_time, "max_start_time": end_time,
        "page_size": 100, "filter": f"and({filter}, {bound})" if filter else bound,
        "selects": ["ID", "TRACE_ID", "THREAD_ID", "START_TIME"],
    }
    # Accumulate distinct conversations in the order LangSmith returns roots and
    # stop paging once the limit is reached; later pages are never requested.
    matches, conversations, cursors, pages = {}, {}, set(), 0
    while True:
        page = _api(workspace_id, "POST", "/api/v2/runs/query", body, runner=runner)
        pages += 1
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise PipelineError("LangSmith returned an invalid root query page")
        # Validate duplicate roots across pages as well as within each page.
        for trace_id, match in _matches(page["items"]).items():
            if trace_id in matches and matches[trace_id] != match:
                raise PipelineError(f"conflicting query results for trace {trace_id}; select again")
            matches[trace_id] = match
            conversation = _conversation(match)
            conversations.setdefault((conversation["key"], conversation["id"]), conversation)
        if len(conversations) >= limit:
            break
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError("LangSmith returned an invalid or repeated query cursor")
        cursors.add(cursor)
        body["cursor"] = cursor
    roots = list(matches.values())
    selected = list(conversations.values())[:limit]
    value = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": _utc_now(),
        "workspace_id": workspace_id, "project_id": project_id,
        "query": {"start_time": start_time, "end_time": end_time, "filter": filter, "limit": limit},
        "pages_fetched": pages, "matches": roots, "selected": selected,
    }
    _write_new(output, value)
    selected_set = {(item["key"], item["id"]) for item in selected}
    return {
        "selection": str(output), "matching_roots": len(roots), "pages_fetched": pages,
        "distinct_conversations": len(conversations), "selected_examples": len(selected),
        "trace_keyed_examples": sum(item["key"] == "trace_id" for item in selected),
        "preview": [
            {**item, "selected": tuple(_conversation(item).values()) in selected_set}
            for item in roots[:PREVIEW_ROOTS]
        ],
    }


def _create_dataset(
    *, workspace_id: str, project_id: str, start_time: str | None = None, end_time: str | None = None,
    limit: int, name: str | None = None, dataset_id: str | None = None,
    filter: str | None = None, output: Path | None = None,
    run_dir: Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY, runner: Callable[..., Any] = _run,
) -> dict:
    """Filter root runs and import the selected conversations in one operation."""
    name, dataset_id = _destination(name, dataset_id)
    _check_concurrency(concurrency)
    if output is not None and run_dir is not None:
        raise PipelineError("use --run-dir or --output, not both")
    run_dir = output.parent if output is not None else run_dir or new_run_directory()
    output = output if output is not None else run_dir / "selection.json"
    if output.exists():
        value = _load_json(output)
        query = value["query"]
        start_time = _time(start_time) if start_time else query["start_time"]
        end_time = _time(end_time) if end_time else query["end_time"]
        if value.get("workspace_id") != _uuid(workspace_id, "workspace_id") or value.get("project_id") != _uuid(project_id, "project_id") or query != {
            "start_time": start_time, "end_time": end_time, "filter": filter, "limit": limit,
        }:
            raise PipelineError("saved selection uses different source filters; use the original settings or a new run directory")
        matches = _matches(value["matches"])
        items = _selected(value["selected"], matches)
        selected = {"selected_examples": len(items), "matching_roots": len(matches),
                    "distinct_conversations": len({tuple(_conversation(m).values()) for m in matches.values()}),
                    "trace_keyed_examples": sum(item["key"] == "trace_id" for item in items)}
    else:
        start_time, end_time = resolve_time_window(start_time, end_time)
        selected = _select_dataset(
            workspace_id=workspace_id, project_id=project_id,
            start_time=start_time, end_time=end_time, output=output,
            filter=filter, limit=limit, runner=runner,
        )
    if selected["selected_examples"] == 0:
        raise PipelineError(f"no conversations matched the filters; no dataset was created or updated; selection={output}")
    imported = _import_selection(selection=output, name=name, dataset_id=dataset_id, concurrency=concurrency, runner=runner)
    return {
        **imported, "selection": str(output), "run_dir": str(run_dir),
        "matching_roots": selected["matching_roots"],
        "distinct_conversations": selected["distinct_conversations"],
        "trace_keyed_examples": selected["trace_keyed_examples"],
    }


def create_dataset(*, run_dir=None, output=None, **kwargs):
    if output is not None and run_dir is not None:
        raise PipelineError("use --run-dir or --output, not both")
    directory = output.parent if output is not None else run_dir or new_run_directory()
    with output_lock(directory):
        return _create_dataset(output=output, run_dir=directory if output is None else None, **kwargs)


def _selected(value: Any, matches: dict[str, dict]) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise PipelineError("selection must contain selected conversations; empty selections cannot be imported")
    candidates = {tuple(_conversation(match).values()) for match in matches.values()}
    selected, seen = [], set()
    for item in value:
        if not isinstance(item, dict) or item.get("key") not in TRAJECTORY_KEYS:
            raise PipelineError("each selected conversation must name a thread_id or trace_id key")
        key = item["key"]
        identity = _uuid(item.get("id"), "selected trace ID") if key == "trace_id" else _text(item.get("id"), "selected thread ID")
        if (key, identity) in seen or (key, identity) not in candidates:
            raise PipelineError("selected conversations must be unique and belong to the saved matches")
        seen.add((key, identity))
        selected.append({"key": key, "id": identity})
    return selected


def _fetch_trajectory(
    workspace_id: str, project_id: str, item: dict[str, str], *,
    runner: Callable[..., Any], retain_empty: bool = False,
) -> dict:
    from smithtune.bindings import trajectory_bindings

    body = {"project_id": project_id, item["key"]: item["id"],
            "format": "ui", "include": {"system_messages": True}}
    items, cursors = [], set()
    while True:
        attempt = 1
        while True:
            try:
                trajectory = _api(workspace_id, "POST", "/v1/trajectory", body, runner=runner)
                break
            except _TrajectoryPageTooLarge:
                if body.get("page_size") == 1:
                    # Keep only the rejection, never a prefix from earlier pages.
                    return {"messages": [], "source": None, "trace_ids": [],
                            "training_error": f"{item['key']} {item['id']} exceeds the trajectory fetch limit "
                                              "even with page_size=1; whole trajectory excluded"}
                # Narrow only the transport page. Preserve the cursor, all saved
                # messages, and system-message inclusion; never truncate a turn.
                body["page_size"] = 1
            except PipelineError as exc:
                if attempt == FETCH_ATTEMPTS or "returned invalid JSON" in str(exc):
                    raise PipelineError(f"{exc} after {attempt} attempt(s)") from exc
                _sleep(FETCH_BACKOFF_SECONDS * attempt)
                attempt += 1
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("items"), list):
            raise PipelineError(f"{item['key']} {item['id']} returned invalid trajectory items; expected format=ui")
        for entry in trajectory["items"]:
            if not isinstance(entry, dict) or entry.get("type", "message") != "message" or not isinstance(entry.get("message"), dict):
                raise PipelineError(f"{item['key']} {item['id']} returned an unsupported trajectory item")
            items.append(entry)
        cursor = trajectory.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError(f"{item['key']} {item['id']} returned an invalid or repeated continuation cursor")
        cursors.add(cursor)
        body["cursor"] = cursor
    if not items:
        if not retain_empty:
            raise PipelineError(f"{item['key']} {item['id']} returned no messages")
        return {"messages": [], "source": None, "trace_ids": [],
                "training_error": f"{item['key']} {item['id']} returned no messages"}
    # Tool configuration is evidence, not conversation content. Keep the native
    # message list and the existing per-assistant metadata representation.
    messages = [{key: value for key, value in entry["message"].items() if key != "available_tools"} for entry in items]
    trace_ids = sorted({metadata["trace_id"] for entry in items
                        if isinstance(metadata := entry.get("metadata"), dict)
                        and isinstance(metadata.get("trace_id"), str) and metadata["trace_id"]})
    if item["key"] == "trace_id" and any(tid != item["id"] for tid in trace_ids):
        raise PipelineError("trajectory evidence belongs to another trace")
    source, error = None, None
    try:
        source = trajectory_bindings(items)
    except PipelineError as exc:
        error = str(exc)
    return {"messages": messages, "source": source, "trace_ids": trace_ids, "training_error": error}


def _check_concurrency(concurrency: Any) -> None:
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise PipelineError(f"concurrency must be an integer between 1 and {MAX_CONCURRENCY}")


def _source_example(workspace, project, item):
    return {"inputs": {}, "outputs": None, "metadata": {
        "trajectory_format": "messages", "conversation_scope": "root",
        "source_workspace_id": workspace, "source_project_id": project,
        "source_scope": item["key"].removesuffix("_id"), "source_scope_id": item["id"],
    }}


def _download_examples(workspace, project, selected, run_dir, concurrency, *, runner, checkpoint=None, saved_inputs=None, destination=None):
    from smithtune.dataset import _source_key
    from smithtune.dataset_artifacts import load_conversation
    from smithtune.dataset_import import _action, import_rejection

    checkpoint = checkpoint if checkpoint is not None else storage.open_checkpoint(
        run_dir, "create", {"workspace_id": workspace, "project_id": project, "selected": selected})

    def download(item):
        saved = storage.downloaded(run_dir, checkpoint, item)
        if saved is not None and (saved["contract"] is not None or saved["rejection"] is not None):
            return item, saved
        example = saved["example"] if saved is not None else _source_example(workspace, project, item)
        if saved is None:
            trajectory = _fetch_trajectory(workspace, project, item, runner=runner)
            example["inputs"]["messages"] = trajectory["messages"]
            if trajectory["source"] is not None:
                example["metadata"]["smithtune_source"] = trajectory["source"]
            if trajectory["training_error"]:
                return item, {"example": example, "contract": None,
                              "rejection": {"code": "invalid_import_trajectory_excluded", "reason": trajectory["training_error"]}}
        if destination is not None:
            index = destination["index"]
            if index is None:  # Completed import: verify files without refreshing tools.
                if saved is None:
                    raise PipelineError("completed import is missing a saved trajectory")
                return item, saved
            _, existing_path = index.get(_source_key(example, workspace, None), (None, None))
            if existing_path is not None and _action(example, load_conversation(existing_path), triaged=False)[0] == "skipped":
                return item, {"example": example, "contract": None, "rejection": None}
        contract = {}
        rejection = import_rejection(example, workspace)
        return item, {"example": example, "contract": contract, "rejection": rejection}

    remaining = iter(selected)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = deque(pool.submit(download, item) for item in islice(remaining, concurrency))
        try:
            while pending:
                item, unit = pending.popleft().result()
                storage.record_download(run_dir, checkpoint, item, unit)
                if saved_inputs is not None:
                    from smithtune.inference_contract import json_sha256
                    saved_inputs[_source_key(unit["example"], workspace, None)] = run_dir / checkpoint["downloads"][json_sha256(item)]
                yield unit["example"]
                item = next(remaining, None)
                if item is not None:
                    pending.append(pool.submit(download, item))
        finally:
            for future in pending:
                if future.cancel():
                    continue
                try:
                    item, unit = future.result()
                except Exception:
                    continue
                storage.record_download(run_dir, checkpoint, item, unit)


def _import_selection(
    *, selection: Path, name: str | None = None, dataset_id: str | None = None, concurrency: int = DEFAULT_CONCURRENCY,
    runner: Callable[..., Any] = _run,
) -> dict:
    from smithtune.dataset import _source_key
    from smithtune.dataset_import import import_dataset

    _check_concurrency(concurrency)
    value = _load_json(selection)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError(f"selection must use schema_version {SCHEMA_VERSION}; regenerate it with dataset create")
    workspace = _uuid(value.get("workspace_id"), "workspace_id")
    project = _uuid(value.get("project_id"), "project_id")
    selected = _selected(value.get("selected"), _matches(value.get("matches")))
    name, dataset_id = _destination(name, dataset_id)
    receipt_path = selection.with_suffix(".import.json")
    if receipt_path == selection:
        raise PipelineError("selection path must differ from its import receipt path")
    checkpoint = storage.open_checkpoint(selection.parent, "create", value)
    keys = {_source_key(_source_example(workspace, project, item), workspace, None) for item in selected}
    saved_inputs, destination = {}, {}
    examples = _download_examples(workspace, project, selected, selection.parent, concurrency, runner=runner,
                                  checkpoint=checkpoint, saved_inputs=saved_inputs, destination=destination)

    def validation(example):
        metadata = example["metadata"]
        item = {"key": metadata["source_scope"] + "_id", "id": metadata["source_scope_id"]}
        unit = storage.downloaded(selection.parent, checkpoint, item)
        if unit is None or unit["example"] != example:
            raise PipelineError("saved trajectory changed before import")
        return unit["rejection"]

    try:
        return import_dataset(workspace, examples, keys, selection.parent, receipt_path, name=name, dataset_id=dataset_id,
                              runner=runner, validation=validation, saved_inputs=saved_inputs, on_index=lambda index: destination.update(index=index))
    finally:
        examples.close()

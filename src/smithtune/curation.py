"""Select conversations through root trace filters and import their trajectories into LangSmith datasets."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import UUID

from smithtune.artifacts import _json_dump, _load_json, _run, _utc_now
from smithtune.dataset_artifacts import load_conversation, new_run_directory, save_conversation
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
# Trajectory reads are idempotent, so transient failures are retried; example writes are not.
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = 1.0
_sleep = time.sleep


def _api(workspace_id, method, path, body=None, *, runner=_run):
    command = ["langsmith", "api", path, "--workspace", workspace_id, "--method", method]
    if body is not None:
        command += ["--input", "-"]
    try:
        result = runner(command, capture=True, input=json.dumps(body) if body is not None else None)
    except subprocess.CalledProcessError as exc:
        # API errors can echo message contents; only expose the HTTP status.
        status = re.search(r"\bHTTP [45]\d\d\b", exc.stderr or "")
        detail = status.group() if status else "request failed"
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


def _write_new(path: Path, value: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise PipelineError(f"cannot create {path}; use a new writable path") from exc


def _save_receipt(path: Path, value: dict) -> None:
    # Replace atomically so an interruption preserves the last known progress.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        _json_dump(temporary, value)
        temporary.replace(path)
    except OSError as exc:
        raise PipelineError(f"cannot update import receipt {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def create_dataset(
    *, workspace_id: str, project_id: str, start_time: str, end_time: str,
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
    runner: Callable[..., Any],
) -> list:
    body = {"project_id": project_id, item["key"]: item["id"],
            "format": "messages", "include": {"system_messages": True}}
    messages, cursors = [], set()
    while True:
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            try:
                trajectory = _api(workspace_id, "POST", "/v1/trajectory", body, runner=runner)
                break
            except PipelineError as exc:
                if attempt == FETCH_ATTEMPTS or "returned invalid JSON" in str(exc):
                    raise PipelineError(f"{exc} after {attempt} attempt(s)") from exc
                _sleep(FETCH_BACKOFF_SECONDS * attempt)
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
            raise PipelineError(f"{item['key']} {item['id']} returned invalid messages")
        messages.extend(trajectory["messages"])
        cursor = trajectory.get("next_cursor")
        if cursor is None:
            if not messages:
                raise PipelineError(f"{item['key']} {item['id']} returned no messages")
            return messages
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError(f"{item['key']} {item['id']} returned an invalid or repeated continuation cursor")
        cursors.add(cursor)
        body["cursor"] = cursor


def _check_concurrency(concurrency: Any) -> None:
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise PipelineError(f"concurrency must be an integer between 1 and {MAX_CONCURRENCY}")


def _source_example(workspace, project, item):
    return {"inputs": {}, "outputs": None, "metadata": {
        "trajectory_format": "messages", "conversation_scope": "root",
        "source_workspace_id": workspace, "source_project_id": project,
        "source_scope": item["key"].removesuffix("_id"), "source_scope_id": item["id"],
    }}


def _download_examples(workspace, project, selected, run_dir, concurrency, *, runner):
    def download(item):
        example = _source_example(workspace, project, item)
        example["inputs"]["messages"] = _fetch_trajectory(workspace, project, item, runner=runner)
        return save_conversation(run_dir, example)

    remaining = iter(selected)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = deque(pool.submit(download, item) for item in islice(remaining, concurrency))
        try:
            while pending:
                yield load_conversation(pending.popleft().result())
                item = next(remaining, None)
                if item is not None:
                    pending.append(pool.submit(download, item))
        finally:
            for future in pending:
                future.cancel()


def _import_selection(
    *, selection: Path, name: str | None = None, dataset_id: str | None = None, concurrency: int = DEFAULT_CONCURRENCY,
    runner: Callable[..., Any] = _run,
) -> dict:
    from smithtune.dataset_import import import_rejection

    _check_concurrency(concurrency)
    value = _load_json(selection)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError(
            f"selection must use schema_version {SCHEMA_VERSION}; regenerate it with dataset create"
        )
    workspace_id = _uuid(value.get("workspace_id"), "workspace_id")
    project_id = _uuid(value.get("project_id"), "project_id")
    matches = _matches(value.get("matches"))
    selected = _selected(value.get("selected"), matches)
    name, dataset_id = _destination(name, dataset_id)
    receipt_path = selection.with_suffix(".import.json")
    if receipt_path == selection:
        raise PipelineError("selection path must differ from its import receipt path")
    if dataset_id is not None:
        from smithtune.dataset import _source_key
        from smithtune.dataset_import import update_dataset

        keys = {_source_key(_source_example(workspace_id, project_id, item), workspace_id, None) for item in selected}
        examples = _download_examples(workspace_id, project_id, selected, selection.parent, concurrency, runner=runner)
        try:
            return update_dataset(workspace_id, dataset_id, examples, keys, selection.parent, receipt_path, runner=runner)
        finally:
            examples.close()
    # in_flight lists conversations whose fetch or write has started but not been
    # confirmed; after a failure it names exactly the sources with unknown outcomes.
    receipt = {
        "selection": str(selection.resolve()), "workspace_id": workspace_id,
        "project_id": project_id, "dataset_name": name, "concurrency": concurrency,
        "dataset_id": None, "status": "in_progress", "example_ids": [], "rejections": [],
        "in_flight": [], "pending_write": "dataset", "created_at_utc": _utc_now(),
    }
    _write_new(receipt_path, receipt)
    lock = threading.Lock()

    def update(**changes) -> None:
        with lock:
            receipt.update(changes)
            _save_receipt(receipt_path, receipt)

    def import_one(item: dict[str, str]) -> None:
        entry = {**item, "pending_write": None}
        with lock:
            receipt["in_flight"].append(entry)
            _save_receipt(receipt_path, receipt)
        messages = _fetch_trajectory(workspace_id, project_id, item, runner=runner)
        example = _source_example(workspace_id, project_id, item)
        example["inputs"]["messages"] = messages
        conversation_path = save_conversation(selection.parent, example)
        example = load_conversation(conversation_path)
        with lock:
            entry["conversation"] = str(conversation_path)
            _save_receipt(receipt_path, receipt)
        rejection = import_rejection(example, workspace_id, runner=runner)
        with lock:
            if rejection is not None:
                receipt["rejections"].append({**rejection, "conversation": str(conversation_path)})
                receipt["in_flight"].remove(entry)
                _save_receipt(receipt_path, receipt)
                return
            entry["pending_write"] = "example"
            _save_receipt(receipt_path, receipt)
        result = _api(workspace_id, "POST", "/api/v1/examples", {
            **example, "dataset_id": dataset_id,
        }, runner=runner)
        example_id = _uuid(result.get("id") if isinstance(result, dict) else None, "returned example ID")
        with lock:
            if example_id in receipt["example_ids"]:
                raise PipelineError("LangSmith returned a duplicate example ID")
            receipt["example_ids"].append(example_id)
            receipt["in_flight"].remove(entry)
            _save_receipt(receipt_path, receipt)

    try:
        dataset = _api(workspace_id, "POST", "/api/v1/datasets", {"name": name, "data_type": "kv"}, runner=runner)
        dataset_id = _uuid(dataset.get("id") if isinstance(dataset, dict) else None, "returned dataset ID")
        update(dataset_id=dataset_id, pending_write=None)
        remaining, pending, failure = iter(selected), set(), None
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while True:
                # Keep at most `concurrency` conversations in flight; after a
                # failure, submit nothing new and let started work resolve.
                while failure is None and len(pending) < concurrency:
                    item = next(remaining, None)
                    if item is None:
                        break
                    pending.add(pool.submit(import_one, item))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except PipelineError as exc:
                        failure = failure or exc
        if failure is not None:
            raise failure
        update(status="complete")
    except PipelineError as exc:
        receipt["status"] = "failed"
        try:
            _save_receipt(receipt_path, receipt)
        except PipelineError:
            pass  # The last atomic receipt still identifies the incomplete import.
        unresolved = receipt["in_flight"]
        pending_note = "; current write outcome may be unknown" if receipt["pending_write"] or any(
            entry["pending_write"] for entry in unresolved) else ""
        source = ", ".join(f"{entry['key']}={entry['id']}" for entry in unresolved) or None
        raise PipelineError(
            f"{exc}; dataset={receipt['dataset_id'] or name}; "
            f"confirmed={len(receipt['example_ids'])}; source={source}; "
            f"receipt={receipt_path}{pending_note}. Inspect the import before starting a new attempt."
        ) from exc
    return {"dataset_id": dataset_id, "example_count": len(receipt["example_ids"]),
            "rejected": len(receipt["rejections"]), "receipt": str(receipt_path)}

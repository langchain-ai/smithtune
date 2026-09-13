"""Select threads through root trace filters and import whole conversations into LangSmith datasets."""

from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from artifacts import _json_dump, _load_json, _run, _utc_now
from providers.base import PipelineError


DEFAULT_SELECTION_DIR = Path(__file__).resolve().parent / "data" / "selections"


def _api(workspace_id, method, path, body=None, *, runner=_run):
    command = ["langsmith", "api", path, "--workspace", workspace_id, "--method", method]
    if body is not None:
        command += ["--input", "-"]
    try:
        result = runner(command, capture=True, input=json.dumps(body) if body is not None else None)
    except subprocess.CalledProcessError as exc:
        # API errors can echo message contents; only expose the HTTP status.
        status = re.search(r"\bHTTP [45]\d\d\b", exc.stderr or "")
        detail = status.group() if status else "request failed; outcome may be unknown"
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


def _time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError
        return parsed.astimezone(timezone.utc).isoformat()
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
        match = {"trace_id": trace_id, "thread_id": thread_id,
                 "start_time": _time(item.get("start_time")),
                 "feedback_stats": item.get("feedback_stats")}
        if match["feedback_stats"] is not None and not isinstance(match["feedback_stats"], dict):
            raise PipelineError("feedback_stats must be an object or null")
        if trace_id in matches and matches[trace_id] != match:
            raise PipelineError(f"conflicting query results for trace {trace_id}; select again")
        matches[trace_id] = match
    return matches


def _select_dataset(
    *, workspace_id: str, project_id: str, start_time: str, end_time: str,
    output: Path, filter: str | None = None, limit: int | None = None,
    seed: int = 42, runner: Callable[..., Any] = _run,
) -> dict:
    workspace_id = _uuid(workspace_id, "workspace_id")
    project_id = _uuid(project_id, "project_id")
    start_time, end_time = _time(start_time), _time(end_time)
    if datetime.fromisoformat(start_time) >= datetime.fromisoformat(end_time):
        raise PipelineError("start_time must precede end_time")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise PipelineError("limit must be a positive integer")
    if type(seed) is not int:
        raise PipelineError("seed must be an integer")
    if filter is not None:
        _text(filter, "filter")
    if output.exists():
        raise PipelineError(f"selection already exists: {output}; use a new path")
    bound = f"lt(start_time, {json.dumps(end_time)})"
    body = {
        "project_ids": [project_id], "is_root": True,
        "min_start_time": start_time, "max_start_time": end_time,
        "page_size": 100, "filter": f"and({filter}, {bound})" if filter else bound,
        "selects": ["ID", "TRACE_ID", "THREAD_ID", "START_TIME", "FEEDBACK_STATS"],
    }
    matches, cursors = {}, set()
    while True:
        page = _api(workspace_id, "POST", "/api/v2/runs/query", body, runner=runner)
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise PipelineError("LangSmith returned an invalid root query page")
        # Validate duplicate roots across pages as well as within each page.
        for trace_id, match in _matches(page["items"]).items():
            if trace_id in matches and matches[trace_id] != match:
                raise PipelineError(f"conflicting query results for trace {trace_id}; select again")
            matches[trace_id] = match
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError("LangSmith returned an invalid or repeated query cursor")
        cursors.add(cursor)
        body["cursor"] = cursor
    roots = sorted(matches.values(), key=lambda item: item["trace_id"])
    threads = {item["thread_id"] for item in roots if item["thread_id"] is not None}
    candidates = sorted(threads)
    selected = candidates if limit is None else sorted(random.Random(seed).sample(candidates, min(limit, len(candidates))))
    value = {
        "schema_version": 1, "created_at_utc": _utc_now(),
        "workspace_id": workspace_id, "project_id": project_id, "scope": "thread",
        "query": {"start_time": start_time, "end_time": end_time,
                  "filter": filter, "limit": limit, "seed": seed},
        "matches": roots, "selected_ids": selected,
    }
    _write_new(output, value)
    selected_set = set(selected)
    return {
        "selection": str(output), "scope": "thread", "matching_roots": len(roots),
        "distinct_threads": len(threads),
        "excluded_unthreaded_roots": sum(item["thread_id"] is None for item in roots),
        "eligible_examples": len(candidates), "selected_examples": len(selected),
        "preview": [{**item, "selected": item["thread_id"] in selected_set} for item in roots[:20]],
    }


def create_dataset(
    *, workspace_id: str, project_id: str, start_time: str, end_time: str,
    name: str, filter: str | None = None, limit: int | None = None,
    seed: int = 42, output: Path | None = None, runner: Callable[..., Any] = _run,
) -> dict:
    """Filter root runs and import the selected conversations in one operation."""
    name = _text(name, "name")
    if output is None:
        output = DEFAULT_SELECTION_DIR / f"{uuid4()}.json"
    selected = _select_dataset(
        workspace_id=workspace_id, project_id=project_id,
        start_time=start_time, end_time=end_time, output=output,
        filter=filter, limit=limit, seed=seed, runner=runner,
    )
    if selected["selected_examples"] == 0:
        raise PipelineError(
            "no conversation threads matched the filters; no dataset was created; "
            f"excluded_unthreaded_roots={selected['excluded_unthreaded_roots']}; selection={output}"
        )
    imported = _import_selection(selection=output, name=name, runner=runner)
    return {
        **imported, "selection": str(output),
        "matching_roots": selected["matching_roots"],
        "distinct_threads": selected["distinct_threads"],
        "excluded_unthreaded_roots": selected["excluded_unthreaded_roots"],
    }


def _import_selection(*, selection: Path, name: str, runner: Callable[..., Any] = _run) -> dict:
    value = _load_json(selection)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise PipelineError("selection must use schema_version 1")
    workspace_id = _uuid(value.get("workspace_id"), "workspace_id")
    project_id = _uuid(value.get("project_id"), "project_id")
    if value.get("scope") != "thread":
        raise PipelineError(
            "only thread selections are supported; regenerate with dataset create and project filters"
        )
    matches = _matches(value.get("matches"))
    ids = value.get("selected_ids")
    if not isinstance(ids, list) or not ids:
        raise PipelineError("selection must contain selected_ids; empty selections cannot be imported")
    ids = [_text(item, "selected thread ID") for item in ids]
    candidates = {item["thread_id"] for item in matches.values() if item["thread_id"] is not None}
    if len(ids) != len(set(ids)) or not set(ids) <= candidates:
        raise PipelineError("selected_ids must be unique and belong to the saved matches")
    name = _text(name, "name")
    receipt_path = selection.with_suffix(".import.json")
    if receipt_path == selection:
        raise PipelineError("selection path must differ from its import receipt path")
    receipt = {
        "selection": str(selection.resolve()), "workspace_id": workspace_id,
        "project_id": project_id, "scope": "thread", "dataset_name": name,
        "dataset_id": None, "status": "in_progress", "example_ids": [],
        "current_source_id": None, "pending_write": "dataset", "created_at_utc": _utc_now(),
    }
    _write_new(receipt_path, receipt)
    try:
        dataset = _api(workspace_id, "POST", "/api/v1/datasets", {"name": name, "data_type": "kv"}, runner=runner)
        dataset_id = _uuid(dataset.get("id") if isinstance(dataset, dict) else None, "returned dataset ID")
        receipt.update(dataset_id=dataset_id, pending_write=None)
        _save_receipt(receipt_path, receipt)
        common = {"trajectory_format": "messages", "conversation_scope": "root",
                  "source_project_id": project_id, "selection_scope": "thread"}
        for source_id in ids:
            receipt.update(current_source_id=source_id, pending_write="example")
            _save_receipt(receipt_path, receipt)
            result = _api(workspace_id, "POST", f"/v1/platform/datasets/{dataset_id}/examples/thread-imports", {
                "project_id": project_id, "thread_ids": [source_id], "metadata": common,
            }, runner=runner)
            if not isinstance(result, dict) or result.get("count") != 1 or not isinstance(result.get("example_ids"), list) or len(result["example_ids"]) != 1:
                raise PipelineError("thread import did not confirm exactly one example")
            example_id = _uuid(result["example_ids"][0], "returned example ID")
            if example_id in receipt["example_ids"]:
                raise PipelineError("LangSmith returned a duplicate example ID")
            receipt["example_ids"].append(example_id)
            receipt.update(pending_write=None, current_source_id=None)
            _save_receipt(receipt_path, receipt)
        receipt["status"] = "complete"
        _save_receipt(receipt_path, receipt)
    except PipelineError as exc:
        receipt["status"] = "failed"
        try:
            _save_receipt(receipt_path, receipt)
        except PipelineError:
            pass  # The last atomic receipt still identifies the incomplete import.
        pending = "; current write outcome may be unknown" if receipt["pending_write"] else ""
        raise PipelineError(
            f"{exc}; dataset={receipt['dataset_id'] or name}; "
            f"confirmed={len(receipt['example_ids'])}; source={receipt['current_source_id']}; "
            f"receipt={receipt_path}{pending}. Inspect the import before starting a new attempt."
        ) from exc
    return {"dataset_id": dataset_id, "example_count": len(receipt["example_ids"]), "receipt": str(receipt_path)}

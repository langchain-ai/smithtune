"""Query source roots and fetch complete trajectories for dataset curation."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from smithtune.artifacts import _run, _utc_now
from smithtune.providers.base import PipelineError


# The trajectory endpoint scopes one conversation by either key.
TRAJECTORY_KEYS = ("thread_id", "trace_id")
# Each conversation is fetched and written one at a time; keep runs bounded.
MAX_LIMIT = 2000
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

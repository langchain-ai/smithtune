"""LangSmith wire helpers for the checkpoint curation stages."""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from smithtune.artifacts import _run, _utc_now
from smithtune.providers.base import PipelineError

DEFAULT_SOURCE_WINDOW = timedelta(days=1)

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
    runner: Callable[..., Any],
) -> list:
    body = {"project_id": project_id, item["key"]: item["id"],
            "format": "messages", "include": {"system_messages": True}}
    messages, cursors = [], set()
    while True:
        trajectory = _api(workspace_id, "POST", "/v1/trajectory", body, runner=runner)
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
            raise PipelineError(f"{item['key']} {item['id']} returned invalid messages")
        messages.extend(trajectory["messages"])
        cursor = trajectory.get("next_cursor")
        if cursor is None:
            return messages
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError(f"{item['key']} {item['id']} returned an invalid or repeated continuation cursor")
        if len(cursors) >= 1000:
            raise PipelineError("trajectory exceeds 1000 pages; narrow the source")
        cursors.add(cursor)
        body["cursor"] = cursor

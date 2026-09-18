"""Read source evidence in memory and save one bound conversation at a time."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid5

from smithtune.artifacts import _run, _utc_now
from smithtune.curation import _api, _fetch_trajectory, _matches, _uuid, resolve_time_window
from smithtune.dataset import _project_start_time, _query_contract_runs, validate_import_messages
from smithtune.dataset_artifacts import save_conversation
from smithtune.inference_contract import json_sha256
from smithtune.bindings import capture_bindings, validate_bound_messages
from smithtune.providers.base import PipelineError


MAX_SOURCE_PAGES = 1000
MAX_EXPANDED_TRACES = 10_000


def _fetch(command, *, runner, **kwargs):
    """One retry layer for idempotent source reads; no response cache."""
    for attempt in range(3):
        try:
            return runner(command, **kwargs)
        except (subprocess.CalledProcessError, PipelineError, OSError):
            if attempt == 2:
                raise
            print(f"Retrying source read (attempt {attempt + 2}/3).", file=sys.stderr)
            time.sleep(2 ** attempt)


def query_runs(workspace: str, project: str, query: dict, *, runner=_run) -> list[dict]:
    tid = _uuid(query.get("trace"), "trace id")
    selects = ["ID", "TRACE_ID", "PARENT_RUN_IDS", "PROJECT_ID", "IS_ROOT", "NAME", "RUN_TYPE",
               "START_TIME", "END_TIME", "INPUTS", "OUTPUTS", "ERROR", "EXTRA", "ATTACHMENTS"]
    path = f"/api/v2/traces/{tid}/runs?" + urlencode([("project_id", project), *[("selects", field) for field in selects]])
    # QueryTraceRequestQueryParams has no cursor; QueryTraceResponseBody is
    # one complete items array (unlike /runs/query and /v1/trajectory).
    # Omit both time bounds to retain evidence outside the selection window.
    # Fail closed if a future server starts returning a continuation cursor.
    page = _api(workspace, "GET", path, runner=runner)
    if not isinstance(page, dict) or not isinstance(page.get("items"), list) or page.get("next_cursor"):
        raise PipelineError("invalid or incomplete V2 trace run response")
    rows = {}
    for row in page["items"]:
        if not isinstance(row, dict) or row.get("project_id") != project or row.get("trace_id") != tid:
            raise PipelineError("V2 run response returned another project or trace")
        rid = _uuid(row.get("id"), "run id")
        parents = row.get("parent_run_ids", [])
        if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
            raise PipelineError("invalid V2 run ancestry")
        if row.get("is_root") is False and not parents:
            raise PipelineError("V2 child run has no parent")
        normalized = {**row, "session_id": project, "parent_run_id": parents[-1] if parents else None,
                      "run_type": str(row.get("run_type", "")).lower()}
        if rid in rows and rows[rid] != normalized:
            raise PipelineError("source run changed during snapshot")
        rows[rid] = normalized
    return sorted(rows.values(), key=lambda r: (r.get("start_time") or "", r["id"]))


def multimodal_types(trace: dict) -> list[str]:
    """Detect media payloads, not ordinary text that mentions images or audio."""
    found = set()
    kinds = {"image", "image_url", "input_image", "output_image", "audio", "audio_url", "input_audio",
             "output_audio", "video", "video_url", "file", "input_file", "document"}

    def visit(value):
        if isinstance(value, dict):
            kind = value.get("type")
            kind = kind.lower() if isinstance(kind, str) else None
            file_payload = kind not in {"file", "input_file", "document"} or any(key in value for key in ("file", "file_id", "file_data", "source", "source_type", "data"))
            if isinstance(kind, str) and kind.lower() in kinds and file_payload:
                found.add(kind.lower())
            for key in ("mime_type", "media_type", "mimeType"):
                mime = value.get(key)
                if isinstance(mime, str) and mime.split("/")[0] in {"image", "audio", "video"}:
                    found.add(mime.split("/")[0])
                elif mime == "application/pdf":
                    found.add("document")
            for key in ("image", "image_url", "audio", "input_audio", "video"):
                payload = value.get(key)
                if isinstance(payload, dict) and any(field in payload for field in ("data", "url", "file_id", "source")):
                    found.add(key)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            if re.search(r"data:(?:(?:image|audio|video)/[^;,\s]+|application/pdf);base64,[A-Za-z0-9+/]{20}", value):
                found.add("embedded media")
            # Tool results can hold a JSON-encoded content block.
            if value.lstrip().startswith(("{", "[")) and re.search(r'"type"\s*:\s*"(?:image|audio|video|input_image|input_audio|image_url|file|document)"', value):
                try:
                    visit(json.loads(value))
                except ValueError:
                    pass

    visit(trace.get("messages", []))
    text_suffixes = {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml", ".py", ".js", ".ts", ".html", ".xml"}
    for run in trace.get("runs", []):
        visit(run.get("inputs"))
        visit(run.get("outputs"))
        if any(Path(name).suffix.lower() not in text_suffixes for name in (run.get("attachments") or {})):
            found.add("attachment")
    return sorted(found)


def _root_selection(source: dict, *, runner) -> list[dict]:
    bound = f"lt(start_time,{json.dumps(source['end_time'])})"
    user_filter = source.get("filter")
    body = {"project_ids": [source["project_id"]], "is_root": True,
            "min_start_time": source["start_time"], "max_start_time": source["end_time"],
            "filter": f"and({user_filter},{bound})" if user_filter else bound,
            "page_size": 100, "selects": ["ID", "TRACE_ID", "THREAD_ID", "START_TIME", "FEEDBACK_STATS"]}
    rows, cursors = {}, set()
    while True:
        if len(cursors) >= MAX_SOURCE_PAGES:
            raise PipelineError("root query exceeds the page limit; narrow the time window or filter")
        page = _api(source["workspace_id"], "POST", "/api/v2/runs/query", body, runner=runner)
        if not isinstance(page, dict):
            raise PipelineError("invalid root query page")
        for tid, row in _matches(page.get("items")).items():
            if tid in rows and rows[tid] != row:
                raise PipelineError("root changed during selection")
            rows[tid] = row
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError("invalid root query cursor")
        cursors.add(cursor)
        body["cursor"] = cursor
    # Select distinct conversations before sampling, so busy threads do not
    # consume multiple slots or gain extra sampling probability.
    groups = {}
    for root in sorted(rows.values(), key=lambda r: r["trace_id"]):
        key = ("thread", root["thread_id"]) if root["thread_id"] else ("trace", root["trace_id"])
        groups.setdefault(key, root)
    roots = [groups[key] for key in sorted(groups)]
    return sorted(random.Random(source["seed"]).sample(roots, min(source["limit"], len(roots))),
                  key=lambda r: (r["thread_id"] or "", r["trace_id"]))


def thread_trace_ids(workspace: str, project: str, thread: str, *, start_time: str, end_time: str, runner) -> list[str]:
    # Query metadata over the full project history, not the selected time window.
    # Messages come only from /v1/trajectory; these IDs locate tools and media.
    roots = _query_contract_runs(workspace, {
        "project_ids": [project], "is_root": True,
        "filter": f"eq(thread_id,{json.dumps(thread)})",
        "min_start_time": start_time, "max_start_time": end_time,
        "selects": ["ID", "TRACE_ID", "THREAD_ID", "PROJECT_ID", "START_TIME"],
    }, runner=runner, read_attempts=1)
    if any(root.get("thread_id") != thread for root in roots):
        raise PipelineError("thread source query returned another thread")
    return list(dict.fromkeys(_uuid(root.get("trace_id"), "trace id")
                             for root in sorted(roots, key=lambda r: (r.get("start_time") or "", r["id"]))))


def source_options(workspace_id, project_id, start_time=None, end_time=None, *, filter=None, limit=100, seed=42) -> dict:
    start_time, end_time = resolve_time_window(start_time, end_time)
    value = {"workspace_id": _uuid(workspace_id, "workspace id"), "project_id": _uuid(project_id, "project id"),
             "start_time": start_time, "end_time": end_time, "filter": filter, "limit": limit, "seed": seed}
    if type(limit) is not int or not 1 <= limit <= 2000:
        raise PipelineError("conversation limit must be between 1 and 2000")
    if type(seed) is not int or (filter is not None and (not isinstance(filter, str) or not filter.strip())):
        raise PipelineError("invalid source seed or filter")
    return value




class SourceExclusion(PipelineError):
    """Terminal evidence failure for one selected conversation."""


def _download(source, selected, *, runner):
    workspace, project = source["workspace_id"], source["project_id"]
    scope, scope_id = selected["scope"], selected["scope_id"]
    thread = scope == "thread"
    start = _project_start_time(workspace, project, runner=runner, read_attempts=1) if thread else None
    traces = thread_trace_ids(workspace, project, scope_id, start_time=start, end_time=_utc_now(), runner=runner) if thread else [scope_id]
    if selected["trace_id"] not in traces:
        raise SourceExclusion("selected root no longer belongs to the conversation")
    if len(traces) > MAX_EXPANDED_TRACES:
        raise SourceExclusion("conversation exceeds 10000 traces")
    messages = _fetch_trajectory(workspace, project, {"key": scope + "_id", "id": scope_id}, runner=runner)
    if not messages:
        raise SourceExclusion(f"{scope} {scope_id} returned no messages")
    runs = [run for tid in traces for run in query_runs(workspace, project, {"trace": tid}, runner=runner)]
    if thread and set(traces) != set(thread_trace_ids(workspace, project, scope_id, start_time=start, end_time=_utc_now(), runner=runner)):
        raise SourceExclusion("thread membership changed during download; use a fresh checkpoint for this conversation")
    media = multimodal_types({"messages": messages, "runs": runs})
    if media:
        raise SourceExclusion("multimodal content: " + ", ".join(media))
    example = {"id": str(uuid5(NAMESPACE_URL, json_sha256([workspace, project, scope, scope_id]))),
               "inputs": {"messages": messages}, "outputs": None,
               "metadata": {"trajectory_format": "messages", "conversation_scope": "root",
                            "source_workspace_id": workspace, "source_project_id": project,
                            "source_scope": scope, "source_scope_id": scope_id}}
    try:
        validate_import_messages(example)
        example["metadata"]["smithtune_source"] = capture_bindings(messages, runs)
        validate_bound_messages(example)
    except PipelineError as exc:
        raise SourceExclusion(str(exc)) from exc
    return example


def pull(directory, checkpoint, *, runner=_run, concurrency=4):
    """Caller holds the command lock. Completed selections make no source reads."""
    from smithtune.checkpoint import save, conversations

    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise PipelineError("pull concurrency must be between 1 and 4")
    # Verify saved evidence even on a no-op; never replace it after voting.
    conversations(directory, checkpoint)
    read = partial(_fetch, runner=runner)
    if checkpoint["selection"] is None:
        roots = _root_selection(checkpoint["source"], runner=read)
        checkpoint["selection"] = [{"scope": "thread" if r["thread_id"] else "trace",
                                    "scope_id": r["thread_id"] or r["trace_id"],
                                    "trace_id": r["trace_id"], "status": "pending"} for r in roots]
        save(directory, checkpoint)
    pending = [s for s in checkpoint["selection"] if s["status"] == "pending"]
    if pending and (directory / "triage.jsonl").exists():
        raise PipelineError("cannot refetch conversations after voting starts")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_download, checkpoint["source"], selected, runner=read): selected for selected in pending}
        try:
            for future in as_completed(futures):
                selected = futures[future]
                try:
                    example = future.result()
                    path = save_conversation(directory, example)
                    selected.update(status="complete", file=str(path.relative_to(directory)))
                    selected.pop("error", None)
                except SourceExclusion as exc:
                    selected.update(status="excluded", reason=str(exc))
                except Exception as exc:
                    selected["error"] = f"source read incomplete ({type(exc).__name__}); retry pull"
                save(directory, checkpoint)
        finally:
            for future in futures:
                future.cancel()
    return checkpoint

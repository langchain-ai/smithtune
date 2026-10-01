"""Freeze LangSmith trajectory messages and tool availability before judging."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from smithtune import checkpoint as storage
from smithtune.artifacts import _atomic_text, _json_dump, _load_json, _run, _utc_now
from smithtune.curation import MAX_LIMIT, _api, _fetch_trajectory, _matches, _trajectory_page_too_large, _uuid, resolve_time_window
from smithtune.dataset import _project_start_time, _query_runs, validate_import_messages
from smithtune.inference_contract import ContractError, json_sha256, parse_inference_contract
from smithtune.bindings import validate_bound_messages
from smithtune.dataset_artifacts import LazySequence
from smithtune.providers.base import PipelineError


MAX_COLLECTION_ROUNDS = 3
MAX_SOURCE_PAGES = 1000
MAX_EXPANDED_TRACES = 10_000


def _fetch(command, *, runner, cache_dir, use_cache=True, **kwargs):
    """Retry idempotent source reads; never wrap dataset writes."""
    path = cache_dir / (json_sha256({"command": command, "input": kwargs.get("input")}) + ".json")
    if use_cache and path.exists():
        return subprocess.CompletedProcess(command, 0, stdout=path.read_text(encoding="utf-8"), stderr="")
    for attempt in range(3):
        try:
            result = runner(command, **kwargs)
        except subprocess.CalledProcessError as exc:
            if command[:3] == ["langsmith", "api", "/v1/trajectory"] and _trajectory_page_too_large(exc):
                # Let the trajectory paginator narrow this deterministic failure.
                raise
            if attempt == 2:
                raise
            # The LangSmith CLI can write request errors to stdout.
            error = (exc.stdout or "") + (exc.stderr or "")
            limited = bool(re.search(r"\b429\b|rate.limit", error, re.IGNORECASE))
            delay = 30 * (attempt + 1) if limited else 2 ** attempt
            print(f"{'LangSmith rate limit. ' if limited else ''}Retrying trace download in {delay}s (attempt {attempt + 2}/3).", file=sys.stderr)
            time.sleep(delay)
            continue
        # Cache only successful JSON responses, never request errors.
        try:
            json.loads(result.stdout)
        except (ValueError, TypeError):
            return result
        if use_cache:
            _atomic_text(path, result.stdout)
        return result


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


def _root_query(source: dict) -> dict:
    bound = f"lt(start_time,{json.dumps(source['end_time'])})"
    user_filter = source.get("filter")
    body = {"project_ids": [source["project_id"]], "is_root": True,
            "min_start_time": source["start_time"], "max_start_time": source["end_time"],
            "filter": f"and({user_filter},{bound})" if user_filter else bound,
            "page_size": 100, "selects": ["ID", "TRACE_ID", "THREAD_ID", "START_TIME", "FEEDBACK_STATS"]}
    return body


def _root_selection(source: dict, *, runner) -> list[dict]:
    body = _root_query(source)
    rows, cursors, trajectories = {}, set(), {}
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
            key = ("thread", row["thread_id"]) if row["thread_id"] else ("trace", tid)
            trajectories.setdefault(key, row)
        if source.get("selection_mode") == "trajectories" and len(trajectories) >= source["limit"]:
            break
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError("invalid root query cursor")
        cursors.add(cursor)
        body["cursor"] = cursor
    if source.get("selection_mode") == "trajectories":
        return list(trajectories.values())[:source["limit"]]
    roots = sorted(rows.values(), key=lambda r: r["trace_id"])
    return sorted(random.Random(source["seed"]).sample(roots, min(source["limit"], len(roots))), key=lambda r: r["trace_id"]) if source["limit"] else roots


def _candidate_roots(source: dict, output_dir: Path, checkpoint: dict, *, candidate_limit: int, runner):
    """Resume candidate discovery from saved pages; count each scope once."""
    roots = checkpoint["roots"]
    yield from list(roots)
    seen = {_scope_key(root) for root in roots}
    selection = checkpoint.setdefault("root_selection", {
        "pending": [], "cursor": None, "exhausted": False, "cursors": [],
    })
    while len(roots) < candidate_limit:
        if selection["pending"]:
            root = selection["pending"].pop(0)
            key = _scope_key(root)
            if key in seen:
                continue
            roots.append(root)
            seen.add(key)
            storage.save(output_dir, checkpoint)
            yield root
            continue
        if selection["exhausted"]:
            return
        if len(selection["cursors"]) >= MAX_SOURCE_PAGES:
            raise PipelineError("root query exceeds the page limit; narrow the time window or filter")
        body = _root_query(source)
        if selection["cursor"] is not None:
            body["cursor"] = selection["cursor"]
        page = _api(source["workspace_id"], "POST", "/api/v2/runs/query", body, runner=runner)
        if not isinstance(page, dict):
            raise PipelineError("invalid root query page")
        cursor = page.get("next_cursor")
        if cursor is not None:
            if not isinstance(cursor, str) or not cursor or cursor in selection["cursors"]:
                raise PipelineError("invalid root query cursor")
            selection["cursors"].append(cursor)
        selection.update(pending=list(_matches(page.get("items")).values()),
                         cursor=cursor, exhausted=cursor is None)
        storage.save(output_dir, checkpoint)


def _scope_key(root):
    return ("thread", root["thread_id"]) if root["thread_id"] else ("trace", root["trace_id"])


def thread_trace_ids(workspace: str, project: str, thread: str, *, start_time: str, end_time: str, runner) -> list[str]:
    # Query metadata over the full project history, not the selected time window.
    # Messages and tools come from /v1/trajectory; these IDs verify thread membership.
    roots = _query_runs(workspace, {
        "project_ids": [project], "is_root": True,
        "filter": f"eq(thread_id,{json.dumps(thread)})",
        "min_start_time": start_time, "max_start_time": end_time,
        "selects": ["ID", "TRACE_ID", "THREAD_ID", "PROJECT_ID", "START_TIME"],
    }, runner=runner)
    if any(root.get("thread_id") != thread for root in roots):
        raise PipelineError("thread source query returned another thread")
    return list(dict.fromkeys(_uuid(root.get("trace_id"), "trace id")
                             for root in sorted(roots, key=lambda r: (r.get("start_time") or "", r["id"]))))


def source_options(workspace_id, project_id, start_time=None, end_time=None, *, filter=None, limit=100, seed=42, target_count=None, max_candidates=None, review_mode=None) -> dict:
    start_time, end_time = resolve_time_window(start_time, end_time)
    value = {"workspace_id": _uuid(workspace_id, "workspace id"), "project_id": _uuid(project_id, "project id"),
             "start_time": start_time, "end_time": end_time, "filter": filter, "limit": limit, "seed": seed}
    if limit is not None and (type(limit) is not int or limit < 1):
        raise PipelineError("trace limit must be positive")
    if type(seed) is not int or (filter is not None and (not isinstance(filter, str) or not filter.strip())):
        raise PipelineError("invalid source seed or filter")
    if target_count is not None:
        if type(target_count) is not int or not 1 <= target_count <= MAX_LIMIT:
            raise PipelineError(f"target count must be between 1 and {MAX_LIMIT}")
        if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_LIMIT:
            raise PipelineError(f"max candidates must be between 1 and {MAX_LIMIT}")
        value.pop("limit")
        value.pop("seed")
        value.update(target_count=target_count, max_candidates=max_candidates)
        if review_mode is not None:
            if review_mode not in {"council", "none"}:
                raise PipelineError("invalid review mode")
            value["review_mode"] = review_mode
    return value


def training_error(unit: dict) -> str | None:
    """Check saved conversations with preparation's provider-neutral validation."""
    if unit["training_error"]:
        return unit["training_error"]
    try:
        if "smithtune_source" in unit["example"].get("metadata", {}):
            validate_bound_messages(unit["example"])
        else:
            messages = validate_import_messages(unit["example"])
            contract = parse_inference_contract(unit["contract"])
            contract.validate_messages(messages)
    except (PipelineError, ContractError) as exc:
        return str(exc)
    return None


def snapshot(source: dict, output_dir: Path, *, runner=_run, concurrency=1, extend=False) -> dict:
    if type(concurrency) is not int or not 1 <= concurrency <= 4:
        raise PipelineError("download concurrency must be between 1 and 4")
    path = output_dir / "snapshot.json"
    checkpoint = storage.open_checkpoint(output_dir, "triage", source)
    value = load_snapshot(output_dir) if path.exists() else None
    if value is not None and value["source"] != source:
        raise PipelineError("triage snapshot uses a different source query; use a new output directory")
    collection = checkpoint.get("collection")
    if value is not None:
        active_round = collection["round"] if collection else None
        complete = value.get("selection_result", {}).get("round") == active_round
        if complete and not extend:
            return value
        if complete and extend:
            if collection is None or collection["round"] >= MAX_COLLECTION_ROUNDS:
                raise PipelineError("collection limit reached; use the saved dataset or start a new curation run")
            collection.update(round=collection["round"] + 1, start=len(checkpoint["roots"]))
            storage.save(output_dir, checkpoint)
    if source.get("review_mode") and collection is None:
        collection = checkpoint["collection"] = {"round": 1, "start": 0}
        storage.save(output_dir, checkpoint)
    # Raw API pages and run trees are temporary. Resume reuses complete units,
    # never stitches an interrupted conversation to newly fetched pages.
    runner = partial(_fetch, runner=runner, cache_dir=output_dir, use_cache=False)
    goal = source.get("target_count")
    target = None if source.get("review_mode") == "council" else goal
    if "roots" not in checkpoint:
        checkpoint.update(roots=[] if goal is not None else _root_selection(source, runner=runner), captured_at=_utc_now())
        storage.save(output_dir, checkpoint)
    roots = checkpoint["roots"]
    workspace, project = source["workspace_id"], source["project_id"]
    if goal is not None:
        candidate_limit = (collection["start"] if collection else 0) + source["max_candidates"]
        candidates = _candidate_roots(source, output_dir, checkpoint, candidate_limit=candidate_limit, runner=runner)
        print(f"Collecting up to {source['max_candidates']} new candidates" +
              (f" for round {collection['round']}..." if collection else "..."), file=sys.stderr)
    else:
        candidates = iter(roots)
        print(f"Downloading trajectories for {len(roots)} selected roots...", file=sys.stderr)
    def download(entry):
        key, root = entry
        thread, tid = root["thread_id"], root["trace_id"]
        saved = storage.downloaded(output_dir, checkpoint, key)
        if saved is not None:
            return key, saved
        trajectory = _fetch_trajectory(workspace, project, {"key": "thread_id" if thread else "trace_id", "id": thread or tid},
                                       runner=runner, retain_empty=True)
        # Validate the downloaded evidence's source, without requiring a live
        # thread to remain unchanged while its pages are read.
        source_traces = thread_trace_ids(workspace, project, thread, start_time=checkpoint["project_start"],
                                        end_time=_utc_now(), runner=runner) if thread else [tid]
        if tid not in source_traces:
            raise PipelineError("selected root was not found in its conversation")
        downloaded_traces = set(trajectory["trace_ids"])
        if not downloaded_traces <= set(source_traces):
            # LangSmith can attribute runs to a thread whose root is outside this project.
            # Never train on that evidence; exclude this trajectory, not the whole pull.
            trajectory = {**trajectory, "messages": [], "source": None,
                          "training_error": f"thread {thread} includes evidence from traces outside its source; whole trajectory excluded"}
            downloaded_traces &= set(source_traces)
        # Empty/rejected payloads retain the selected root for recovery metadata.
        unit_traces = [trace_id for trace_id in source_traces if trace_id in downloaded_traces] or [tid]
        unit_records = [{"trace_id": trace_id, "thread_id": thread, "project_id": project} for trace_id in unit_traces]
        all_messages = trajectory["messages"]
        example_id = str(uuid5(NAMESPACE_URL, json_sha256({"workspace": workspace, "project": project, "key": key, "messages": all_messages})))
        example = {"id": example_id, "inputs": {"messages": all_messages}, "outputs": None,
                   "metadata": {"trajectory_format": "messages", "conversation_scope": "root",
                                "source_project_id": project, "source_thread_id": thread,
                                "source_workspace_id": workspace, "source_scope": "thread" if thread else "trace",
                                "source_scope_id": thread or unit_traces[0],
                                "source_trace_id": unit_traces[0], "triage_trace_ids": unit_traces}}
        if trajectory["source"] is not None:
            example["metadata"]["smithtune_source"] = trajectory["source"]
        unit = {"example": example, "trace_ids": unit_traces, "contract": None,
                "training_error": trajectory["training_error"], "traces": unit_records,
                "multimodal_types": multimodal_types({"messages": all_messages})}
        unit["training_error"] = training_error(unit)
        return key, unit

    selected = {}
    downloaded = trace_count = usable = 0
    exhausted = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = deque()
        try:
            while True:
                # Reserve one remaining target slot per in-flight download, so
                # successful workers cannot overshoot the requested count.
                while not exhausted and len(pending) < concurrency and (target is None or usable + len(pending) < target):
                    root = next(candidates, None)
                    if root is None:
                        exhausted = True
                        break
                    key = _scope_key(root)
                    if key in selected:
                        continue
                    if root["thread_id"] and "project_start" not in checkpoint:
                        checkpoint["project_start"] = _project_start_time(workspace, project, runner=runner)
                        storage.save(output_dir, checkpoint)
                    selected[key] = root
                    pending.append(pool.submit(download, (key, root)))
                if not pending:
                    break
                key, unit = pending.popleft().result()
                trace_count += len(unit["traces"])
                if trace_count > MAX_EXPANDED_TRACES:
                    raise PipelineError("thread expansion exceeds 10000 traces; select fewer roots")
                storage.record_download(output_dir, checkpoint, key, unit)
                downloaded += 1
                usable += not unit["training_error"] and not unit.get("multimodal_types")
                if downloaded % 10 == 0:
                    print(f"Downloaded {downloaded} trajectories, {trace_count} traces.", file=sys.stderr)
        finally:
            for future in pending:
                if future.cancel():
                    continue
                try:
                    key, unit = future.result()
                except Exception:
                    continue
                storage.record_download(output_dir, checkpoint, key, unit)
    if not selected:
        raise PipelineError("no traces match the source query; check the project, time window, and filter")
    value = {"schema_version": 3, "source": source,
             "selected_trace_ids": [root["trace_id"] for root in roots],
             "unit_files": [checkpoint["downloads"][json_sha256(key)] for key in selected]}
    if goal is not None:
        selection = checkpoint.get("root_selection", {})
        source_exhausted = bool(selection.get("exhausted") and
                                not any(_scope_key(root) not in selected for root in selection.get("pending", [])))
        value["selection_result"] = {
            "target_count": goal, "max_candidates": source["max_candidates"],
            "examined": downloaded, "usable": usable, "source_exhausted": source_exhausted,
            "stop_reason": "target_reached" if target is not None and usable >= target else
                           "source_exhausted" if source_exhausted else "candidate_cap",
        }
        if target is not None:
            value["selection_result"]["target_met"] = usable >= target
        if collection:
            value["selection_result"].update(round=collection["round"], round_examined=downloaded - collection["start"])
    value["snapshot_sha256"] = json_sha256(value)
    _json_dump(path, value)
    return load_snapshot(output_dir)


def load_snapshot(output_dir: Path) -> dict:
    value = _load_json(output_dir / "snapshot.json")
    if isinstance(value, dict) and value.get("schema_version") == 1:
        raise PipelineError("this triage snapshot predates system-message capture; use a new output directory to download and judge again")
    if not isinstance(value, dict) or value.get("schema_version") not in (2, 3):
        raise PipelineError("unsupported triage snapshot")
    expected = value.get("snapshot_sha256")
    if expected != json_sha256({key: item for key, item in value.items() if key != "snapshot_sha256"}):
        raise PipelineError("triage snapshot hash mismatch")
    if value["schema_version"] == 3:
        paths = value.get("unit_files")
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
            raise PipelineError("invalid triage snapshot file references")
        units = LazySequence(len(paths), lambda index: storage.read_file(output_dir, paths[index]))
        value = {**value, "units": units, "traces": [trace for unit in units for trace in unit["traces"]]}
    return value


def conversation_trajectories(frozen: dict, *, summaries=False) -> list[dict]:
    """Use the existing training conversations as the judging units."""
    traces = {trace["trace_id"]: trace for trace in frozen["traces"]}
    trajectories = []
    for unit in frozen["units"]:
        trajectory = {"trajectory_id": unit["example"]["id"],
                      "messages": unit["example"]["inputs"]["messages"]}
        if "multimodal_types" in unit:
            trajectory["multimodal_types"] = unit["multimodal_types"]
        else:
            runs = [run for tid in unit["trace_ids"] for run in traces[tid]["runs"]]
            trajectory["multimodal_types"] = multimodal_types({**trajectory, "runs": runs})
        source = unit["example"].get("metadata", {}).get("smithtune_source")
        if summaries:
            trajectory.pop("messages")
            trajectory["has_assistant_runs"] = source is not None
        elif source is not None:
            trajectory["assistant_runs"] = source["assistant_runs"]
        trajectories.append(trajectory)
    return trajectories

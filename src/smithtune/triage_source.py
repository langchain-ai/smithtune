"""Freeze LangSmith conversation messages and run evidence before judging."""

from __future__ import annotations

import copy
import json
import random
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid5

from smithtune.artifacts import _atomic_text, _json_dump, _load_json, _run
from smithtune.curation import _api, _matches, _time, _uuid
from smithtune.dataset import convert_message, validate_trajectories
from smithtune.dataset_artifacts import save_conversation
from smithtune.inference_contract import ContractError, contract_from_runs, json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError


MAX_SOURCE_PAGES = 1000
MAX_EXPANDED_TRACES = 10_000


def _fetch(command, *, runner, cache_dir, **kwargs):
    """Retry idempotent source reads; never wrap dataset writes."""
    path = cache_dir / (json_sha256({"command": command, "input": kwargs.get("input")}) + ".json")
    if path.exists():
        return subprocess.CompletedProcess(command, 0, stdout=path.read_text(encoding="utf-8"), stderr="")
    for attempt in range(3):
        try:
            result = runner(command, **kwargs)
        except subprocess.CalledProcessError as exc:
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
        _atomic_text(path, result.stdout)
        return result


def query_runs(workspace: str, project: str, query: dict, *, runner=_run) -> list[dict]:
    tid = _uuid(query.get("trace"), "trace id")
    selects = ["ID", "TRACE_ID", "PARENT_RUN_IDS", "PROJECT_ID", "IS_ROOT", "NAME", "RUN_TYPE",
               "START_TIME", "END_TIME", "INPUTS", "OUTPUTS", "ERROR", "EXTRA", "ATTACHMENTS"]
    path = f"/api/v2/traces/{tid}/runs?" + urlencode([("project_id", project), *[("selects", field) for field in selects]])
    # V2's trace endpoint returns the complete tree. Omit time bounds so turns
    # outside the root-selection window retain their full evidence too.
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
    roots = sorted(rows.values(), key=lambda r: r["trace_id"])
    return sorted(random.Random(source["seed"]).sample(roots, min(source["limit"], len(roots))), key=lambda r: r["trace_id"]) if source["limit"] else roots


def group_messages(groups: list[dict]) -> list[dict]:
    """Read the API's normalized messages; never flatten child LLM runs."""
    messages = []
    if not isinstance(groups, list):
        raise PipelineError("invalid conversation groups")
    for group in groups:
        if not isinstance(group, dict):
            raise PipelineError("invalid conversation group")
        if group.get("type") == "message":
            messages.append(copy.deepcopy(group.get("message")))
        elif group.get("type") == "tool_interaction":
            ai = copy.deepcopy(group.get("aiMessage"))
            if not isinstance(ai, dict) or ai.get("role") != "ai":
                raise PipelineError("invalid tool interaction")
            messages.append(ai)
            calls = group.get("toolCalls")
            if not isinstance(calls, list):
                raise PipelineError("tool interaction has no call list")
            for call in calls:
                if not isinstance(call, dict):
                    raise PipelineError("invalid tool call")
                result = copy.deepcopy(call.get("result"))
                if result is not None:
                    if not isinstance(result, dict) or result.get("role") != "tool":
                        raise PipelineError("invalid tool result")
                    if result.get("tool_call_id", call.get("id")) != call.get("id"):
                        raise PipelineError("tool result has a different call id")
                    result["tool_call_id"] = call.get("id")
                    messages.append(result)
        else:
            raise PipelineError("unsupported conversation group; cannot preserve training messages")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise PipelineError("conversation message requires an object with a role")
    return messages


def thread_turns(workspace: str, project: str, thread: str, *, runner=_run) -> list[dict]:
    # Follow both directions: the initial API page can start near the end of a
    # conversation. Turn indexes prove that no earlier page was omitted.
    queue, visited, turns = deque([None]), set(), {}
    while queue:
        cursor = queue.popleft()
        if cursor in visited:
            continue
        if len(visited) >= MAX_SOURCE_PAGES:
            raise PipelineError("thread exceeds the page limit")
        visited.add(cursor)
        command = ["langsmith", "thread", "messages", thread, "--project-id", project,
                   "--workspace", workspace, "--format", "json", "--limit", "100"]
        if cursor:
            command += ["--cursor", cursor]
        try:
            page = json.loads(runner(command, capture=True).stdout)
        except Exception:
            raise PipelineError("cannot read complete thread messages; check LangSmith CLI access") from None
        if not isinstance(page, dict) or page.get("thread_id") != thread or not isinstance(page.get("groups"), list):
            raise PipelineError("invalid thread message page")
        current = None
        page_turns = []
        for group in page["groups"]:
            if not isinstance(group, dict):
                raise PipelineError("invalid conversation group")
            if group.get("type") == "turn_boundary":
                boundary = group.get("turnBoundary") or {}
                if not isinstance(boundary, dict):
                    raise PipelineError("invalid thread boundary")
                index = boundary.get("turn_index")
                if type(index) is not int or index < 0:
                    raise PipelineError("invalid thread turn index")
                current = {"index": index, "trace_id": _uuid(boundary.get("trace_id"), "trace id"), "groups": []}
                page_turns.append(current)
            elif current is None:
                raise PipelineError("thread page has messages without a trace boundary")
            else:
                current["groups"].append(group)
        for turn in page_turns:
            index = turn["index"]
            if index in turns and turns[index] != turn:
                raise PipelineError("thread messages changed during snapshot")
            turns[index] = turn
        cursors = page.get("cursors")
        if not isinstance(cursors, dict):
            raise PipelineError("thread page has no pagination metadata")
        for direction in ("prev", "next"):
            value = cursors.get(direction)
            if value is not None and value != "":
                if not isinstance(value, str):
                    raise PipelineError("invalid thread cursor")
                if value not in visited:
                    queue.append(value)
    ordered = [turns[index] for index in sorted(turns)]
    if not ordered or sorted(turns) != list(range(len(turns))):
        raise PipelineError("thread message snapshot is incomplete")
    if len({turn["trace_id"] for turn in ordered}) != len(ordered):
        raise PipelineError("thread contains repeated trace boundaries")
    return ordered


def source_options(workspace_id, project_id, start_time, end_time, *, filter=None, limit=100, seed=42) -> dict:
    value = {"workspace_id": _uuid(workspace_id, "workspace id"), "project_id": _uuid(project_id, "project id"),
             "start_time": _time(start_time), "end_time": _time(end_time), "filter": filter, "limit": limit, "seed": seed}
    if datetime.fromisoformat(value["start_time"]) >= datetime.fromisoformat(value["end_time"]):
        raise PipelineError("start time must precede end time")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise PipelineError("trace limit must be positive")
    if type(seed) is not int or (filter is not None and (not isinstance(filter, str) or not filter.strip())):
        raise PipelineError("invalid source seed or filter")
    return value


def training_error(unit: dict) -> str | None:
    """Check saved conversations with preparation's provider-neutral validation."""
    if unit["training_error"]:
        return unit["training_error"]
    try:
        validate_trajectories([unit["example"]], 1)
        contract = parse_inference_contract(unit["contract"])
        contract.validate_messages([convert_message(message) for message in unit["example"]["inputs"]["messages"]])
    except (PipelineError, ContractError) as exc:
        return str(exc)
    return None


def snapshot(source: dict, output_dir: Path, *, runner=_run) -> dict:
    path = output_dir / "snapshot.json"
    if path.exists():
        value = load_snapshot(output_dir)
        if value["source"] != source:
            raise PipelineError("triage snapshot uses a different source query; use a new output directory")
        return value
    cache_dir = output_dir / "download"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = cache_dir / "source.json"
    if source_path.exists() and _load_json(source_path) != source:
        raise PipelineError("partial download uses a different source query; use a new output directory")
    _json_dump(source_path, source)
    runner = partial(_fetch, runner=runner, cache_dir=cache_dir)
    roots = _root_selection(source, runner=runner)
    if not roots:
        raise PipelineError("no traces match the source query; check the project, time window, and filter")
    workspace, project = source["workspace_id"], source["project_id"]
    print(f"Downloading conversations for {len(roots)} selected traces...", file=sys.stderr)
    units, traces, visited = [], [], set()
    for root in roots:
        thread, tid = root["thread_id"], root["trace_id"]
        key = ("thread", thread) if thread else ("trace", tid)
        if key in visited:
            continue
        visited.add(key)
        if thread:
            turns = thread_turns(workspace, project, thread, runner=runner)
        else:
            page = _api(workspace, "POST", "/api/v2/traces/messages", {
                "project_ids": [project], "ids": [tid], "min_start_time": root["start_time"], "page_size": 1,
            }, runner=runner)
            items = page.get("items") if isinstance(page, dict) else None
            if not isinstance(items, list) or len(items) != 1 or items[0].get("trace_id") != tid or page.get("next_cursor"):
                raise PipelineError("trace messages are missing or incomplete")
            turns = [{"index": 0, "trace_id": tid, "groups": items[0].get("groups")}]
        if tid not in {turn["trace_id"] for turn in turns}:
            raise PipelineError("selected root was not found in its conversation")
        all_messages, all_runs, unit_traces = [], [], []
        for turn in turns:
            if len(traces) >= MAX_EXPANDED_TRACES:
                raise PipelineError("thread expansion exceeds 10000 traces; select fewer roots")
            trace_id = turn["trace_id"]
            messages = group_messages(turn["groups"])
            runs = query_runs(workspace, project, {"trace": trace_id}, runner=runner)
            if not runs or any(run.get("trace_id") != trace_id for run in runs):
                raise PipelineError("trace run evidence is missing or belongs to another trace")
            roots_for_trace = [run for run in runs if not run.get("parent_run_id")]
            if len(roots_for_trace) > 1:
                raise PipelineError("trace has multiple root runs")
            all_messages.extend(messages)
            all_runs.extend(runs)
            record = {"trace_id": trace_id, "root_run_id": roots_for_trace[0]["id"] if roots_for_trace else None, "thread_id": thread,
                      "project_id": project, "messages": copy.deepcopy(all_messages),
                      "turn_start": len(all_messages) - len(messages), "runs": runs}
            if not roots_for_trace:
                record["source_warnings"] = ["Root run missing from the saved source; assess only the available evidence."]
            record["multimodal_types"] = multimodal_types(record)
            record["source_sha256"] = json_sha256(record)
            traces.append(record)
            unit_traces.append(trace_id)
        example_id = str(uuid5(NAMESPACE_URL, json_sha256({"workspace": workspace, "project": project, "key": key, "messages": all_messages})))
        example = {"id": example_id, "inputs": {"messages": all_messages}, "outputs": None,
                   "metadata": {"trajectory_format": "messages", "conversation_scope": "root",
                                "source_project_id": project, "source_thread_id": thread,
                                "source_workspace_id": workspace, "source_scope": "thread" if thread else "trace",
                                "source_scope_id": thread or unit_traces[0],
                                "source_trace_id": unit_traces[0], "triage_trace_ids": unit_traces}}
        save_conversation(output_dir, example)
        contract = None
        error = None
        try:
            contract = contract_from_runs([run for run in all_runs if run.get("run_type") == "llm"], workspace_id=workspace, thread_id=thread)
        except ContractError:
            error = "tool schemas cannot be represented by the current training contract"
        if any(trace["root_run_id"] is None for trace in traces if trace["trace_id"] in unit_traces):
            error = "conversation source has a missing root run"
        unit = {"example": example, "trace_ids": unit_traces, "contract": contract, "training_error": error}
        unit["training_error"] = training_error(unit)
        units.append(unit)
        if len(units) % 10 == 0:
            print(f"Downloaded {len(units)} conversations, {len(traces)} traces.", file=sys.stderr)
    value = {"schema_version": 1, "source": source, "selected_trace_ids": [root["trace_id"] for root in roots],
             "traces": traces, "units": units}
    value["snapshot_sha256"] = json_sha256(value)
    _json_dump(path, value)
    return value


def load_snapshot(output_dir: Path) -> dict:
    value = _load_json(output_dir / "snapshot.json")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PipelineError("unsupported triage snapshot")
    expected = value.get("snapshot_sha256")
    if expected != json_sha256({key: item for key, item in value.items() if key != "snapshot_sha256"}):
        raise PipelineError("triage snapshot hash mismatch")
    return value


def conversation_trajectories(frozen: dict) -> list[dict]:
    """Use the existing training conversations as the judging units."""
    traces = {trace["trace_id"]: trace for trace in frozen["traces"]}
    trajectories = []
    for unit in frozen["units"]:
        trajectory = {"trajectory_id": unit["example"]["id"],
                      "messages": unit["example"]["inputs"]["messages"]}
        runs = [run for tid in unit["trace_ids"] for run in traces[tid]["runs"]]
        trajectory["multimodal_types"] = multimodal_types({**trajectory, "runs": runs})
        trajectories.append(trajectory)
    return trajectories

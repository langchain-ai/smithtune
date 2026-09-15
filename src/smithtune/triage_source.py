"""Freeze LangSmith conversation messages and run evidence before judging."""

from __future__ import annotations

import copy
import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from smithtune.artifacts import _json_dump, _load_json, _run
from smithtune.curation import _api, _matches, _time, _uuid
from smithtune.dataset import _validate_source_message, validate_trajectories
from smithtune.inference_contract import ContractError, contract_from_runs, json_sha256
from smithtune.providers.base import PipelineError


MAX_SOURCE_PAGES = 1000
MAX_EXPANDED_TRACES = 10_000


def query_runs(workspace: str, project: str, query: dict, *, runner=_run) -> list[dict]:
    body = {"session": [project], "limit": 100,
            "select": ["id", "trace_id", "parent_run_id", "session_id", "name", "run_type", "start_time", "end_time", "inputs", "outputs", "error", "extra"], **query}
    rows, cursors = {}, set()
    while True:
        if len(cursors) >= MAX_SOURCE_PAGES:
            raise PipelineError("run query exceeds the page limit; use a smaller source")
        page = _api(workspace, "POST", "/api/v1/runs/query", body, runner=runner)
        if not isinstance(page, dict) or not isinstance(page.get("runs"), list):
            raise PipelineError("invalid LangSmith run page")
        for row in page["runs"]:
            if not isinstance(row, dict) or row.get("session_id") != project:
                raise PipelineError("run query returned another project")
            rid = _uuid(row.get("id"), "run id")
            if rid in rows and rows[rid] != row:
                raise PipelineError("source run changed during snapshot")
            rows[rid] = row
        page_cursors = page.get("cursors") or {}
        if not isinstance(page_cursors, dict):
            raise PipelineError("invalid LangSmith run cursors")
        cursor = page_cursors.get("next")
        if cursor is None:
            return sorted(rows.values(), key=lambda r: (r.get("start_time") or "", r["id"]))
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise PipelineError("invalid or repeated LangSmith run cursor")
        cursors.add(cursor)
        body["cursor"] = cursor


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
        _validate_source_message(message)
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


def snapshot(source: dict, output_dir: Path, *, runner=_run) -> dict:
    path = output_dir / "snapshot.json"
    if path.exists():
        value = load_snapshot(output_dir)
        if value["source"] != source:
            raise PipelineError("triage snapshot uses a different source query; use a new output directory")
        return value
    roots = _root_selection(source, runner=runner)
    if not roots:
        raise PipelineError("no traces match the source query; check the project, time window, and filter")
    workspace, project = source["workspace_id"], source["project_id"]
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
            if not messages:
                raise PipelineError(f"trace {trace_id} has no conversation messages")
            runs = query_runs(workspace, project, {"trace": trace_id}, runner=runner)
            if not runs or any(run.get("trace_id") != trace_id for run in runs):
                raise PipelineError("trace run evidence is missing or belongs to another trace")
            roots_for_trace = [run for run in runs if not run.get("parent_run_id")]
            if len(roots_for_trace) != 1 or not roots_for_trace[0].get("end_time"):
                raise PipelineError("trace is still running or has no unique root")
            all_messages.extend(messages)
            all_runs.extend(runs)
            record = {"trace_id": trace_id, "root_run_id": roots_for_trace[0]["id"], "thread_id": thread,
                      "project_id": project, "messages": copy.deepcopy(all_messages),
                      "turn_start": len(all_messages) - len(messages), "runs": runs}
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
        validate_trajectories([example], 1)
        contract = None
        error = None
        try:
            contract = contract_from_runs([run for run in all_runs if run.get("run_type") == "llm"], workspace_id=workspace, thread_id=thread)
        except ContractError:
            error = "tool schemas cannot be represented by the current training contract"
        units.append({"example": example, "trace_ids": unit_traces, "contract": contract, "training_error": error})
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

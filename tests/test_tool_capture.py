from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from smithtune import dataset
from smithtune.providers.base import PipelineError


def tool(name):
    return {"type": "function", "function": {
        "name": name, "description": f"Call {name}.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }}


def llm(run_id, tools, *, trace_id="trace-1"):
    return {"id": run_id, "trace_id": trace_id, "session_id": "project-1", "run_type": "llm",
            "name": "ChatModel", "start_time": "2026-09-11T12:00:00Z",
            "extra": {"invocation_params": {"tools": tools, "temperature": 0.2}}}


def page(runs, cursor=None):
    items = []
    for run in runs:
        if not isinstance(run, dict):
            items.append(run)
            continue
        item = copy.deepcopy(run)
        item["project_id"] = item.pop("session_id", "project-1")
        if isinstance(item.get("run_type"), str):
            item["run_type"] = item["run_type"].upper()
        if isinstance(item.get("extra"), dict) and "metadata" in item["extra"]:
            item["metadata"] = item["extra"].pop("metadata")
        items.append(item)
    return {"items": items, "next_cursor": cursor}


def catalog_tool(entries, *, suffix=""):
    value = tool("load_integration_tools")
    value["function"]["description"] = (
        "Load integration tools on demand.\nAvailable tools:\n"
        + "\n".join(f"- {name} (integration: {integration})" for name, integration in entries)
        + suffix
    )
    return value


def test_v2_capture_pages_through_all_project_history_and_normalizes_runs(monkeypatch):
    monkeypatch.setattr(dataset, "_utc_now", lambda: "2026-09-15T00:00:00Z")
    older = llm("old", [tool("weather")])
    boundary = llm("boundary", [tool("swell")])
    recent = llm("recent", [tool("lookup")])
    pages = iter([page([older], "next"), page([boundary]), page([boundary, recent])])
    bodies = []

    def runner(command, capture=False):
        assert command[2] == "/api/v2/runs/query"
        body = json.loads(command[command.index("--body") + 1])
        bodies.append(body)
        assert body["project_ids"] == ["project-1"]
        assert body["run_type"] == "LLM"
        assert set(body["selects"]) == {"ID", "TRACE_ID", "PROJECT_ID", "NAME", "RUN_TYPE", "START_TIME", "EXTRA", "METADATA"}
        return SimpleNamespace(stdout=json.dumps(next(pages)))

    runs = dataset._query_runs("workspace-1", {
        "project_ids": ["project-1"], "run_type": "LLM", "min_start_time": "2025-01-01T00:00:00Z",
    }, runner=runner)
    assert runs == [older, boundary, recent]
    assert bodies[0]["min_start_time"] == "2025-01-01T00:00:00+00:00"
    assert bodies[0]["max_start_time"] == "2026-02-05T00:00:00+00:00"
    assert bodies[1] == {**bodies[0], "cursor": "next"}
    assert bodies[2]["min_start_time"] == bodies[0]["max_start_time"]
    assert bodies[2]["max_start_time"] == "2026-09-15T00:00:00+00:00"
    assert "cursor" not in bodies[2]


@pytest.mark.parametrize("start", [None, "invalid", "2026-09-01T00:00:00"])
def test_project_lookup_rejects_missing_or_ambiguous_history_start(start):
    def runner(command, capture=False):
        return SimpleNamespace(stdout=json.dumps({"id": "project-1", "start_time": start}))

    with pytest.raises(PipelineError, match="no valid start time"):
        dataset._project_start_time("workspace-1", "project-1", runner=runner)



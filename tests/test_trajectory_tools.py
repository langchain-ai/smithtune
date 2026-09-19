"""The trajectory UI response is the only automatic tool/provenance source."""
import copy
import json
from types import SimpleNamespace

import pytest

from smithtune import curation, dataset, triage, triage_source
from smithtune.providers.base import PipelineError
from test_assistant_bindings import tool
from test_per_assistant_preparation import example
from test_triage import API, source, uid
from trajectory_fixtures import items


def test_ui_pages_preserve_messages_and_global_assistant_positions():
    messages = [
        {"role": "system", "content": "Recorded instructions."},
        {"role": "human", "content": "Find a result."},
        {"role": "ai", "content": "First answer."},
        {"role": "human", "content": "Check again."},
        {"role": "ai", "content": "Final answer."},
    ]
    wire = items(messages, trace_id=uid(1), tools=[tool()])
    wire[-1]["message"]["available_tools"] = []
    wire[-1]["metadata"]["run_id"] = "second-run"
    original = copy.deepcopy(wire)
    requests = []

    def api(command, *, input, **kwargs):
        assert command[2] == "/v1/trajectory"
        body = json.loads(input)
        requests.append(body)
        if "cursor" not in body:
            return SimpleNamespace(stdout=json.dumps({"items": wire[:3], "next_cursor": "page-2"}))
        return SimpleNamespace(stdout=json.dumps({"items": wire[3:], "next_cursor": None}))

    result = curation._fetch_trajectory(uid(100), uid(101), {"key": "trace_id", "id": uid(1)}, runner=api)
    assert result["messages"] == messages
    assert result["training_error"] is None
    assert [(b["message_index"], len(b["tools"])) for b in result["source"]["assistant_runs"]] == [(2, 1), (4, 0)]
    assert result["source"]["assistant_runs"][-1]["run_id"] == "second-run"
    assert requests == [
        {"project_id": uid(101), "trace_id": uid(1), "format": "ui", "include": {"system_messages": True}},
        {"project_id": uid(101), "trace_id": uid(1), "format": "ui", "include": {"system_messages": True}, "cursor": "page-2"},
    ]
    assert wire == original


@pytest.mark.parametrize("entry", [None, {}, {"type": "event", "message": {}}, {"message": []}])
def test_invalid_ui_items_are_not_silently_dropped(entry):
    def api(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps({"items": [entry], "next_cursor": None}))
    with pytest.raises(PipelineError, match="unsupported trajectory item"):
        curation._fetch_trajectory("workspace", "project", {"key": "thread_id", "id": "thread"}, runner=api)


@pytest.mark.parametrize("scope", ["trace", "thread"])
def test_foreign_trace_metadata_stops_snapshot_before_judging(tmp_path, scope):
    api = API()
    if scope == "trace":
        api.root_pages[0][0]["thread_id"] = None

    def foreign(command, **kwargs):
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory":
            body = json.loads(response.stdout)
            for item in body["items"]:
                item["metadata"]["trace_id"] = uid(999)
            response.stdout = json.dumps(body)
        return response

    with pytest.raises(PipelineError, match="another trace"):
        triage_source.snapshot(source(), tmp_path, runner=foreign)
    assert not (tmp_path / "snapshot.json").exists()


def test_missing_assistant_metadata_is_saved_and_excluded_before_council(tmp_path):
    api = API()
    def missing(command, **kwargs):
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory":
            body = json.loads(response.stdout)
            for item in body["items"]:
                if item["message"].get("role") == "ai":
                    del item["metadata"]
            response.stdout = json.dumps(body)
        return response
    result = triage.run_triage(source(), tmp_path, runner=missing, confirm=True,
                              judge_call=lambda *_: pytest.fail("missing provenance reached paid judging"))
    assert result["filtered_training"] == 1
    assert "run/trace provenance" in triage_source.load_snapshot(tmp_path)["units"][0]["training_error"]


@pytest.mark.parametrize("change", ["text", "appended", "message_id", "numeric_type"])
def test_unbound_export_requires_exact_source_messages(change):
    value = example(1)
    del value["metadata"]["smithtune_source"]
    messages = copy.deepcopy(value["inputs"]["messages"])
    if change == "text":
        messages[-1]["content"] = "changed"
    elif change == "appended":
        messages.append({"role": "human", "content": "later turn"})
    elif change == "message_id":
        messages[-1]["id"] = "new-id"
    else:
        value["inputs"]["messages"][-1]["additional_kwargs"] = {"value": 1}
        messages[-1]["additional_kwargs"] = {"value": True}
    def api(command, **kwargs):
        assert command[2] == "/v1/trajectory"
        return SimpleNamespace(stdout=json.dumps({"items": items(messages), "next_cursor": None}))
    enriched, errors = dataset.capture_example_bindings([value], "workspace", runner=api)
    assert enriched == [] and "messages differ" in errors[0]["reason"]
    assert "smithtune_source" not in value["metadata"]


def test_unbound_exports_route_same_thread_to_its_own_project_and_workspace():
    values = [example(i, thread="same", project=f"project-{i}") for i in (1, 2)]
    for i, value in enumerate(values, 1):
        del value["metadata"]["smithtune_source"]
        value["metadata"]["source_workspace_id"] = f"source-{i}"
    requests = []
    def api(command, *, input, **kwargs):
        assert command[2] == "/v1/trajectory"
        workspace = command[command.index("--workspace") + 1]
        body = json.loads(input)
        requests.append((workspace, body["project_id"], body["thread_id"]))
        i = int(body["project_id"].rsplit("-", 1)[1])
        return SimpleNamespace(stdout=json.dumps({"items": items(values[i-1]["inputs"]["messages"],
            tools=[tool(name=f"tool_{i}")]), "next_cursor": None}))
    enriched, errors = dataset.capture_example_bindings(values, "destination", runner=api)
    assert not errors
    assert requests == [("source-1", "project-1", "same"), ("source-2", "project-2", "same")]
    assert [v["metadata"]["smithtune_source"]["assistant_runs"][0]["tools"][0]["function"]["name"]
            for v in enriched] == ["tool_1", "tool_2"]

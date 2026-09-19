import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from smithtune.dataset_artifacts import load_conversation
from smithtune.providers.base import PipelineError
from test_curation import API as CurationAPI, COMMON_METADATA, create, root, uid
from test_dataset_import import API as DestinationAPI, example, update
from test_tool_capture import tool
from binding_fixtures import bound_example
from trajectory_fixtures import items


def tool_turn(name="search", args=None):
    return [
        {"role": "ai", "content": [
            {"type": "tool_call", "id": f"call-{name}", "name": name,
             "args": {"query": "unchanged"} if args is None else args},
        ]},
        {"role": "tool", "tool_call_id": f"call-{name}", "content": "recorded result"},
        {"role": "ai", "content": "Recorded answer."},
    ]


def incoming_examples():
    incoming = [example(n, turns=2, **COMMON_METADATA) for n in (1, 2, 3)]
    for item in (incoming[0], incoming[2]):
        item["metadata"].update(source_scope="trace", source_scope_id=item["id"])
    incoming[2]["inputs"]["messages"].insert(0, {"role": "system", "content": "Keep all turns."})
    incoming[2]["inputs"]["messages"].extend(tool_turn() + tool_turn("weather"))
    incoming[2]["inputs"]["messages"][-1]["content"] = [
        {"type": "text", "text": "Recorded answer."},
        {"type": "reasoning", "reasoning": "Recorded late reasoning.", "signature": "opaque-state"},
    ]
    return bind(incoming)


def bind(incoming):
    for i, value in enumerate(incoming):
        bound_example(value, tools=[tool("search"), tool("weather")] if i == 2 else [tool("search")])
        for binding in value["metadata"]["smithtune_source"]["assistant_runs"]:
            binding["trace_id"] = value["id"]
    return incoming


class SourceAPI(CurationAPI):
    def __init__(self, incoming, *, failure=None):
        super().__init__([[root(n, item["metadata"]["source_scope_id"]
                               if item["metadata"]["source_scope"] == "thread" else None)
                          for n, item in enumerate(incoming, 1)]])
        self.trajectories = {item["metadata"]["source_scope_id"]: item for item in incoming}
        self.tool_queries = []
        self.tool_failure = failure

    def __call__(self, command, *, capture=False, input=None):
        assert command[command.index("--method") + 1] != "DELETE"
        path = command[2]
        if path != "/v1/trajectory":
            return super().__call__(command, capture=capture, input=input)
        body = json.loads(input)
        self.tool_queries.append(copy.deepcopy(body))
        if self.tool_failure == "403":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 403 Forbidden")
        if self.tool_failure == "pagination":
            return SimpleNamespace(stdout=json.dumps({"items": None, "next_cursor": "more"}))
        value = self.trajectories[body.get("thread_id") or body["trace_id"]]
        self.messages = value["inputs"]["messages"]
        response = super().__call__(command, capture=capture, input=input)
        wire = items(self.messages, trace_id=value["id"])
        for binding in value["metadata"]["smithtune_source"]["assistant_runs"]:
            item = wire[binding["message_index"]]
            item["message"]["available_tools"] = binding["tools"]
            item["metadata"]["run_id"] = binding["run_id"]
        response.stdout = json.dumps({"items": wire, "next_cursor": None})
        return response


def import_api(destination, incoming, **kwargs):
    source = SourceAPI(incoming, **kwargs)
    if destination == "new":
        return source
    existing = copy.deepcopy(incoming) if destination == "existing-patch" else []
    for item in existing:
        messages = item["inputs"]["messages"]
        item["inputs"]["messages"] = messages[:3 if messages[0]["role"] == "system" else 2]
        item["metadata"]["smithtune_source"]["assistant_runs"] = [
            b for b in item["metadata"]["smithtune_source"]["assistant_runs"] if b["message_index"] < len(item["inputs"]["messages"])]
    api = DestinationAPI(existing)
    api.source = source
    return api


def run_import(tmp_path, destination, api, incoming):
    return create(tmp_path, api) if destination == "new" else update(tmp_path, api, incoming)


def uploaded(api, destination):
    if destination == "new":
        return [body for path, body in api.calls if path == "/api/v1/examples"]
    assert all(method == ("PATCH" if destination == "existing-patch" else "POST")
               for method, _, _ in api.writes)
    return [body for _, _, body in api.writes]


def rejection_entries(receipt, destination):
    if destination == "new":
        assert receipt["pending_write"] is None
        return receipt["rejections"]
    actions = [json.loads(line) for line in Path(receipt["actions"]).read_text().splitlines()]
    assert [item["action"] for item in actions] == ["rejected"] + [
        "updated" if destination == "existing-patch" else "created"
    ] * receipt["created" if destination == "existing-post" else "updated"]
    return [item for item in actions if item["action"] == "rejected"]


@pytest.mark.parametrize("destination", ["new", "existing-post", "existing-patch"])
@pytest.mark.parametrize("case,reason", [
    ("missing-result", "unmatched_tool_calls_or_results"),
    ("orphan-result", "unmatched_tool_calls_or_results"),
    ("repeated-call-id", "repeated_tool_call_id"),
    ("image", "unsupported_native_content"),
    ("invalid-role", "invalid_message"),
    ("misplaced-system", "misplaced_system_message"),
    ("unknown-tool", "unknown tool unknown"),
    ("invalid-schema-args", "do not match its JSON Schema"),
    ("string-args", "invalid_message"),
    ("text-after-tool", "invalid_message"),
])
def test_mixed_import_prunes_before_writes_and_preserves_whole_conversations(tmp_path, destination, case, reason):
    incoming = incoming_examples()
    messages = incoming[0]["inputs"]["messages"]
    if case == "missing-result":
        messages.extend(tool_turn()[:1])
    elif case == "orphan-result":
        messages.extend(tool_turn()[1:])
    elif case == "repeated-call-id":
        messages.extend(tool_turn() * 2)
    elif case == "image":
        messages.append({"role": "human", "content": [{"type": "image", "url": "https://example.test/image.png"}]})
    elif case == "invalid-role":
        messages.append({"role": [], "content": "Invalid role."})
    elif case == "misplaced-system":
        messages.append({"role": "system", "content": "Late instructions."})
    elif case == "unknown-tool":
        messages.extend(tool_turn("unknown"))
    elif case == "invalid-schema-args":
        messages.extend(tool_turn(args={"query": 7}))
    elif case == "string-args":
        messages.extend(tool_turn(args='{"query":"unchanged"}'))
    elif case == "text-after-tool":
        turn = tool_turn()
        turn[0]["content"].append({"type": "text", "text": "Would require reordering."})
        messages.extend(turn)
    bind(incoming)
    original = copy.deepcopy(incoming)
    api = import_api(destination, incoming)
    previous = copy.deepcopy(api.existing) if destination != "new" else None

    result = run_import(tmp_path, destination, api, incoming)

    assert (result["example_count"], result["rejected"]) == (2, 1)
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete" and receipt["pending_write"] is None
    rejection, = rejection_entries(receipt, destination)
    assert {key: rejection[key] for key in ("code", "source")} == {
        "code": "invalid_import_trajectory_excluded",
        "source": {"workspace": uid(100), "project": uid(101), "scope": "trace", "scope_id": uid(1)},
    }
    assert reason in rejection["reason"]
    assert (lambda v: v.get("example", v))(load_conversation(Path(rejection["conversation"])))["inputs"] == original[0]["inputs"]
    saved = [value.get("example", value) for path in (tmp_path / "conversations").glob("*.json") if (value := load_conversation(path))]
    assert {item["metadata"]["source_scope_id"]: item["inputs"] for item in saved} == {
        item["metadata"]["source_scope_id"]: item["inputs"] for item in original
    }
    writes = uploaded(api, destination)
    assert [item["inputs"] for item in writes] == [item["inputs"] for item in original[1:]]
    assert [item["metadata"]["source_scope_id"] for item in writes] == ["thread-2", uid(3)]
    assert all(item["outputs"] is None for item in writes)
    source = api if destination == "new" else api.source
    if destination == "new":
        assert len(source.tool_queries) == 3
    else:
        assert source.tool_queries == []  # Upload uses the saved tool evidence.
    if destination == "new":
        assert receipt["confirmed_example_ids"] == [item["id"] for item in api.examples]
    else:
        assert (result["created"], result["updated"], result["skipped"]) == (
            (2, 0, 0) if destination == "existing-post" else (0, 2, 0)
        )
        if destination == "existing-patch":
            assert api.existing[0] == previous[0]  # Rejected source was not overwritten.
        else:
            assert len(api.existing) == len(previous) + 2
    assert incoming == original


@pytest.mark.parametrize("destination", ["new", "existing-post", "existing-patch"])
def test_all_rejected_import_completes_without_example_writes(tmp_path, destination):
    incoming = incoming_examples()[:1]
    incoming[0]["metadata"].update(source_scope="thread", source_scope_id="thread-1")
    incoming[0]["inputs"]["messages"].extend(tool_turn()[:1])
    bind(incoming)
    original = copy.deepcopy(incoming)
    api = import_api(destination, incoming)

    result = run_import(tmp_path, destination, api, incoming)

    assert (result["example_count"], result["rejected"]) == (0, 1)
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete" and receipt["pending_write"] is None
    rejection, = rejection_entries(receipt, destination)
    assert rejection["reason"] == "unmatched_tool_calls_or_results"
    assert rejection["code"] == "invalid_import_trajectory_excluded"
    assert rejection["source"] == {
        "workspace": uid(100), "project": uid(101), "scope": "thread", "scope_id": "thread-1",
    }
    assert (lambda v: v.get("example", v))(load_conversation(Path(rejection["conversation"])))["inputs"] == original[0]["inputs"]
    assert uploaded(api, destination) == []
    assert incoming == original


@pytest.mark.parametrize("failure,error", [("403", "HTTP 403"), ("pagination", "invalid trajectory items")])
def test_trajectory_read_failure_stops_import_without_pruning_or_example_writes(tmp_path, monkeypatch, failure, error):
    from smithtune import curation
    monkeypatch.setattr(curation, "_sleep", lambda _: None)
    incoming = incoming_examples()[:2]
    api = SourceAPI(incoming, failure=failure)
    with pytest.raises(PipelineError, match=error):
        create(tmp_path, api)
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete"
    assert receipt["pending_write"] is None and receipt["rejections"] == []
    assert api.examples == []


@pytest.mark.parametrize("destination", ["existing-post", "existing-patch"])
def test_saved_tools_upload_without_source_access(tmp_path, destination):
    incoming = incoming_examples()[:2]
    api = import_api(destination, incoming, failure="403")
    result = run_import(tmp_path, destination, api, incoming)
    assert result["example_count"] == 2 and result["rejected"] == 0
    assert api.source.tool_queries == []


def test_unchanged_examples_skip_without_source_tool_reads(tmp_path):
    incoming = incoming_examples()[:1]
    api = DestinationAPI(copy.deepcopy(incoming))
    api.source = SourceAPI(incoming, failure="403")

    result = update(tmp_path, api, incoming)

    assert (result["skipped"], result["rejected"]) == (1, 0)
    assert api.writes == [] and api.source.contract_calls == []


def test_unresolvable_schema_is_not_reported_as_invalid_arguments(tmp_path):
    incoming = incoming_examples()[:1]
    incoming[0]["inputs"]["messages"].extend(tool_turn())
    unresolved = tool("search")
    unresolved["function"]["parameters"] = {"$ref": "https://example.invalid/schema"}
    bind(incoming)
    for binding in incoming[0]["metadata"]["smithtune_source"]["assistant_runs"]:
        binding["tools"] = [unresolved]
    api = SourceAPI(incoming)

    result = create(tmp_path, api)
    assert result["rejected"] == 1

    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "complete"
    assert "cannot resolve schema reference" in receipt["rejections"][0]["reason"]
    assert "invalid_tool_arguments" not in receipt["rejections"][0]["reason"]
    assert api.examples == []

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
    return incoming


class SourceAPI(CurationAPI):
    def __init__(self, incoming, *, tool_pages=None, failure=None):
        super().__init__([[root(n, item["metadata"]["source_scope_id"]
                               if item["metadata"]["source_scope"] == "thread" else None)
                          for n, item in enumerate(incoming, 1)]])
        self.trajectories = {item["metadata"]["source_scope_id"]: item["inputs"]["messages"]
                             for item in incoming}
        self.tool_pages = tool_pages or {}
        self.tool_queries = []
        self.tool_failure = failure

    def __call__(self, command, *, capture=False, input=None):
        assert command[command.index("--method") + 1] != "DELETE"
        body = json.loads(input) if input is not None else (
            json.loads(command[command.index("--body") + 1]) if "--body" in command else None
        )
        path = command[2]
        if path == "/v1/trajectory":
            self.messages = copy.deepcopy(self.trajectories[body.get("thread_id") or body["trace_id"]])
        response = super().__call__(command, capture=capture, input=input)
        if path != "/api/v2/runs/query" or body.get("run_type") != "LLM":
            return response
        self.tool_queries.append(copy.deepcopy(body))
        if self.tool_failure == "403":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 403 Forbidden")
        if self.tool_failure == "pagination":
            return SimpleNamespace(stdout=json.dumps({"items": None, "next_cursor": "more"}))
        pages = self.tool_pages.get(body.get("trace_id"))
        if pages is None:
            return response
        position = int(body.get("cursor", 0))
        run = {"id": uid(500 + position), "trace_id": body["trace_id"], "project_id": uid(101),
               "run_type": "LLM", "start_time": "2026-09-02T12:00:00Z",
               "extra": {"invocation_params": {"tools": pages[position]}}}
        return SimpleNamespace(stdout=json.dumps({
            "items": [run], "next_cursor": str(position + 1) if position + 1 < len(pages) else None,
        }))


def import_api(destination, incoming, **kwargs):
    source = SourceAPI(incoming, **kwargs)
    if destination == "new":
        return source
    existing = copy.deepcopy(incoming) if destination == "existing-patch" else []
    for item in existing:
        messages = item["inputs"]["messages"]
        item["inputs"]["messages"] = messages[:3 if messages[0]["role"] == "system" else 2]
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
    ("unknown-tool", "unknown_tool"),
    ("invalid-schema-args", "invalid_tool_arguments"),
    ("string-args", "invalid_message"),
    ("text-after-tool", "invalid_message"),
    ("uncalled-conflict", "conflicting_tool_definitions"),
])
def test_mixed_import_prunes_before_writes_and_preserves_whole_conversations(tmp_path, destination, case, reason):
    incoming = incoming_examples()
    messages = incoming[0]["inputs"]["messages"]
    pages = {uid(1): [[tool("search")]], uid(3): [[tool("search")], [tool("weather")]]}
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
    else:
        conflicting = tool("search")
        conflicting["function"]["parameters"]["properties"]["query"]["type"] = "integer"
        pages[uid(1)].append([conflicting])
    original = copy.deepcopy(incoming)
    api = import_api(destination, incoming, tool_pages=pages)
    previous = copy.deepcopy(api.existing) if destination != "new" else None

    result = run_import(tmp_path, destination, api, incoming)

    assert (result["example_count"], result["rejected"]) == (2, 1)
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete" and receipt["pending_write"] is None
    rejection, = rejection_entries(receipt, destination)
    assert {key: rejection[key] for key in ("code", "source", "reason")} == {
        "code": "invalid_import_trajectory_excluded",
        "source": {"workspace": uid(100), "project": uid(101), "scope": "trace", "scope_id": uid(1)},
        "reason": reason,
    }
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
    assert [body.get("cursor") for body in source.tool_queries if body.get("trace_id") == uid(3)] == [None, "1"]
    if case == "uncalled-conflict":
        assert [body.get("cursor") for body in source.tool_queries if body.get("trace_id") == uid(1)] == [None, "1"]
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


@pytest.mark.parametrize("destination", ["new", "existing-post", "existing-patch"])
@pytest.mark.parametrize("failure,error", [("403", "HTTP 403"), ("pagination", "invalid run query page")])
def test_tool_read_failure_stops_import_without_pruning_or_example_writes(tmp_path, destination, failure, error):
    incoming = incoming_examples()[:2]
    api = import_api(destination, incoming, failure=failure)

    with pytest.raises(PipelineError, match=error) as exc:
        run_import(tmp_path, destination, api, incoming)

    assert "outcome may be unknown" not in str(exc.value)
    path = tmp_path / ("selection.import.json" if destination == "new" else "receipt.json")
    receipt = json.loads(path.read_text())
    assert receipt["pending_write"] is None
    assert uploaded(api, destination) == []
    source = api if destination == "new" else api.source
    assert len(source.tool_queries) == 1
    if destination == "new":
        assert receipt["status"] == "incomplete" and receipt["rejections"] == [] and receipt["confirmed_example_ids"] == []
    else:
        assert receipt["status"] == "incomplete"
        assert (receipt["created"], receipt["updated"], receipt["skipped"], receipt["rejected"]) == (0, 0, 0, 0)
        assert Path(receipt["actions"]).read_text() == ""
    if destination != "new":
        saved, = (tmp_path / "conversations").glob("*.json")
        assert load_conversation(saved)["inputs"] == incoming[0]["inputs"]


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
    api = SourceAPI(incoming, tool_pages={uid(1): [[unresolved]]})

    with pytest.raises(PipelineError, match="cannot resolve schema reference"):
        create(tmp_path, api)

    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["rejections"] == []
    assert api.examples == []

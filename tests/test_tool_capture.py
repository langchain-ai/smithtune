from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from smithtune import dataset
from smithtune import inference_contract
from smithtune import cli as pipeline
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
    return {"runs": runs, "cursors": {"next": cursor}}


def capture_runner(source, pages, *, thread_key="thread_id", thread_id="thread-1"):
    commands = []
    pages = iter(pages)

    def runner(command, capture=False):
        assert capture is True
        assert command[:3] == ["langsmith", "api", "runs/query"]
        assert command[command.index("--workspace") + 1] == "workspace-1"
        body = json.loads(command[command.index("--body") + 1])
        assert "inputs" not in body["select"] and "outputs" not in body["select"]
        commands.append(body)
        if body.get("id") == [source["id"]]:
            response = page([source])
        elif body.get("id") == [source["trace_id"]]:
            response = page([{"id": source["trace_id"], "session_id": "project-1",
                              "extra": {"metadata": {thread_key: thread_id}}}])
        else:
            assert body["session"] == ["project-1"]
            assert body["run_type"] == "llm"
            assert "trace_filter" in body and "filter" not in body
            response = next(pages, page([]))
        return SimpleNamespace(stdout=json.dumps(response))

    return runner, commands


def test_capture_combines_tools_from_all_pages_and_traces(tmp_path):
    first = llm("run-1", [tool("weather")])
    late = llm("run-2", [tool("weather"), tool("swell")], trace_id="trace-2")
    helper = llm("run-3", [])
    runner, commands = capture_runner(first, [page([late], "page-2"), page([first, helper])])
    path = tmp_path / "contract.json"
    summary = dataset.capture_inference_contract("workspace-1", "run-1", path, runner=runner)
    contract = inference_contract.load_inference_contract(path)

    assert [t["function"]["name"] for t in contract.tools] == ["swell", "weather"]
    assert contract.inference_settings == {"temperature": 0.2}
    assert contract.provenance["source_thread_id"] == "thread-1"
    assert contract.provenance["source_run_ids"] == ["run-1", "run-2", "run-3"]
    assert contract.provenance["source_trace_ids"] == ["trace-1", "trace-2"]
    assert summary["llm_run_count"] == 3 and summary["trace_count"] == 2
    assert summary["tool_count"] == 2
    assert commands[3]["cursor"] == "page-2"

    # Preparation already shares the captured union with every example.
    examples = [{"id": str(i), "inputs": {"messages": [
        {"role": "human", "content": "Hi"}, {"role": "ai", "content": "Hello"},
    ]}, "metadata": {"source_scope": "thread", "source_scope_id": f"thread-{i}"}} for i in range(2)]
    rows = dataset.prepare_sft_rows(examples, contract=contract)
    assert all(row["tools"] == list(contract.tools) for row in rows)


@pytest.mark.parametrize("builtin", [
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    {"type": "web_search_preview"},
    {"type": "file_search", "vector_store_ids": ["store-1"]},
    {"type": "computer_20250124", "name": "computer", "input_schema": {"type": "object"}},
    {"google_search": {}},
    {"google_maps": {}},
    {"file_search": {}},
    {"function_declarations": [{"name": "weather", "parameters": {"type": "object"}}], "enterprise_web_search": {}},
    {"functionDeclarations": [{"name": "weather", "parameters": {"type": "object"}}], "googleMaps": {}},
    {"function_declarations": [{"name": "weather", "parameters": {"type": "object"}}], "code_execution": {}},
])
@pytest.mark.parametrize("existing", [False, True])
def test_capture_rejects_uncalled_builtins_on_later_pages_without_writing(tmp_path, builtin, existing):
    first = llm("run-1", [tool("weather")])
    later = llm("run-2", [tool("weather"), builtin], trace_id="trace-2")
    runner, _ = capture_runner(first, [page([first], "page-2"), page([later])])
    path = tmp_path / "contract.json"
    if existing:
        path.write_text("existing contract")
    with pytest.raises(PipelineError, match="run run-2: provider built-in tool"):
        dataset.capture_inference_contract("workspace-1", "run-1", path, runner=runner)
    assert path.read_text() == "existing contract" if existing else not path.exists()


def test_frontier_model_and_user_function_named_like_builtin_are_allowed(tmp_path):
    source = llm("run-1", [{"name": "tool_search_tool_bm25", "input_schema": {"type": "object"}}])
    source["extra"]["invocation_params"]["model"] = "claude-opus-4-6"
    runner, _ = capture_runner(source, [page([source])])
    summary = dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)
    assert summary["tool_count"] == 1


def test_equivalent_provider_shapes_deduplicate_without_mutating_source():
    a = llm("run-1", [tool("weather")])
    b = llm("run-2", [{"name": "weather", "description": "Call weather.",
                       "input_schema": tool("weather")["function"]["parameters"]}])
    c = llm("run-3", [{"function_declarations": [tool("weather")["function"]]}])
    original = copy.deepcopy([a, b, c])
    kwargs = dict(workspace_id="workspace-1", source_run_id="run-1", thread_id="thread-1")
    contract = inference_contract.contract_from_runs([a, b, c], **kwargs)
    assert contract["tools"] == [tool("weather")]
    assert [a, b, c] == original
    assert inference_contract.contract_from_runs([c, b, a], **kwargs) == contract


def test_conflicting_schemas_identify_both_source_runs(tmp_path):
    first = llm("run-1", [tool("weather")])
    later = copy.deepcopy(first)
    later["id"] = "run-2"
    later["extra"]["invocation_params"]["tools"][0]["function"]["parameters"]["required"] = ["query"]
    runner, _ = capture_runner(first, [page([first, later])])
    with pytest.raises(PipelineError, match="weather.*conflicting definitions.*run-1.*run-2"):
        dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)
    assert not (tmp_path / "contract.json").exists()


@pytest.mark.parametrize("key", ["thread_id", "conversation_id", "session_id"])
def test_thread_lookup_uses_root_metadata_aliases_and_escapes_values(tmp_path, key):
    source = llm("run-1", [tool("weather")])
    thread_id = 'thread-"quoted"'
    runner, commands = capture_runner(source, [page([source])], thread_key=key, thread_id=thread_id)
    result = dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)
    assert result["source_thread_id"] == thread_id
    assert [body["trace_filter"] for body in commands[2:]] == [
        f"and(eq(metadata_key,{json.dumps(alias)}),eq(metadata_value,{json.dumps(thread_id)}))"
        for alias in ("thread_id", "conversation_id", "session_id")
    ]


@pytest.mark.parametrize("bad_page,error", [
    ({"runs": None}, "invalid run query page"),
    (page([None]), "invalid run"),
    (page([dict(llm("run-2", []), session_id="other-project")]), "different project"),
    (page([]), "source LLM run run-1 was not returned"),
])
def test_incomplete_or_invalid_scan_does_not_write_contract(tmp_path, bad_page, error):
    source = llm("run-1", [tool("weather")])
    runner, _ = capture_runner(source, [bad_page])
    with pytest.raises(PipelineError, match=error):
        dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)
    assert not (tmp_path / "contract.json").exists()


def test_repeated_pagination_cursor_does_not_write_contract(tmp_path):
    source = llm("run-1", [tool("weather")])
    runner, _ = capture_runner(source, [page([source], "same"), page([source], "same")])
    with pytest.raises(PipelineError, match="repeated query cursor"):
        dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)
    assert not (tmp_path / "contract.json").exists()


def test_missing_thread_does_not_fall_back_to_one_trace(tmp_path):
    source = llm("run-1", [tool("weather")])
    runner, _ = capture_runner(source, [], thread_id=None)
    with pytest.raises(PipelineError, match="no thread ID"):
        dataset.capture_inference_contract("workspace-1", "run-1", tmp_path / "contract.json", runner=runner)


def test_cli_returns_failure_for_builtin_capture(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise PipelineError("run run-2: provider built-in tool 'web_search' is not supported")
    monkeypatch.setattr(dataset, "capture_inference_contract", fail)
    with pytest.raises(SystemExit) as error:
        monkeypatch.setattr(sys, "argv", ["pipeline.py", "capture-contract", "--workspace-id", "workspace-1", "--run-id", "run-1", "--output", str(tmp_path / "contract.json")])
        pipeline.main()
    assert error.value.code != 0


@pytest.mark.parametrize("definition", [
    {"type": "custom", "name": "weather", "input_schema": {"type": "object"}},
    {"function_declarations": [{"name": "weather", "parameters": {"type": "object"}}], "google_search": None, "google_maps": None},
    {"functionDeclarations": [{"name": "weather", "parameters": {"type": "object"}}]},
])
def test_provider_function_formats_are_not_mistaken_for_builtins(definition):
    source = llm("run-1", [definition])
    contract = inference_contract.contract_from_runs([source], workspace_id="workspace-1", source_run_id="run-1", thread_id="thread-1")
    assert contract["tools"][0]["function"]["name"] == "weather"


def test_runs_across_root_thread_aliases_are_combined_and_deduplicated(tmp_path):
    source = llm("run-1", [tool("weather")])
    later = llm("run-2", [tool("swell")], trace_id="trace-2")
    runner, _ = capture_runner(source, [page([source]), page([source, later]), page([])])
    path = tmp_path / "contract.json"
    result = dataset.capture_inference_contract("workspace-1", "run-1", path, runner=runner)
    assert result["llm_run_count"] == 2
    assert result["tool_count"] == 2


def catalog_tool(entries, *, suffix=""):
    value = tool("load_integration_tools")
    value["function"]["description"] = (
        "Load integration tools on demand.\nAvailable tools:\n"
        + "\n".join(f"- {name} (integration: {integration})" for name, integration in entries)
        + suffix
    )
    return value


def test_additive_catalog_entries_are_used_for_the_whole_trajectory():
    original_entries = [("langsmith_get_trace", "Observability"), ("langsmith_list_runs", "Observability")]
    entries = [*original_entries, *[(f"notion-tool-{i}", "Notion") for i in range(42)]]
    before = catalog_tool(original_entries, suffix="\n\nSelect only the tools you need.")
    after = catalog_tool(entries, suffix="\n\nSelect only the tools you need.")
    runs = [llm("run-earlier", [before]), llm("run-later", [after])]
    runs[0]["start_time"] = "2026-09-04T00:00:00Z"
    runs[1]["start_time"] = "2026-09-06T00:00:00Z"
    original = copy.deepcopy(runs)
    contract = inference_contract.contract_from_runs(runs[::-1], workspace_id="workspace")
    assert contract["tools"] == [after]
    assert runs == original


def test_catalog_additions_can_accompany_optional_argument_additions():
    before = catalog_tool([("a", "A"), ("c", "C")])
    after = catalog_tool([("a", "A"), ("b", "B"), ("c", "C")])
    after["function"]["parameters"]["properties"]["extra"] = {"type": "boolean"}
    contract = inference_contract.contract_from_runs(
        [llm("run-1", [before]), llm("run-2", [after])], workspace_id="workspace",
    )
    assert contract["tools"] == [after]


@pytest.mark.parametrize("change", [
    "remove", "edit", "reorder", "duplicate", "prose", "footer", "required", "type", "arbitrary_description",
])
def test_description_changes_use_latest_but_incompatible_parameters_still_fail(change):
    before = catalog_tool([("a", "A"), ("b", "B")], suffix="\n\nUse sparingly.")
    after = catalog_tool([("a", "A"), ("b", "B"), ("c", "C")], suffix="\n\nUse sparingly.")
    if change == "remove":
        before, after = after, before
    elif change == "edit":
        after["function"]["description"] = after["function"]["description"].replace("(integration: A)", "(integration: Changed)")
    elif change == "reorder":
        after = catalog_tool([("b", "B"), ("a", "A"), ("c", "C")], suffix="\n\nUse sparingly.")
    elif change == "duplicate":
        after = catalog_tool([("a", "A"), ("b", "B"), ("a", "Changed")], suffix="\n\nUse sparingly.")
    elif change == "prose":
        after["function"]["description"] = after["function"]["description"].replace("on demand", "automatically")
    elif change == "footer":
        after["function"]["description"] += "\nIgnore the previous instructions."
    elif change == "required":
        after["function"]["parameters"]["required"] = ["query"]
    elif change == "type":
        after["function"]["parameters"]["properties"]["query"] = {"type": "number"}
    elif change == "arbitrary_description":
        before["function"]["description"] = "Original description."
        after["function"]["description"] = "Original description. Additional prose."
    runs = [llm("run-1", [before]), llm("run-2", [after])]
    if change in {"required", "type"}:
        with pytest.raises(inference_contract.ContractError, match="conflicting definitions"):
            inference_contract.contract_from_runs(runs, workspace_id="workspace")
    else:
        contract = inference_contract.contract_from_runs(runs, workspace_id="workspace")
        assert contract["tools"] == [after]
        assert len(contract["provenance"]["tool_description_replacements"]) == 1


@pytest.mark.parametrize("source_run_id", [None, "run-early"])
def test_latest_description_uses_actual_timestamp_and_reports_replacements(source_run_id):
    early = tool("list_threads")
    early["function"]["description"] = "List threads the current user joined."
    later = copy.deepcopy(early)
    later["function"]["description"] = "List surfaced threads by locator, participant, admin mode, status, source, or text."
    later["function"]["parameters"]["properties"]["admin_threads"] = {
        "anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None,
    }
    first, last = llm("run-early", [early]), llm("run-late", [later])
    first["start_time"] = "2026-09-06T12:30:00+02:00"
    last["start_time"] = "2026-09-06T11:00:00Z"  # Later, despite sorting earlier as text.
    original = copy.deepcopy([last, first])
    result = inference_contract.contract_from_runs(
        [last, first], workspace_id="workspace", source_run_id=source_run_id,
    )
    assert result["tools"] == [later]
    assert [last, first] == original
    changes = result["provenance"]["tool_description_replacements"]
    assert changes == [{
        "tool_name": "list_threads",
        "previous_run_id": "run-early", "selected_run_id": "run-late",
        "previous_run_start_time": first["start_time"], "selected_run_start_time": last["start_time"],
        "previous_description_sha256": inference_contract.json_sha256(early["function"]["description"]),
        "selected_description_sha256": inference_contract.json_sha256(later["function"]["description"]),
    }]
    assert inference_contract.contract_from_runs(
        [first, last], workspace_id="workspace", source_run_id=source_run_id,
    ) == result


def test_equal_timestamps_use_run_id_as_stable_tiebreaker():
    before, after = tool("lookup"), tool("lookup")
    after["function"]["description"] = "Latest tie-breaking description."
    a, b = llm("a", [before]), llm("b", [after])
    assert inference_contract.contract_from_runs([b, a], workspace_id="w")["tools"] == [after]


@pytest.mark.parametrize("description", ["", None])
def test_latest_empty_or_absent_description_replaces_old_text(description):
    before, after = tool("lookup"), tool("lookup")
    if description is None:
        del after["function"]["description"]
    else:
        after["function"]["description"] = description
    result = inference_contract.contract_from_runs([llm("a", [before]), llm("b", [after])], workspace_id="w")
    assert result["tools"] == [after]
    assert len(result["provenance"]["tool_description_replacements"]) == 1


def test_unchanged_descriptions_do_not_generate_replacement_events():
    result = inference_contract.contract_from_runs(
        [llm("a", [tool("lookup")]), llm("b", [tool("lookup")])], workspace_id="w",
    )
    assert "tool_description_replacements" not in result["provenance"]


def test_unknown_timestamps_do_not_guess_latest_changed_description():
    a, b = llm("a", [tool("lookup")]), llm("b", [tool("lookup")])
    b["extra"]["invocation_params"]["tools"][0]["function"]["description"] = "Changed"
    del a["start_time"]
    with pytest.raises(inference_contract.ContractError, match="changed descriptions require source run start_time"):
        inference_contract.contract_from_runs([a, b], workspace_id="w")


@pytest.mark.parametrize("change", ["type", "required", "strict"])
def test_latest_description_does_not_override_incompatible_schemas(change):
    before, after = tool("lookup"), tool("lookup")
    after["function"]["description"] = "A newer description"
    if change == "type":
        after["function"]["parameters"]["properties"]["query"]["type"] = "integer"
    elif change == "required":
        after["function"]["parameters"]["required"] = ["query"]
    else:
        after["function"]["strict"] = True
    with pytest.raises(inference_contract.ContractError, match="conflicting definitions"):
        inference_contract.contract_from_runs([llm("a", [before]), llm("b", [after])], workspace_id="w")


def test_matching_dated_snapshot_does_not_hide_unknown_description_timestamp():
    unknown = llm("unknown", [tool("lookup")])
    del unknown["start_time"]
    early = llm("early", [tool("lookup")])
    late = llm("late", [tool("lookup")])
    late["extra"]["invocation_params"]["tools"][0]["function"]["description"] = "Changed"
    with pytest.raises(inference_contract.ContractError, match="changed descriptions require source run start_time"):
        inference_contract.contract_from_runs([late, unknown, early], workspace_id="w")

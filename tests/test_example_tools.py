from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from smithtune import dataset
from smithtune import evaluation
from smithtune.artifacts import _json_dump, _jsonl_dump, _load_jsonl
from smithtune.inference_contract import ContractError, TOOL_MERGE_POLICY, contract_from_runs, json_sha256
from smithtune.providers import fireworks
from smithtune.providers.base import PipelineError
from test_tool_capture import llm, tool, page


def example(index, *, thread=None, project="project-1"):
    return {
        "id": f"example-{index}",
        "inputs": {"messages": [{"role": "system", "content": f"policy {index}"},
                                {"role": "human", "content": f"question {index}"},
                                {"role": "ai", "content": f"answer {index}"}]},
        "outputs": None, "metadata": {"source_scope": "thread", "source_scope_id": thread or f"thread-{index}",
                                      "source_project_id": project,
                                      "trajectory_format": "messages", "conversation_scope": "root"},
    }


def source_runner(groups):
    queries = []

    def run(command, capture=False):
        if command[2].startswith("/api/v1/sessions/"):
            return SimpleNamespace(stdout=json.dumps({"id": command[2].rsplit("/", 1)[1],
                                                      "start_time": "2026-09-01T00:00:00Z"}))
        body = json.loads(command[command.index("--body") + 1])
        assert command[command.index("--workspace") + 1] == "workspace-id"
        queries.append(body)
        project = body["project_ids"][0]
        found = []
        for (source_project, thread), runs in groups.items():
            if project != source_project:
                continue
            if body.get("is_root"):
                if json.dumps(thread) in body["filter"] and '\"thread_id\"' in body["filter"]:
                    found.extend({"id": f"{thread}:{tid}", "session_id": project}
                                 for tid in sorted({r["trace_id"] for r in runs}))
            else:
                for item in runs:
                    trace_id = f"{thread}:{item['trace_id']}"
                    if json.dumps(trace_id) in body["filter"]:
                        found.append({**item, "trace_id": trace_id})
        return SimpleNamespace(stdout=json.dumps(page(found)))

    return run, queries


def write_empty_tool_snapshot(root, examples, dataset_id="dataset-id", workspace_id="workspace-id"):
    """Fixture for previously captured, tool-free source conversations."""
    contracts = {}
    for ex in examples:
        payload = contract_from_runs([llm(f"run-{ex['id']}", [])], workspace_id=workspace_id)
        payload["provenance"]["source_example_id"] = ex["id"]
        contracts[ex["id"]] = payload
    _json_dump(root / "raw" / "example_contracts.json", {
        "schema_version": 1, "workspace_id": workspace_id, "dataset_id": dataset_id,
        "tool_merge_policy": TOOL_MERGE_POLICY,
        "source_examples_sha256": hashlib.sha256((root / "raw" / "examples.json").read_bytes()).hexdigest(),
        "contracts": contracts, "contracts_sha256": json_sha256(contracts),
    })


def test_examples_get_their_own_complete_union_and_repeated_sources_are_cached():
    examples = [example(1), example(2), example(3, thread="thread-1")]
    original = copy.deepcopy(examples)
    runner, queries = source_runner({
        ("project-1", "thread-1"): [llm("run-1", [tool("weather")]),
                                    llm("run-2", [tool("weather"), tool("swell")], trace_id="trace-2")],
        ("project-1", "thread-2"): [llm("run-3", [tool("lookup")])],
    })
    contracts = dataset.capture_example_contracts("workspace-id", examples, runner=runner)
    rows = dataset.prepare_sft_rows(examples, example_contracts=contracts)
    assert [[t["function"]["name"] for t in row["tools"]] for row in rows] == [
        ["swell", "weather"], ["lookup"], ["swell", "weather"],
    ]
    assert contracts["example-1"].provenance["source_run_ids"] == ["run-1", "run-2"]
    assert contracts["example-3"].provenance["source_example_id"] == "example-3"
    assert all(contract.inference_settings == {} for contract in contracts.values())
    assert examples == original
    assert len(queries) == 8


def test_same_thread_name_in_different_projects_stays_separate():
    examples = [example(1, thread="same"), example(2, thread="same", project="project-2")]
    second_run = {**llm("run-2", [tool("swell")]), "session_id": "project-2"}
    runner, _ = source_runner({("project-1", "same"): [llm("run-1", [tool("weather")])],
                               ("project-2", "same"): [second_run]})
    contracts = dataset.capture_example_contracts("workspace-id", examples, runner=runner)
    assert contracts["example-1"].tools != contracts["example-2"].tools


@pytest.mark.parametrize("native", [False, True])
def test_trace_only_example_queries_every_llm_in_that_trace(native):
    ex = example(1)
    ex["metadata"].update(source_scope="trace", source_scope_id="trace-1")
    if native:
        del ex["metadata"]["source_project_id"]
        ex["source_session_id"] = "project-1"
    queries = []

    def runner(command, capture=False):
        if command[2].startswith("/api/v1/sessions/"):
            return SimpleNamespace(stdout=json.dumps({"id": command[2].rsplit("/", 1)[1],
                                                      "start_time": "2026-09-01T00:00:00Z"}))
        body = json.loads(command[command.index("--body") + 1])
        queries.append(body)
        assert body["project_ids"] == ["project-1"]
        assert body["run_type"] == "LLM"
        assert body["trace_id"] == "trace-1"
        return SimpleNamespace(stdout=json.dumps(page([llm("run-1", [tool("weather")]), llm("run-2", [tool("swell")])])))

    contracts = dataset.capture_example_contracts("workspace-id", [ex], runner=runner)
    assert len(contracts[ex["id"]].tools) == 2
    assert len(queries) == 1


def test_toolless_source_is_valid_but_missing_source_calls_are_not():
    runner, _ = source_runner({("project-1", "thread-1"): [llm("run-1", [])]})
    contracts = dataset.capture_example_contracts("workspace-id", [example(1)], runner=runner)
    assert contracts["example-1"].tools == ()
    with pytest.raises(PipelineError, match="example example-2.*no LLM runs"):
        dataset.capture_example_contracts("workspace-id", [example(2)], runner=runner)


def setup_preparation(tmp_path, monkeypatch, *, builtin=False):
    examples = [example(1), example(2)]
    raw = tmp_path / "raw"
    _json_dump(raw / "examples.json", examples)
    _json_dump(raw / "dataset-export.json", [{"inputs": ex["inputs"]} for ex in examples])
    _json_dump(raw / "dataset.json", {"id": "dataset-id", "example_count": 2})
    second_tools = [{"type": "web_search"}] if builtin else [tool("swell")]
    runner, _ = source_runner({("project-1", "thread-1"): [llm("run-1", [tool("weather")])],
                               ("project-1", "thread-2"): [llm("run-2", second_tools)]})
    collect = dataset.capture_example_contracts
    monkeypatch.setattr(dataset, "capture_example_contracts", lambda workspace, examples, **kwargs: collect(workspace, examples, runner=runner, **kwargs))
    monkeypatch.setattr(dataset, "download_dataset", lambda *args: None)
    return examples


def prepare(tmp_path, **kwargs):
    return dataset.prepare_dataset("workspace-id", "dataset-id", fireworks.DEFAULT_MODEL, tmp_path,
                                   test_fraction=1, validation_fraction=0, check_render=False, **kwargs)


def test_builtin_in_second_example_aborts_before_publishing_artifacts(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch, builtin=True)
    with pytest.raises(PipelineError, match="example example-2.*run run-2.*provider built-in.*web_search"):
        prepare(tmp_path)
    assert not (tmp_path / "raw" / "example_contracts.json").exists()
    assert not (tmp_path / "prepared").exists()


def test_per_example_capture_roundtrips_offline_without_network(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    manifest = prepare(tmp_path)
    online = (tmp_path / "prepared" / "test.jsonl").read_bytes()
    contracts = dataset._prepared_example_contracts(tmp_path, manifest)
    assert set(contracts) == {"example-1", "example-2"}

    def unexpected(*args):
        pytest.fail("offline preparation attempted a network call")

    monkeypatch.setattr(dataset, "capture_example_contracts", unexpected)
    monkeypatch.setattr(dataset, "download_dataset", unexpected)
    offline = prepare(tmp_path, fetch=False)
    assert offline["example_contracts"] == manifest["example_contracts"]
    assert (tmp_path / "prepared" / "test.jsonl").read_bytes() == online


@pytest.mark.parametrize("corruption", ["missing", "export", "workspace", "coverage", "hash"])
def test_offline_capture_requires_complete_matching_snapshot(tmp_path, monkeypatch, corruption):
    setup_preparation(tmp_path, monkeypatch)
    prepare(tmp_path)
    path = tmp_path / "raw" / "example_contracts.json"
    snapshot = json.loads(path.read_text())
    if corruption == "missing":
        path.unlink()
    elif corruption == "export":
        raw_path = tmp_path / "raw" / "examples.json"
        raw_path.write_text(raw_path.read_text() + "\n")
    else:
        if corruption == "workspace":
            snapshot["workspace_id"] = "other-workspace"
        elif corruption == "coverage":
            del snapshot["contracts"]["example-2"]
            snapshot["contracts_sha256"] = json_sha256(snapshot["contracts"])
        else:
            snapshot["contracts_sha256"] = "0" * 64
        _json_dump(path, snapshot)
    with pytest.raises(PipelineError, match="(cached|cover every example)"):
        prepare(tmp_path, fetch=False)


def test_replay_dispatches_matching_contracts_and_rejects_stale_results(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    prepare(tmp_path)
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, *args: (
        [{**case, "prompt_tokens": 10} for case in cases], [],
    ))
    monkeypatch.setattr(evaluation, "calibrate_judge", lambda *args: [{"actual": True, "expected": True}])
    monkeypatch.setattr(evaluation, "judge_replay_candidate", lambda *args: {"pass": True, "reason": "ok"})
    requests = []

    def chat(model, messages, max_tokens, json_mode, contract):
        requests.append((messages[0]["content"], [t["function"]["name"] for t in contract.tools]))
        return {"role": "assistant", "content": "answer"}

    output = tmp_path / "replay"
    kwargs = dict(confirm=True, chat=chat, concurrency=1)
    summary = evaluation.run_replay_evaluation(tmp_path, output, "tuned", "judge", **kwargs)
    assert sorted(requests) == [("policy 1", ["weather"]), ("policy 2", ["swell"])]
    assert "example_contracts_sha256" in summary
    requests.clear()
    evaluation.run_replay_evaluation(tmp_path, output, "tuned", "judge", **kwargs)
    assert requests == []
    results = _load_jsonl(output / "results.jsonl")
    results[0]["contract_sha256"] = "0" * 64
    _jsonl_dump(output / "results.jsonl", results)
    with pytest.raises(PipelineError, match="different inference contract"):
        evaluation.run_replay_evaluation(tmp_path, output, "tuned", "judge", **kwargs)


def test_replay_rejects_row_tools_that_differ_from_saved_example(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    prepare(tmp_path)
    path = tmp_path / "prepared" / "test.jsonl"
    rows = _load_jsonl(path)
    rows[0]["tools"] = rows[1]["tools"]
    _jsonl_dump(path, rows)
    with pytest.raises(PipelineError, match="different tool schemas"):
        evaluation.prepare_replay_evaluation(tmp_path, tmp_path / "replay")


def test_prepared_contract_mapping_rejects_tampering(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    manifest = prepare(tmp_path)
    path = tmp_path / "prepared" / "example_contracts.json"
    payload = json.loads(path.read_text())
    payload["example-1"] = payload["example-2"]
    _json_dump(path, payload)
    with pytest.raises(PipelineError, match="hash differs"):
        dataset._prepared_example_contracts(tmp_path, manifest)



@pytest.mark.parametrize("temperature", [None, 0.7])
def test_replay_temperature_default_survives_automatic_contracts(monkeypatch, temperature):
    import io
    from smithtune import inference
    from smithtune.inference_contract import parse_inference_contract
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "test-session")
    payload = contract_from_runs([llm("run-1", [])], workspace_id="workspace-id")
    if temperature is not None:
        payload["inference_settings"] = {"temperature": temperature}
        del payload["contract_sha256"]
    contract = parse_inference_contract(payload)
    requests = []

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())

    monkeypatch.setattr(inference.urllib.request, "urlopen", urlopen)
    inference._fireworks_chat_completion("model", [{"role": "user", "content": "hi"}], 128, request_contract=contract)
    assert requests[0]["temperature"] == (0 if temperature is None else temperature)


def test_same_tool_name_may_differ_between_examples_but_not_within_one():
    changed = tool("weather")
    changed["function"]["parameters"]["required"] = ["query"]
    runner, _ = source_runner({("project-1", "thread-1"): [llm("run-1", [tool("weather")])],
                               ("project-1", "thread-2"): [llm("run-2", [changed])]})
    contracts = dataset.capture_example_contracts("workspace-id", [example(1), example(2)], runner=runner)
    assert contracts["example-1"].tools != contracts["example-2"].tools
    runner, _ = source_runner({("project-1", "thread-1"): [llm("run-1", [tool("weather")]), llm("run-2", [changed])]})
    with pytest.raises(PipelineError, match="example example-1.*weather.*conflicting definitions.*run-1.*run-2"):
        dataset.capture_example_contracts("workspace-id", [example(1)], runner=runner)


@pytest.mark.parametrize("expanded_first", [False, True])
@pytest.mark.parametrize("additional_properties", [None, False, True])
def test_optional_tool_argument_expansion_preserves_unused_tools(expanded_first, additional_properties):
    original = tool("list_threads")
    parameters = original["function"]["parameters"]
    parameters["required"] = ["query"]
    if additional_properties is not None:
        parameters["additionalProperties"] = additional_properties
    expanded = copy.deepcopy(original)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    definitions = [expanded, original] if expanded_first else [original, expanded]
    runs = [llm(f"run-{i}", [definition]) for i, definition in enumerate(definitions)]
    before = copy.deepcopy(runs)
    runner, _ = source_runner({("project-1", "thread-1"): runs})
    examples = [example(1)]  # No tool calls in the conversation.
    contracts = dataset.capture_example_contracts("workspace-id", examples, runner=runner)
    rows = dataset.prepare_sft_rows(examples, example_contracts=contracts)
    assert rows[0]["tools"] == [expanded]
    assert runs == before
    assert contracts["example-1"].tools_sha256 == json_sha256([expanded])


def test_optional_arguments_are_combined_across_multiple_snapshots():
    base = tool("list_threads")
    admin = copy.deepcopy(base)
    admin["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    limit = copy.deepcopy(base)
    limit["function"]["parameters"]["properties"]["limit"] = {"type": "integer"}
    merged = contract_from_runs([llm("a", [admin]), llm("b", [limit]), llm("c", [base])], workspace_id="workspace")
    assert merged["tools"][0]["function"]["parameters"]["properties"] == {
        "query": {"type": "string"}, "admin_threads": {"type": ["boolean", "null"]}, "limit": {"type": "integer"},
    }
    reversed_merge = contract_from_runs([llm("a", [limit]), llm("b", [base]), llm("c", [admin])], workspace_id="workspace")
    assert merged["contract_sha256"] == reversed_merge["contract_sha256"]


@pytest.mark.parametrize("change", ["required", "existing_type", "nested", "additional_properties"])
def test_optional_expansion_still_rejects_incompatible_changes(change):
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["filters"] = {"type": "object", "properties": {}}
    expanded = copy.deepcopy(base)
    parameters = expanded["function"]["parameters"]
    parameters["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    if change == "required":
        parameters["required"] = ["admin_threads"]
    elif change == "existing_type":
        parameters["properties"]["query"]["type"] = "integer"
    elif change == "nested":
        parameters["properties"]["filters"]["properties"]["new"] = {"type": "string"}
    else:
        parameters["additionalProperties"] = False
    with pytest.raises(ContractError, match="list_threads.*conflicting definitions.*a.*b"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


@pytest.mark.parametrize("constraint", [
    {"maxProperties": 1}, {"anyOf": [{"required": ["query"]}]},
    {"dependentRequired": {"query": ["admin_threads"]}},
    {"additionalProperties": {"type": "string"}}, {"required": ["admin_threads"]},
])
def test_optional_expansion_rejects_interacting_root_constraints(constraint):
    base = tool("list_threads")
    base["function"]["parameters"].update(constraint)
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")
    # Unchanged complex schemas remain supported.
    assert contract_from_runs([llm("a", [base]), llm("b", [base])], workspace_id="workspace")["tools"] == [base]


def test_optional_expansion_rejects_schema_references():
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["query"] = {"$ref": "#/properties/admin_threads"}
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


def test_optional_expansion_rejects_draft3_property_level_required():
    base = tool("list_threads")
    base["function"]["parameters"]["$schema"] = "http://json-schema.org/draft-03/schema#"
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {
        "type": ["boolean", "null"], "required": True,
    }
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


def test_optional_expansion_distinguishes_boolean_and_numeric_constraints():
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["flag"] = {"const": True}
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["flag"] = {"const": 1}
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


def test_mixed_workspaces_route_and_cache_sources_separately():
    examples = [example(i, thread="same") for i in range(1, 5)]
    examples[1]["metadata"]["source_workspace_id"] = "other"
    examples[2]["metadata"]["source_workspace_id"] = "other"
    examples[3]["metadata"]["source_workspace_id"] = "workspace-id"
    calls = []

    def runner(command, capture=False):
        workspace = command[command.index("--workspace") + 1]
        if command[2].startswith("/api/v1/sessions/"):
            return SimpleNamespace(stdout=json.dumps({"id": command[2].rsplit("/", 1)[1],
                                                      "start_time": "2026-09-01T00:00:00Z"}))
        body = json.loads(command[command.index("--body") + 1])
        calls.append(workspace)
        if body.get("is_root"):
            runs = [{"id": "trace-1", "session_id": "project-1"}] if '"thread_id"' in body["filter"] else []
        else:
            runs = [llm(f"run-{workspace}", [tool(workspace)])]
        return SimpleNamespace(stdout=json.dumps(page(runs)))

    contracts = dataset.capture_example_contracts(
        "workspace-id", examples, source_workspace_id="default-source", runner=runner,
    )
    assert calls == ["default-source"] * 4 + ["other"] * 4 + ["workspace-id"] * 4
    for ex, workspace in zip(examples, ["default-source", "other", "other", "workspace-id"], strict=True):
        contract = contracts[ex["id"]]
        assert contract.provenance["source_workspace_id"] == workspace
        assert contract.tools[0]["function"]["name"] == workspace


def test_trace_with_explicit_project_uses_source_workspace():
    ex = example(1)
    ex["metadata"].update(source_scope="trace", source_scope_id="trace-1")
    queries = []

    def runner(command, capture=False):
        assert command[command.index("--workspace") + 1] == "source-workspace"
        if command[2].startswith("/api/v1/sessions/"):
            return SimpleNamespace(stdout=json.dumps({"id": command[2].rsplit("/", 1)[1],
                                                      "start_time": "2026-09-01T00:00:00Z"}))
        body = json.loads(command[command.index("--body") + 1])
        queries.append(body)
        runs = [llm("run-1", [])]
        return SimpleNamespace(stdout=json.dumps(page(runs)))

    contracts = dataset.capture_example_contracts(
        "dataset-workspace", [ex], source_workspace_id="source-workspace", runner=runner,
    )
    assert len(queries) == 1
    assert contracts[ex["id"]].provenance["source_workspace_id"] == "source-workspace"


@pytest.mark.parametrize("value", [None, "", " ", " padded ", 123, [], {}])
def test_invalid_source_workspace_metadata_does_not_fall_back(value):
    ex = example(1)
    ex["metadata"]["source_workspace_id"] = value

    def unexpected(*args, **kwargs):
        pytest.fail("invalid source workspace must not trigger a query")

    with pytest.raises(PipelineError, match="source workspace ID"):
        dataset.capture_example_contracts("workspace-id", [ex], runner=unexpected)


def test_source_workspace_access_failure_does_not_fall_back():
    calls = []

    def runner(command, capture=False):
        calls.append(command[command.index("--workspace") + 1])
        raise PipelineError("access denied")

    with pytest.raises(PipelineError, match="example example-1.*access denied"):
        dataset.capture_example_contracts(
            "dataset-workspace", [example(1)], source_workspace_id="source-workspace", runner=runner,
        )
    assert calls == ["source-workspace"]


def test_offline_capture_rejects_changed_source_workspace(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    prepare(tmp_path)
    with pytest.raises(PipelineError, match="different source workspace"):
        prepare(tmp_path, fetch=False, source_workspace_id="different")
    # Explicitly choosing the original workspace keeps existing caches usable.
    prepare(tmp_path, fetch=False, source_workspace_id="workspace-id")


def test_preparation_reports_description_replacements_and_reuses_them_offline(tmp_path, monkeypatch):
    collect = dataset.capture_example_contracts
    setup_preparation(tmp_path, monkeypatch)
    before, after = tool("list_threads"), tool("list_threads")
    after["function"]["description"] = "List surfaced threads by locator, participant or admin mode."
    after["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    runner, _ = source_runner({
        ("project-1", "thread-1"): [llm("r1", [before]), llm("r2", [after])],
        ("project-1", "thread-2"): [llm("r3", [])],
    })
    monkeypatch.setattr(dataset, "capture_example_contracts",
                        lambda workspace, examples, **kwargs: collect(workspace, examples, runner=runner, **kwargs))
    raw_before = (tmp_path / "raw" / "examples.json").read_bytes()
    manifest = prepare(tmp_path)
    assert manifest["prepared"] == {"accepted": 2, "rejected": 0}
    assert manifest["audit"]["tool_description_replacements"] == 1
    report = json.loads((tmp_path / "prepared" / "tool_description_replacements.json").read_text())
    assert len(report) == 1
    assert report[0]["example_id"] == "example-1"
    assert report[0]["tool_name"] == "list_threads"
    assert report[0]["previous_run_id"] == "r1"
    assert report[0]["selected_run_id"] == "r2"
    assert (tmp_path / "raw" / "examples.json").read_bytes() == raw_before
    contracts = dataset._prepared_example_contracts(tmp_path, manifest)
    assert list(contracts["example-1"].tools) == [after]

    def unexpected(*args, **kwargs):
        pytest.fail("offline preparation called the network")
    monkeypatch.setattr(dataset, "capture_example_contracts", unexpected)
    monkeypatch.setattr(dataset, "download_dataset", unexpected)
    offline = prepare(tmp_path, fetch=False)
    assert offline["audit"]["tool_description_replacements"] == 1
    assert json.loads((tmp_path / "prepared" / "tool_description_replacements.json").read_text()) == report


def test_previous_capture_policy_requires_fresh_capture(tmp_path, monkeypatch):
    setup_preparation(tmp_path, monkeypatch)
    prepare(tmp_path)
    path = tmp_path / "raw" / "example_contracts.json"
    snapshot = json.loads(path.read_text())
    del snapshot["tool_merge_policy"]
    _json_dump(path, snapshot)
    with pytest.raises(PipelineError, match="prepare again without --no-fetch"):
        prepare(tmp_path, fetch=False)


def test_interrupted_capture_resumes_and_publishes_complete_snapshot(tmp_path, monkeypatch, capsys):
    examples = [example(1), example(2)]
    calls = []
    fail_second = True

    def query(workspace, project, thread, **kwargs):
        calls.append(thread)
        if thread == "thread-2" and fail_second:
            raise PipelineError("context deadline exceeded")
        return [llm(f"run-{thread}", [tool("weather")])]

    monkeypatch.setattr(dataset, "_project_start_time", lambda *args, **kwargs: "2026-09-01T00:00:00Z")
    monkeypatch.setattr(dataset, "_query_thread_llm_runs", query)

    def capture(fetch=True):
        return dataset._example_contract_snapshot("workspace-id", "dataset-id", examples,
                                                  "source-sha", tmp_path, fetch=fetch)

    with pytest.raises(PipelineError, match="example example-2.*deadline exceeded"):
        capture()
    partial = tmp_path / "example_contracts.partial.json"
    checkpoint = json.loads(partial.read_text())
    assert set(checkpoint["contracts"]) == {"example-1"}
    assert not (tmp_path / "example_contracts.json").exists()
    with pytest.raises(PipelineError, match="no cached example inference contracts"):
        capture(fetch=False)

    fail_second = False
    contracts = capture()
    assert calls == ["thread-1", "thread-2", "thread-2"]
    assert set(contracts) == {"example-1", "example-2"}
    assert "1/2 examples already saved" in capsys.readouterr().err
    assert not partial.exists()
    assert {key: value.to_dict() for key, value in capture(fetch=False).items()} == {
        key: value.to_dict() for key, value in contracts.items()
    }
    assert calls == ["thread-1", "thread-2", "thread-2"]
    capture()  # A completed checkpoint does not turn a fresh fetch into a stale-cache read.
    assert calls[-2:] == ["thread-1", "thread-2"]


@pytest.mark.parametrize("change", ["contents", "source_project", "source_workspace", "dataset_workspace", "policy"])
def test_capture_checkpoint_invalidated_by_changed_inputs(tmp_path, monkeypatch, change):
    examples = [example(1), example(2)]
    path = tmp_path / "partial.json"
    calls = []

    def query(workspace, project, thread, **kwargs):
        calls.append((workspace, project, thread))
        if thread == "thread-2":
            raise PipelineError("request failed")
        return [llm("run-1", [])]

    monkeypatch.setattr(dataset, "_project_start_time", lambda *args, **kwargs: "2026-09-01T00:00:00Z")
    monkeypatch.setattr(dataset, "_query_thread_llm_runs", query)
    with pytest.raises(PipelineError, match="request failed"):
        dataset.capture_example_contracts("workspace-id", examples, checkpoint_path=path)
    kwargs = {}
    workspace = "workspace-id"
    if change == "contents":
        examples[0]["inputs"]["messages"][0]["content"] = "updated policy"
    elif change == "source_project":
        examples[0]["metadata"]["source_project_id"] = "other-project"
    elif change == "source_workspace":
        kwargs["source_workspace_id"] = "other-workspace"
    elif change == "dataset_workspace":
        workspace = "other-workspace"
    else:
        monkeypatch.setattr(dataset, "TOOL_MERGE_POLICY", "next-policy")
    calls.clear()
    with pytest.raises(PipelineError, match="request failed"):
        dataset.capture_example_contracts(workspace, examples, checkpoint_path=path, **kwargs)
    assert [call[2] for call in calls] == ["thread-1", "thread-2"]


def test_capture_rejects_corrupted_checkpoint(tmp_path, monkeypatch):
    examples = [example(1), example(2)]
    path = tmp_path / "partial.json"

    def query(workspace, project, thread, **kwargs):
        if thread == "thread-2":
            raise PipelineError("request failed")
        return [llm("run-1", [])]

    monkeypatch.setattr(dataset, "_project_start_time", lambda *args, **kwargs: "2026-09-01T00:00:00Z")
    monkeypatch.setattr(dataset, "_query_thread_llm_runs", query)
    with pytest.raises(PipelineError, match="request failed"):
        dataset.capture_example_contracts("workspace-id", examples, checkpoint_path=path)
    checkpoint = json.loads(path.read_text())
    checkpoint["contracts"]["example-1"]["provenance"]["source_workspace_id"] = "wrong-workspace"
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(PipelineError, match="checkpoint hash mismatch"):
        dataset.capture_example_contracts("workspace-id", examples, checkpoint_path=path)


def test_project_history_start_is_fetched_once_per_workspace_and_project():
    examples = [example(1), example(2)]
    inner, _ = source_runner({
        ("project-1", "thread-1"): [llm("run-1", [])],
        ("project-1", "thread-2"): [llm("run-2", [])],
    })
    lookups = []

    def runner(command, capture=False):
        if command[2].startswith("/api/v1/sessions/"):
            lookups.append(command[2])
        return inner(command, capture=capture)

    assert len(dataset.capture_example_contracts("workspace-id", examples, runner=runner)) == 2
    assert lookups == ["/api/v1/sessions/project-1"]


@pytest.mark.parametrize("scope", ["thread", "trace"])
def test_missing_source_project_fails_without_discovery(scope):
    ex = example(1)
    del ex["metadata"]["source_project_id"]
    ex["metadata"].update(source_scope=scope, source_scope_id=f"{scope}-1")

    def unexpected(*args, **kwargs):
        pytest.fail("missing source project must not trigger a network request")

    with pytest.raises(PipelineError, match=r"example example-1: cannot collect tools: missing source project ID; add metadata.source_project_id"):
        dataset.capture_example_contracts("workspace-id", [ex], runner=unexpected)

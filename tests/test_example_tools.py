from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import dataset
import evaluation
from artifacts import _json_dump, _jsonl_dump, _load_jsonl
from inference_contract import contract_from_runs, json_sha256
from providers import fireworks
from providers.base import PipelineError
from test_tool_capture import llm, tool, page


def example(index, *, thread=None, project="project-1"):
    return {
        "id": f"example-{index}", "source_thread_id": thread or f"thread-{index}",
        "inputs": {"messages": [{"role": "system", "content": f"policy {index}"},
                                {"role": "human", "content": f"question {index}"},
                                {"role": "ai", "content": f"answer {index}"}]},
        "outputs": None, "metadata": {"source_project_id": project,
                                      "trajectory_format": "messages", "conversation_scope": "root"},
    }


def source_runner(groups):
    queries = []

    def run(command, capture=False):
        body = json.loads(command[command.index("--body") + 1])
        assert command[command.index("--workspace") + 1] == "workspace-id"
        queries.append(body)
        project = body["session"][0]
        for (source_project, thread), runs in groups.items():
            if project == source_project and json.dumps(thread) in body.get("trace_filter", ""):
                if '"thread_id"' in body["trace_filter"]:
                    return SimpleNamespace(stdout=json.dumps(page(runs)))
        return SimpleNamespace(stdout=json.dumps(page([])))

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
    assert len(queries) == 6


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
    del ex["source_thread_id"]
    del ex["metadata"]["source_project_id"]
    (ex if native else ex["metadata"])["source_trace_id"] = "trace-1"
    if native:
        ex["source_session_id"] = "project-1"
    queries = []

    def runner(command, capture=False):
        body = json.loads(command[command.index("--body") + 1])
        queries.append(body)
        if body.get("id"):
            return SimpleNamespace(stdout=json.dumps(page([{"id": "trace-1", "session_id": "project-1"}])))
        assert body["session"] == ["project-1"]
        assert body["run_type"] == "llm"
        assert body["filter"] == 'eq(trace_id,"trace-1")'
        return SimpleNamespace(stdout=json.dumps(page([llm("run-1", [tool("weather")]), llm("run-2", [tool("swell")])])))

    contracts = dataset.capture_example_contracts("workspace-id", [ex], runner=runner)
    assert len(contracts[ex["id"]].tools) == 2
    assert len(queries) == (1 if native else 2)


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
    monkeypatch.setattr(dataset, "capture_example_contracts", lambda workspace, examples: collect(workspace, examples, runner=runner))
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
    monkeypatch.setattr(evaluation, "calibrate_judge", lambda *args: [])
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
    import inference
    from inference_contract import parse_inference_contract
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

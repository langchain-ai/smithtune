"""Bound allocations by active evidence, not the number of saved trajectories."""

import gc
import hashlib
import json
import tracemalloc

import pytest

from smithtune import artifacts, checkpoint, dataset_workflow, triage, triage_source
from smithtune.dataset_artifacts import LazySequence, save_conversation
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError
from test_triage import API, judge_call, source, uid

MIB = 1024 * 1024


def measured(call):
    gc.collect()
    tracemalloc.start()
    try:
        result = call()
        retained, peak = tracemalloc.get_traced_memory()
        return result, retained, peak
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("value", [
    None, True, False, 0, -12, 2**80, -0.0, 1e-100,
    float("inf"), float("-inf"), float("nan"), "", [], {},
    {"z": [None, True, 0.25, {"b": "café 中文 🙂", "a": '\n\t"\\\u0000\u2028\u2029'}], "a": {}},
    {3: "three", 1: "one"}, ("tuple", {"nested": [1, 2, 3]}),
])
def test_incremental_hash_preserves_saved_fingerprints(value):
    expected = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert json_sha256(value) == expected


def test_hash_and_writes_do_not_copy_entire_documents(tmp_path):
    # Distinct small tokens represent repeated LLM context in a large trace.
    value = [{"inputs": "x" * (MIB // 16)} for _ in range(256)]
    _, _, peak = measured(lambda: json_sha256(value))
    assert peak < MIB
    for write in (lambda: artifacts._json_dump(tmp_path / "object.json", value),
                  lambda: artifacts._jsonl_dump(tmp_path / "rows.jsonl", iter(value))):
        _, _, peak = measured(write)
        assert peak < MIB
    assert json.loads((tmp_path / "object.json").read_text()) == value
    assert artifacts._load_jsonl(tmp_path / "rows.jsonl") == value


def test_streamed_atomic_writes_keep_format_and_previous_file_on_failure(tmp_path):
    value = {"b": ["café", 1], "a": "line\n"}
    path = tmp_path / "saved.json"
    artifacts._json_dump(path, value)
    original = path.read_bytes()
    assert original.decode() == json.dumps(value, indent=2, sort_keys=True) + "\n"
    with pytest.raises(TypeError):
        artifacts._json_dump(path, [value, object()])
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_lazy_sequence_does_not_cache_mutable_bodies():
    calls = []
    def load(index):
        calls.append(index)
        return {"index": index}
    values = LazySequence(3, load)
    assert len(values) == 3 and not calls
    assert values[-1] == {"index": 2}
    assert list(values[1:]) == [{"index": 1}, {"index": 2}]
    values[0]["index"] = "local edit"
    assert values[0] == {"index": 0}
    with pytest.raises(IndexError):
        values[3]


@pytest.mark.parametrize("count", [4, 16])
def test_snapshot_load_verifies_files_without_retaining_bodies(tmp_path, count):
    paths = []
    for index in range(count):
        unit = {"example": {"id": str(index), "inputs": {"messages": [{"content": "x" * MIB}]}}, "traces": []}
        paths.append(str(save_conversation(tmp_path, unit).relative_to(tmp_path)))
    manifest = {"schema_version": 3, "source": {}, "selected_trace_ids": [], "unit_files": paths}
    manifest["snapshot_sha256"] = json_sha256(manifest)
    artifacts._json_dump(tmp_path / "snapshot.json", manifest)
    frozen, retained, peak = measured(lambda: triage_source.load_snapshot(tmp_path))
    assert retained < MIB and peak < 6 * MIB
    assert len(frozen["units"]) == count
    assert frozen["units"][count - 1]["example"]["id"] == str(count - 1)
    path = tmp_path / paths[-1]
    path.write_text('{"traces": [], "changed": true}')
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        triage_source.load_snapshot(tmp_path)
    # On-demand accesses still recheck integrity after initial validation.
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        frozen["units"][-1]


@pytest.mark.parametrize("runner_mode", ["api", "deepagent"])
def test_queued_council_tasks_hold_only_summaries(tmp_path, monkeypatch, runner_mode):
    api = API()
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"})
    seen = []
    def judge(slot, messages, tokens, **kwargs):
        body = json.loads(messages[-1]["content"])
        assert body["untrusted_trajectory"] and body["untrusted_assistant_tool_bindings"]
        seen.append(slot["name"])
        return judge_call(slot, messages, tokens)
    def dispatch(pending, run_task, save_record, *args, **kwargs):
        assert len(pending) == 6
        assert all(set(trajectory) == {"trajectory_id", "multimodal_types", "has_assistant_runs"}
                   for trajectory, _ in pending)
        for trajectory, slot in pending:
            save_record(run_task(trajectory, slot))
    monkeypatch.setattr(triage, "_run_direct", dispatch)
    monkeypatch.setattr(triage, "check_credentials", lambda _: None)
    monkeypatch.setattr(triage, "api_judge", judge)
    monkeypatch.setattr(triage, "deepagent_judge", judge)
    if runner_mode == "deepagent":
        from smithtune import triage_agent, triage_coordinator
        monkeypatch.setattr(triage_agent, "check_installation", lambda: None)
        monkeypatch.setattr(triage_coordinator, "coordinate", dispatch)
    summary = triage.run_triage(source(), tmp_path, runner=api, runner_mode=runner_mode, confirm=True, attempts=1)
    assert summary["kept"] == 2 and len(seen) == 6
    examples = triage.selected_examples(tmp_path, require_complete=True)
    assert isinstance(examples, LazySequence) and len(examples) == 2
    assert all(example["metadata"]["smithtune_triage"] for example in examples)


@pytest.mark.parametrize("council", [False, True])
def test_upload_selection_stays_lazy_and_tamper_checks_precede_writes(tmp_path, council):
    api = API()
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"})
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    if council:
        triage.run_triage(source(), tmp_path, runner=api, judge_call=judge_call, confirm=True)
    state = {"stages": ["triage", "push"] if council else ["push"], "destination": {"name": "selected"}}
    examples = dataset_workflow._examples(tmp_path, frozen, state)
    assert isinstance(examples, LazySequence) and len(examples) == 2
    path = tmp_path / frozen["unit_files"][-1]
    unit = checkpoint.read_file(tmp_path, frozen["unit_files"][-1])
    unit["example"]["inputs"]["messages"][-1]["content"] = "changed"
    path.write_text(json.dumps(unit))
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        dataset_workflow._push(tmp_path, frozen, state, confirm=True,
                               runner=lambda *_a, **_kw: pytest.fail("tampered input reached a remote write"))


def test_download_does_not_fetch_raw_run_trees(tmp_path):
    api = API()  # Rejects any raw trace-run request.
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    assert len(frozen["traces"]) == 2
    assert all(not command[2].startswith("/api/v2/traces/") for command, _ in api.calls)
    bindings = frozen["units"][0]["example"]["metadata"]["smithtune_source"]["assistant_runs"]
    assert [b["run_id"] for b in bindings] == [uid(1001), uid(1002)]

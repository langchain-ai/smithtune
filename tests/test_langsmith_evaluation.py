"""Exercise split and experiment publication with an in-memory LangSmith service boundary."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import pytest
from langsmith import Client
from langsmith.schemas import Example, Run, TracerSession
from langsmith.utils import LangSmithNotFoundError

from smithtune import dataset
from smithtune.evaluation import replay as evaluation
from smithtune.evaluation import langsmith as reporting
from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _load_jsonl
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL
from test_pipeline import example, write_raw


pytestmark = pytest.mark.sdk_integration
WORKSPACE = str(UUID(int=900))
DATASET = str(UUID(int=901))


class MemoryClient(Client):
    def __init__(self, examples):
        super().__init__(api_url="http://localhost:9999", api_key="unit-test", workspace_id=WORKSPACE,
                         auto_batch_tracing=False, info={"version": "0.16.0", "instance_flags": {}})
        self.examples = {item["id"]: copy.deepcopy(item) for item in examples}
        self.version = datetime(2026, 9, 1, tzinfo=UTC)
        self.versions = {}
        self.projects = {}
        self.runs_store = {}
        self.feedback_store = []
        self.split_writes = []
        self.fail_split = None
        self.drop_feedback = None
        self.run_batches = []
        self._snapshot()

    def _snapshot(self):
        self.version += timedelta(seconds=1)
        self.versions[self.version.isoformat()] = copy.deepcopy(self.examples)

    def list_examples(self, dataset_id=None, splits=None, as_of=None, **kwargs):
        assert str(dataset_id) == DATASET
        values = self.versions[as_of] if as_of else self.examples
        for value in values.values():
            metadata = value.get("metadata", {})
            if splits and not set(splits).intersection(metadata.get("dataset_split", [])):
                continue
            yield Example(**copy.deepcopy(value), dataset_id=DATASET,
                          created_at=datetime(2026, 8, 1, tzinfo=UTC), modified_at=self.version)

    def read_dataset_version(self, **kwargs):
        return SimpleNamespace(as_of=self.version)

    def update_dataset_splits(self, *, dataset_id, split_name, example_ids, remove=False):
        if split_name == self.fail_split:
            raise ConnectionError("interrupted split write")
        self.split_writes.append((split_name, list(example_ids), remove))
        for example_id in example_ids:
            metadata = self.examples[str(example_id)]["metadata"]
            splits = set(metadata.get("dataset_split", []))
            (splits.discard if remove else splits.add)(split_name)
            metadata["dataset_split"] = sorted(splits)
        self._snapshot()

    def read_project(self, *, project_name=None, project_id=None, **kwargs):
        for project in self.projects.values():
            if project.name == project_name or str(project.id) == str(project_id):
                return project
        raise LangSmithNotFoundError("project missing")

    def create_project(self, project_name, *, reference_dataset_id=None, metadata=None, **kwargs):
        project = TracerSession(id=uuid4(), name=project_name, tenant_id=WORKSPACE,
                               reference_dataset_id=reference_dataset_id, extra={"metadata": metadata or {}},
                               _host_url="https://smith.langchain.com")
        self.projects[str(project.id)] = project
        return project

    def update_project(self, project_id, *, metadata=None, **kwargs):
        project = self.read_project(project_id=project_id)
        if metadata is not None:
            project.extra["metadata"] = copy.deepcopy(metadata)
        return project

    def create_run(self, name, inputs, run_type, *, project_name=None, **kwargs):
        project_name = project_name or kwargs.pop("session_name", None) or "evaluators"
        try:
            project = self.read_project(project_name=project_name)
        except LangSmithNotFoundError:
            project = self.create_project(project_name)
        value = {**kwargs, "name": name, "inputs": inputs, "run_type": run_type, "session_id": project.id}
        value.setdefault("trace_id", value["id"])
        self.runs_store[str(value["id"])] = Run(**value)

    def batch_ingest_runs(self, create=None, update=None):
        assert not update
        self.run_batches.append([str(run["id"]) for run in create])
        for run in create:
            value = copy.deepcopy(run)
            project = self.read_project(project_id=value.pop("session_id"))
            self.create_run(project_name=project.name, **value)

    def update_run(self, run_id, **kwargs):
        value = self.runs_store[str(run_id)].model_dump()
        value.update(kwargs)
        self.runs_store[str(run_id)] = Run(**value)

    def read_run(self, run_id, **kwargs):
        if str(run_id) not in self.runs_store:
            raise LangSmithNotFoundError("run missing")
        return self.runs_store[str(run_id)].model_copy(deep=True)

    def list_runs(self, *, project_id=None, project_name=None, is_root=None, run_ids=None, **kwargs):
        project = self.read_project(project_id=project_id, project_name=project_name)
        for run in self.runs_store.values():
            if (run.session_id == project.id and (is_root is None or is_root == (run.parent_run_id is None))
                    and (run_ids is None or str(run.id) in {str(value) for value in run_ids})):
                yield run.model_copy(deep=True)

    def create_feedback(self, run_id=None, key="unnamed", *, score=None, comment=None, feedback_id=None, **kwargs):
        if key == self.drop_feedback:
            raise ConnectionError("feedback unavailable")
        assert feedback_id is not None
        item = SimpleNamespace(id=feedback_id, run_id=UUID(str(run_id)), key=key, score=score, comment=comment)
        if not any(saved.id == item.id for saved in self.feedback_store):
            self.feedback_store.append(item)
        return item

    def list_feedback(self, *, run_ids=None, **kwargs):
        ids = {str(run_id) for run_id in run_ids}
        return iter(item for item in self.feedback_store if str(item.run_id) in ids)

    def flush(self, **kwargs):
        pass


def toy_examples(count=12):
    values = []
    for index in range(count):
        item = example(index)
        item["id"] = str(UUID(int=index + 1))
        item["inputs"]["messages"] += [
            {"role": "human", "content": f"followup {index}", "id": f"u-{index}"},
            {"role": "ai", "content": f"second answer {index}", "id": f"a-{index}"},
        ]
        values.append(item)
    return values


def empty_contract():
    return parse_inference_contract({"schema_version": 1, "format": "main_model_inference_contract",
        "tools": [], "tools_sha256": json_sha256([]), "provenance": {}, "inference_settings": {}})


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    examples = toy_examples()
    client = MemoryClient(examples)
    data = tmp_path / "data"
    data.mkdir()
    write_raw(data, examples, dataset_id=DATASET, workspace_id=WORKSPACE)
    monkeypatch.setattr(reporting, "make_client", lambda _: client)
    monkeypatch.setattr(reporting.time, "sleep", lambda _: None)
    model = DEFAULT_MODEL
    if getattr(request, "param", None) == "baseten":
        from smithtune.providers.baseten import DEFAULT_MODEL as model
    manifest = dataset.prepare_dataset(WORKSPACE, DATASET, model, data, fetch=False,
        inference_contract=empty_contract(), check_render=False, validation_fraction=.2, test_fraction=.3)
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, *_args, **_kwargs: ([{**case, "prompt_tokens": 8} for case in cases], []))
    return data, manifest, client


def test_split_membership_is_exact_versioned_and_idempotent(prepared):
    data, manifest, client = prepared
    receipt = _load_json(data / "prepared/langsmith-splits.json")
    assert all(receipt["splits"][name] for name in reporting.SPLITS)
    original_inputs = {key: copy.deepcopy(value["inputs"]) for key, value in client.examples.items()}
    original_writes = len(client.split_writes)
    synchronized = reporting.synchronize_splits(data, manifest, _load_json(data / "raw/examples.json"), client=client)
    assert synchronized == manifest["langsmith"]["split_sync"]
    assert len(client.split_writes) == original_writes
    assert {key: value["inputs"] for key, value in client.examples.items()} == original_inputs
    # Future dataset edits cannot change the old pinned test cohort.
    chosen = receipt["splits"]["test"][0]
    client.examples[chosen]["inputs"]["messages"][0]["content"] = "changed later"
    client._snapshot()
    _, frozen, _ = reporting.verify_test_split(data, manifest, client=client)
    assert {str(ex.id) for ex in frozen} == set(receipt["splits"]["test"])
    assert next(ex for ex in frozen if str(ex.id) == chosen).inputs == original_inputs[chosen]
    with pytest.raises(PipelineError, match="differ from the prepared snapshot"):
        reporting.synchronize_splits(data, manifest, _load_json(data / "raw/examples.json"), client=client)


def test_reprepare_after_publishing_splits_preserves_snapshot_identity(prepared):
    data, manifest, client = prepared
    # A subsequent SDK export includes metadata added by split publication.
    raw = [example.model_dump(mode="json") for example in client.list_examples(dataset_id=DATASET)]
    raw.reverse()
    _json_dump(data / "raw/examples.json", raw)
    repeated = dataset.prepare_dataset(WORKSPACE, DATASET, DEFAULT_MODEL, data, fetch=False,
        inference_contract=empty_contract(), check_render=False, validation_fraction=.2, test_fraction=.3)
    assert repeated["langsmith"]["split_sync"] == manifest["langsmith"]["split_sync"]


def test_split_failure_resumes_and_rejected_membership_is_removed(prepared):
    data, manifest, client = prepared
    receipt = _load_json(data / "prepared/langsmith-splits.json")
    for name in reporting.SPLITS:
        for example_id in receipt["splits"][name]:
            client.examples[example_id]["metadata"]["dataset_split"] = []
    client._snapshot()
    client.fail_split = "test"
    examples = _load_json(data / "raw/examples.json")
    with pytest.raises(PipelineError, match="synchronization failed"):
        reporting.synchronize_splits(data, manifest, examples, client=client)
    assert _load_json(data / "prepared/langsmith-splits.json")["status"] == "syncing"
    client.fail_split = None
    reporting.synchronize_splits(data, manifest, examples, client=client)
    rows = _load_jsonl(data / "prepared/test.jsonl")
    removed = rows.pop()["_source"]["example_id"]
    _jsonl_dump(data / "prepared/test.jsonl", rows)
    manifest["split"]["test"] -= 1
    reporting.synchronize_splits(data, manifest, examples, client=client)
    assert "test" not in client.examples[removed]["metadata"]["dataset_split"]


def test_conflicting_membership_and_local_tampering_fail(prepared):
    data, manifest, client = prepared
    receipt = _load_json(data / "prepared/langsmith-splits.json")
    chosen = receipt["splits"]["test"][0]
    client.examples[chosen]["metadata"]["dataset_split"].append("train")
    client._snapshot()
    with pytest.raises(PipelineError, match="conflicting"):
        reporting.synchronize_splits(data, manifest, _load_json(data / "raw/examples.json"), client=client)
    rows = _load_jsonl(data / "prepared/test.jsonl")
    rows[0]["messages"][0]["content"] = "tampered"
    _jsonl_dump(data / "prepared/test.jsonl", rows)
    with pytest.raises(PipelineError, match="differs from its LangSmith split receipt"):
        reporting.verify_test_split(data, manifest, client=client)


def toy_chat(calls):
    def chat(model, messages, max_tokens, json_mode=False, request_contract=None):
        calls.append(model)
        if json_mode:
            evidence = json.loads(messages[-1]["content"])
            passed = evidence["reference_next_action"] == evidence["candidate_next_action"]
            return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "toy comparison"})}
        question = messages[-1]["content"]
        index = question.split()[-1]
        answer = f"answer {index}" if question.startswith("question") else f"second answer {index}"
        return {"role": "assistant", "content": answer if model == "tuned" else "incorrect"}
    return chat


def test_publication_attaches_child_and_root_feedback_and_resumes(prepared, tmp_path):
    data, manifest, client = prepared
    calls = []
    output = tmp_path / "eval"
    kwargs = dict(base_model="base", chat=toy_chat(calls), confirm=True)
    summary = evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert summary["tuned_pass_rate"] == 1
    assert summary["base_pass_rate"] == 0
    assert summary["langsmith"]["dataset_version"] == manifest["langsmith"]["split_sync"]["dataset_version"]
    experiments = _load_json(output / "langsmith-experiments.json")["experiments"]
    selected = parse_qs(urlsplit(summary["langsmith"]["comparison_url"]).query)["selectedSessions"][0]
    assert selected == experiments["base"]["id"] + "," + experiments["tuned"]["id"]
    assert "experiments" not in summary["langsmith"]
    for label, info in experiments.items():
        runs = list(client.list_runs(project_id=info["id"]))
        roots = [run for run in runs if run.parent_run_id is None]
        children = [run for run in runs if run.run_type == "llm"]
        assert len(roots) == manifest["split"]["test"]
        assert len(children) == len(roots) * 2
        assert all(len(run.outputs["steps"]) == 2 for run in roots)
        scores = list(client.list_feedback(run_ids=[run.id for run in children]))
        assert len(scores) == len(children)
        assert {item.key for item in scores} == {"teacher_agreement"}
        assert all(item.score == (label == "tuned") for item in scores)
        assert all(item.comment == "toy comparison" for item in scores)
        root_scores = list(client.list_feedback(run_ids=[run.id for run in roots]))
        assert len(root_scores) == len(roots)
        assert {item.key for item in root_scores} == {"trajectory_teacher_agreement"}
        assert all(item.score == (label == "tuned") for item in root_scores)
        assert all(run.start_time <= run.end_time for run in runs)
        assert all(run.reference_example_id is not None for run in roots)
    before = (len(calls), len(client.feedback_store), len(client.runs_store))
    resumed = evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert resumed == summary
    assert before == (len(calls), len(client.feedback_store), len(client.runs_store))


def test_partial_batch_failure_resumes_only_missing_runs_without_inference(prepared, tmp_path, monkeypatch):
    data, _, client = prepared
    calls, batches = [], []
    output = tmp_path / "eval"
    upload = client.batch_ingest_runs

    def interrupted(create=None, update=None):
        batches.append([str(run["id"]) for run in create])
        if len(batches) == 1:
            upload(create=create[:1])
            raise reporting.PublicationRequestError("LangSmith POST /runs/batch: HTTP 429; usage limit; Retry-After: 3600s")
        return upload(create=create)

    monkeypatch.setattr(client, "batch_ingest_runs", interrupted)
    kwargs = dict(base_model="base", chat=toy_chat(calls), confirm=True)
    with pytest.raises(PipelineError, match="base:run_upload.*POST /runs/batch.*429"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    receipt = _load_json(output / "langsmith-experiments.json")
    assert "Retry-After: 3600s" in receipt["last_error"]
    assert receipt["status"] == "interrupted"
    before = len(calls)
    summary = evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(calls) == before
    assert batches[0][0] not in {run_id for batch in batches[1:] for run_id in batch}
    assert summary["langsmith"]["comparison_url"]
    assert "last_error" not in _load_json(output / "langsmith-experiments.json")


def test_existing_conflicting_run_is_not_overwritten(prepared, tmp_path):
    data, _, client = prepared
    output = tmp_path / "eval"
    kwargs = dict(chat=toy_chat([]), confirm=True)
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    run = next(iter(client.runs_store.values()))
    run.outputs = {"changed": True}
    batches_before = len(client.run_batches)
    with pytest.raises(PipelineError, match="run differs"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(client.run_batches) == batches_before


def test_large_publication_resumes_after_indexing_lag_without_reupload(prepared, monkeypatch):
    _, _, client = prepared
    project = client.create_project("batch-test")
    now = datetime.now(UTC)
    values = []
    for _ in range(205):
        run_id = str(uuid4())
        values.append({"id": run_id, "name": "replay_trajectory", "run_type": "chain",
                       "project_name": project.name, "trace_id": run_id,
                       "inputs": {}, "outputs": {}, "start_time": now, "end_time": now,
                       "dotted_order": now.strftime("%Y%m%dT%H%M%S%fZ") + run_id,
                       "extra": {"metadata": {"smithtune_evaluation_id": "eval-test"}}})
    original = client.list_runs
    monkeypatch.setattr(client, "list_runs", lambda **kwargs: iter([]))
    with pytest.raises(PipelineError, match="not fully indexed"):
        reporting._ensure_runs(client, values, project.id)
    assert len(client.runs_store) == 100
    monkeypatch.setattr(client, "list_runs", original)
    reporting._ensure_runs(client, values, project.id)
    assert [len(batch) for batch in client.run_batches] == [100, 100, 5]
    assert len(client.runs_store) == 205


def test_feedback_upload_failure_does_not_repeat_inference_or_successful_feedback(prepared, tmp_path):
    data, _, client = prepared
    calls = []
    output = tmp_path / "eval"
    client.drop_feedback = "teacher_agreement"
    kwargs = dict(chat=toy_chat(calls), confirm=True)
    with pytest.raises(PipelineError, match="feedback"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert _load_json(output / "evaluation-state.json")["phase"] == "langsmith_publication"
    assert _load_jsonl(output / "results.jsonl")
    before = len(calls)
    successful = {(str(item.run_id), item.key) for item in client.feedback_store}
    client.drop_feedback = None
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(calls) == before
    for run_id, key in successful:
        assert sum(str(item.run_id) == run_id and item.key == key for item in client.feedback_store) == 1


def test_unsynchronized_data_fails_before_inference(prepared, tmp_path):
    data, manifest, _ = prepared
    manifest["langsmith"]["split_sync"] = {"status": "pending"}
    _json_dump(data / "prepared/manifest.json", manifest)
    with pytest.raises(PipelineError, match="not synchronized"):
        evaluation.run_replay_evaluation(data, tmp_path / "eval", "tuned", "judge", confirm=True,
            chat=lambda *_: pytest.fail("must validate LangSmith before inference"))


def test_rejected_replay_points_do_not_add_feedback_or_fail_agreement(prepared, tmp_path, monkeypatch):
    data, manifest, client = prepared

    def reject_followups(cases, *_args, **_kwargs):
        return ([{**case, "prompt_tokens": 8} for case in cases if case["message_index"] == 1],
                [{"id": case["id"], "example_id": case["example_id"], "reason": "context limit"}
                 for case in cases if case["message_index"] != 1])

    monkeypatch.setattr(evaluation, "validate_replay_context", reject_followups)
    summary = evaluation.run_replay_evaluation(data, tmp_path / "eval", "tuned", "judge", confirm=True, chat=toy_chat([]))
    assert summary["tuned_pass_rate"] == 1
    assert summary["rejected"] == manifest["split"]["test"]
    roots = list(client.list_runs(project_id=_load_json(tmp_path / "eval/langsmith-experiments.json")["experiments"]["tuned"]["id"], is_root=True))
    scores = list(client.list_feedback(run_ids=[root.id for root in roots]))
    assert len(scores) == len(roots)
    assert {score.key for score in scores} == {"trajectory_teacher_agreement"}
    assert all(score.score == 1 for score in scores)


def test_empty_trajectory_has_no_agreement_feedback():
    assert reporting._expected_feedback(str(uuid4()), []) == []


@pytest.mark.parametrize("passes,total", [(4, 12), (8, 17), (9, 17), (0, 3), (3, 3)])
def test_fractional_feedback_matches_sdk_storage_precision(passes, total):
    root = str(uuid4())
    steps = [{"run_id": str(uuid4()), "judgment": {"pass": index < passes, "reason": "judged"}}
             for index in range(total)]
    expected = reporting._expected_feedback(root, steps)
    # The SDK's create_feedback serializes float scores with round(score, 4).
    stored = SimpleNamespace(run_id=UUID(root), key="trajectory_teacher_agreement",
                             score=round(passes / total, 4), comment=None)
    client = SimpleNamespace(list_feedback=lambda **kwargs: iter([stored]))
    assert reporting._saved_feedback(client, expected) == {(root, stored.key)}
    assert expected[-1]["score"] == stored.score
    stored.score = round(stored.score + .0001, 4)
    with pytest.raises(PipelineError, match="conflicting LangSmith feedback"):
        reporting._saved_feedback(client, expected)


def test_changed_dataset_version_cannot_resume_an_existing_evaluation(prepared, tmp_path):
    data, manifest, client = prepared
    calls = []
    output = tmp_path / "eval"
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", confirm=True, chat=toy_chat(calls))
    before = len(calls)
    client._snapshot()
    manifest["langsmith"]["split_sync"] = reporting.synchronize_splits(data, manifest, _load_json(data / "raw/examples.json"), client=client)
    _json_dump(data / "prepared/manifest.json", manifest)
    with pytest.raises(PipelineError, match="different prepared snapshot"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", confirm=True, chat=toy_chat(calls))
    assert len(calls) == before


class ReplaySampler:
    checkpoint = "tuned"

    def __init__(self, provider, events):
        self.events = events
        self.config = {"provider": provider, "checkpoint": self.checkpoint,
                       "serving_mode": "sampler" if provider == "baseten" else "serverless"}

    def __enter__(self):
        self.events.append("open")
        return self.checkpoint

    def __exit__(self, *_):
        self.events.append("closed")

    def generate(self, *args):
        self.events.append("generate")
        return toy_chat([])(*args)


@pytest.mark.parametrize("prepared", ["fireworks", "baseten"], indirect=True)
def test_sampler_publication_failure_resumes_after_cleanup_without_inference(prepared, tmp_path, monkeypatch):
    data, manifest, client = prepared
    provider = dataset._model_from_manifest(manifest).provider
    events, calls = [], []
    sampler = ReplaySampler(provider, events)
    output = tmp_path / "replay"
    original_feedback = client.create_feedback

    def feedback(*args, **kwargs):
        assert events[-1] == "closed"
        return original_feedback(*args, **kwargs)

    monkeypatch.setattr(client, "create_feedback", feedback)
    client.drop_feedback = "teacher_agreement"
    kwargs = dict(base_model="base", chat=toy_chat(calls), replay_sampler=sampler, confirm=True)
    with pytest.raises(PipelineError, match="feedback"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert _load_json(output / "summary.json")["tuned_pass_rate"] == 1
    assert _load_json(output / "evaluation-state.json")["phase"] == "langsmith_publication"
    before = (list(events), len(calls))
    client.drop_feedback = None
    summary = evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert (events, len(calls)) == before
    assert summary["langsmith"]["comparison_url"]
    for label, info in _load_json(output / "langsmith-experiments.json")["experiments"].items():
        project = client.read_project(project_id=info["id"])
        assert project.metadata["generation_source"] == "sampler"
        assert project.metadata["serving_mode"] == sampler.config["serving_mode"]
        children = [run for run in client.list_runs(project_id=info["id"]) if run.parent_run_id]
        expected = {row["case"]["id"]: row for row in _load_jsonl(output / "generations.jsonl") if row["label"] == label}
        for child in children:
            recorded = expected[child.extra["metadata"]["case_id"]]
            assert child.start_time == datetime.fromisoformat(recorded["started_at"])
            assert child.end_time == datetime.fromisoformat(recorded["ended_at"])


def test_judge_retry_preserves_generation_timestamps_for_publication(prepared, tmp_path, monkeypatch):
    data, _, client = prepared
    events = []
    output = tmp_path / "replay"
    sampler = ReplaySampler("fireworks", events)
    original_judge = evaluation.judge_replay_candidate
    fail = True

    def judge(*args):
        if events and fail:
            raise PipelineError("judge unavailable")
        return original_judge(*args)

    monkeypatch.setattr(evaluation, "judge_replay_candidate", judge)
    kwargs = dict(chat=toy_chat([]), replay_sampler=sampler, confirm=True, concurrency=1)
    with pytest.raises(PipelineError, match="cases failed"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    generations = _load_jsonl(output / "generations.jsonl")
    before = list(events)
    fail = False
    summary = evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert events == before
    assert _load_jsonl(output / "generations.jsonl") == generations
    expected = {row["case"]["id"]: row for row in generations}
    info = _load_json(output / "langsmith-experiments.json")["experiments"]["tuned"]
    assert parse_qs(urlsplit(summary["langsmith"]["comparison_url"]).query)["selectedSessions"] == [info["id"]]
    for run in client.list_runs(project_id=info["id"]):
        if run.parent_run_id:
            recorded = expected[run.extra["metadata"]["case_id"]]
            assert run.start_time == datetime.fromisoformat(recorded["started_at"])
            assert run.end_time == datetime.fromisoformat(recorded["ended_at"])


def test_unsynchronized_splits_block_evaluation_before_sampling(prepared, tmp_path):
    data, manifest, client = prepared
    manifest["langsmith"]["split_sync"] = {"status": "pending"}
    _json_dump(data / "prepared/manifest.json", manifest)
    events, calls = [], []
    with pytest.raises(PipelineError, match="splits are not synchronized"):
        evaluation.run_replay_evaluation(
            data, tmp_path / "replay", "tuned", "judge", chat=toy_chat(calls),
            replay_sampler=ReplaySampler("fireworks", events), confirm=True,
        )
    assert events == calls == []
    assert client.projects == {}


@pytest.mark.parametrize("command", ["evaluate", "train", "plan"])
def test_cli_rejects_langsmith_opt_out(command, capsys):
    from smithtune import cli

    args = [command, "--no-langsmith"]
    if command != "evaluate":
        args.append("--evaluate")
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    assert "unrecognized arguments: --no-langsmith" in capsys.readouterr().err


@pytest.mark.parametrize("training", [{}, {"parent_training_run_id": "weather-sft", "checkpoint_epoch": 2}])
def test_experiments_record_available_training_origin(prepared, tmp_path, training):
    data, _, client = prepared
    output = tmp_path / "replay"
    kwargs = dict(base_model="base", chat=toy_chat([]), confirm=True, training=training)
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(client.projects) == 2
    for project in client.projects.values():
        for key in ("parent_training_run_id", "checkpoint_epoch"):
            if key in training:
                assert project.metadata[key] == training[key]
            else:
                assert key not in project.metadata
    if training:
        with pytest.raises(PipelineError, match="settings"):
            evaluation.run_replay_evaluation(
                data, output, "tuned", "judge", **{**kwargs, "training": {**training, "checkpoint_epoch": 3}},
            )



def test_lost_feedback_response_and_delayed_visibility_reuse_feedback_id(prepared, tmp_path, monkeypatch):
    data, _, client = prepared
    output, calls = tmp_path / "replay", []
    create = client.create_feedback
    read = client.list_feedback
    submitted = []
    visible = False

    def upload(*args, **kwargs):
        nonlocal visible
        submitted.append(kwargs["feedback_id"])
        result = create(*args, **kwargs)
        if len(submitted) == 1:
            raise ConnectionError("response lost after accepted feedback")
        visible = True
        return result

    monkeypatch.setattr(client, "create_feedback", upload)
    monkeypatch.setattr(client, "list_feedback", lambda **kwargs: read(**kwargs) if visible else iter([]))
    kwargs = dict(chat=toy_chat(calls), confirm=True)
    with pytest.raises(PipelineError, match="feedback_upload"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    before = len(calls)
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(calls) == before
    assert submitted[0] == submitted[1]
    assert len(client.feedback_store) == len(set(submitted))


def test_resume_older_publication_adds_training_origin_without_resampling(prepared, tmp_path):
    data, _, client = prepared
    output, calls = tmp_path / "replay", []
    kwargs = dict(base_model="base", chat=toy_chat(calls), confirm=True)
    client.drop_feedback = "teacher_agreement"
    with pytest.raises(PipelineError, match="feedback_upload"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    original_projects = set(client.projects)
    before = len(calls)
    client.drop_feedback = None
    training = {"parent_training_run_id": "weather-sft", "checkpoint_epoch": 2}
    summary = evaluation.run_replay_evaluation(data, output, "tuned", "judge", training=training, **kwargs)
    assert len(calls) == before
    assert original_projects <= client.projects.keys()
    assert len(client.projects) == 2
    assert len(parse_qs(urlsplit(summary["langsmith"]["comparison_url"]).query)["selectedSessions"][0].split(",")) == 2
    for project in client.projects.values():
        assert all(project.metadata[key] == value for key, value in training.items())


@pytest.mark.parametrize("host", ["https://smith.langchain.com", "https://eu.smith.langchain.com", "https://smith.example.com/langsmith"])
@pytest.mark.parametrize("include_base", [False, True])
def test_comparison_url_preserves_host_workspace_and_dataset(host, include_base):
    path = f"/o/{WORKSPACE}/datasets/{DATASET}/compare"
    experiments = {"tuned": {"id": "tuned-id", "url": f"{host}{path}?selectedSessions=tuned-id"}}
    if include_base:
        experiments["base"] = {"id": "base-id", "url": f"{host}{path}?selectedSessions=base-id"}
    selected = "base-id,tuned-id" if include_base else "tuned-id"
    assert reporting._comparison_url(experiments) == f"{host}{path}?selectedSessions={selected}"


@pytest.mark.parametrize("prepared", ["fireworks", "baseten"], indirect=True)
@pytest.mark.parametrize("compare_base", [False, True])
def test_experiment_names_use_prepared_model_for_both_roles(prepared, tmp_path, compare_base):
    data, manifest, client = prepared
    output = tmp_path / "replay"
    kwargs = dict(chat=toy_chat([]), confirm=True)
    if compare_base:
        kwargs["base_model"] = "base"
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    receipt = _load_json(output / "langsmith-experiments.json")
    short_model = manifest["model"]["base_model"].rsplit("/", 1)[-1]
    for project in client.projects.values():
        role = project.metadata["model_role"]
        assert project.name == f"smithtune-{role}-{short_model}-{receipt['evaluation_id']}"
        assert project.metadata["model"] == role


def test_legacy_experiment_names_survive_interrupted_publication(prepared, tmp_path, monkeypatch):
    data, _, client = prepared
    output, calls = tmp_path / "replay", []
    verify = reporting.verify_test_split

    def legacy_context(*args, **kwargs):
        client, examples, splits = verify(*args, **kwargs)
        return client, examples, {**splits, "base_model": ""}

    monkeypatch.setattr(reporting, "verify_test_split", legacy_context)
    client.drop_feedback = "teacher_agreement"
    kwargs = dict(base_model="base", chat=toy_chat(calls), confirm=True)
    with pytest.raises(PipelineError, match="feedback_upload"):
        evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    path = output / "langsmith-experiments.json"
    receipt = _load_json(path)
    receipt.pop("model_name")
    _json_dump(path, receipt)
    original_projects, before = set(client.projects), len(calls)
    monkeypatch.setattr(reporting, "verify_test_split", verify)
    client.drop_feedback = None
    evaluation.run_replay_evaluation(data, output, "tuned", "judge", **kwargs)
    assert len(calls) == before
    assert original_projects <= client.projects.keys()
    assert len(client.projects) == 2
    for project in client.projects.values():
        role = project.metadata["model_role"]
        assert project.name == f"smithtune-{role}-{receipt['evaluation_id']}"

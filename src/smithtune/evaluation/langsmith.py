"""Versioned dataset splits and resumable LangSmith replay experiments.

Inference and judging are checkpointed by replay.py. Publication uses those
saved predictions so a reporting failure never requires another paid model call.
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from contextlib import contextmanager
from threading import Condition, Event, Thread
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4, uuid5

from langsmith.utils import LangSmithNotFoundError

from smithtune.evaluation.langsmith_client import PublicationRequestError, PublishingClient
from smithtune.artifacts import _json_dump, _load_json, _load_jsonl, _utc_now, output_lock
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError


SPLITS = ("train", "validation", "test")
SPLIT_RECEIPT = "langsmith-splits.json"


def make_client(workspace_id: str) -> PublishingClient:
    """Use environment credentials, explicitly scoped to the dataset workspace."""
    return PublishingClient(workspace_id=workspace_id)


def _example_hash(example) -> str:
    value = example if isinstance(example, dict) else example.model_dump(mode="json")
    metadata = {key: value for key, value in (value.get("metadata") or {}).items() if key != "dataset_split"}
    return json_sha256({"inputs": value["inputs"], "outputs": value.get("outputs"), "metadata": metadata})


def split_snapshot(data_dir: Path, manifest: dict, examples: list[dict]) -> dict:
    rows = {name: _load_jsonl(data_dir / "prepared" / f"{name}.jsonl") for name in SPLITS}
    splits = {name: sorted(row["_source"]["example_id"] for row in values) for name, values in rows.items()}
    ids = [example_id for members in splits.values() for example_id in members]
    if len(ids) != len(set(ids)):
        raise PipelineError("prepared LangSmith splits contain duplicate example IDs")
    hashes = {str(example["id"]): _example_hash(example) for example in examples}
    if not set(ids) <= hashes.keys():
        raise PipelineError("prepared split refers to an example missing from the raw export")
    for name in SPLITS:
        if len(splits[name]) != manifest["split"][name]:
            raise PipelineError(f"prepared {name} split differs from its manifest")
        # Publishing splits adds this derived metadata to future exports. It
        # must not change the identity of the messages that were prepared.
        for row in rows[name]:
            row["_source"].get("metadata", {}).pop("dataset_split", None)
            row["_source"].pop("source_index", None)
        rows[name].sort(key=lambda row: row["_source"]["example_id"])
    identity = {
        "workspace_id": manifest["langsmith"]["workspace_id"],
        "dataset_id": manifest["langsmith"]["dataset_id"],
        "assignments_sha256": manifest["split"].get("assignments_sha256"),
        "prepared_sha256": json_sha256(rows),
        "example_sha256": hashes,
        "splits": splits,
    }
    return {**identity, "identity_sha256": json_sha256(identity)}


def _check_examples(examples, expected: dict[str, str]) -> None:
    actual = {str(example.id): _example_hash(example) for example in examples}
    if actual != expected or len(examples) != len(expected):
        changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
        raise PipelineError(f"LangSmith examples differ from the prepared snapshot: {', '.join(changed[:10])}; prepare again")


def synchronize_splits(data_dir: Path, manifest: dict, examples: list[dict], *, client=None, enabled=True) -> dict:
    """Publish exact accepted membership without overwriting foreign assignments."""
    desired = split_snapshot(data_dir, manifest, examples)
    path = data_dir / "prepared" / SPLIT_RECEIPT
    previous = _load_json(path) if path.exists() else {}
    if previous and any(previous.get(key) != desired[key] for key in ("workspace_id", "dataset_id")):
        raise PipelineError(f"LangSmith split receipt belongs to another dataset: {path}")
    if not enabled:
        if previous.get("status") == "complete" and previous.get("identity_sha256") == desired["identity_sha256"]:
            return {"status": "complete", "dataset_version": previous["dataset_version"],
                    "identity_sha256": desired["identity_sha256"]}
        return {"status": "pending"}
    client = client or make_client(desired["workspace_id"])
    dataset_id = desired["dataset_id"]
    try:
        _check_examples(list(client.list_examples(dataset_id=dataset_id)), desired["example_sha256"])
        actual = {name: {str(ex.id) for ex in client.list_examples(dataset_id=dataset_id, splits=[name])} for name in SPLITS}
        owned = previous.get("managed_splits", previous.get("splits", {}))
        for name in SPLITS:
            wanted = set(desired["splits"][name])
            foreign = actual[name] - wanted - set(owned.get(name, []))
            conflicting = wanted & set().union(*(actual[other] for other in SPLITS if other != name))
            if foreign or conflicting:
                raise PipelineError(f"conflicting existing LangSmith {name} membership: {', '.join(sorted(foreign | conflicting)[:10])}")
        receipt = {**desired, "status": "syncing", "managed_splits": {
            name: sorted(set(owned.get(name, [])) | set(desired["splits"][name])) for name in SPLITS
        }}
        # Save ownership before writes; additions/removals are safe to repeat.
        _json_dump(path, receipt)
        for name in SPLITS:
            wanted = set(desired["splits"][name])
            for remove, ids in ((True, actual[name] - wanted), (False, wanted - actual[name])):
                ordered = sorted(ids)
                for offset in range(0, len(ordered), 100):
                    client.update_dataset_splits(dataset_id=dataset_id, split_name=name,
                                                 example_ids=ordered[offset:offset + 100], remove=remove)
        version = client.read_dataset_version(dataset_id=dataset_id, tag="latest").as_of.isoformat()
        frozen = list(client.list_examples(dataset_id=dataset_id, as_of=version))
        _check_examples(frozen, desired["example_sha256"])
        for name in SPLITS:
            members = list(client.list_examples(dataset_id=dataset_id, splits=[name], as_of=version))
            if sorted(str(ex.id) for ex in members) != desired["splits"][name]:
                raise PipelineError(f"LangSmith {name} split verification failed; rerun prepare to resume: {path}")
        receipt.update(status="complete", dataset_version=version, managed_splits=desired["splits"])
        _json_dump(path, receipt)
        return {"status": "complete", "dataset_version": version, "identity_sha256": desired["identity_sha256"]}
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"LangSmith split synchronization failed ({type(exc).__name__}); local data is saved; rerun prepare: {path}") from None


def publish_prepared_splits(data_dir: Path) -> dict:
    """Publish and verify existing prepared split artifacts without preparing again."""
    with output_lock(data_dir):
        manifest_path = data_dir / "prepared" / "manifest.json"
        manifest = _load_json(manifest_path)
        examples = _load_json(data_dir / "raw" / "examples.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("langsmith"), dict):
            raise PipelineError(f"prepared manifest has no LangSmith dataset identity: {manifest_path}")
        if not isinstance(manifest.get("split"), dict):
            raise PipelineError(f"prepared manifest has no split definition: {manifest_path}")
        if not isinstance(examples, list):
            raise PipelineError("raw examples artifact must be an array")
        synchronized = synchronize_splits(data_dir, manifest, examples)
        manifest["langsmith"]["split_sync"] = synchronized
        _json_dump(manifest_path, manifest)
        return {
            "status": synchronized["status"],
            "data_dir": str(data_dir.resolve()),
            "langsmith": synchronized,
            "split": manifest["split"],
        }


def verify_test_split(data_dir: Path, manifest: dict, *, client=None):
    """Resolve the registered test cohort and verify it before paid inference."""
    path = data_dir / "prepared" / SPLIT_RECEIPT
    if not path.exists() or manifest.get("langsmith", {}).get("split_sync", {}).get("status") != "complete":
        raise PipelineError("LangSmith splits are not synchronized; run prepare (or prepare --no-fetch) before evaluation")
    receipt = _load_json(path)
    desired = split_snapshot(data_dir, manifest, _load_json(data_dir / "raw" / "examples.json"))
    if (receipt.get("status") != "complete" or receipt.get("identity_sha256") != desired["identity_sha256"]
            or any(receipt.get(key) != value for key, value in desired.items())
            or manifest["langsmith"]["split_sync"].get("dataset_version") != receipt.get("dataset_version")):
        raise PipelineError(f"prepared data differs from its LangSmith split receipt: {path}")
    client = client or make_client(receipt["workspace_id"])
    try:
        examples = list(client.list_examples(dataset_id=receipt["dataset_id"], splits=["test"], as_of=receipt["dataset_version"]))
        _check_examples(examples, {key: receipt["example_sha256"][key] for key in receipt["splits"]["test"]})
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"cannot verify the registered LangSmith test split ({type(exc).__name__}): {path}") from None
    return client, examples, {**receipt, "base_model": manifest["model"]["base_model"]}


def bind_evaluation_snapshot(output_dir: Path, config: dict, cases: list[dict], context) -> str:
    """Pin the dataset version before inference, including interrupted evaluations."""
    splits = context[2]
    # Training lineage is descriptive metadata, not prediction identity.
    prediction_config = {key: value for key, value in config.items() if key != "training"}
    value = {"config": prediction_config, "cases_sha256": json_sha256(sorted(cases, key=lambda case: case["id"])), "split_identity": splits["identity_sha256"],
             "dataset_id": splits["dataset_id"], "dataset_version": splits["dataset_version"]}
    path = output_dir / "langsmith-evaluation-input.json"
    if path.exists() and _load_json(path) != value:
        raise PipelineError(f"LangSmith evaluation uses a different prepared snapshot; use a new output directory: {path}")
    _json_dump(path, value)
    return json_sha256(value)


def _feedback(run_id: str, key: str, score, comment=None) -> dict:
    # Match the SDK's create_feedback precision for uploads and readback checks.
    if isinstance(score, float):
        score = round(score, 4)
    result = {"key": key, "score": score, "target_run_id": run_id}
    if comment is not None:
        result["comment"] = comment
    return result


def _expected_feedback(root_id: str, steps: list[dict]) -> list[dict]:
    feedback = []
    for step in steps:
        feedback.append(_feedback(step["run_id"], "teacher_agreement", int(step["judgment"]["pass"]), step["judgment"]["reason"]))
    if steps:
        feedback.append(_feedback(root_id, "trajectory_teacher_agreement", sum(step["judgment"]["pass"] for step in steps) / len(steps)))
    return feedback


def _saved_feedback(client, expected: list[dict], mutable_runs=()) -> set[tuple[str, str]]:
    """Read back scores; a successful SDK return alone does not prove upload."""
    wanted = {(item["target_run_id"], item["key"]): item for item in expected}
    found = set()
    ids = sorted({key[0] for key in wanted})
    for offset in range(0, len(ids), 100):
        for feedback in client.list_feedback(run_ids=ids[offset:offset + 100]):
            key = (str(feedback.run_id), feedback.key)
            if key in wanted:
                item = wanted[key]
                if feedback.score != item["score"] or (item.get("comment") is not None and feedback.comment != item["comment"]):
                    if key[0] in mutable_runs and str(feedback.id) == str(uuid5(UUID(key[0]), key[1])):
                        continue
                    raise PipelineError(f"conflicting LangSmith feedback for run {key[0]}, key {key[1]}")
                found.add(key)
    return found


def _check_run(run, value: dict, project_id) -> None:
    if (run.inputs != value["inputs"] or run.outputs != value["outputs"]
            or str(run.session_id) != str(project_id)
            or str(run.trace_id) != value["trace_id"]
            or str(run.parent_run_id or "") != str(value.get("parent_run_id") or "")
            or run.extra.get("metadata", {}).get("smithtune_evaluation_id") != value["extra"]["metadata"]["smithtune_evaluation_id"]):
        raise PipelineError(f"LangSmith run differs from saved replay: {value['id']}")


def _wait_for_indexing(ready, label: str) -> None:
    if ready():
        return
    # Ingestion acknowledgement can precede query visibility. Check without
    # resubmitting accepted writes, allowing one minute of backoff in total.
    for delay in (1, 2, 4, 8, 15, 30):
        print(f"Waiting for LangSmith {label} to become searchable; checking again in {delay}s.", file=sys.stderr)
        time.sleep(delay)
        if ready():
            return
    raise PipelineError(f"LangSmith {label} are not fully indexed after 60s of waiting; "
                        "saved results are preserved; rerun evaluate to resume publication")


def _ensure_runs(client, values: list[dict], project_id, check_cancelled=lambda: None) -> None:
    for offset in range(0, len(values), 100):
        check_cancelled()
        batch = values[offset:offset + 100]
        expected = {value["id"]: value for value in batch}

        def read(expected=expected, batch=batch):
            check_cancelled()
            found = {}
            for run in client.list_runs(project_id=project_id, run_ids=list(expected),
                                        start_time=min(value["start_time"] for value in batch),
                                        limit=len(batch)):
                run_id = str(run.id)
                if run_id not in expected:
                    raise PipelineError("LangSmith returned a run outside the requested publication batch")
                _check_run(run, expected[run_id], project_id)
                found[run_id] = run
            return found

        found = read()
        missing = [value for value in batch if value["id"] not in found]
        if missing:
            check_cancelled()
            client.batch_ingest_runs(create=[
                {**{key: value for key, value in run.items() if key != "project_name"}, "session_id": project_id}
                for run in missing
            ])
            _wait_for_indexing(lambda read=read, expected=expected: len(read()) == len(expected), "runs")


def _comparison_url(experiments: dict) -> str:
    selected = [experiments[label] for label in ("base", "tuned") if label in experiments]
    if not selected or not selected[0].get("url"):
        raise PipelineError("LangSmith did not return an experiment URL; rerun evaluate to resume")
    parts = urlsplit(selected[0]["url"])
    query = dict(parse_qsl(parts.query))
    query["selectedSessions"] = ",".join(item["id"] for item in selected)
    return urlunsplit(parts._replace(query=urlencode(query, safe=",")))


class EvaluationPublisher:
    """One serialized writer for evolving conversation runs and immutable actions."""

    def __init__(self, output_dir: Path, config: dict, cases: list[dict], context, results=()):
        self.output_dir, self.config, self.cases = output_dir, config, cases
        self.client, self.examples, self.splits = context
        self.path = output_dir / "langsmith-experiments.json"
        identity = bind_evaluation_snapshot(output_dir, config, cases, context)
        self.receipt = _load_json(self.path) if self.path.exists() else {
            "schema_version": 1, "evaluation_id": str(uuid4()), "created_at": min((row[label].get("started_at", _utc_now())
                for row in results for label in config["models"]), default=_utc_now()),
            "identity_sha256": identity, "experiments": {},
            "model_name": self.splits["base_model"].rstrip("/").rsplit("/", 1)[-1],
        }
        if self.receipt.get("identity_sha256") != identity:
            raise PipelineError(f"LangSmith experiments use different evaluation settings: {self.path}")
        self.cancelled = Event()
        self.verified_children = set()
        self.published = {}
        self.projects = {}
        self.phase = "experiment_creation"
        self.report = None

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise PipelineError("LangSmith publication stopped; rerun evaluate to resume saved results")

    def _save(self):
        self._check_cancelled()
        _json_dump(self.path, self.receipt)

    @contextmanager
    def _operation(self):
        try:
            self._check_cancelled()
            yield
        except Exception as exc:
            self.receipt.update(status="interrupted", failure_phase=self.phase, error_type=type(exc).__name__)
            if isinstance(exc, PublicationRequestError):
                self.receipt["last_error"] = str(exc)
            if not self.cancelled.is_set():
                self._save()
            if isinstance(exc, PipelineError) and not isinstance(exc, PublicationRequestError):
                raise
            detail = str(exc) if isinstance(exc, PublicationRequestError) else type(exc).__name__
            raise PipelineError(f"LangSmith experiment publication failed during {self.phase}: {detail}; "
                                f"replay results are saved; rerun evaluate to resume publication: {self.path}") from None

    def _metadata(self, label):
        config, splits = self.config, self.splits
        namespace = self.receipt["evaluation_id"]
        model = config["models"][label]
        metadata = {**config.get("training", {}), "smithtune_evaluation_id": str(namespace), "model": model, "model_role": label,
                    "judge_model": config["judge_model"], "evaluation_mode": "recorded_prefix",
                    "serving_mode": config["serving_mode"],
                    "generation_source": "sampler" if config.get("sampler") else "endpoint",
                    "provider": config.get("sampler", {}).get("provider", "baseten" if config.get("baseten_endpoint") else "fireworks"),
                    "dataset_version": splits["dataset_version"], "dataset_splits": ["test"],
                    "smithtune_dataset_version": splits["dataset_version"],
                    "prepared_identity": splits["identity_sha256"], "cached_predictions": True}
        return metadata

    def start(self):
        """Create every experiment before any generation or publication starts."""
        with self._operation():
            self.receipt["status"] = "publishing"
            self._save()
            model_suffix = f"-{self.receipt['model_name']}" if self.receipt.get("model_name") else ""
            for label in self.config["models"]:
                self.phase = f"{label}:experiment_creation"
                name = f"smithtune-{label}{model_suffix}-{self.receipt['evaluation_id']}"
                metadata = self._metadata(label)
                try:
                    project = self.client.read_project(project_name=name)
                except LangSmithNotFoundError:
                    self._check_cancelled()
                    project = self.client.create_project(name, reference_dataset_id=self.splits["dataset_id"], metadata=metadata)
                if (str(project.reference_dataset_id) != self.splits["dataset_id"]
                        or any(project.metadata.get(key) != value for key, value in metadata.items()
                               if key not in ("dataset_version", "dataset_splits")
                               and not (key in self.config.get("training", {}) and key not in project.metadata))):
                    raise PipelineError(f"LangSmith experiment ownership/settings conflict: {name}")
                self._check_cancelled()
                self.client.update_project(project.id, metadata=metadata)
                self.projects[label] = project
                url = project.url
                if url:
                    url = url.replace(f"/projects/p/{project.id}", f"/datasets/{self.splits['dataset_id']}/compare?selectedSessions={project.id}")
                self.receipt["experiments"][label] = {"id": str(project.id), "name": name, "url": url}
                self._save()
            self.receipt["comparison_url"] = _comparison_url(self.receipt["experiments"])
            self._save()
            self.report = {"dataset_id": self.splits["dataset_id"], "dataset_version": self.splits["dataset_version"],
                           "split": "test", "comparison_url": self.receipt["comparison_url"]}
        return self.report

    def _runs(self, label, example, results):
        namespace = UUID(self.receipt["evaluation_id"])
        started = datetime.fromisoformat(self.receipt["created_at"])
        example_id = str(example.id)
        total = sum(case["example_id"] == example_id for case in self.cases)
        metadata, name = self._metadata(label), self.projects[label].name
        root_id = str(uuid5(namespace, f"{label}:{example_id}"))
        root_order = started.strftime("%Y%m%dT%H%M%S%fZ") + root_id
        steps, children = [], []
        for result in sorted(results, key=lambda row: row["case"]["message_index"]):
            case, value = result["case"], result[label]
            child_id = str(uuid5(UUID(root_id), case["id"]))
            step = {"case_id": case["id"], "message_index": case["message_index"], "run_id": child_id,
                    "candidate": value["candidate"], "judgment": value["judgment"],
                    "deterministic_metrics": value["deterministic_metrics"]}
            steps.append(step)
            begin = datetime.fromisoformat(value.get("started_at", self.receipt["created_at"]))
            end = datetime.fromisoformat(value.get("ended_at", value.get("started_at", self.receipt["created_at"])))
            children.append({"id": child_id, "name": "replay_assistant", "run_type": "llm",
                "inputs": {"messages": case["messages"], "tools": case["tools"]},
                "outputs": {"message": value["candidate"]}, "start_time": begin, "end_time": end,
                "parent_run_id": root_id, "trace_id": root_id,
                "dotted_order": root_order + "." + begin.strftime("%Y%m%dT%H%M%S%fZ") + child_id,
                "project_name": name, "extra": {"metadata": {**metadata, "case_id": case["id"],
                    "message_index": case["message_index"], "serving_route": value["serving_route"]}}})
        root = {"id": root_id, "name": "replay_trajectory", "run_type": "chain", "inputs": example.inputs,
                "outputs": {"steps": steps, "completed_actions": len(steps), "total_actions": total,
                            "status": "complete" if len(steps) == total else "partial"}, "reference_example_id": example.id, "trace_id": root_id,
                "dotted_order": root_order, "project_name": name, "start_time": started,
                "end_time": max((child["end_time"] for child in children), default=started),
                "extra": {"metadata": metadata}}
        return root, children, steps

    def _ensure_root(self, root, project_id):
        self._check_cancelled()
        found = list(self.client.list_runs(project_id=project_id, run_ids=[root["id"]],
                                          start_time=root["start_time"], limit=1))
        if not found:
            _ensure_runs(self.client, [root], project_id, self._check_cancelled)
            return
        run = found[0]
        _check_run(run, {**root, "outputs": run.outputs}, project_id)
        wanted = {step["case_id"]: step for step in root["outputs"]["steps"]}
        previous = (run.outputs or {}).get("steps")
        if (not isinstance(previous, list) or any(wanted.get(step.get("case_id")) != step for step in previous)
                or len({step["case_id"] for step in previous}) != len(previous)):
            raise PipelineError(f"LangSmith run differs from saved replay: {root['id']}")
        if run.outputs != root["outputs"]:
            self._check_cancelled()
            self.client.update_run(root["id"], outputs=root["outputs"], end_time=root["end_time"],
                                   trace_id=root["trace_id"], dotted_order=root["dotted_order"])

            def visible():
                self._check_cancelled()
                runs = list(self.client.list_runs(project_id=project_id, run_ids=[root["id"]],
                                                   start_time=root["start_time"], limit=1))
                return bool(runs and runs[0].outputs == root["outputs"])

            _wait_for_indexing(visible, "conversation results")

    def _publish_feedback(self, root, children, steps, project_id):
        expected = [item for item in _expected_feedback(root["id"], steps)
                    if item["target_run_id"] not in self.verified_children]
        times = {value["id"]: value["start_time"] for value in [root, *children]}
        mutable = {root["id"]}
        found = _saved_feedback(self.client, expected, mutable)
        root_scores = list(self.client.list_feedback(run_ids=[root["id"]]))
        for item in expected:
            self._check_cancelled()
            target, key = item["target_run_id"], item["key"]
            if (target, key) in found:
                continue
            feedback_id = uuid5(UUID(target), key)
            if target == root["id"] and any(str(score.id) == str(feedback_id) for score in root_scores):
                self.client.update_feedback(feedback_id, score=item["score"])
            else:
                self.client.create_feedback(
                    run_id=target, key=key, score=item["score"], comment=item.get("comment"),
                    feedback_id=feedback_id, feedback_source_type="model",
                    trace_id=root["id"], session_id=project_id, start_time=times[target],
                    source_info={"judge_model": self.config["judge_model"], "cached_judgment": True},
                )
        self.client.flush()
        self.phase = self.phase.replace("feedback_upload", "feedback_verification")

        def visible():
            self._check_cancelled()
            return len(_saved_feedback(self.client, expected, mutable)) == len(expected)

        _wait_for_indexing(visible, "feedback scores")

    def publish(self, results):
        """Publish a cumulative snapshot; skip unchanged conversations in this session."""
        with self._operation():
            by_example = {str(example.id): [] for example in self.examples}
            for result in sorted(results, key=lambda item: item["case"]["id"]):
                by_example[result["case"]["example_id"]].append(result)
            for label, project in self.projects.items():
                for example in self.examples:
                    self._check_cancelled()
                    rows = by_example[str(example.id)]
                    if not rows:
                        continue
                    fingerprint = json_sha256(rows)
                    key = (label, str(example.id))
                    if self.published.get(key) == fingerprint:
                        continue
                    root, children, steps = self._runs(label, example, rows)
                    self.phase = f"{label}:run_upload"
                    self._ensure_root(root, project.id)
                    missing = [child for child in children if child["id"] not in self.verified_children]
                    _ensure_runs(self.client, missing, project.id, self._check_cancelled)
                    self.phase = f"{label}:feedback_upload"
                    # Child predictions and feedback are immutable once verified.
                    self._publish_feedback(root, children, steps, project.id)
                    self.verified_children.update(child["id"] for child in missing)
                    self.published[key] = fingerprint
            self.receipt.update(status="complete" if len(results) == len(self.cases) else "partial",
                                completed=len(results), total=len(self.cases))
            for field in ("failure_phase", "error_type", "last_error"):
                self.receipt.pop(field, None)
            self._save()
        return self.report


def publish_evaluation(output_dir: Path, config: dict, results: list[dict], cases: list[dict], context) -> dict:
    """Publish saved results without making model calls, including legacy callers."""
    with output_lock(output_dir / "publication"):
        publisher = EvaluationPublisher(output_dir, config, cases, context, results)
        publisher.start()
        return publisher.publish(results)


class BackgroundPublisher:
    """Coalesce saved results into small batches, independently of model workers.

    Only this worker writes publication receipts. Its lock survives a timed-out
    close until the in-flight SDK request returns, preventing concurrent resume.
    """

    batch_size = 10
    interval = 5.0
    close_timeout = 75.0

    def __init__(self, output_dir, config, cases, context, results):
        self.condition = Condition()
        self.pending = deque()
        self.initial = copy.deepcopy(results)
        self.cancelled = Event()
        self.closing = False
        self.error = None
        self.ready = Event()
        self.publisher = None
        self.report = None
        self.args = (output_dir, config, cases, context, self.initial)
        self.closed = False
        self.thread = Thread(target=self._run, name="langsmith-publisher", daemon=True)
        self.thread.start()
        # Experiment setup is a preflight requirement, before paid work.
        try:
            self.ready.wait()
        except BaseException:
            self.cancelled.set()
            with self.condition:
                self.closing = True
                self.condition.notify()
            raise
        if self.error:
            raise self.error
        print(f"LangSmith comparison: {self.report['comparison_url']}", file=sys.stderr)

    def submit(self, result):
        with self.condition:
            if self.error is None:
                self.pending.append(copy.deepcopy(result))
                self.condition.notify()

    def _run(self):
        try:
            with output_lock(self.args[0] / "publication"):
                self.publisher = EvaluationPublisher(*self.args)
                self.publisher.cancelled = self.cancelled
                self.report = self.publisher.start()
                self.ready.set()
                # Reconcile the entire saved snapshot first on resume: a remote
                # conversation may already contain more than one batch of actions.
                published = {row["case"]["id"]: row for row in self.initial}
                if published:
                    self.publisher.publish(list(published.values()))
                last_upload = 0.0
                while True:
                    with self.condition:
                        while True:
                            delay = self.interval - (time.monotonic() - last_upload)
                            if self.closing or (self.pending and (len(self.pending) >= self.batch_size or delay <= 0)):
                                break
                            self.condition.wait(timeout=max(delay, .01) if self.pending else None)
                        if not self.pending and self.closing:
                            # Also writes an honest empty/partial receipt on cancellation.
                            self.publisher.publish(list(published.values()))
                            return
                        batch = [self.pending.popleft() for _ in range(min(len(self.pending), self.batch_size))]
                    snapshot = {**published, **{row["case"]["id"]: row for row in batch}}
                    self.publisher.publish(list(snapshot.values()))
                    published = snapshot
                    last_upload = time.monotonic()
        except Exception as exc:
            self.error = exc
            print(f"LangSmith publication paused: {exc}", file=sys.stderr)
        finally:
            self.ready.set()

    def close(self):
        if self.closed:
            return self.report
        self.closed = True
        with self.condition:
            self.closing = True
            self.condition.notify()
        try:
            self.thread.join(timeout=self.close_timeout)
        except BaseException:
            self.cancelled.set()
            raise
        if self.thread.is_alive():
            self.cancelled.set()
            raise PipelineError("LangSmith publication is still pending; saved results are preserved. "
                                "Rerun evaluate with the same settings to resume publication.")
        if self.error:
            raise self.error
        return self.report

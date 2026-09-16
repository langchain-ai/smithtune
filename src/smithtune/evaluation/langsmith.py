"""Versioned dataset splits and resumable LangSmith replay experiments.

Inference and judging are checkpointed by replay.py. Publication uses those
saved predictions so a reporting failure never requires another paid model call.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4, uuid5

from langsmith.utils import LangSmithNotFoundError

from smithtune.evaluation.langsmith_client import PublicationRequestError, PublishingClient
from smithtune.artifacts import _json_dump, _load_json, _load_jsonl, _utc_now
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


def _saved_feedback(client, expected: list[dict]) -> set[tuple[str, str]]:
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


def _ensure_runs(client, values: list[dict], project_id) -> None:
    for offset in range(0, len(values), 100):
        batch = values[offset:offset + 100]
        expected = {value["id"]: value for value in batch}

        def read(expected=expected, batch=batch):
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
            client.batch_ingest_runs(create=[
                {**{key: value for key, value in run.items() if key != "project_name"}, "session_id": project_id}
                for run in missing
            ])
            for attempt in range(5):
                if len(read()) == len(expected):
                    break
                if attempt < 4:
                    time.sleep(1)
            else:
                raise PipelineError("LangSmith runs are not fully indexed; rerun evaluate to resume publication")


def _comparison_url(experiments: dict) -> str:
    selected = [experiments[label] for label in ("base", "tuned") if label in experiments]
    if not selected or not selected[0].get("url"):
        raise PipelineError("LangSmith did not return an experiment URL; rerun evaluate to resume")
    parts = urlsplit(selected[0]["url"])
    query = dict(parse_qsl(parts.query))
    query["selectedSessions"] = ",".join(item["id"] for item in selected)
    return urlunsplit(parts._replace(query=urlencode(query, safe=",")))


def publish_evaluation(output_dir: Path, config: dict, results: list[dict], cases: list[dict], context) -> dict:
    """Publish saved predictions and judgments with stable run and feedback IDs."""
    client, examples, splits = context
    path = output_dir / "langsmith-experiments.json"
    identity = bind_evaluation_snapshot(output_dir, config, cases, context)
    inference_start = min((result[label].get("started_at", _utc_now()) for result in results for label in config["models"]), default=_utc_now())
    receipt = _load_json(path) if path.exists() else {
        "schema_version": 1, "evaluation_id": str(uuid4()), "created_at": inference_start,
        "identity_sha256": identity, "experiments": {},
        "model_name": splits["base_model"].rstrip("/").rsplit("/", 1)[-1],
    }
    if receipt.get("identity_sha256") != identity:
        raise PipelineError(f"LangSmith experiments use different evaluation settings: {path}")
    receipt["status"] = "publishing"
    _json_dump(path, receipt)
    namespace = UUID(receipt["evaluation_id"])
    # Receipts without a model name retain their original naming on retries.
    model_suffix = f"-{receipt['model_name']}" if receipt.get("model_name") else ""
    started = datetime.fromisoformat(receipt["created_at"])
    by_example = {str(example.id): [] for example in examples}
    for result in results:
        by_example[result["case"]["example_id"]].append(result)
    phase = "experiment_creation"
    try:
        for label, model in config["models"].items():
            phase = f"{label}:experiment_creation"
            name = f"smithtune-{label}{model_suffix}-{namespace}"
            metadata = {**config.get("training", {}), "smithtune_evaluation_id": str(namespace), "model": model, "model_role": label,
                        "judge_model": config["judge_model"], "evaluation_mode": "recorded_prefix",
                        "serving_mode": config["serving_mode"],
                        "generation_source": "sampler" if config.get("sampler") else "endpoint",
                        "provider": config.get("sampler", {}).get("provider", "baseten" if config.get("baseten_endpoint") else "fireworks"),
                        "dataset_version": splits["dataset_version"], "dataset_splits": ["test"],
                        "smithtune_dataset_version": splits["dataset_version"],
                        "prepared_identity": splits["identity_sha256"], "cached_predictions": True}
            try:
                project = client.read_project(project_name=name)
            except LangSmithNotFoundError:
                project = client.create_project(name, reference_dataset_id=splits["dataset_id"], metadata=metadata)
            if (str(project.reference_dataset_id) != splits["dataset_id"]
                    or any(project.metadata.get(key) != value for key, value in metadata.items()
                           if key not in ("dataset_version", "dataset_splits")
                           and not (key in config.get("training", {}) and key not in project.metadata))):
                raise PipelineError(f"LangSmith experiment ownership/settings conflict: {name}")
            client.update_project(project.id, metadata=metadata)
            url = project.url
            if url:
                url = url.replace(f"/projects/p/{project.id}", f"/datasets/{splits['dataset_id']}/compare?selectedSessions={project.id}")
            receipt["experiments"][label] = {"id": str(project.id), "name": name, "url": url}
            _json_dump(path, receipt)
            expected_by_root = {}
            feedback_times = {}
            phase = f"{label}:run_upload"
            for example in examples:
                example_id = str(example.id)
                root_id = str(uuid5(namespace, f"{label}:{example_id}"))
                root_order = started.strftime("%Y%m%dT%H%M%S%fZ") + root_id
                steps, children = [], []
                for result in by_example[example_id]:
                    case, value = result["case"], result[label]
                    child_id = str(uuid5(UUID(root_id), case["id"]))
                    step = {"case_id": case["id"], "message_index": case["message_index"], "run_id": child_id,
                            "candidate": value["candidate"], "judgment": value["judgment"],
                            "deterministic_metrics": value["deterministic_metrics"]}
                    steps.append(step)
                    begin = datetime.fromisoformat(value.get("started_at", receipt["created_at"]))
                    end = datetime.fromisoformat(value.get("ended_at", value.get("started_at", receipt["created_at"])))
                    children.append({"id": child_id, "name": "replay_assistant", "run_type": "llm",
                        "inputs": {"messages": case["messages"], "tools": case["tools"]},
                        "outputs": {"message": value["candidate"]}, "start_time": begin, "end_time": end,
                        "parent_run_id": root_id, "trace_id": root_id,
                        "dotted_order": root_order + "." + begin.strftime("%Y%m%dT%H%M%S%fZ") + child_id,
                        "project_name": name, "extra": {"metadata": {**metadata, "case_id": case["id"],
                            "message_index": case["message_index"], "serving_route": value["serving_route"]}}})
                root = {"id": root_id, "name": "replay_trajectory", "run_type": "chain", "inputs": example.inputs,
                        "outputs": {"steps": steps}, "reference_example_id": example.id, "trace_id": root_id,
                        "dotted_order": root_order, "project_name": name, "start_time": started,
                        "end_time": max((child["end_time"] for child in children), default=started),
                        "extra": {"metadata": metadata}}
                for value in [root, *children]:
                    feedback_times[value["id"]] = value["start_time"]
                _ensure_runs(client, [root, *children], project.id)
                expected_by_root[root_id] = _expected_feedback(root_id, steps)
            expected = [item for items in expected_by_root.values() for item in items]
            phase = f"{label}:feedback_readback"
            found = _saved_feedback(client, expected)

            if len(found) != len(expected):
                phase = f"{label}:feedback_upload"
                for root_id, items in expected_by_root.items():
                    for item in items:
                        target = item["target_run_id"]
                        if (target, item["key"]) in found:
                            continue
                        # A retry must keep the same ID even if an accepted write
                        # lost its response or has not appeared in list feedback yet.
                        client.create_feedback(
                            run_id=target, key=item["key"], score=item["score"], comment=item.get("comment"),
                            feedback_id=uuid5(UUID(target), item["key"]), feedback_source_type="model",
                            trace_id=root_id, session_id=project.id, start_time=feedback_times[target],
                            source_info={"judge_model": config["judge_model"], "cached_judgment": True},
                        )
                client.flush()
                phase = f"{label}:feedback_verification"
                for attempt in range(5):
                    if len(_saved_feedback(client, expected)) == len(expected):
                        break
                    if attempt < 4:
                        time.sleep(1)
                else:
                    raise PipelineError(f"LangSmith feedback upload is incomplete; rerun evaluate: {path}")
        phase = "comparison_url"
        receipt["comparison_url"] = _comparison_url(receipt["experiments"])
        receipt["status"] = "complete"
        receipt.pop("failure_phase", None)
        receipt.pop("error_type", None)
        receipt.pop("last_error", None)
        _json_dump(path, receipt)
        return {"dataset_id": splits["dataset_id"], "dataset_version": splits["dataset_version"],
                "split": "test", "comparison_url": receipt["comparison_url"]}
    except PublicationRequestError as exc:
        receipt.update(status="interrupted", failure_phase=phase, error_type=type(exc).__name__, last_error=str(exc))
        _json_dump(path, receipt)
        raise PipelineError(f"LangSmith experiment publication failed during {phase}: {exc}; "
                            f"replay results are saved; rerun evaluate to resume publication: {path}") from None
    except PipelineError as exc:
        receipt.update(status="interrupted", failure_phase=phase, error_type=type(exc).__name__)
        _json_dump(path, receipt)
        raise
    except Exception as exc:
        receipt.update(status="interrupted", failure_phase=phase, error_type=type(exc).__name__)
        _json_dump(path, receipt)
        raise PipelineError(f"LangSmith experiment publication failed during {phase} ({type(exc).__name__}); replay results are saved; rerun evaluate: {path}") from None

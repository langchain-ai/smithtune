"""Resumable full-conversation labels and a frozen dataset handoff."""

from __future__ import annotations

import copy
import json
import re
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from smithtune.artifacts import _atomic_text, _json_dump, _jsonl_dump, _load_json, _load_jsonl, _run, exclusive_output
from smithtune.curation import _api, _uuid, _write_new
from smithtune.dataset import validate_trajectories
from smithtune.dataset_artifacts import load_conversation, save_conversation
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.triage_judges import PROVIDERS, api_judge, check_credentials, context_window_exceeded, deepagent_judge, judge_messages, rubric_text, validate_judgment
from smithtune.triage_source import conversation_trajectories, load_snapshot, multimodal_types, snapshot, training_error


JUDGE_ALIASES = {
    "deepseek-v4.1-flash": ("fireworks", "accounts/fireworks/models/deepseek-v4p1-flash"),
    "glm-5.3-flash": ("fireworks", "accounts/fireworks/models/glm-5p3-flash"),
    "gpt-5.6-terra": ("openai", "gpt-5.6-terra"),
}


def load_config(path: Path | None) -> dict:
    value = _load_json(path) if path else json.loads(files("smithtune").joinpath("skills/sft-trace-triage/config.example.json").read_text(encoding="utf-8"))
    return validate_config(value)


def validate_config(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"judges", "rules"}:
        raise PipelineError("triage config must contain judges and optional rules")
    judges = value.get("judges")
    if not isinstance(judges, list) or not 1 <= len(judges) <= 16:
        raise PipelineError("configure between 1 and 16 judge slots")
    for judge in judges:
        if not isinstance(judge, dict) or set(judge) != {"name", "provider", "model"}:
            raise PipelineError("each judge requires name, provider, and model")
        if not isinstance(judge["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", judge["name"]):
            raise PipelineError("judge name must use lowercase letters, digits, or hyphens")
        if judge["provider"] not in PROVIDERS or not isinstance(judge["model"], str) or not judge["model"].strip() or len(judge["model"]) > 256:
            raise PipelineError("judge provider or model is invalid")
    if len({judge["name"] for judge in judges}) != len(judges):
        raise PipelineError("judge slot names must be unique")
    rules = value.get("rules", [])
    if not isinstance(rules, list) or len(rules) > 30 or any(not isinstance(rule, str) or not 1 <= len(rule) <= 2000 for rule in rules):
        raise PipelineError("rules must be a list of short strings")
    return {"judges": judges, "rules": rules}


def council_settings(output_dir: Path, *, judges=None, rules=None, config_path=None, **overrides) -> dict:
    """Reuse the preview on confirm/resume; explicit options replace defaults."""
    path = output_dir / "plan.json"
    saved = _load_json(path) if path.exists() else {}
    if not isinstance(saved, dict):
        raise PipelineError("invalid saved council plan")
    config = copy.deepcopy(saved["config"]) if "config" in saved else load_config(None)
    if config_path is not None:
        if judges is not None or rules is not None:
            raise PipelineError("use --config or --judges/--rule, not both")
        config = load_config(config_path)
    if judges is not None:
        slots = []
        for index, value in enumerate(judges, 1):
            value = value.strip()
            if value.lower() in JUDGE_ALIASES:
                provider, model = JUDGE_ALIASES[value.lower()]
            else:
                provider, separator, model = value.partition(":")
                provider, model = provider.strip(), model.strip()
                if not separator or provider not in PROVIDERS or not model:
                    raise PipelineError("--judges needs comma-separated model names: " + ",".join(JUDGE_ALIASES)
                                        + "; for other models use provider:model")
            slots.append({"name": f"judge-{index}", "provider": provider, "model": model})
        config["judges"] = slots
    if rules is not None:
        config["rules"] = rules
    settings = {"config": validate_config(config),
                "runner_mode": saved.get("runner", "deepagent"),
                "concurrency": saved.get("concurrency", 4),
                "attempts": saved.get("max_attempts_per_task", 3),
                "max_output_tokens": saved.get("max_output_tokens", 4096)}
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def _label(trajectory: dict, records: dict, judges: list[dict]) -> dict:
    media = trajectory["multimodal_types"] if "multimodal_types" in trajectory else multimodal_types(trajectory)
    if media:
        return {"trajectory_id": trajectory["trajectory_id"], "keep": 0, "status": "filtered", "votes": [],
                "disagreement": False, "reason": "Filtered before judging: multimodal content (" + ", ".join(media) + ")."}
    votes = [records.get((trajectory["trajectory_id"], judge["name"])) for judge in judges]
    context_error = next((vote for vote in votes if vote and vote["status"] == "context_exceeded"), None)
    if context_error:
        return {"trajectory_id": trajectory["trajectory_id"], "keep": 0, "status": "context_exceeded",
                "votes": votes, "disagreement": False, "reason": "Filtered: " + context_error["error"]}
    valid = [record for record in votes if record and record["status"] == "complete"]
    complete = len(valid) == len(judges)
    keeps = sum(record["judgment"]["keep"] for record in valid)
    return {"trajectory_id": trajectory["trajectory_id"], "keep": int(complete and keeps > len(judges) / 2),
            "status": "complete" if complete else "incomplete", "valid_judges": len(valid), "expected_judges": len(judges),
            "disagreement": len({record["judgment"]["keep"] for record in valid}) > 1,
            "votes": votes}


def _result(label: dict) -> dict:
    """A small public result, explained by the votes without another model call."""
    if label["status"] in {"filtered", "context_exceeded"}:
        reason = label["reason"]
    elif label["status"] != "complete":
        reason = (f"Labeling incomplete: {label['valid_judges']}/{label['expected_judges']} judges finished. "
                  "Not selected; rerun the command to finish labeling.")
    else:
        votes = [vote["judgment"] for vote in label["votes"]]
        matching = [vote for vote in votes if vote["keep"] == label["keep"]]
        reasons = list(dict.fromkeys(" ".join(vote["reason"].split()) for vote in matching))
        tied = len(matching) * 2 == len(votes)
        reason = ("Tied vote; no majority for 1. " if tied else f"{len(matching)}/{len(votes)} judges voted {label['keep']}. ") + " ".join(reasons)
    return {"trajectory_id": label["trajectory_id"], "keep": label["keep"], "reason": reason}


@exclusive_output("output_dir")
def run_triage(source: dict, output_dir: Path, *, config_path: Path | None = None, runner_mode="api", dry_run=False,
               confirm=False, concurrency=4, max_output_tokens=4096, attempts=3,
               runner=_run, judge_call=None, sleeper=time.sleep, config: dict | None = None) -> dict:
    config = validate_config(config) if config is not None else load_config(config_path)
    if runner_mode not in {"api", "deepagent"}:
        raise PipelineError("triage runner must be api or deepagent")
    if not 1 <= concurrency <= 16 or not 1 <= attempts <= 5 or not 128 <= max_output_tokens <= 16384:
        raise PipelineError("invalid triage concurrency, retry, or input/output limits")
    if not dry_run and not confirm:
        raise PipelineError("trajectory judging incurs cost; review --dry-run, then use --confirm")
    frozen = snapshot(source, output_dir, runner=runner)
    judging = conversation_trajectories(frozen)
    filtered = {trajectory["trajectory_id"] for trajectory in judging
                if (trajectory["multimodal_types"] if "multimodal_types" in trajectory else multimodal_types(trajectory))}
    rubric = rubric_text()
    identity = {"snapshot_sha256": frozen["snapshot_sha256"], "config": config, "rubric_sha256": json_sha256(rubric),
                "runner": runner_mode, "max_output_tokens": max_output_tokens,
                "prefilter": "multimodal-and-provider-context-v1", "judging_unit": "conversation-v1"}
    if runner_mode == "deepagent":
        # The coordinator skill changes scheduling decisions and belongs in
        # the resume identity just like the judge rubric.
        skill = files("smithtune").joinpath("skills/sft-trace-triage/SKILL.md").read_text(encoding="utf-8")
        identity.update(agent_version=10, skill_sha256=json_sha256(skill))
    plan = {**identity, "selected_traces": len(frozen["selected_trace_ids"]), "source_traces": len(frozen["traces"]), "trajectories": len(judging),
            "conversation_units": len(frozen["units"]), "judges": len(config["judges"]),
            "filtered_multimodal": len(filtered),
            "judge_tasks": (len(judging) - len(filtered)) * len(config["judges"]), "max_attempts_per_task": attempts,
            "concurrency": concurrency, "aggregation": "all slots required; strict majority; ties drop",
            "training_selection": "whole frozen conversation with a complete majority keep vote",
            "cost": "one request per trajectory/judge attempt, plus coordinator calls in deepagent mode"}
    if runner_mode == "deepagent":
        plan["coordinator"] = config["judges"][0]
        plan["code_mode"] = "sandboxed Python; bounded judge batches; no host filesystem or network"
    manifest_path = output_dir / "triage-config.json"
    if manifest_path.exists() and _load_json(manifest_path) != identity:
        raise PipelineError("existing triage run uses different input, rubric, model, or limits; use a new output directory")
    _json_dump(output_dir / "plan.json", plan)
    if dry_run:
        print(f"Preview: {plan['trajectories']} trajectories, {plan['judges']} judges, {plan['judge_tasks']} votes. No judge calls made.", file=sys.stderr)
        print(f"Filtered {len(filtered)} trajectories with multimodal content before judging.", file=sys.stderr)
        print(f"Run: smithtune dataset triage {shlex.quote(str(output_dir))} --confirm", file=sys.stderr)
        return plan
    identity_hash = json_sha256(identity)
    trajectories = {trajectory["trajectory_id"]: trajectory for trajectory in judging}
    if len(trajectories) != len(judging):
        raise PipelineError("snapshot contains duplicate trajectory identities")
    records = {}
    results_path = output_dir / "judgments.jsonl"
    if results_path.exists():
        for record in _load_jsonl(results_path):
            if not isinstance(record, dict) or not isinstance(record.get("trajectory_id"), str) or not isinstance(record.get("judge"), str):
                raise PipelineError("saved judgment has an invalid identity")
            key = (record.get("trajectory_id"), record.get("judge"))
            if key in records or key[0] not in trajectories or key[1] not in {judge["name"] for judge in config["judges"]}:
                raise PipelineError("saved judgments contain duplicate or unknown identities")
            if record.get("identity_sha256") != identity_hash or record.get("status") not in {"complete", "error", "context_exceeded"}:
                raise PipelineError("saved judgment has different run identity or invalid status")
            if record["status"] == "complete":
                validate_judgment(record.get("judgment"))
            records[key] = record
    call = judge_call or (deepagent_judge if runner_mode == "deepagent" else api_judge)

    def judge_one(trajectory, judge):
        record = {"trajectory_id": trajectory["trajectory_id"], "judge": judge["name"], "identity_sha256": identity_hash}
        messages = judge_messages(trajectory, rubric, config["rules"])
        base_prompt = messages[0]["content"]
        feedback = ""
        for attempt in range(attempts):
            messages[0] = {"role": "system", "content": base_prompt + feedback}
            record["attempts"] = attempt + 1
            error_kind = "request_failed"
            try:
                if runner_mode == "deepagent" and judge_call is None:
                    diagnostics = {}
                    record["agent"] = diagnostics
                    response = call(judge, messages, max_output_tokens, diagnostics=diagnostics)
                else:
                    response = call(judge, messages, max_output_tokens)
                error_kind = "invalid_result"
                result = validate_judgment(response)
                return {**record, "status": "complete", "judgment": result}
            except Exception as exc:
                if context_window_exceeded(exc):
                    return {**record, "status": "context_exceeded",
                            "error": f"Full trajectory exceeds the context window of {judge['model']}."}
                # Save useful failure categories, never raw provider bodies or
                # request headers that could contain private evidence or keys.
                failure = {"error_type": type(exc).__name__}
                if error_kind == "invalid_result" and isinstance(exc, PipelineError):
                    failure["validation_error"] = str(exc)
                    feedback = "\nYour previous attempt failed validation. Return only JSON with keep (0 or 1) and a short reason."
                elif isinstance(exc, PipelineError) and str(exc) == "judge returned invalid JSON":
                    feedback = "\nYour previous attempt returned invalid JSON. Return one JSON object matching the required schema, with no code fences or other text."
                status = getattr(exc, "status_code", None)
                if isinstance(status, int):
                    failure["http_status"] = status
                if attempt + 1 < attempts:
                    print(f"Retrying {judge['name']} for {trajectory['trajectory_id']}: {type(exc).__name__}.", file=sys.stderr)
                    sleeper(2 ** attempt)
        return {**record, **failure, "status": "error", "error_kind": error_kind, "error": "judge request or result validation failed; rerun to retry"}

    context_filtered = {r["trajectory_id"] for r in records.values() if r["status"] == "context_exceeded"}
    pending = [(trajectory, judge) for trajectory in judging for judge in config["judges"]
               if trajectory["trajectory_id"] not in filtered | context_filtered and records.get((trajectory["trajectory_id"], judge["name"]), {}).get("status") != "complete"]
    if pending and judge_call is None:
        check_credentials(config["judges"])
        if runner_mode == "deepagent":
            from smithtune.triage_agent import check_installation
            check_installation()
    _json_dump(manifest_path, identity)
    if not results_path.exists():
        _jsonl_dump(results_path, [])
    total = plan["judge_tasks"]
    print(f"Filtered {len(filtered)} trajectories with multimodal content before judging.", file=sys.stderr)
    print(f"Council: {len(config['judges'])} judges; {len(pending)}/{total} votes pending.", file=sys.stderr)
    record_lock = Lock()

    def save_record(record):
        with record_lock:
            records[(record["trajectory_id"], record["judge"])] = record
            _jsonl_dump(results_path, [records[key] for key in sorted(records)])
            completed = sum(r["status"] == "complete" for r in records.values())
            print(f"Saved judge result: {record['status']}; {completed}/{total} valid votes.", file=sys.stderr)

    try:
        if runner_mode == "deepagent" and pending and judge_call is None:
            from smithtune.triage_coordinator import coordinate
            coordinate(pending, judge_one, save_record, output_dir, concurrency=concurrency,
                       max_tokens=max_output_tokens, coordinator_judge=config["judges"][0])
        elif pending:
            _run_direct(pending, judge_one, save_record, concurrency)
    finally:
        labels = [_label(trajectory, records, config["judges"]) for trajectory in judging]
        results = [_result(label) for label in labels]
        _jsonl_dump(output_dir / "labels.jsonl", results)
        summary = {"trajectories": len(labels), "kept": sum(label["keep"] for label in labels),
                   "dropped": sum(label["status"] == "complete" and not label["keep"] for label in labels),
                   "incomplete": sum(label["status"] == "incomplete" for label in labels),
                   "filtered_multimodal": len(filtered),
                   "filtered_context": sum(label["status"] == "context_exceeded" for label in labels),
                   "disagreement": sum(label["disagreement"] for label in labels),
                   "identity_sha256": identity_hash, "labels_sha256": json_sha256(results),
                   "labels": str(output_dir / "labels.jsonl"), "report": str(output_dir / "report.md")}
        summary["status"] = "complete" if summary["incomplete"] == 0 else "incomplete"
        by_id = {label["trajectory_id"]: label for label in labels}
        # Recheck cached snapshots without changing their frozen evidence or hashes.
        training_errors = [training_error(unit) for unit in frozen["units"]]
        summary["eligible_conversations"] = sum(
            not error and by_id[unit["example"]["id"]]["keep"]
            for unit, error in zip(frozen["units"], training_errors, strict=True)
        )
        summary["unsupported_training_conversations"] = sum(bool(error) for error in training_errors)
        summary["unsupported_tool_contracts"] = sum(error == "tool schemas cannot be represented by the current training contract" for error in training_errors)
        _json_dump(output_dir / "summary.json", summary)
        eligible = summary["trajectories"] - len(filtered) - summary["filtered_context"]
        explanation = f"Labeled {eligible - summary['incomplete']}/{eligible} text trajectories: {summary['kept']} with 1, {summary['dropped']} with 0. Filtered {len(filtered)} multimodal trajectories before judging."
        if summary["filtered_context"]:
            explanation += f" Filtered {summary['filtered_context']} trajectories that exceed a council model's context window."
        if summary["incomplete"]:
            explanation += f" {summary['incomplete']} still need labeling; rerun the command to retry."
        report = ["# Trajectory labels", "", explanation, "", "1 = use for SFT. 0 = do not use for SFT.", ""]
        for result in results:
            report.append(f"- {result['trajectory_id']}: {result['keep']} — {result['reason']}")
        _atomic_text(output_dir / "report.md", "\n".join(report) + "\n")
        print(explanation + f" Labels and reasons: {output_dir / 'labels.jsonl'}.", file=sys.stderr)
    return summary


def _run_direct(pending, judge_one, save_record, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        try:
            for trajectory, judge in pending:
                futures.append(executor.submit(judge_one, trajectory, judge))
            for future in as_completed(futures):
                save_record(future.result())
        finally:
            for future in futures:
                future.cancel()


def selected_examples(triage_dir: Path) -> list[dict]:
    frozen = load_snapshot(triage_dir)
    judging = conversation_trajectories(frozen)
    identity = _load_json(triage_dir / "triage-config.json")
    if not isinstance(identity, dict) or identity.get("judging_unit") != "conversation-v1":
        raise PipelineError("this run has per-trace votes; judge whole conversations in a new run directory before import")
    summary = _load_json(triage_dir / "summary.json")
    labels = _load_jsonl(triage_dir / "labels.jsonl")
    if not isinstance(identity, dict) or not isinstance(summary, dict) or any(not isinstance(label, dict) or not isinstance(label.get("trajectory_id"), str) for label in labels):
        raise PipelineError("invalid triage artifacts")
    if identity.get("snapshot_sha256") != frozen["snapshot_sha256"] or summary.get("identity_sha256") != json_sha256(identity) or summary.get("labels_sha256") != json_sha256(labels):
        raise PipelineError("triage labels or source snapshot have changed")
    trajectories = {trajectory["trajectory_id"]: trajectory for trajectory in judging}
    by_id = {label.get("trajectory_id"): label for label in labels}
    if len(by_id) != len(labels) or set(by_id) != set(trajectories):
        raise PipelineError("labels must cover every snapshot trajectory exactly once")
    records = {}
    for record in _load_jsonl(triage_dir / "judgments.jsonl"):
        if not isinstance(record, dict) or not isinstance(record.get("trajectory_id"), str) or not isinstance(record.get("judge"), str):
            raise PipelineError("saved judgment has an invalid identity")
        key = (record.get("trajectory_id"), record.get("judge"))
        if key in records or key[0] not in trajectories or key[1] not in {judge["name"] for judge in identity["config"]["judges"]}:
            raise PipelineError("invalid or duplicate saved judge identity")
        if record.get("identity_sha256") != json_sha256(identity):
            raise PipelineError("judgment identity mismatch")
        if record.get("status") == "complete":
            validate_judgment(record.get("judgment"))
        records[key] = record
    for tid, trajectory in trajectories.items():
        calculated = _label(trajectory, records, identity["config"]["judges"])
        if _result(calculated) != by_id[tid]:
            raise PipelineError("labels do not match validated judge votes")
        by_id[tid] = calculated
    selected = []
    for unit in frozen["units"]:
        label = by_id[unit["example"]["id"]]
        if training_error(unit) or label["keep"] != 1 or label["status"] != "complete":
            continue
        example = load_conversation(save_conversation(triage_dir, unit["example"]))
        contract = parse_inference_contract(unit["contract"])
        example["metadata"]["smithtune_triage"] = {"identity_sha256": json_sha256(identity),
            "messages_sha256": json_sha256(example["inputs"]["messages"]), "contract": contract.to_dict()}
        selected.append(example)
    if not selected:
        raise PipelineError("no complete, kept conversations are eligible for training")
    validate_trajectories(selected, len(selected))
    return selected


@exclusive_output("triage_dir")
def create_triaged_dataset(triage_dir: Path, name: str, *, confirm: bool, runner=_run) -> dict:
    if not confirm:
        raise PipelineError("creating a LangSmith dataset requires --confirm")
    if not name or not name.strip():
        raise PipelineError("dataset name must be nonempty")
    examples = selected_examples(triage_dir)
    frozen = load_snapshot(triage_dir)
    workspace = frozen["source"]["workspace_id"]
    receipt_path = triage_dir / "dataset-import.json"
    receipt = {"dataset_name": name, "dataset_id": None, "status": "creating", "confirmed_example_ids": [],
               "examples_sha256": json_sha256(examples), "pending_write": "dataset"}
    _write_new(receipt_path, receipt)
    try:
        created = _api(workspace, "POST", "/api/v1/datasets", {"name": name, "data_type": "kv"}, runner=runner)
        dataset_id = _uuid(created.get("id"), "dataset id")
        receipt.update(dataset_id=dataset_id, status="importing", pending_write=None)
        _json_dump(receipt_path, receipt)
        for example in examples:
            # IDs include the destination dataset so the same reviewed snapshot
            # can be imported into independent datasets without ID collisions.
            example_id = str(uuid5(NAMESPACE_URL, dataset_id + ":" + example["id"]))
            receipt["pending_write"] = example_id
            _json_dump(receipt_path, receipt)
            result = _api(workspace, "POST", "/api/v1/examples", {**example, "id": example_id, "dataset_id": dataset_id}, runner=runner)
            if not isinstance(result, dict) or result.get("id") != example_id:
                raise PipelineError("example import did not confirm the requested ID")
            receipt["confirmed_example_ids"].append(example_id)
            receipt["pending_write"] = None
            _json_dump(receipt_path, receipt)
        receipt["status"] = "complete"
        _json_dump(receipt_path, receipt)
    except BaseException:
        receipt["status"] = "incomplete"
        _json_dump(receipt_path, receipt)
        raise PipelineError(f"dataset import incomplete; inspect {receipt_path} before retrying") from None
    return {"dataset_id": dataset_id, "example_count": len(examples), "receipt": str(receipt_path)}


def export_skill(output: Path) -> dict:
    destination = output / "sft-trace-triage"
    if destination.exists():
        raise PipelineError("skill destination already exists; choose a new directory")
    source = files("smithtune").joinpath("skills/sft-trace-triage")
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            (destination / item.name).write_bytes(item.read_bytes())
    return {"skill": str(destination / "SKILL.md")}

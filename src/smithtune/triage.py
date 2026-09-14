"""Resumable trace labels and a frozen, all-pass conversation dataset handoff."""

from __future__ import annotations

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from smithtune.artifacts import _atomic_text, _json_dump, _jsonl_dump, _load_json, _load_jsonl, _run, exclusive_output
from smithtune.curation import _api, _uuid, _write_new
from smithtune.dataset import validate_trajectories
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.triage_judges import PROVIDERS, IncompleteJudgment, api_judge, check_credentials, deepagent_judge, judge_messages, rubric_text, validate_judgment
from smithtune.triage_source import load_snapshot, snapshot


def load_config(path: Path | None) -> dict:
    value = _load_json(path) if path else {"judges": [{"name": "judge-1", "provider": "anthropic-gateway", "model": "claude-sonnet-5"}]}
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


def _label(trace: dict, records: dict, judges: list[dict]) -> dict:
    votes = [records.get((trace["trace_id"], judge["name"])) for judge in judges]
    valid = [record for record in votes if record and record["status"] == "complete"]
    complete = len(valid) == len(judges)
    keeps = sum(record["judgment"]["keep"] for record in valid)
    return {"schema_version": 1, "trace_id": trace["trace_id"], "root_run_id": trace["root_run_id"],
            "thread_id": trace["thread_id"], "project_id": trace["project_id"],
            "source_sha256": trace["source_sha256"], "keep": int(complete and keeps > len(judges) / 2),
            "status": "complete" if complete else "incomplete", "valid_judges": len(valid), "expected_judges": len(judges),
            "disagreement": len({record["judgment"]["keep"] for record in valid}) > 1,
            "votes": votes}


@exclusive_output("output_dir")
def run_triage(source: dict, output_dir: Path, *, config_path: Path | None = None, runner_mode="api", dry_run=False,
               confirm=False, concurrency=4, max_input_chars=200_000, max_output_tokens=4096, attempts=3,
               runner=_run, judge_call=None, sleeper=time.sleep) -> dict:
    config = load_config(config_path)
    if runner_mode not in {"api", "deepagent"}:
        raise PipelineError("triage runner must be api or deepagent")
    if not 1 <= concurrency <= 16 or not 1 <= attempts <= 5 or not 1000 <= max_input_chars <= 2_000_000 or not 128 <= max_output_tokens <= 16384:
        raise PipelineError("invalid triage concurrency, retry, or input/output limits")
    if not dry_run and not confirm:
        raise PipelineError("trace judging incurs cost; review --dry-run, then use --confirm")
    if not dry_run and judge_call is None:
        check_credentials(config["judges"])
        if runner_mode == "deepagent":
            from smithtune.triage_agent import check_installation
            check_installation()
    frozen = snapshot(source, output_dir, runner=runner)
    rubric = rubric_text()
    identity = {"snapshot_sha256": frozen["snapshot_sha256"], "config": config, "rubric_sha256": json_sha256(rubric),
                "runner": runner_mode, "max_input_chars": max_input_chars, "max_output_tokens": max_output_tokens}
    if runner_mode == "deepagent":
        # The coordinator skill changes scheduling decisions and belongs in
        # the resume identity just like the judge rubric.
        skill = files("smithtune").joinpath("skills/sft-trace-triage/SKILL.md").read_text(encoding="utf-8")
        identity.update(agent_version=2, skill_sha256=json_sha256(skill))
    plan = {**identity, "selected_traces": len(frozen["selected_trace_ids"]), "traces": len(frozen["traces"]),
            "conversation_units": len(frozen["units"]), "judges": len(config["judges"]),
            "judge_tasks": len(frozen["traces"]) * len(config["judges"]), "max_attempts_per_task": attempts,
            "concurrency": concurrency, "aggregation": "all slots required; strict majority; ties drop",
            "training_selection": "whole frozen conversation only if every trace passes",
            "cost": "provider input/output rates; deepagent adds coordinator calls and up to 12 judge graph steps per attempt"}
    if runner_mode == "deepagent":
        plan["coordinator"] = config["judges"][0]
        plan["code_mode"] = "sandboxed Python; bounded judge batches; no host filesystem or network"
    _json_dump(output_dir / "plan.json", plan)
    if dry_run:
        return plan
    manifest_path = output_dir / "triage-config.json"
    if manifest_path.exists() and _load_json(manifest_path) != identity:
        raise PipelineError("existing triage run uses different input, rubric, model, or limits; use a new output directory")
    _json_dump(manifest_path, identity)
    identity_hash = json_sha256(identity)
    traces = {trace["trace_id"]: trace for trace in frozen["traces"]}
    if len(traces) != len(frozen["traces"]):
        raise PipelineError("snapshot contains duplicate trace identities")
    records = {}
    results_path = output_dir / "judgments.jsonl"
    if results_path.exists():
        for record in _load_jsonl(results_path):
            if not isinstance(record, dict) or not isinstance(record.get("trace_id"), str) or not isinstance(record.get("judge"), str):
                raise PipelineError("saved judgment has an invalid identity")
            key = (record.get("trace_id"), record.get("judge"))
            if key in records or key[0] not in traces or key[1] not in {judge["name"] for judge in config["judges"]}:
                raise PipelineError("saved judgments contain duplicate or unknown identities")
            if record.get("identity_sha256") != identity_hash or record.get("status") not in {"complete", "error", "input_too_large"}:
                raise PipelineError("saved judgment has different run identity or invalid status")
            if record["status"] == "complete":
                validate_judgment(record.get("judgment"), traces[key[0]])
            records[key] = record
    call = judge_call or (deepagent_judge if runner_mode == "deepagent" else api_judge)

    def judge_one(trace, judge):
        record = {"trace_id": trace["trace_id"], "judge": judge["name"], "identity_sha256": identity_hash}
        messages = judge_messages(trace, rubric, config["rules"])
        if sum(len(message["content"]) for message in messages) > max_input_chars:
            return {**record, "status": "input_too_large", "error": "input exceeds configured limit; no evidence was truncated"}
        for attempt in range(attempts):
            error_kind = "request_failed"
            try:
                response = call(judge, messages, max_output_tokens)
                error_kind = "invalid_result"
                result = validate_judgment(response, trace)
                return {**record, "status": "complete", "judgment": result}
            except IncompleteJudgment as exc:
                return {**record, "status": "error", "error_kind": "insufficient_evidence", "error": str(exc)}
            except Exception:
                if attempt + 1 < attempts:
                    sleeper(2 ** attempt)
        return {**record, "status": "error", "error_kind": error_kind, "error": "judge request or result validation failed; rerun to retry"}

    pending = [(trace, judge) for trace in frozen["traces"] for judge in config["judges"]
               if records.get((trace["trace_id"], judge["name"]), {}).get("status") != "complete"]
    record_lock = Lock()

    def save_record(record):
        with record_lock:
            records[(record["trace_id"], record["judge"])] = record
            _jsonl_dump(results_path, [records[key] for key in sorted(records)])

    try:
        if runner_mode == "deepagent" and pending and judge_call is None:
            from smithtune.triage_coordinator import coordinate
            coordinate(pending, judge_one, save_record, output_dir, concurrency=concurrency,
                       max_tokens=max_output_tokens, coordinator_judge=config["judges"][0])
        elif pending:
            _run_direct(pending, judge_one, save_record, concurrency)
    finally:
        labels = [_label(trace, records, config["judges"]) for trace in frozen["traces"]]
        for label in labels:
            label["identity_sha256"] = identity_hash
        _jsonl_dump(output_dir / "labels.jsonl", labels)
        summary = {"traces": len(labels), "kept": sum(label["keep"] for label in labels),
                   "dropped": sum(label["status"] == "complete" and not label["keep"] for label in labels),
                   "incomplete": sum(label["status"] != "complete" for label in labels),
                   "disagreement": sum(label["disagreement"] for label in labels),
                   "identity_sha256": identity_hash, "labels_sha256": json_sha256(labels),
                   "labels": str(output_dir / "labels.jsonl"), "report": str(output_dir / "report.md")}
        summary["status"] = "complete" if summary["incomplete"] == 0 else "incomplete"
        by_id = {label["trace_id"]: label for label in labels}
        summary["eligible_conversations"] = sum(
            not unit["training_error"] and all(by_id[tid]["keep"] for tid in unit["trace_ids"])
            for unit in frozen["units"]
        )
        summary["unsupported_tool_contracts"] = sum(bool(unit["training_error"]) for unit in frozen["units"])
        _json_dump(output_dir / "summary.json", summary)
        report = ["# SFT trace selection", "", f"Traces: {summary['traces']}. Kept: {summary['kept']}. Dropped: {summary['dropped']}. Incomplete: {summary['incomplete']}.",
                  f"Disagreement: {summary['disagreement']}. Ties drop. Errors are not quality votes.", "",
                  f"Eligible training conversations: {summary['eligible_conversations']}. Unsupported tool contracts: {summary['unsupported_tool_contracts']}.",
                  "Training uses only whole frozen conversations whose traces all pass.", "", "Sample decisions:"]
        for keep in (1, 0):
            for label in [item for item in labels if item["keep"] == keep][:3]:
                reasons = [vote["judgment"]["reason"] for vote in label["votes"] if vote and vote["status"] == "complete"]
                report.append(f"- {label['trace_id']}: keep={keep}, {label['status']}. " + " ".join(reasons).replace("\n", " "))
        _atomic_text(output_dir / "report.md", "\n".join(report) + "\n")
    return summary


def _run_direct(pending, judge_one, save_record, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        try:
            for trace, judge in pending:
                futures.append(executor.submit(judge_one, trace, judge))
            for future in as_completed(futures):
                save_record(future.result())
        finally:
            for future in futures:
                future.cancel()


def selected_examples(triage_dir: Path) -> list[dict]:
    frozen = load_snapshot(triage_dir)
    identity = _load_json(triage_dir / "triage-config.json")
    summary = _load_json(triage_dir / "summary.json")
    labels = _load_jsonl(triage_dir / "labels.jsonl")
    if not isinstance(identity, dict) or not isinstance(summary, dict) or any(not isinstance(label, dict) or not isinstance(label.get("trace_id"), str) for label in labels):
        raise PipelineError("invalid triage artifacts")
    if identity.get("snapshot_sha256") != frozen["snapshot_sha256"] or summary.get("identity_sha256") != json_sha256(identity) or summary.get("labels_sha256") != json_sha256(labels):
        raise PipelineError("triage labels or source snapshot have changed")
    traces = {trace["trace_id"]: trace for trace in frozen["traces"]}
    by_id = {label.get("trace_id"): label for label in labels}
    if len(by_id) != len(labels) or set(by_id) != set(traces):
        raise PipelineError("labels must cover every snapshot trace exactly once")
    records = {}
    for record in _load_jsonl(triage_dir / "judgments.jsonl"):
        if not isinstance(record, dict) or not isinstance(record.get("trace_id"), str) or not isinstance(record.get("judge"), str):
            raise PipelineError("saved judgment has an invalid identity")
        key = (record.get("trace_id"), record.get("judge"))
        if key in records or key[0] not in traces or key[1] not in {judge["name"] for judge in identity["config"]["judges"]}:
            raise PipelineError("invalid or duplicate saved judge identity")
        if record.get("identity_sha256") != json_sha256(identity):
            raise PipelineError("judgment identity mismatch")
        if record.get("status") == "complete":
            validate_judgment(record.get("judgment"), traces[key[0]])
        records[key] = record
    for tid, trace in traces.items():
        calculated = _label(trace, records, identity["config"]["judges"])
        calculated["identity_sha256"] = json_sha256(identity)
        if calculated != by_id[tid]:
            raise PipelineError("labels do not match validated judge votes")
    selected = []
    for unit in frozen["units"]:
        if unit["training_error"] or not all(by_id[tid]["keep"] == 1 and by_id[tid]["status"] == "complete" for tid in unit["trace_ids"]):
            continue
        example = copy.deepcopy(unit["example"])
        contract = parse_inference_contract(unit["contract"])
        example["metadata"]["smithtune_triage"] = {"identity_sha256": json_sha256(identity),
            "messages_sha256": json_sha256(example["inputs"]["messages"]), "contract": contract.to_dict()}
        selected.append(example)
    if not selected:
        raise PipelineError("no complete, all-pass conversations are eligible for training")
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

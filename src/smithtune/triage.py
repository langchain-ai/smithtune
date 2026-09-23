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

from smithtune.artifacts import _atomic_text, _json_dump, _jsonl_dump, _load_json, _load_jsonl, _run, exclusive_output
from smithtune.curation import _destination
from smithtune.dataset import _source_key, validate_trajectories
from smithtune.dataset_artifacts import LazySequence, load_conversation, save_conversation
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.triage_judges import BASETEN_REASONING, FIREWORKS_REASONING, PROVIDERS, api_judge, check_credentials, context_window_exceeded, deepagent_judge, judge_messages, rubric_text, validate_judgment
from smithtune.triage_source import conversation_trajectories, load_snapshot, multimodal_types, snapshot, training_error


# Accept known whole-file hashes for vote recovery only when the coordinator
# instructions match. Operator guidance does not affect council decisions.
LEGACY_COORDINATOR_SKILLS = {"2c32f652cb3c3ae3e2155125c892a1fff2e24d4912dea4a9bbcb75135617632b": "a50b912941ff1bd49ceb4cc7e6b0dd0c084590c6651716ed2042cc26c89b1627"}


JUDGE_ALIASES = {
    "muse-glimmer-30b": ("fireworks", "accounts/fireworks/models/muse-glimmer-30b"),
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


def council_settings(output_dir: Path, *, judges=None, rules=None, config_path=None, rubric_path=None, saved_settings=None, **overrides) -> dict:
    """Reuse the preview on confirm/resume; explicit options replace defaults."""
    path = output_dir / "plan.json"
    saved = _load_json(path) if path.exists() else {}
    if saved_settings is not None:
        saved = {**saved_settings, "runner": saved_settings["runner_mode"], "max_attempts_per_task": saved_settings["attempts"]}
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
    selection_rubric = saved.get("selection_rubric")
    if rubric_path is not None:
        try:
            selection_rubric = rubric_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PipelineError("--rubric must be a readable UTF-8 file") from exc
        if not selection_rubric.strip():
            raise PipelineError("--rubric must contain non-empty text")
    settings = {"config": validate_config(config),
                "runner_mode": saved.get("runner", "deepagent"),
                "concurrency": saved.get("concurrency", 4),
                "attempts": saved.get("max_attempts_per_task", 3),
                "max_output_tokens": saved.get("max_output_tokens", 4096)}
    if selection_rubric is not None:
        settings["selection_rubric"] = selection_rubric
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


def _label(trajectory: dict, records: dict, judges: list[dict], *, error: str | None = None) -> dict:
    media = trajectory["multimodal_types"] if "multimodal_types" in trajectory else multimodal_types(trajectory)
    if media:
        return {"trajectory_id": trajectory["trajectory_id"], "keep": 0, "status": "filtered", "votes": [],
                "disagreement": False, "reason": "Filtered before judging: multimodal content (" + ", ".join(media) + ")."}
    if error:
        return {"trajectory_id": trajectory["trajectory_id"], "keep": 0, "status": "filtered", "votes": [],
                "disagreement": False, "reason": "Filtered before judging: " + error}
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


def _result(label: dict, *, target_met=False) -> dict:
    """A small public result, explained by the votes without another model call."""
    if label["status"] in {"filtered", "context_exceeded"}:
        reason = label["reason"]
    elif label["status"] != "complete" and target_met:
        reason = "Not selected: the council-approved target has already been reached."
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


def _run_triage(source: dict, output_dir: Path, *, config_path: Path | None = None, runner_mode="api", dry_run=False,
               confirm=False, concurrency=4, max_output_tokens=4096, attempts=3,
               runner=_run, judge_call=None, sleeper=time.sleep, config: dict | None = None, frozen: dict | None = None,
               selection_rubric: str | None = None) -> dict:
    config = validate_config(config) if config is not None else load_config(config_path)
    if selection_rubric is not None and (not isinstance(selection_rubric, str) or not selection_rubric.strip()):
        raise PipelineError("selection rubric must be non-empty text")
    if runner_mode not in {"api", "deepagent"}:
        raise PipelineError("triage runner must be api or deepagent")
    if not 1 <= concurrency <= 16 or not 1 <= attempts <= 5 or not 128 <= max_output_tokens <= 16384:
        raise PipelineError("invalid triage concurrency, retry, or input/output limits")
    if not dry_run and not confirm:
        raise PipelineError("trajectory judging incurs cost; review --dry-run, then use --confirm")
    frozen = snapshot(source, output_dir, runner=runner) if frozen is None else frozen
    if frozen["source"] != source:
        raise PipelineError("triage snapshot uses a different source query; use a new directory")
    judging = conversation_trajectories(frozen, summaries=True)
    filtered = {trajectory["trajectory_id"] for trajectory in judging
                if (trajectory["multimodal_types"] if "multimodal_types" in trajectory else multimodal_types(trajectory))}
    multimodal_filtered = filtered.copy()
    # Recheck cached snapshots before paid work without changing frozen evidence
    # or vote identity: provider-neutral eligibility does not change judge inputs.
    target = source.get("target_count") if source.get("review_mode") == "council" else None
    training_errors, trajectory_hashes = {}, {}
    for unit in frozen["units"]:
        tid = unit["example"]["id"]
        training_errors[tid] = training_error(unit)
        if target is not None:
            trajectory_hashes[tid] = json_sha256(unit)
    training_filtered = {tid for tid, error in training_errors.items() if error} - filtered
    filtered |= training_filtered
    rejections = [_result(_label(trajectory, {}, config["judges"], error=training_errors[trajectory["trajectory_id"]]))
                  for trajectory in judging if trajectory["trajectory_id"] in filtered]
    rubric = rubric_text()
    if selection_rubric is not None:
        rubric += "\nTask-specific selection rubric:\n" + selection_rubric
    identity = {"snapshot_sha256": frozen["snapshot_sha256"], "config": config, "rubric_sha256": json_sha256(rubric),
                "runner": runner_mode, "max_output_tokens": max_output_tokens,
                "reasoning": {"fireworks": "none", "gpt-5.6-terra": "none"},
                "prefilter": "multimodal-and-provider-context-v1", "judging_unit": "conversation-v1"}
    if target is not None:
        identity.pop("snapshot_sha256")
        identity.update(source=source, evidence_binding="per-trajectory-v1")
    if selection_rubric is not None:
        identity["selection_rubric"] = selection_rubric
    if target is not None or any(trajectory["has_assistant_runs"] for trajectory in judging):
        identity["tool_evidence"] = "per-assistant-v1"
    identity["reasoning"].update({judge["model"]: FIREWORKS_REASONING[judge["model"]] for judge in config["judges"]
                                  if judge["provider"] == "fireworks" and judge["model"] in FIREWORKS_REASONING})
    if any(judge["provider"] == "baseten" for judge in config["judges"]):
        identity["reasoning"]["baseten"] = "none"
        identity["reasoning"].update({judge["model"]: BASETEN_REASONING[judge["model"]] for judge in config["judges"]
                                      if judge["provider"] == "baseten" and judge["model"] in BASETEN_REASONING})
    if runner_mode == "deepagent":
        # The coordinator skill changes scheduling decisions and belongs in
        # the resume identity just like the judge rubric.
        skill = files("smithtune").joinpath("skills/sft-trace-triage/SKILL.md").read_text(encoding="utf-8")
        skill_hash = json_sha256(skill.partition("## Agent helping a user")[0])
        saved_identity = _load_json(output_dir / "triage-config.json") if (output_dir / "triage-config.json").exists() else {}
        previous = saved_identity.get("skill_sha256")
        if LEGACY_COORDINATOR_SKILLS.get(previous) == skill_hash:
            skill_hash = previous
        identity.update(agent_version=11, skill_sha256=skill_hash)
    plan = {**identity, "selected_traces": len(frozen["selected_trace_ids"]), "source_traces": len(frozen["traces"]), "trajectories": len(judging),
            "conversation_units": len(frozen["units"]), "judges": len(config["judges"]),
            "filtered_multimodal": len(multimodal_filtered),
            "filtered_training": len(training_filtered),
            "judge_tasks": (len(judging) - len(filtered)) * len(config["judges"]), "max_attempts_per_task": attempts,
            "rejections": rejections,
            "concurrency": concurrency, "aggregation": "all slots required; strict majority; ties drop",
            "training_selection": "whole frozen conversation with a complete majority keep vote",
            "cost": "one request per trajectory/judge attempt, plus coordinator calls in deepagent mode"}
    if target is not None:
        plan["target_count"] = target
        plan["training_selection"] = "stop at the cumulative approved target; unreviewed candidates remain saved"
    if runner_mode == "deepagent":
        plan["coordinator"] = config["judges"][0]
        plan["code_mode"] = "sandboxed Python; bounded judge batches; no host filesystem or network"
    manifest_path = output_dir / "triage-config.json"
    if manifest_path.exists() and _load_json(manifest_path) != identity:
        raise PipelineError("existing triage run uses different input, rubric, model, or limits; use a new output directory")
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
            if target is not None and record.get("trajectory_sha256") != trajectory_hashes[key[0]]:
                raise PipelineError("saved judgment has different trajectory content or tool evidence")
            if record["status"] == "complete":
                validate_judgment(record.get("judgment"))
            records[key] = record
    call = judge_call or (deepagent_judge if runner_mode == "deepagent" else api_judge)

    positions = {trajectory["trajectory_id"]: index for index, trajectory in enumerate(judging)}

    def judge_one(trajectory, judge):
        # Queued tasks contain only IDs/media. Load full evidence inside the
        # bounded worker, and release it when that judgment returns.
        unit = frozen["units"][positions[trajectory["trajectory_id"]]]
        trajectory = {"trajectory_id": trajectory["trajectory_id"], "messages": unit["example"]["inputs"]["messages"]}
        source_evidence = unit["example"].get("metadata", {}).get("smithtune_source")
        if source_evidence is not None:
            trajectory["assistant_runs"] = source_evidence["assistant_runs"]
        record = {"trajectory_id": trajectory["trajectory_id"], "judge": judge["name"], "identity_sha256": identity_hash}
        if target is not None:
            record["trajectory_sha256"] = trajectory_hashes[trajectory["trajectory_id"]]
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
    def approved_count():
        return sum(_label(trajectory, records, config["judges"], error=training_errors[trajectory["trajectory_id"]])["keep"]
                   for trajectory in judging)

    if target is not None and approved_count() >= target:
        pending = []
    plan["pending_judge_tasks"] = len(pending)
    _json_dump(output_dir / "plan.json", plan)
    if dry_run:
        print(f"Preview: {plan['trajectories']} trajectories, {plan['judges']} judges, up to {plan['pending_judge_tasks']} pending votes. No judge calls made.", file=sys.stderr)
        print(f"Filtered before judging: {len(multimodal_filtered)} multimodal, {len(training_filtered)} with structural or tool errors.", file=sys.stderr)
        print(f"Run: smithtune dataset triage {shlex.quote(str(output_dir))} --confirm", file=sys.stderr)
        return plan
    if pending and judge_call is None:
        check_credentials(config["judges"])
        if runner_mode == "deepagent":
            from smithtune.triage_agent import check_installation
            check_installation()
    _json_dump(manifest_path, identity)
    if not results_path.exists():
        _jsonl_dump(results_path, [])
    total = plan["judge_tasks"]
    print(f"Filtered before judging: {len(multimodal_filtered)} multimodal, {len(training_filtered)} with structural or tool errors.", file=sys.stderr)
    print(f"Council: {len(config['judges'])} judges; {len(pending)}/{total} votes pending.", file=sys.stderr)
    record_lock = Lock()

    def save_record(record):
        with record_lock:
            records[(record["trajectory_id"], record["judge"])] = record
            _jsonl_dump(results_path, [records[key] for key in sorted(records)])
            completed = sum(r["status"] == "complete" and r["trajectory_id"] not in filtered for r in records.values())
            print(f"Saved judge result: {record['status']}; {completed}/{total} valid votes.", file=sys.stderr)

    def run_batch(batch):
        if runner_mode == "deepagent" and judge_call is None:
            from smithtune.triage_coordinator import coordinate
            coordinate(batch, judge_one, save_record, output_dir, concurrency=concurrency,
                       max_tokens=max_output_tokens, coordinator_judge=config["judges"][0])
        else:
            _run_direct(batch, judge_one, save_record, concurrency)

    try:
        if target is None and pending:
            run_batch(pending)
        elif target is not None:
            # A batch cannot approve more trajectories than the remaining target.
            # Completed votes are retained; errored slots retry only on a later run.
            while pending and (remaining := target - approved_count()) > 0:
                ids = set(list(dict.fromkeys(trajectory["trajectory_id"] for trajectory, _ in pending))[:remaining])
                batch = [task for task in pending if task[0]["trajectory_id"] in ids]
                pending = [task for task in pending if task[0]["trajectory_id"] not in ids]
                run_batch(batch)
    finally:
        labels = [_label(trajectory, records, config["judges"], error=training_errors[trajectory["trajectory_id"]]) for trajectory in judging]
        target_met = target is not None and sum(label["keep"] for label in labels) >= target
        results = [_result(label, target_met=target_met) for label in labels]
        _jsonl_dump(output_dir / "labels.jsonl", results)
        summary = {"trajectories": len(labels), "kept": sum(label["keep"] for label in labels),
                   "dropped": sum(label["status"] == "complete" and not label["keep"] for label in labels),
                   "incomplete": sum(label["status"] == "incomplete" for label in labels),
                   "filtered_multimodal": len(multimodal_filtered),
                   "filtered_training": len(training_filtered),
                   "filtered_context": sum(label["status"] == "context_exceeded" for label in labels),
                   "disagreement": sum(label["disagreement"] for label in labels),
                   "identity_sha256": identity_hash, "labels_sha256": json_sha256(results),
                   "labels": str(output_dir / "labels.jsonl"), "report": str(output_dir / "report.md")}
        summary["status"] = "complete" if target_met or summary["incomplete"] == 0 else "incomplete"
        if target is not None:
            summary.update(snapshot_sha256=frozen["snapshot_sha256"], target_count=target, target_met=target_met,
                           unreviewed=sum(label["status"] == "incomplete" and not any(label["votes"]) for label in labels))
        by_id = {label["trajectory_id"]: label for label in labels}
        summary["eligible_conversations"] = sum(
            not error and by_id[tid]["keep"] for tid, error in training_errors.items()
        )
        summary["unsupported_training_conversations"] = sum(bool(error) for error in training_errors.values())
        summary["unsupported_tool_contracts"] = sum(error == "tool schemas cannot be represented by the current training contract" for error in training_errors.values())
        _json_dump(output_dir / "summary.json", summary)
        eligible = summary["trajectories"] - len(filtered) - summary["filtered_context"]
        explanation = f"Labeled {eligible - summary['incomplete']}/{eligible} text trajectories: {summary['kept']} with 1, {summary['dropped']} with 0. Filtered before judging: {len(multimodal_filtered)} multimodal, {len(training_filtered)} with structural or tool errors."
        if summary["filtered_context"]:
            explanation += f" Filtered {summary['filtered_context']} trajectories that exceed a council model's context window."
        if target_met:
            explanation += f" Target reached; {summary['incomplete']} remaining candidates were not selected."
        elif summary["incomplete"]:
            explanation += f" {summary['incomplete']} still need labeling; rerun the command to retry."
        report = ["# Trajectory labels", "", explanation, "", "1 = use for SFT. 0 = do not use for SFT.", ""]
        for result in results:
            report.append(f"- {result['trajectory_id']}: {result['keep']} — {result['reason']}")
        _atomic_text(output_dir / "report.md", "\n".join(report) + "\n")
        print(explanation + f" Labels and reasons: {output_dir / 'labels.jsonl'}.", file=sys.stderr)
    return summary


run_triage = exclusive_output("output_dir")(_run_triage)


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


def selected_examples(triage_dir: Path, *, frozen: dict | None = None, require_complete=False, allow_empty=False):
    frozen = load_snapshot(triage_dir) if frozen is None else frozen
    judging = conversation_trajectories(frozen, summaries=True)
    identity = _load_json(triage_dir / "triage-config.json")
    if not isinstance(identity, dict) or identity.get("judging_unit") != "conversation-v1":
        raise PipelineError("this run has per-trace votes; judge whole conversations in a new run directory before import")
    summary = _load_json(triage_dir / "summary.json")
    labels = _load_jsonl(triage_dir / "labels.jsonl")
    if not isinstance(identity, dict) or not isinstance(summary, dict) or any(not isinstance(label, dict) or not isinstance(label.get("trajectory_id"), str) for label in labels):
        raise PipelineError("invalid triage artifacts")
    targeted = identity.get("evidence_binding") == "per-trajectory-v1"
    snapshot_hash = summary.get("snapshot_sha256") if targeted else identity.get("snapshot_sha256")
    if targeted and identity.get("source") != frozen["source"]:
        raise PipelineError("triage source selection has changed")
    if snapshot_hash != frozen["snapshot_sha256"] or summary.get("identity_sha256") != json_sha256(identity) or summary.get("labels_sha256") != json_sha256(labels):
        raise PipelineError("triage labels or source snapshot have changed")
    trajectories = {trajectory["trajectory_id"]: trajectory for trajectory in judging}
    by_id = {label.get("trajectory_id"): label for label in labels}
    if len(trajectories) != len(judging) or len(by_id) != len(labels) or set(by_id) != set(trajectories):
        raise PipelineError("labels must cover every snapshot trajectory exactly once")
    hashes = {unit["example"]["id"]: json_sha256(unit) for unit in frozen["units"]} if targeted else {}
    records = {}
    for record in _load_jsonl(triage_dir / "judgments.jsonl"):
        if not isinstance(record, dict) or not isinstance(record.get("trajectory_id"), str) or not isinstance(record.get("judge"), str):
            raise PipelineError("saved judgment has an invalid identity")
        key = (record.get("trajectory_id"), record.get("judge"))
        if key in records or key[0] not in trajectories or key[1] not in {judge["name"] for judge in identity["config"]["judges"]}:
            raise PipelineError("invalid or duplicate saved judge identity")
        if record.get("identity_sha256") != json_sha256(identity):
            raise PipelineError("judgment identity mismatch")
        if targeted and record.get("trajectory_sha256") != hashes[key[0]]:
            raise PipelineError("saved judgment has different trajectory content or tool evidence")
        if record.get("status") == "complete":
            validate_judgment(record.get("judgment"))
        records[key] = record
    training_errors = {unit["example"]["id"]: training_error(unit) for unit in frozen["units"]}
    calculated_labels = {tid: _label(trajectory, records, identity["config"]["judges"], error=training_errors[tid])
                         for tid, trajectory in trajectories.items()}
    target_met = targeted and sum(label["keep"] for label in calculated_labels.values()) >= identity["source"]["target_count"]
    for tid, trajectory in trajectories.items():
        calculated = calculated_labels[tid]
        if _result(calculated, target_met=target_met) != by_id[tid]:
            # Older runs could judge invalid trajectories. Verify their historical
            # quality label against the saved votes, but still exclude them.
            historical = _label(trajectory, records, identity["config"]["judges"])
            if not training_errors[tid] or _result(historical) != by_id[tid]:
                raise PipelineError("labels do not match validated judge votes")
        if require_complete and not target_met and calculated["status"] == "incomplete":
            raise PipelineError("council judging is incomplete; run dataset resume DIR --confirm before pushing")
        by_id[tid] = calculated
    selected = []
    for index, unit in enumerate(frozen["units"]):
        label = by_id[unit["example"]["id"]]
        if training_errors[unit["example"]["id"]] or label["keep"] != 1 or label["status"] != "complete":
            continue
        selected.append(index)

    def load_selected(index):
        unit = frozen["units"][selected[index]]
        if frozen["schema_version"] == 3:
            example = {**unit["example"], "metadata": dict(unit["example"]["metadata"])}
        else:
            example = load_conversation(save_conversation(triage_dir, unit["example"]))
        evidence = {"identity_sha256": json_sha256(identity),
                    "messages_sha256": json_sha256(example["inputs"]["messages"])}
        if "smithtune_source" in example["metadata"]:
            from smithtune.bindings import evidence_hash
            evidence["evidence_sha256"] = evidence_hash(example)
        else:
            evidence["contract"] = parse_inference_contract(unit["contract"]).to_dict()
        example["metadata"]["smithtune_triage"] = evidence
        return example

    examples = LazySequence(len(selected), load_selected)
    if not examples and not allow_empty:
        raise PipelineError("no complete, kept conversations are eligible for training")
    if examples:
        validate_trajectories(examples, len(examples))
    return examples


@exclusive_output("triage_dir")
def create_triaged_dataset(triage_dir: Path, name: str | None = None, *, dataset_id: str | None = None, confirm: bool, runner=_run) -> dict:
    if not confirm:
        raise PipelineError("importing into a LangSmith dataset requires --confirm")
    name, dataset_id = _destination(name, dataset_id)
    frozen = load_snapshot(triage_dir)
    examples = selected_examples(triage_dir, frozen=frozen)
    workspace = frozen["source"]["workspace_id"]
    receipt_path = triage_dir / "dataset-import.json"
    from smithtune.dataset_import import import_dataset

    keys = {_source_key(example, workspace, None) for example in examples}
    saved_inputs = {_source_key(unit["example"], workspace, None): triage_dir / path
                    for unit, path in zip(frozen["units"], frozen.get("unit_files", []), strict=False)}
    return import_dataset(workspace, examples, keys, triage_dir, receipt_path, name=name, dataset_id=dataset_id,
                          runner=runner, triaged=True, saved_inputs=saved_inputs)


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

"""Local whole-conversation council votes, saved once per judge slot."""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from threading import Lock

from smithtune.artifacts import _jsonl_dump, _load_json
from smithtune.checkpoint import CheckpointError, conversations, load, pull_complete, save, votes
from smithtune.bindings import evidence_hash
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError
from smithtune.triage_judges import PROVIDERS, api_judge, check_credentials, context_window_exceeded, deepagent_judge, judge_messages, rubric_text, validate_judgment
from smithtune.triage_source import multimodal_types


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


def council_settings(output_dir: Path, *, judges=None, rules=None, config_path=None, **overrides) -> dict:
    """Reuse the preview on confirm/resume; explicit options replace defaults."""
    path = output_dir / "checkpoint.json"
    saved = (load(output_dir).get("council") or {}) if path.exists() else {}
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
                "runner_mode": saved.get("runner_mode", "deepagent"),
                "concurrency": saved.get("concurrency", 4),
                "attempts": saved.get("attempts", 3),
                "max_output_tokens": saved.get("max_output_tokens", 4096)}
    settings.update({key: value for key, value in overrides.items() if value is not None})
    settings["rubric"] = saved.get("rubric", rubric_text())
    if settings["runner_mode"] not in {"api", "deepagent"} or not 1 <= settings["concurrency"] <= 16 or not 1 <= settings["attempts"] <= 5 or not 128 <= settings["max_output_tokens"] <= 16384:
        raise CheckpointError("invalid council runner, concurrency, attempts, or max output tokens")
    if (output_dir / "triage.jsonl").exists() and (output_dir / "triage.jsonl").stat().st_size:
        for key in ("config", "rubric", "max_output_tokens"):
            if settings[key] != saved.get(key):
                raise CheckpointError("council models, rubric, rules and request settings are frozen after voting starts; use a new checkpoint")
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


def labels(directory, checkpoint):
    examples = conversations(directory, checkpoint)
    records = votes(directory, checkpoint, examples)
    council = checkpoint.get("council")
    if council is None:
        raise CheckpointError("triage settings have not been saved")
    return [_label({"trajectory_id": tid, "messages": example["inputs"]["messages"], "multimodal_types": []},
                   records, council["config"]["judges"]) for tid, (example, _) in examples.items()]


def triage_summary(directory, checkpoint):
    calculated = labels(directory, checkpoint)
    incomplete = sum(label["status"] == "incomplete" for label in calculated)
    results = [_result(label) for label in calculated]
    return {"status": "incomplete" if incomplete else "complete", "trajectories": len(calculated),
            "kept": sum(label["keep"] for label in calculated),
            "dropped": sum(label["status"] == "complete" and not label["keep"] for label in calculated),
            "incomplete": incomplete, "filtered_context": sum(label["status"] == "context_exceeded" for label in calculated),
            "rejected": [{"scope_id": s["scope_id"], "reason": s["reason"]} for s in checkpoint["selection"] if s["status"] == "excluded"],
            "results": results}


def run_triage(directory, checkpoint, *, confirm=False, judge_call=None, sleeper=time.sleep):
    """Judge only local files. The command, not this stage, owns the lock."""
    if not pull_complete(checkpoint):
        raise PipelineError("pull is incomplete; finish dataset pull before triage")
    settings = checkpoint["council"]
    config, rubric = settings["config"], settings["rubric"]
    runner_mode, attempts = settings["runner_mode"], settings["attempts"]
    max_output_tokens, concurrency = settings["max_output_tokens"], settings["concurrency"]
    examples = conversations(directory, checkpoint)
    records = votes(directory, checkpoint, examples)
    trajectories = [{"trajectory_id": tid, "messages": example["inputs"]["messages"],
                     "assistant_runs": example["metadata"]["smithtune_source"]["assistant_runs"],
                     "conversation_sha256": digest} for tid, (example, digest) in examples.items()]
    context_filtered = {r["trajectory_id"] for r in records.values() if r["status"] == "context_exceeded"}
    pending = [(trajectory, judge) for trajectory in trajectories for judge in config["judges"]
               if trajectory["trajectory_id"] not in context_filtered and records.get((trajectory["trajectory_id"], judge["name"]), {}).get("status") != "complete"]
    if not confirm:
        return {**triage_summary(directory, checkpoint), "status": "preview", "candidates": len(trajectories),
                "pending_votes": len(pending), "accepted": None if pending else sum(label["keep"] for label in labels(directory, checkpoint))}
    if pending and judge_call is None:
        check_credentials(config["judges"])
        if runner_mode == "deepagent":
            from smithtune.triage_agent import check_installation
            check_installation()
    save(directory, checkpoint)
    path = directory / "triage.jsonl"
    if not path.exists():
        _jsonl_dump(path, [])
    call = judge_call or (deepagent_judge if runner_mode == "deepagent" else api_judge)

    def judge_one(trajectory, judge):
        record = {"trajectory_id": trajectory["trajectory_id"], "judge": judge["name"],
                  "conversation_sha256": trajectory["conversation_sha256"]}
        messages = judge_messages(trajectory, rubric, config["rules"])
        for attempt in range(attempts):
            try:
                judgment = validate_judgment(call(judge, messages, max_output_tokens))
                return {**record, "status": "complete", "judgment": judgment}
            except Exception as exc:
                if context_window_exceeded(exc):
                    return {**record, "status": "context_exceeded", "error": f"Full trajectory exceeds the context window of {judge['model']}."}
                error = type(exc).__name__
                if attempt + 1 < attempts:
                    sleeper(2 ** attempt)
        return {**record, "status": "error", "error_kind": error, "error": "judge request or result validation failed; resume to retry"}

    record_lock = Lock()

    def save_record(record):
        with record_lock:
            records[(record["trajectory_id"], record["judge"])] = record
            _jsonl_dump(path, [records[key] for key in sorted(records)])

    if runner_mode == "deepagent" and pending and judge_call is None:
        from smithtune.triage_coordinator import coordinate
        coordinate(pending, judge_one, save_record, concurrency=concurrency,
                   max_tokens=max_output_tokens, coordinator_judge=config["judges"][0])
    elif pending:
        _run_direct(pending, judge_one, save_record, concurrency)
    summary = triage_summary(directory, checkpoint)
    print(f"Council: {summary['kept']} kept, {summary['dropped']} dropped, {summary['incomplete']} incomplete, "
          f"{summary['filtered_context']} context exclusions. Reasons are in the command result.", file=sys.stderr)
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


def selected_examples(directory, checkpoint):
    """Refuse incomplete councils; attach compact evidence-bound triage provenance."""
    if not pull_complete(checkpoint):
        raise PipelineError("pull is incomplete")
    examples = conversations(directory, checkpoint)
    if "triage" not in checkpoint["stages"]:
        return [example for example, _ in examples.values()]
    calculated = labels(directory, checkpoint)
    if any(label["status"] == "incomplete" for label in calculated):
        raise PipelineError("council work is incomplete; resume triage before push")
    selected = []
    for label in calculated:
        if not label["keep"]:
            continue
        example = copy.deepcopy(examples[label["trajectory_id"]][0])
        example["metadata"]["smithtune_triage"] = {
            "messages_sha256": json_sha256(example["inputs"]["messages"]),
            "evidence_sha256": evidence_hash(example),
            "council": {k: checkpoint["council"][k] for k in ("config", "rubric", "max_output_tokens")},
            "votes": [{"judge": v["judge"], **v["judgment"]} for v in label["votes"]],
        }
        selected.append(example)
    return selected


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

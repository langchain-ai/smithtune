"""Compose curation stages over the same frozen files and upload receipts."""

import copy
import shlex
import sys
from collections import Counter
from pathlib import Path

from smithtune import checkpoint as storage, dataset_import, triage, triage_source
from smithtune.artifacts import _load_json, _run, output_lock
from smithtune.curation import _destination, _time, _uuid
from smithtune.dataset import _malformed_trajectory_reason, _source_key
from smithtune.dataset_artifacts import LazySequence, new_run_directory
from smithtune.providers.base import PipelineError


STAGES = ("pull", "triage", "push")
SOURCE_FLAGS = ("workspace_id", "project_id", "start_time", "end_time", "filter", "target_count", "max_candidates")
COUNCIL_FLAGS = ("judges", "rules", "config_path", "rubric_path", "runner_mode", "concurrency", "attempts", "max_output_tokens")


def _open(directory, command, options):
    if (directory / "checkpoint.json").exists():
        checkpoint = storage.load(directory)
        if checkpoint["kind"] != "triage":
            raise PipelineError("unsupported curation checkpoint; use a new directory")
    elif (directory / "snapshot.json").exists():
        # Existing judged snapshots retain their evidence and vote identities.
        source = triage_source.load_snapshot(directory)["source"]
        checkpoint = storage.open_checkpoint(directory, "triage", source)
    else:
        if command != "pull" or not all(options.get(key) for key in ("workspace_id", "project_id")):
            raise PipelineError("start with dataset pull DIR --workspace-id WORKSPACE --project-id PROJECT")
        if any(path.name != ".smithtune.lock" for path in directory.iterdir()):
            raise PipelineError("new curation requires an empty directory")
        source = triage_source.source_options(options["workspace_id"], options["project_id"],
            options.get("start_time"), options.get("end_time"), filter=options.get("filter"),
            target_count=options.get("target_count") if options.get("target_count") is not None else 100,
            max_candidates=options.get("max_candidates") if options.get("max_candidates") is not None else 1000,
            review_mode="none" if options.get("no_triage") else "council")
        source["selection_mode"] = "trajectories"
        checkpoint = storage.open_checkpoint(directory, "triage", source)
    if options.get("no_triage") and checkpoint["source"].get("review_mode") != "none":
        raise PipelineError("--no-triage conflicts with the saved review mode; use a new directory")
    for key in SOURCE_FLAGS:
        value = options.get(key)
        if value is not None:
            value = _time(value) if key in {"start_time", "end_time"} else value
            value = _uuid(value, key) if key in {"workspace_id", "project_id"} else value
            if value != checkpoint["source"].get(key):
                raise PipelineError(f"--{key.replace('_', '-')} conflicts with the frozen selection; use a new directory")
    return checkpoint


def _settings(directory, checkpoint, command, options):
    state = copy.deepcopy(checkpoint.get("workflow") or {"stages": ["pull"], "destination": {}, "council": None})
    concurrency = options.get("concurrency")
    if concurrency is not None and (type(concurrency) is not int or not 1 <= concurrency <= 16):
        raise PipelineError("concurrency must be between 1 and 16; downloads cap at 4")
    if command == "pull":
        state["download_concurrency"] = min(concurrency, 4) if concurrency is not None else state.get("download_concurrency", 4)
    # Older triage checkpoints include council intent, even if downloading stopped before the plan.
    if not checkpoint.get("workflow") and checkpoint["source"].get("selection_mode") != "trajectories":
        state["stages"].append("triage")
        state["council"] = triage.council_settings(directory)
    if not checkpoint.get("workflow") and (directory / "dataset-import.json").exists():
        receipt = _load_json(directory / "dataset-import.json")
        if receipt.get("schema_version") != 2:
            raise PipelineError("legacy upload receipt needs inspection; use a new directory with --dataset-id")
        state["stages"].append("push")
        state["destination"] = {"name": receipt.get("dataset_name"),
                                "dataset_id": None if receipt.get("dataset_name") else receipt["dataset_id"]}
    review_mode = checkpoint["source"].get("review_mode")
    if review_mode == "none" and command == "triage":
        raise PipelineError("this directory uses --no-triage; use a new directory for council review")
    if review_mode == "council" and "triage" not in state["stages"]:
        state["stages"].append("triage")
    has_council_options = command == "triage" and any(options.get(key) is not None for key in COUNCIL_FLAGS)
    stages = set(state["stages"])
    if command in STAGES:
        stages.add(command)
    if "triage" in stages and "triage" not in state["stages"] and (directory / "dataset-import.json").exists():
        raise PipelineError("cannot add council judging after upload started; use a new directory")
    state["stages"] = [stage for stage in STAGES if stage in stages]
    name, dataset_id = options.get("name"), options.get("dataset_id")
    if name is not None or dataset_id is not None:
        name, dataset_id = _destination(name, dataset_id)
        destination = {"name": name, "dataset_id": dataset_id}
        if state["destination"] and destination != state["destination"]:
            raise PipelineError("destination conflicts with saved settings; use a new directory")
        state["destination"] = destination
    if "triage" in stages:
        if state["council"] is None or has_council_options:
            settings = triage.council_settings(directory, saved_settings=state["council"], **({key: options.get(key) for key in COUNCIL_FLAGS} if command == "triage" else {}))
            if state["council"] is not None and settings != state["council"] and (directory / "triage-config.json").exists():
                raise PipelineError("council settings conflict with saved votes; use a new directory")
            state["council"] = settings
        state["selection"] = {"mode": "council", "reason": "Filters select candidates; the council applies the judging criteria."}
    else:
        state["selection"] = {"mode": "filters" if checkpoint["source"].get("filter") else "unreviewed",
                              "reason": "Use the explicit source selection without model calls."}
    checkpoint["workflow"] = state
    storage.save(directory, checkpoint)
    return state


def _pending(directory, state):
    pending = []
    snapshot_path = directory / "snapshot.json"
    snapshot = _load_json(snapshot_path) if snapshot_path.exists() else {}
    collection = storage.load(directory).get("collection", {})
    if not snapshot or (collection and collection["round"] != snapshot.get("selection_result", {}).get("round")):
        pending.append("pull")
    if "triage" in state["stages"]:
        summary = _load_json(directory / "summary.json") if (directory / "summary.json").exists() else {}
        if pending or summary.get("status") != "complete" or (snapshot.get("source", {}).get("review_mode") and summary.get("snapshot_sha256") != snapshot.get("snapshot_sha256")):
            pending.append("triage")
    if "push" in state["stages"]:
        receipt = _load_json(directory / "dataset-import.json") if (directory / "dataset-import.json").exists() else {}
        if pending or (receipt.get("status") != "complete" and not state.get("empty_push")):
            pending.append("push")
    return pending


def _download_summary(frozen):
    reasons = Counter()
    threads = traces = 0
    for unit in frozen["units"]:
        threads += bool(unit["example"].get("metadata", {}).get("source_thread_id"))
        traces += len(unit["trace_ids"])
        error = triage_source.training_error(unit)
        if unit.get("multimodal_types"):
            reasons["multimodal"] += 1
        elif error:
            reason = _malformed_trajectory_reason(unit["example"]["id"], PipelineError(error))
            if reason is None:
                if "trajectory fetch limit" in error:
                    reason = "trajectory_fetch_limit"
                elif "returned no messages" in error:
                    reason = "missing_messages"
                elif "unknown tool availability" in error:
                    reason = "missing_tool_availability"
                elif "unsupported available_tools" in error:
                    reason = "unsupported_tool_definitions"
                elif "evidence from traces outside its source" in error:
                    reason = "foreign_trace_evidence"
                else:
                    reason = "other_structural_or_tool_error"
            reasons[reason] += 1
    downloaded = len(frozen["units"])
    excluded = sum(reasons.values())
    summary = {"selected_roots": len(frozen["selected_trace_ids"]), "threads": threads,
               "standalone_traces": downloaded - threads, "traces": traces,
               "structurally_usable": downloaded - excluded, "excluded": excluded,
               "exclusion_reasons": dict(sorted(reasons.items()))}
    print(f"Selected roots: {summary['selected_roots']}. Full threads: {threads}; "
          f"standalone traces: {summary['standalone_traces']}; total traces: {traces}.", file=sys.stderr)
    if threads:
        print("Filters and time bounds select roots; full threads can include other runs outside those criteria.", file=sys.stderr)
    print(f"Downloaded {downloaded} trajectories: {summary['structurally_usable']} structurally usable, "
          f"{excluded} excluded. Model-specific checks run during prepare.", file=sys.stderr)
    if reasons:
        print("Exclusions: " + "; ".join(f"{count} {reason.replace('_', ' ')}"
                                        for reason, count in summary["exclusion_reasons"].items()) + ".", file=sys.stderr)
        print("Per-trajectory errors and source IDs are saved in the files referenced by snapshot.json.", file=sys.stderr)
    if selection := frozen.get("selection_result"):
        summary.update(selection)
        reason = {"target_reached": "Target reached.", "candidate_cap": "Stopped at the candidate cap.",
                  "source_exhausted": "No more matching candidates in the selected time window."}[selection["stop_reason"]]
        print(f"Examined {selection['examined']} candidates; saved {selection['usable']} structurally usable "
              f"trajectories. {reason}", file=sys.stderr)
    return summary


def _examples(directory, frozen, state):
    if "triage" in state["stages"]:
        return triage.selected_examples(directory, frozen=frozen, require_complete=True, allow_empty=True)
    indices = [index for index, unit in enumerate(frozen["units"])
               if not unit.get("multimodal_types") and not triage_source.training_error(unit)]
    return LazySequence(len(indices), lambda index: frozen["units"][indices[index]]["example"])


def _push(directory, frozen, state, *, confirm, runner):
    examples = _examples(directory, frozen, state)
    result = {"status": "preview" if not confirm else "complete", "eligible": len(examples),
              "rejected": sum(bool(unit.get("multimodal_types") or triage_source.training_error(unit)) for unit in frozen["units"])}
    if not examples:
        if confirm:
            state["empty_push"] = True
        return {**result, "example_count": 0}
    destination = state["destination"]
    if not destination:
        return {**result, "status": "incomplete" if confirm else "preview",
                "next_command": f"smithtune dataset push {shlex.quote(str(directory))} --name NAME --confirm"}
    workspace = frozen["source"]["workspace_id"]
    keys = {_source_key(example, workspace, None) for example in examples}
    triaged = "triage" in state["stages"]
    if not confirm:
        index = dataset_import._destination_index(workspace, destination["dataset_id"], keys, directory, runner=runner) if destination.get("dataset_id") else {}
        counts = dict.fromkeys(("created", "updated", "skipped"), 0)
        from smithtune.dataset_artifacts import load_conversation
        for example in examples:
            _, path = index.get(_source_key(example, workspace, None), (None, None))
            action, _ = dataset_import._action(example, load_conversation(path) if path else None, triaged=triaged)
            counts[action] += 1
        return {**result, **counts}
    saved_inputs = {_source_key(unit["example"], workspace, None): directory / path
                    for unit, path in zip(frozen["units"], frozen.get("unit_files", []), strict=False)}
    imported = dataset_import.import_dataset(workspace, examples, keys, directory, directory / "dataset-import.json",
        runner=runner, triaged=triaged, saved_inputs=saved_inputs, validation=lambda _: None, **destination)
    return {**imported, **result}


def _collection_status(directory, frozen, state):
    source = frozen["source"]
    mode = source.get("review_mode")
    if mode is None:
        return None
    selection = frozen["selection_result"]
    summary = _load_json(directory / "summary.json") if (directory / "summary.json").exists() else {}
    approved = summary.get("eligible_conversations", 0) if mode == "council" else selection["usable"]
    review_complete = mode == "none" or (summary.get("status") == "complete" and summary.get("snapshot_sha256") == frozen["snapshot_sha256"])
    if "pull" in _pending(directory, state):
        status = "downloading"
    elif not review_complete:
        status = "needs_review"
    elif approved >= source["target_count"]:
        status = "target_reached"
    elif selection["round"] >= triage_source.MAX_COLLECTION_ROUNDS:
        status = "round_limit"
    elif selection["source_exhausted"]:
        status = "source_exhausted"
    else:
        status = "needs_candidates"
    active_round = storage.load(directory)["collection"]["round"] if status == "downloading" else selection["round"]
    return {"status": status, "round": active_round, "eligible": approved,
            "target_count": source["target_count"], "review_mode": mode}


def _collection_guidance(directory, collection):
    quoted = shlex.quote(str(directory))
    count, target = collection["eligible"], collection["target_count"]
    prefix = (f"Council approved {count} of the requested {target} trajectories." if collection["review_mode"] == "council"
              else f"{count} of the requested {target} trajectories passed structural checks. No council review was performed.")
    status = collection["status"]
    if status == "needs_review":
        return "Candidate pool ready for council review.", f"smithtune dataset triage {quoted}"
    if status == "downloading":
        return "Collection is unfinished; resume the current round.", f"smithtune dataset resume {quoted} --confirm"
    if status == "needs_candidates":
        return prefix + f" This candidate pool is exhausted. Run smithtune dataset pull {quoted} to collect more candidates.", f"smithtune dataset pull {quoted}"
    if status == "round_limit":
        return (prefix + f" The collection limit has been reached after {triage_source.MAX_COLLECTION_ROUNDS} rounds. "
                "Your eligible trajectories are saved and can be uploaded. To pursue a larger dataset, "
                "start a new curation run with broader source criteria."), None
    if status == "source_exhausted":
        return (prefix + " No unseen matching candidates remain in the selected time window. "
                "Your eligible trajectories are saved and can be uploaded. To collect more, start a new "
                "curation run with broader source criteria."), None
    return prefix + " Target reached.", None


def run(command, directory=None, *, confirm=False, runner=_run, judge_call=None, **options):
    if command not in (*STAGES, "resume"):
        raise PipelineError(f"unknown dataset command: {command}")
    if directory is None:
        if command != "pull":
            raise PipelineError(f"dataset {command} requires a saved directory")
        directory = new_run_directory()
    directory = Path(directory)
    print(f"Dataset directory: {directory}", file=sys.stderr)
    with output_lock(directory):
        checkpoint = _open(directory, command, options)
        state = _settings(directory, checkpoint, command, options)
        frozen = triage_source.load_snapshot(directory) if (directory / "snapshot.json").exists() else None
        pending = _pending(directory, state)
        if frozen is not None and "triage" in state["stages"] and "triage" not in pending:
            _examples(directory, frozen, state)  # Verify completed votes even for a status-only resume.
        result = {"run_dir": str(directory), "source": checkpoint["source"], "destination": state["destination"],
                  "selection": state["selection"], "pending_stages": pending, "status": "complete"}
        if command == "resume" and not confirm:
            result["status"] = "preview" if pending else "complete"
        else:
            stages = state["stages"] if command == "resume" else [command]
            for stage in stages:
                if command == "resume" and stage == "triage" and stage not in pending:
                    result["triage"] = _load_json(directory / "summary.json")
                    continue
                if stage == "pull":
                    collection = _collection_status(directory, frozen, state) if frozen is not None else None
                    extend = command == "pull" and collection is not None and collection["status"] == "needs_candidates"
                    if extend and (directory / "dataset-import.json").exists():
                        raise PipelineError("cannot collect more candidates after upload started; use a new directory")
                    if frozen is None or "pull" in pending or extend:
                        frozen = triage_source.snapshot(checkpoint["source"], directory, runner=runner,
                                                       concurrency=state.get("download_concurrency", 4), extend=extend)
                    result.update(downloaded=len(frozen["units"]), download_summary=_download_summary(frozen))
                elif frozen is None or "pull" in _pending(directory, state):
                    raise PipelineError("download is incomplete; run dataset resume DIR --confirm or dataset pull DIR first")
                elif stage == "triage":
                    council = triage._run_triage(checkpoint["source"], directory, dry_run=not confirm, confirm=confirm,
                                                runner=runner, judge_call=judge_call, frozen=frozen, **state["council"])
                    result["triage"] = council
                    if not confirm or council["status"] != "complete":
                        result["status"] = "preview" if not confirm else "incomplete"
                        break
                else:
                    if "triage" in pending and "triage" in _pending(directory, state):
                        raise PipelineError("council judging is incomplete; run dataset resume DIR --confirm before pushing")
                    result.update(_push(directory, frozen, state, confirm=confirm, runner=runner))
        if state["selection"]["mode"] == "council":
            council = result.get("triage", {})
            approved = council.get("eligible_conversations") if council.get("status") == "complete" else result.get("eligible")
            if approved is not None and 0 < approved < 100:
                noun = "trajectory" if approved == 1 else "trajectories"
                message = (f"Council approved {approved} {noun} for training. With a small dataset, training gains "
                           "and evaluation results can still vary substantially. We recommend collecting more "
                           "trajectories to improve reliability.")
                result["advisories"] = [message]
                print(message, file=sys.stderr)
        if frozen is not None and checkpoint["source"].get("review_mode"):
            collection = _collection_status(directory, frozen, state)
            message, next_command = _collection_guidance(directory, collection)
            result.update(collection=collection, message=message)
            print(message, file=sys.stderr)
            if next_command:
                result.setdefault("next_command", next_command)
        # Reload download progress rather than overwriting it with the pre-pull object.
        checkpoint = storage.load(directory)
        checkpoint["workflow"] = state
        storage.save(directory, checkpoint)
        result["pending_stages"] = _pending(directory, state)
        if result["pending_stages"]:
            result.setdefault("next_command", f"smithtune dataset resume {shlex.quote(str(directory))} --confirm")
        return result

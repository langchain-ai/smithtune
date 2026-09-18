"""Compose curation stages over the same frozen files and upload receipts."""

import copy
import shlex
import sys
from pathlib import Path

from smithtune import checkpoint as storage, dataset_import, triage, triage_source
from smithtune.artifacts import _load_json, _run, output_lock
from smithtune.curation import MAX_LIMIT, _destination, _time, _uuid
from smithtune.dataset import _source_key
from smithtune.dataset_artifacts import new_run_directory
from smithtune.providers.base import PipelineError


STAGES = ("pull", "triage", "push")
SOURCE_FLAGS = ("workspace_id", "project_id", "start_time", "end_time", "filter", "limit")
COUNCIL_FLAGS = ("judges", "rules", "config_path", "runner_mode", "concurrency", "attempts", "max_output_tokens")


def _open(directory, command, options):
    if (directory / "checkpoint.json").exists():
        checkpoint = storage.load(directory)
        if checkpoint["kind"] != "triage":
            raise PipelineError("this directory uses the older direct-import layout; use dataset resume DIR to finish it")
    elif (directory / "snapshot.json").exists():
        # Existing judged snapshots retain their evidence and vote identities.
        source = triage_source.load_snapshot(directory)["source"]
        checkpoint = storage.open_checkpoint(directory, "triage", source)
    else:
        if command not in {"pull", "create"} or not all(options.get(key) for key in ("workspace_id", "project_id")):
            raise PipelineError("start with dataset pull DIR --workspace-id WORKSPACE --project-id PROJECT")
        if any(path.name != ".smithtune.lock" for path in directory.iterdir()):
            raise PipelineError("new curation requires an empty directory")
        limit = options.get("limit") if options.get("limit") is not None else 100
        if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
            raise PipelineError(f"limit must be between 1 and {MAX_LIMIT}")
        source = triage_source.source_options(options["workspace_id"], options["project_id"],
            options.get("start_time"), options.get("end_time"), filter=options.get("filter"), limit=limit)
        source["selection_mode"] = "trajectories"
        checkpoint = storage.open_checkpoint(directory, "triage", source)
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
    if command in {"pull", "create"}:
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
    has_council_options = any(options.get(key) is not None for key in COUNCIL_FLAGS)
    has_criteria = any(options.get(key) is not None for key in ("judges", "rules", "config_path", "runner_mode"))
    if options.get("no_triage") and (any(options.get(key) is not None for key in COUNCIL_FLAGS if key != "concurrency") or "triage" in state["stages"]):
        raise PipelineError("--no-triage conflicts with council judging or rules; use a new directory")
    stages = set(state["stages"])
    if command == "create":
        if "push" not in stages:
            wants_council = not options.get("no_triage") and (has_criteria or not checkpoint["source"].get("filter"))
            stages.update(("pull", "push"))
            if wants_council:
                stages.add("triage")
        elif has_criteria:
            stages.add("triage")
    elif command in STAGES:
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
            settings = triage.council_settings(directory, saved_settings=state["council"], **{key: options.get(key) for key in COUNCIL_FLAGS})
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
    if not (directory / "snapshot.json").exists():
        pending.append("pull")
    if "triage" in state["stages"]:
        summary = _load_json(directory / "summary.json") if (directory / "summary.json").exists() else {}
        if pending or summary.get("status") != "complete":
            pending.append("triage")
    if "push" in state["stages"]:
        receipt = _load_json(directory / "dataset-import.json") if (directory / "dataset-import.json").exists() else {}
        if pending or (receipt.get("status") != "complete" and not state.get("empty_push")):
            pending.append("push")
    return pending


def _examples(directory, frozen, state):
    if "triage" in state["stages"]:
        return triage.selected_examples(directory, frozen=frozen, require_complete=True, allow_empty=True)
    return [unit["example"] for unit in frozen["units"] if not triage_source.training_error(unit)]


def _push(directory, frozen, state, *, confirm, runner):
    examples = _examples(directory, frozen, state)
    result = {"status": "preview" if not confirm else "complete", "eligible": len(examples),
              "rejected": sum(bool(triage_source.training_error(unit)) for unit in frozen["units"])}
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


def _legacy_resume(directory, checkpoint, *, confirm, runner):
    """Finish a direct import made before the staged command interface."""
    from smithtune.curation import _import_selection

    selections = [path for path in directory.glob("*.json") if path.name != "checkpoint.json" and _load_json(path) == checkpoint["source"]]
    if len(selections) != 1:
        raise PipelineError("cannot locate the original selection file; restore it before resuming")
    selection = selections[0]
    receipt = _load_json(selection.with_suffix(".import.json"))
    if receipt.get("schema_version") != 2:
        raise PipelineError("legacy upload receipt needs inspection; use a new directory with --dataset-id")
    complete = receipt.get("status") == "complete"
    result = {"run_dir": str(directory), "status": "complete" if complete else "preview",
              "pending_stages": [] if complete else ["push"]}
    if confirm:
        destination = {"name": receipt["dataset_name"]} if receipt.get("dataset_name") else {"dataset_id": receipt["dataset_id"]}
        result.update(_import_selection(selection=selection, runner=runner, **destination), status="complete", pending_stages=[])
    if result["pending_stages"]:
        result["next_command"] = f"smithtune dataset resume {shlex.quote(str(directory))} --confirm"
    return result


def run(command, directory=None, *, confirm=False, runner=_run, judge_call=None, **options):
    if directory is None:
        if command not in {"pull", "create"}:
            raise PipelineError(f"dataset {command} requires a saved directory")
        directory = new_run_directory()
    directory = Path(directory)
    print(f"Dataset directory: {directory}", file=sys.stderr)
    with output_lock(directory):
        if command == "resume" and (directory / "checkpoint.json").exists():
            checkpoint = storage.load(directory)
            if checkpoint["kind"] == "create":
                return _legacy_resume(directory, checkpoint, confirm=confirm, runner=runner)
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
            stages = state["stages"] if command in {"create", "resume"} else [command]
            for stage in stages:
                if command in {"create", "resume"} and stage == "triage" and stage not in pending:
                    result["triage"] = _load_json(directory / "summary.json")
                    continue
                if stage == "pull":
                    if frozen is None:
                        frozen = triage_source.snapshot(checkpoint["source"], directory, runner=runner,
                                                       concurrency=state.get("download_concurrency", 4))
                    result.update(downloaded=len(frozen["units"]))
                elif frozen is None:
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
        # Reload download progress rather than overwriting it with the pre-pull object.
        checkpoint = storage.load(directory)
        checkpoint["workflow"] = state
        storage.save(directory, checkpoint)
        result["pending_stages"] = _pending(directory, state)
        if result["pending_stages"]:
            result.setdefault("next_command", f"smithtune dataset resume {shlex.quote(str(directory))} --confirm")
        return result

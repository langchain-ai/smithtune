"""Match saved trajectories to an existing dataset and record every write."""

import json
import sys
from types import SimpleNamespace
from urllib.parse import urlencode

from smithtune.artifacts import _json_dump, _utc_now
from smithtune.curation import _api, _time, _uuid, _write_new
from smithtune.dataset import (
    _malformed_trajectory_reason,
    _recorded_tool_call_reason,
    _source_key,
    capture_example_contracts,
    validate_import_messages,
)
from smithtune.dataset_artifacts import load_conversation, save_conversation
from smithtune.inference_contract import ContractError, json_sha256
from smithtune.providers.base import PipelineError


def import_rejection(example, workspace, *, runner):
    """Return a source-only rejection record before uploading a whole trajectory."""
    source = dict(zip(("workspace", "project", "scope", "scope_id"), _source_key(example, workspace, None), strict=True))
    example = {**example, "id": example.get("id") or json_sha256(source)}
    reason = None
    try:
        messages = validate_import_messages(example)
    except PipelineError as exc:
        reason = _malformed_trajectory_reason(example["id"], exc)
        if reason is None:
            raise
    if reason is None:
        def read(command, *, capture):
            body = json.loads(command[command.index("--body") + 1]) if "--body" in command else None
            value = _api(command[command.index("--workspace") + 1],
                         command[command.index("--method") + 1], command[2], body, runner=runner)
            return SimpleNamespace(stdout=json.dumps(value))

        warnings = []
        contracts = capture_example_contracts(workspace, [example], runner=read, exclusion_warnings=warnings)
        if warnings:
            reason = warnings[0]["reason"]
        else:
            try:
                contracts[example["id"]].validate_messages(messages)
            except ContractError as exc:
                if str(exc).startswith(("cannot resolve schema reference", "cannot validate arguments")):
                    raise PipelineError(f"cannot validate source {source['scope_id']}: {exc}") from exc
                reason = _recorded_tool_call_reason(exc)
    if reason is None:
        return None
    print(f"Warning: excluding {source['scope']} {source['scope_id']} before upload: {reason}", file=sys.stderr)
    return {"code": "invalid_import_trajectory_excluded", "source": source, "reason": reason}


def _destination_index(workspace, dataset_id, source_keys, run_dir, *, runner):
    dataset = _api(workspace, "GET", f"/api/v1/datasets/{dataset_id}", runner=runner)
    if not isinstance(dataset, dict) or dataset.get("id") != dataset_id or dataset.get("data_type") != "kv":
        raise PipelineError("destination must be the requested key-value dataset")
    versions = _api(workspace, "GET", f"/api/v1/datasets/{dataset_id}/versions?limit=1", runner=runner)
    if not isinstance(versions, list) or len(versions) > 1:
        raise PipelineError("destination returned invalid versions")
    if not versions:
        return {}
    # Pin the server's version rather than relying on the client's clock.
    as_of = _time(versions[0].get("as_of") if isinstance(versions[0], dict) else None)
    index, ids, offset = {}, set(), 0
    while True:
        query = urlencode({"dataset": dataset_id, "limit": 100, "offset": offset, "as_of": as_of})
        page = _api(workspace, "GET", f"/api/v1/examples?{query}", runner=runner)
        if not isinstance(page, list) or len(page) > 100:
            raise PipelineError("destination returned an invalid example page")
        for example in page:
            if not isinstance(example, dict) or example.get("dataset_id") != dataset_id:
                raise PipelineError("destination returned an example from another dataset")
            example_id = _uuid(example.get("id"), "destination example ID")
            if example_id in ids:
                raise PipelineError("destination pagination repeated an example")
            ids.add(example_id)
            key = _source_key(example, workspace, None)
            if key in index:
                raise PipelineError(f"destination examples {index[key][0]} and {example_id} have the same source conversation")
            path = save_conversation(run_dir / "destination", example) if key in source_keys else None
            index[key] = (example_id, path)
        if len(page) < 100:
            return index
        offset += len(page)


def _action(incoming, existing, *, triaged):
    inputs = incoming.get("inputs")
    messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if not isinstance(messages, list) or not messages or incoming.get("outputs") not in (None, {}):
        raise PipelineError("incoming example must contain a whole message trajectory without outputs")
    metadata = incoming.get("metadata") or {}
    if metadata.get("smithtune_triage") is not None and not triaged:
        raise PipelineError("triaged updates require --triage-dir")
    if existing is None:
        return "created", {"inputs": inputs, "outputs": None, "metadata": metadata}
    previous = existing.get("inputs")
    previous_messages = previous.get("messages") if isinstance(previous, dict) else None
    if not isinstance(previous_messages, list) or not previous_messages or existing.get("outputs") not in (None, {}):
        raise PipelineError(f"destination example {existing['id']} is not a message trajectory")
    prefix = {**inputs, "messages": messages[:len(previous_messages)]}
    if len(messages) < len(previous_messages) or json_sha256(prefix) != json_sha256(previous):
        raise PipelineError(f"incoming trajectory is shorter or conflicts with destination example {existing['id']}; existing history must be an exact prefix")
    old_metadata = existing.get("metadata") or {}
    same_messages = len(messages) == len(previous_messages)
    if old_metadata.get("smithtune_triage") is not None and not triaged:
        if same_messages:
            return "skipped", None
        raise PipelineError(f"destination example {existing['id']} was triaged; rerun triage on the extended conversation and import with --triage-dir")
    if same_messages and (not triaged or old_metadata.get("smithtune_triage") == metadata.get("smithtune_triage")):
        return "skipped", None
    return "updated", {"inputs": inputs, "outputs": None, "metadata": {**old_metadata, **metadata}}


def update_dataset(workspace, dataset_id, examples, source_keys, run_dir, receipt_path, *, runner, triaged=False):
    """Consume bounded incoming downloads; never retry an ambiguous write."""
    actions_path = receipt_path.with_suffix(".actions.jsonl")
    receipt = {"dataset_id": dataset_id, "status": "indexing", "created": 0, "updated": 0, "skipped": 0, "rejected": 0,
               "actions": str(actions_path), "pending_write": None, "created_at_utc": _utc_now()}
    _write_new(receipt_path, receipt)
    try:
        index = _destination_index(workspace, dataset_id, source_keys, run_dir, runner=runner)
        existing_ids = {item[0] for item in index.values()}
        with actions_path.open("x", encoding="utf-8"):
            pass
        receipt["status"] = "importing"
        _json_dump(receipt_path, receipt)
        seen = set()
        for incoming in examples:
            key = _source_key(incoming, workspace, None)
            if key not in source_keys or key in seen:
                raise PipelineError("incoming conversations must have unique, selected source identities")
            seen.add(key)
            incoming_path = save_conversation(run_dir, incoming)
            incoming = load_conversation(incoming_path)
            example_id, existing_path = index.get(key, (None, None))
            existing = load_conversation(existing_path) if existing_path is not None else None
            action, body = _action(incoming, existing, triaged=triaged)
            rejection = import_rejection(incoming, workspace, runner=runner) if action != "skipped" else None
            if rejection is not None:
                action = "rejected"
            entry = {"action": action, "example_id": example_id, "conversation": str(incoming_path),
                     "source": dict(zip(("workspace", "project", "scope", "scope_id"), key, strict=True)),
                     **(rejection or {})}
            if action in ("created", "updated"):
                receipt["pending_write"] = entry
                _json_dump(receipt_path, receipt)
                if action == "created":
                    result = _api(workspace, "POST", "/api/v1/examples", {**body, "dataset_id": dataset_id}, runner=runner)
                    entry["example_id"] = _uuid(result.get("id") if isinstance(result, dict) else None, "created example ID")
                    if entry["example_id"] in existing_ids:
                        raise PipelineError("destination returned a duplicate example ID")
                    existing_ids.add(entry["example_id"])
                else:
                    result = _api(workspace, "PATCH", f"/api/v1/examples/{example_id}", body, runner=runner)
                    if result != {"message": "Example updated"}:
                        raise PipelineError("example update was not confirmed")
            with actions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            receipt[action] += 1
            receipt["pending_write"] = None
            _json_dump(receipt_path, receipt)
        if seen != source_keys:
            raise PipelineError("incoming download ended before every selected conversation was imported")
        receipt["status"] = "complete"
        _json_dump(receipt_path, receipt)
    except BaseException as exc:
        receipt["status"] = "incomplete"
        _json_dump(receipt_path, receipt)
        detail = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        raise PipelineError(f"{detail}; dataset import incomplete; inspect {receipt_path} before retrying") from exc
    return {"dataset_id": dataset_id, "example_count": sum(receipt[key] for key in ("created", "updated", "skipped")),
            **{key: receipt[key] for key in ("created", "updated", "skipped", "rejected")}, "receipt": str(receipt_path), "run_dir": str(run_dir)}

"""Match saved trajectories to an existing dataset and record every write."""

import json
import shlex
import sys
from types import SimpleNamespace
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid4, uuid5

from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _utc_now
from smithtune.curation import _api, _time, _uuid
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


def import_rejection(example, workspace, *, runner, captured=None):
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
            if captured is not None:
                captured.update(contracts[example["id"]].to_dict())
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


def _payload(example):
    return {"inputs": example.get("inputs"), "outputs": example.get("outputs"), "metadata": example.get("metadata") or {}}


def _lookup(workspace, path, *, runner):
    try:
        return _api(workspace, "GET", path, runner=runner)
    except PipelineError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise


def _resolve_destination(workspace, name, receipt, receipt_path, *, runner):
    if receipt.get("dataset_id"):
        return receipt["dataset_id"]
    pending = receipt.get("pending_write")
    if pending is not None:
        if pending.get("kind") != "dataset" or pending.get("name") != name:
            raise PipelineError("pending dataset creation uses another destination")
        current = _lookup(workspace, f"/api/v1/datasets/{pending['id']}", runner=runner)
        if current is not None:
            if current.get("id") != pending["id"] or current.get("name") != name or current.get("data_type") != "kv":
                raise PipelineError("pending dataset creation conflicts with the recorded destination")
            receipt.update(dataset_id=pending["id"], pending_write=None)
            _json_dump(receipt_path, receipt)
            return pending["id"]
    else:
        pending = {"kind": "dataset", "id": str(uuid4()), "name": name}
        receipt["pending_write"] = pending
        _json_dump(receipt_path, receipt)
    result = _api(workspace, "POST", "/api/v1/datasets", {"id": pending["id"], "name": name, "data_type": "kv"}, runner=runner)
    if not isinstance(result, dict) or result.get("id") != pending["id"]:
        raise PipelineError("dataset creation did not confirm its saved ID; rerun to reconcile")
    receipt.update(dataset_id=pending["id"], pending_write=None)
    _json_dump(receipt_path, receipt)
    return pending["id"]


def import_dataset(workspace, examples, source_keys, run_dir, receipt_path, *, name=None, dataset_id=None,
                   runner, triaged=False, validation=None, saved_inputs=None, on_index=None):
    """Resume frozen uploads, reconciling a pending write by ID before retrying it."""
    request = {"workspace": workspace, "name": name, "dataset_id": dataset_id,
               "sources": sorted(source_keys), "triaged": triaged}
    request_hash = json_sha256(request)
    if receipt_path.exists():
        receipt = _load_json(receipt_path)
        if receipt.get("schema_version") != 2:
            raise PipelineError("legacy import receipt cannot resume automatically; inspect the destination and use a new run directory with --dataset-id")
        if receipt.get("request_sha256") != request_hash or (dataset_id is not None and receipt.get("dataset_id") != dataset_id):
            raise PipelineError("import destination or selection changed; use the original settings or a new run directory")
    else:
        receipt = {"schema_version": 2, "request_sha256": request_hash, "dataset_id": dataset_id,
                   "dataset_name": name, "status": "indexing", "pending_write": None,
                   "outcomes": {}, "created_at_utc": _utc_now()}
        _json_dump(receipt_path, receipt)
    actions_path = receipt_path.with_suffix(".actions.jsonl")

    def save():
        entries = list(receipt["outcomes"].values())
        receipt.update({key: sum(e["action"] == key for e in entries) for key in ("created", "updated", "skipped", "rejected")})
        receipt["actions"] = str(actions_path)
        receipt["rejections"] = [e for e in entries if e["action"] == "rejected"]
        receipt["confirmed_example_ids"] = [e["example_id"] for e in entries if e["action"] != "rejected"]
        # The receipt is authoritative; regenerate the human-readable action log.
        _json_dump(receipt_path, receipt)
        _jsonl_dump(actions_path, entries)

    try:
        was_complete = receipt["status"] == "complete"
        dataset_id = _resolve_destination(workspace, name, receipt, receipt_path, runner=runner)
        # Newly created empty datasets need no index. Resumes query their state.
        index = {} if name and not receipt["outcomes"] and receipt.get("status") == "indexing" and not was_complete else None
        if index is None and not was_complete:
            index = _destination_index(workspace, dataset_id, source_keys, run_dir, runner=runner)
        if on_index is not None:
            on_index(index)
        receipt["status"] = "importing"
        save()
        seen = set()
        for incoming in examples:
            key = _source_key(incoming, workspace, None)
            if key not in source_keys or key in seen:
                raise PipelineError("incoming conversations must have unique, selected source identities")
            seen.add(key)
            identity = json_sha256(key)
            digest = json_sha256(incoming)
            previous = receipt["outcomes"].get(identity)
            if previous is not None:
                if previous["payload_sha256"] != digest:
                    raise PipelineError("saved import trajectory changed; use a new run directory")
                continue
            if was_complete:
                raise PipelineError("completed import is missing a selected source")
            incoming_path = (saved_inputs or {}).get(key)
            if incoming_path is None:
                incoming_path = save_conversation(run_dir, incoming)
                incoming = load_conversation(incoming_path)
            else:
                saved = load_conversation(incoming_path)["example"]
                comparable = {**incoming, "metadata": {k: v for k, v in incoming["metadata"].items() if k != "smithtune_triage"}}
                if comparable != saved:
                    raise PipelineError("saved trajectory differs from import input")
            example_id, existing_path = index.get(key, (None, None))
            existing = load_conversation(existing_path) if existing_path else None
            pending = receipt.get("pending_write")
            if pending is not None:
                if pending.get("identity") != identity or pending.get("payload_sha256") != digest:
                    raise PipelineError("pending upload differs from the saved trajectory; restore the original input")
                example_id = pending["example_id"]
                existing = _lookup(workspace, f"/api/v1/examples/{example_id}", runner=runner)
                if existing is not None:
                    if existing.get("id") != example_id or existing.get("dataset_id") != dataset_id or _source_key(existing, workspace, None) != key:
                        raise PipelineError("pending upload belongs to a different destination or source")
                    current_hash = json_sha256(_payload(existing))
                    if current_hash == pending["body_sha256"]:
                        receipt["outcomes"][identity] = {k: v for k, v in pending.items() if k not in ("before_sha256", "body_sha256", "identity")}
                        receipt["pending_write"] = None
                        save()
                        continue
                    if pending["action"] == "created" or current_hash != pending["before_sha256"]:
                        raise PipelineError("pending upload conflicts with changed destination content; inspect the receipt before retrying")
                elif pending["action"] == "updated":
                    raise PipelineError("pending update destination no longer exists")
            action, body = _action(incoming, existing, triaged=triaged)
            rejection = None
            if action != "skipped":
                rejection = validation(incoming) if validation is not None else import_rejection(incoming, workspace, runner=runner)
            if rejection is not None:
                action = "rejected"
            if pending is not None and action != pending["action"]:
                raise PipelineError("pending upload action changed; inspect the saved receipt")
            if action == "created":
                example_id = str(uuid5(NAMESPACE_URL, dataset_id + ":" + identity))
            entry = {"action": action, "example_id": example_id, "conversation": str(incoming_path), "payload_sha256": digest,
                     "source": dict(zip(("workspace", "project", "scope", "scope_id"), key, strict=True)), **(rejection or {})}
            if action in ("created", "updated"):
                write = {**entry, "identity": identity, "body_sha256": json_sha256(body),
                         "before_sha256": json_sha256(_payload(existing)) if existing else None}
                if pending is not None and write != pending:
                    raise PipelineError("pending upload payload changed; inspect the saved receipt")
                receipt["pending_write"] = write
                save()
                if action == "created":
                    result = _api(workspace, "POST", "/api/v1/examples", {**body, "id": example_id, "dataset_id": dataset_id}, runner=runner)
                    if not isinstance(result, dict) or result.get("id") != example_id:
                        raise PipelineError("example import did not confirm its saved ID; rerun to reconcile")
                else:
                    result = _api(workspace, "PATCH", f"/api/v1/examples/{example_id}", body, runner=runner)
                    if result != {"message": "Example updated"}:
                        raise PipelineError("example update was not confirmed")
            receipt["outcomes"][identity] = entry
            receipt["pending_write"] = None
            save()
        if seen != source_keys:
            raise PipelineError("incoming download ended before every selected conversation was imported")
        if receipt["pending_write"] is not None:
            raise PipelineError("an upload still needs reconciliation")
        receipt["status"] = "complete"
        save()
    except BaseException as exc:
        receipt["status"] = "incomplete"
        save()
        detail = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        raise PipelineError(f"{detail}; dataset import incomplete; run smithtune dataset resume {shlex.quote(str(run_dir))} --confirm; receipt={receipt_path}") from exc
    return {"dataset_id": dataset_id, "example_count": sum(receipt[key] for key in ("created", "updated", "skipped")),
            **{key: receipt[key] for key in ("created", "updated", "skipped", "rejected")}, "receipt": str(receipt_path), "run_dir": str(run_dir)}


def update_dataset(workspace, dataset_id, examples, source_keys, run_dir, receipt_path, *, runner, triaged=False, validation=None, saved_inputs=None):
    return import_dataset(workspace, examples, source_keys, run_dir, receipt_path, dataset_id=dataset_id,
                          runner=runner, triaged=triaged, validation=validation, saved_inputs=saved_inputs)

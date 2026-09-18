"""Sequential, reconciled uploads from a local curation checkpoint."""

from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid4, uuid5

from smithtune.artifacts import _run
from smithtune.bindings import read_bindings, validate_bound_messages
from smithtune.checkpoint import save
from smithtune.curation import _api, _time, _uuid
from smithtune.dataset import _source_key
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError
from smithtune.triage import selected_examples


def _destination_index(workspace, dataset_id, source_keys, *, runner):
    dataset = _api(workspace, "GET", f"/api/v1/datasets/{dataset_id}", runner=runner)
    if not isinstance(dataset, dict) or dataset.get("id") != dataset_id or dataset.get("data_type") != "kv":
        raise PipelineError("destination must be the requested key-value dataset")
    versions = _api(workspace, "GET", f"/api/v1/datasets/{dataset_id}/versions?limit=1", runner=runner)
    if not isinstance(versions, list) or len(versions) > 1:
        raise PipelineError("destination returned invalid versions")
    if not versions:
        return {}, 0
    as_of = _time(versions[0].get("as_of"))
    index, ids, offset = {}, set(), 0
    while True:
        query = urlencode({"dataset": dataset_id, "limit": 100, "offset": offset, "as_of": as_of})
        page = _api(workspace, "GET", f"/api/v1/examples?{query}", runner=runner)
        if not isinstance(page, list) or len(page) > 100:
            raise PipelineError("destination returned an invalid example page")
        for example in page:
            if not isinstance(example, dict) or example.get("dataset_id") != dataset_id:
                raise PipelineError("destination returned an example from another dataset")
            eid = _uuid(example.get("id"), "destination example ID")
            if eid in ids:
                raise PipelineError("destination pagination repeated an example")
            ids.add(eid)
            try:
                key = _source_key(example, workspace, None)
            except PipelineError:
                # Unrelated destination examples still count towards final size.
                continue
            if key in index:
                raise PipelineError("destination has multiple examples for one source conversation")
            # Only selected payloads are retained, and only in memory.
            index[key] = example if key in source_keys else None
        if len(page) < 100:
            return index, len(ids)
        offset += len(page)


def _action(incoming, existing, *, triaged):
    validate_bound_messages(incoming)
    inputs, metadata = incoming["inputs"], incoming["metadata"]
    if existing is None:
        return "created", {"inputs": inputs, "outputs": None, "metadata": metadata}
    previous = existing.get("inputs")
    old_messages = previous.get("messages") if isinstance(previous, dict) else None
    messages = inputs["messages"]
    if not isinstance(old_messages, list) or not old_messages or existing.get("outputs") not in (None, {}):
        raise PipelineError(f"destination example {existing['id']} is not a whole message trajectory")
    if len(messages) < len(old_messages) or {**inputs, "messages": messages[:len(old_messages)]} != previous:
        raise PipelineError(f"existing history must be an exact prefix for destination example {existing['id']}")
    old_metadata = existing.get("metadata") or {}
    if "smithtune_source" not in old_metadata:
        raise PipelineError(f"destination example {existing['id']} lacks bindings; a legacy union cannot prove equivalence. Use a fresh dataset or explicitly validate and upgrade its historical provenance first")
    old_bindings = read_bindings(old_metadata, old_messages)
    new_bindings = read_bindings(metadata, messages)
    if any(binding != new_bindings.get(index) for index, binding in old_bindings.items()):
        raise PipelineError(f"prior tool bindings changed for destination example {existing['id']}")
    same = len(messages) == len(old_messages)
    old_triage, new_triage = old_metadata.get("smithtune_triage"), metadata.get("smithtune_triage")
    if old_triage is not None and not triaged:
        raise PipelineError(f"destination example {existing['id']} was triaged; judge the whole incoming conversation before push")
    if same and old_triage == new_triage:
        return "skipped", None
    return "updated", {"inputs": inputs, "outputs": None, "metadata": {**old_metadata, **metadata}}


def _resolve_destination(directory, checkpoint, *, runner):
    destination = checkpoint["destination"]
    if destination.get("dataset_id"):
        return destination["dataset_id"]
    workspace = checkpoint["source"]["workspace_id"]
    name = destination["name"]
    query = urlencode({"name": name, "limit": 100})
    matches = _api(workspace, "GET", f"/api/v1/datasets?{query}", runner=runner)
    if not isinstance(matches, list):
        raise PipelineError("invalid dataset name lookup")
    matches = [d for d in matches if d.get("name") == name]
    pending = checkpoint.get("pending_write")
    if matches:
        if (len(matches) != 1 or not pending or pending.get("kind") != "dataset"
                or matches[0].get("id") != pending.get("id") or matches[0].get("data_type") != "kv"):
            raise PipelineError("unrelated or ambiguous dataset name collision; explicitly supply --dataset-id after checking the destination")
        dataset_id = _uuid(matches[0]["id"], "dataset id")
    else:
        if pending is None:
            pending = {"kind": "dataset", "id": str(uuid4()), "name": name}
            checkpoint["pending_write"] = pending
            save(directory, checkpoint)
        if pending.get("kind") != "dataset" or pending.get("name") != name:
            raise PipelineError("pending dataset creation uses another destination")
        # The server's DatasetCreate schema supports caller-assigned IDs. The
        # saved ID proves attribution after a timeout; name equality alone cannot.
        result = _api(workspace, "POST", "/api/v1/datasets", {"id": pending["id"], "name": name, "data_type": "kv"}, runner=runner)
        if not isinstance(result, dict) or result.get("id") != pending["id"]:
            raise PipelineError("dataset creation did not confirm its saved ID; resume to reconcile")
        dataset_id = pending["id"]
    destination["dataset_id"] = dataset_id
    checkpoint["pending_write"] = None
    save(directory, checkpoint)
    return dataset_id


def push(directory, checkpoint, *, confirm=False, runner=_run):
    examples = selected_examples(directory, checkpoint)
    for example in examples:
        validate_bound_messages(example)
    result = {"status": "preview" if not confirm else "complete", "eligible": len(examples),
              "existing_before": 0, "created": 0, "updated": 0, "skipped": 0,
              "rejected": sum(s["status"] == "excluded" for s in checkpoint["selection"]),
              "final_size": None,
              "limit_note": "--limit caps this checkpoint's selected conversations, not destination size; counts assume no concurrent external writer"}
    if not examples:
        return result
    destination = checkpoint.get("destination")
    if not destination or not any(destination.get(k) for k in ("name", "dataset_id")):
        return {**result, "status": "incomplete" if confirm else "preview", "next_command": f"smithtune dataset push {directory} --name NAME --confirm (or --dataset-id DATASET_ID)"}
    workspace = checkpoint["source"]["workspace_id"]
    keys = {_source_key(e, workspace, None) for e in examples}
    selected_by_key = {(s["scope"], s["scope_id"]): s for s in checkpoint["selection"]}
    source = None
    try:
        dataset_id = destination.get("dataset_id")
        if dataset_id is None and confirm:
            dataset_id = _resolve_destination(directory, checkpoint, runner=runner)
        result["dataset_id"] = dataset_id
        index, count = _destination_index(workspace, dataset_id, keys, runner=runner) if dataset_id else ({}, 0)
        result["existing_before"] = count
        # Validate all known conflicts before the first example write.
        actions = [(incoming, *_action(incoming, index.get(_source_key(incoming, workspace, None)), triaged="triage" in checkpoint["stages"])) for incoming in examples]
        for incoming, action, body in actions:
            key = _source_key(incoming, workspace, None)
            source = list(key)
            existing = index.get(key)
            example_id = existing["id"] if existing else str(uuid5(NAMESPACE_URL, str(dataset_id) + ":" + json_sha256(key)))
            digest = json_sha256(incoming)
            if confirm and action != "skipped":
                pending = {"kind": "example", "id": example_id, "source": source, "payload_sha256": digest, "action": action}
                checkpoint["pending_write"] = pending
                save(directory, checkpoint)
                try:
                    if action == "created":
                        response = _api(workspace, "POST", "/api/v1/examples", {**body, "id": example_id, "dataset_id": dataset_id}, runner=runner)
                        if not isinstance(response, dict) or response.get("id") != example_id:
                            raise PipelineError("example creation did not confirm its ID")
                    else:
                        response = _api(workspace, "PATCH", f"/api/v1/examples/{example_id}", body, runner=runner)
                        if response != {"message": "Example updated"}:
                            raise PipelineError("example update was not confirmed")
                except Exception:
                    # Reconcile timeout/409/invalid confirmation. Never assume
                    # success and never retry a write within this attempt.
                    remote = _api(workspace, "GET", f"/api/v1/examples/{example_id}", runner=runner)
                    if remote.get("dataset_id") != dataset_id or _source_key(remote, workspace, None) != key:
                        raise PipelineError("ambiguous write returned a different source or destination") from None
                    reconciled, _ = _action(incoming, remote, triaged="triage" in checkpoint["stages"])
                    if reconciled != "skipped":
                        raise PipelineError("remote write remains unconfirmed; resume to reconcile") from None
            result[action] += 1
            if confirm:
                selected_by_key[key[2:]]["upload"] = {"example_id": example_id, "outcome": action, "payload_sha256": digest}
                checkpoint["pending_write"] = None
                save(directory, checkpoint)
        result["final_size"] = count + result["created"]
        if result["final_size"] > checkpoint["source"]["limit"]:
            result["warning"] = "final dataset size exceeds this checkpoint's selection limit"
    except Exception as exc:
        if not confirm:
            raise
        result.update(status="incomplete", source=source,
                      error=str(exc) if isinstance(exc, PipelineError) else type(exc).__name__,
                      next_command=f"smithtune dataset resume {directory} --confirm")
    return result


def push_complete(directory, checkpoint):
    if checkpoint.get("pending_write"):
        return False
    try:
        examples = selected_examples(directory, checkpoint)
    except PipelineError:
        return False
    outcomes = {(s["scope"], s["scope_id"]): s.get("upload", {}) for s in checkpoint["selection"]}
    return all(outcomes[(e["metadata"]["source_scope"], e["metadata"]["source_scope_id"])].get("payload_sha256") == json_sha256(e) for e in examples)

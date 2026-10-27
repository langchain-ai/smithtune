"""Portable assistant target selections, indexed into the unchanged native conversation."""

from itertools import pairwise

from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError


SUPERVISION_SCOPES = ("all-assistant", "skip-initial-history")


def assistant_indices(messages):
    if any(not isinstance(message, dict) for message in messages):
        raise PipelineError("supervision requires message objects")
    return [i for i, message in enumerate(messages) if message.get("role") in ("ai", "assistant")]


def _validate_indices(indices, assistants):
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int for i in indices)
            or any(left >= right for left, right in pairwise(indices))
            or not set(indices) <= set(assistants)):
        raise PipelineError("target_message_indices must be nonempty, sorted, unique assistant message indices")
    return indices


def make_supervision(messages, indices, *, scope="explicit", boundary_message_index=None):
    value = {"schema_version": 1, "scope": scope, "target_message_indices": list(indices),
             "messages_sha256": json_sha256(messages)}
    if boundary_message_index is not None:
        value["boundary_message_index"] = boundary_message_index
    target_indices({"supervision": value}, messages)
    return value


def target_indices(metadata, messages):
    """Read and validate a native example. Absent policy retains legacy semantics."""
    assistants = assistant_indices(messages)
    if "supervision" not in metadata:
        return assistants
    policy = metadata["supervision"]
    if (not isinstance(policy, dict) or type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1 or policy.get("scope") not in (*SUPERVISION_SCOPES, "explicit")):
        raise PipelineError("invalid supervision metadata; expected schema_version 1 and a supported scope")
    if policy.get("messages_sha256") != json_sha256(messages):
        raise PipelineError("supervision messages_sha256 does not match the recorded messages")
    selected = _validate_indices(policy.get("target_message_indices"), assistants)
    if policy["scope"] == "all-assistant" and selected != assistants:
        raise PipelineError("all-assistant supervision must select every assistant message")
    if policy["scope"] == "skip-initial-history":
        boundary = policy.get("boundary_message_index")
        if type(boundary) is not int or boundary not in assistants or selected != [i for i in assistants if i >= boundary]:
            raise PipelineError("skip-initial-history supervision has an invalid boundary or target selection")
    return selected


def trajectory_supervision(items, messages, scope):
    """Infer an opt-in boundary from the first contiguous source run-ID group.

    Run IDs identify source observations, not verified message authors. This
    assumes complete chronological items, one output ending the initial group,
    and no later injected assistant history.
    """
    if scope not in SUPERVISION_SCOPES:
        raise PipelineError(f"unsupported supervision scope: {scope}")
    if scope == "all-assistant":
        return make_supervision(messages, assistant_indices(messages), scope=scope)
    identities = []
    for item in items:
        metadata = item.get("metadata")
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise PipelineError("skip-initial-history requires run_id metadata on every trajectory item")
        identities.append(run_id)
    if not identities:
        raise PipelineError("skip-initial-history requires a nonempty trajectory")
    first = identities[0]
    transition = next((i for i, run_id in enumerate(identities) if run_id != first), len(identities))
    boundary = transition - 1
    if first in identities[transition:]:
        raise PipelineError("skip-initial-history is ambiguous: initial run_id reappears after its first group")
    if boundary not in assistant_indices(messages):
        raise PipelineError("skip-initial-history requires the initial run-ID group to end with an assistant message")
    return make_supervision(messages, [i for i in assistant_indices(messages) if i >= boundary],
                            scope=scope, boundary_message_index=boundary)


def row_target_indices(row):
    """Use original positions saved before reasoning conversion, without re-inference."""
    source = row["_source"]
    assistants = [source["message_positions"][i] for i in assistant_indices(row["messages"])]
    policy = source.get("metadata", {}).get("supervision")
    if "target_message_indices" not in source:
        if "supervision" in source.get("metadata", {}):
            raise PipelineError("prepared row lacks resolved target_message_indices; prepare again")
        return assistants
    selected = _validate_indices(source["target_message_indices"], assistants)
    if "supervision" in source.get("metadata", {}) and (
        not isinstance(policy, dict) or selected != policy.get("target_message_indices")
    ):
        raise PipelineError("prepared target_message_indices conflict with saved supervision; prepare again")
    return selected

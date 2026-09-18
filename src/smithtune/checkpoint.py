"""Small curation checkpoint: identities, file references, settings and outcomes."""

from pathlib import Path

from smithtune.artifacts import _json_dump, _load_json, _load_jsonl
from smithtune.dataset_artifacts import load_conversation
from smithtune.providers.base import PipelineError


class CheckpointError(PipelineError):
    """Usage or incompatible local checkpoint (CLI exit 2)."""


def load(directory):
    if any((directory / name).exists() for name in ("selection.json", "snapshot.json")):
        raise CheckpointError("legacy curation directory is incompatible; run smithtune dataset create NEW_DIR --workspace-id ... --project-id ... --dataset-id EXISTING_ID to extend the same destination")
    try:
        value = _load_json(directory / "checkpoint.json")
    except PipelineError as exc:
        raise CheckpointError(str(exc)) from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise CheckpointError("unsupported checkpoint format; use a new checkpoint directory")
    return value


def save(directory, value):
    _json_dump(directory / "checkpoint.json", value)


def conversations(directory, checkpoint):
    result = {}
    for selected in checkpoint["selection"] or []:
        if selected["status"] != "complete":
            continue
        relative = Path(selected["file"])
        if relative.parent != Path("conversations") or relative.suffix != ".json":
            raise CheckpointError("invalid conversation file reference")
        try:
            example = load_conversation(directory / relative)
        except PipelineError as exc:
            raise CheckpointError(str(exc)) from exc
        if example["metadata"]["source_scope_id"] != selected["scope_id"] or example["metadata"]["source_scope"] != selected["scope"]:
            raise CheckpointError("conversation source differs from checkpoint selection")
        if any(example["metadata"].get("source_" + key) != checkpoint["source"][key]
               for key in ("workspace_id", "project_id")):
            raise CheckpointError("conversation workspace/project differs from checkpoint source")
        if example["id"] in result:
            raise CheckpointError("duplicate saved conversation identity")
        result[example["id"]] = (example, relative.stem)
    return result


def votes(directory, checkpoint, examples):
    from smithtune.triage_judges import validate_judgment

    path = directory / "triage.jsonl"
    rows = _load_jsonl(path) if path.exists() else []
    judges = {j["name"] for j in (checkpoint.get("council") or {}).get("config", {}).get("judges", [])}
    result = {}
    for row in rows:
        key = (row.get("trajectory_id"), row.get("judge"))
        if key in result or key[0] not in examples or key[1] not in judges:
            raise CheckpointError("invalid or duplicate vote identity")
        if row.get("conversation_sha256") != examples[key[0]][1]:
            raise CheckpointError("conversation messages or bindings changed after judging; use a new checkpoint")
        if row.get("status") not in {"complete", "error", "context_exceeded"}:
            raise CheckpointError("invalid saved vote status")
        if row["status"] == "complete":
            validate_judgment(row.get("judgment"))
        result[key] = row
    return result


def pull_complete(checkpoint):
    return checkpoint["selection"] is not None and all(s["status"] != "pending" for s in checkpoint["selection"])

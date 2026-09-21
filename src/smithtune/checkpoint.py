"""Content-verified curation downloads shared by direct creation and triage."""

from pathlib import Path

from smithtune.artifacts import _json_dump, _load_json
from smithtune.dataset_artifacts import load_conversation, save_conversation
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError


def load(directory: Path) -> dict:
    value = _load_json(directory / "checkpoint.json")
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("downloads"), dict):
        raise PipelineError("unsupported curation checkpoint; use a new run directory")
    return value


def open_checkpoint(directory: Path, kind: str, source: dict) -> dict:
    path = directory / "checkpoint.json"
    if path.exists():
        value = load(directory)
        if value.get("kind") != kind or value.get("source") != source:
            raise PipelineError("curation checkpoint uses a different source selection; use a new run directory")
        return value
    value = {"schema_version": 1, "kind": kind, "source": source, "downloads": {}}
    save(directory, value)
    return value


def save(directory: Path, value: dict) -> None:
    _json_dump(directory / "checkpoint.json", value)


def read_file(directory: Path, relative: str) -> dict:
    if not isinstance(relative, str):
        raise PipelineError("invalid curation checkpoint file reference")
    path = Path(relative)
    if path.parent != Path("conversations") or path.suffix != ".json" or (directory / path).is_symlink() or (directory / "conversations").is_symlink():
        raise PipelineError("invalid curation checkpoint file reference")
    return load_conversation(directory / path)


def downloaded(directory: Path, checkpoint: dict, key) -> dict | None:
    relative = checkpoint["downloads"].get(json_sha256(key))
    return read_file(directory, relative) if relative is not None else None


def record_download(directory: Path, checkpoint: dict, key, value: dict) -> None:
    path = save_conversation(directory, value)
    checkpoint["downloads"][json_sha256(key)] = str(path.relative_to(directory))
    save(directory, checkpoint)

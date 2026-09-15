"""Local run directories and frozen conversation files shared by dataset paths."""

from pathlib import Path
from uuid import uuid4

from smithtune.artifacts import _json_dump, _load_json
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError


def new_run_directory() -> Path:
    return Path("data/datasets") / str(uuid4())


def load_conversation(path: Path) -> dict:
    example = _load_json(path)
    if json_sha256(example) != path.stem:
        raise PipelineError(f"saved conversation has changed: {path}")
    return example


def save_conversation(run_dir: Path, example: dict) -> Path:
    """Save a complete native example atomically; never replace changed evidence."""
    path = run_dir / "conversations" / f"{json_sha256(example)}.json"
    try:
        if path.exists():
            load_conversation(path)
        else:
            _json_dump(path, example)
    except OSError as exc:
        raise PipelineError(f"cannot save conversation to {path}") from exc
    return path

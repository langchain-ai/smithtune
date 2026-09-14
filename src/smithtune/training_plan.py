"""Persist reviewed training settings and bind them to the prepared dataset."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate

from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers import get_provider
from smithtune.providers.base import CommonSFTSettings, PipelineError, TrainingOptions, TrainingProvider


_PREPARED_FILES = ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
_FLOAT_SETTINGS = {
    "learning_rate", "early_stopping_min_delta", "max_spend_usd",
    "hourly_rate_usd", "spend_reserve_fraction",
}
_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {
        field.name: {"type": (
            ["number" if field.name in _FLOAT_SETTINGS else "integer"]
            + (["null"] if field.default is None else [])
        )}
        for field in fields(TrainingOptions)
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PlannedTraining:
    provider: TrainingProvider
    data_dir: Path
    run_id: str
    settings: CommonSFTSettings
    init_from_checkpoint: str | None


def _prepared_hashes(data_dir: Path) -> dict[str, str]:
    result = {}
    for name in _PREPARED_FILES:
        path = data_dir / "prepared" / name
        try:
            with path.open("rb") as handle:
                result[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as exc:
            raise PipelineError(f"cannot read prepared data {path}: {exc}") from exc
    return result


def save_training_plan(
    provider: TrainingProvider,
    data_dir: Path,
    run_id: str,
    settings: CommonSFTSettings,
    output: Path,
    *,
    init_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    # Model-dependent defaults must be explicit in the saved settings too.
    preview = provider.plan(data_dir, run_id, settings)
    resolved = {
        field.name: preview["config"][field.name]
        for field in fields(settings)
        if getattr(settings, field.name) is None and field.name in preview["config"]
    }
    settings = replace(settings, **resolved)
    value = {
        "schema_version": 1,
        "training": {
            "provider": provider.name,
            "data_dir": str(data_dir),
            "run_id": run_id,
            "settings": asdict(settings),
            "init_from_checkpoint": init_from_checkpoint,
        },
        "prepared_sha256": _prepared_hashes(data_dir),
        "plan": preview,
    }
    if output.resolve().is_relative_to((data_dir / "prepared").resolve()):
        raise PipelineError("save the training plan outside the prepared data directory")
    _training_request(value)
    try:
        _json_dump(output, value)
    except OSError as exc:
        raise PipelineError(f"cannot save training plan to {output}: {exc}") from exc
    return value


def _training_request(value: Any) -> PlannedTraining:
    try:
        validate(value, {
            "type": "object",
            "required": ["schema_version", "training", "prepared_sha256", "plan"],
            "properties": {
                "schema_version": {"type": "integer", "const": 1},
                "training": {
                    "type": "object",
                    "required": ["provider", "data_dir", "run_id", "settings", "init_from_checkpoint"],
                    "properties": {
                        "provider": {"type": "string", "minLength": 1},
                        "data_dir": {"type": "string", "minLength": 1},
                        "run_id": {"type": "string", "minLength": 1},
                        "settings": _SETTINGS_SCHEMA,
                        "init_from_checkpoint": {"type": ["string", "null"], "minLength": 1},
                    },
                    "additionalProperties": False,
                },
                "prepared_sha256": {"type": "object"},
                "plan": {"type": "object"},
            },
            "additionalProperties": False,
        })
    except ValidationError as exc:
        raise PipelineError(f"invalid saved training plan: {exc.message}; regenerate with smithtune plan") from exc
    training = value["training"]
    provider = get_provider(training["provider"])
    values = training["settings"]
    expected = asdict(provider.settings_from_options(TrainingOptions()))
    if values.keys() != expected.keys():
        raise PipelineError("saved training settings are incomplete or incompatible; regenerate with smithtune plan")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values.values()):
        raise PipelineError("saved training settings must be finite")
    # JSON Schema permits 5.0 as an integer; count settings require Python ints.
    if any(item is not None and type(item) is not int for name, item in values.items() if name not in _FLOAT_SETTINGS):
        raise PipelineError("saved training count settings must be integers")
    settings = provider.settings_from_options(TrainingOptions(**values))
    data_dir = Path(training["data_dir"])
    if not data_dir.is_absolute():
        raise PipelineError("saved training data_dir must be absolute; regenerate with smithtune plan")
    return PlannedTraining(provider, data_dir, training["run_id"], settings, training["init_from_checkpoint"])


def load_training_plan(path: Path) -> PlannedTraining:
    value = _load_json(path)
    request = _training_request(value)
    if _prepared_hashes(request.data_dir) != value["prepared_sha256"]:
        raise PipelineError("prepared data changed since planning; regenerate and review with smithtune plan")
    if request.provider.plan(request.data_dir, request.run_id, request.settings) != value["plan"]:
        raise PipelineError("training plan no longer matches its settings or runtime; regenerate and review with smithtune plan")
    return request

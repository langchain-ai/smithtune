"""Load saved training context shared by provider checkpoint adapters."""

from pathlib import Path
from typing import Any

from smithtune.artifacts import _load_json
from smithtune.dataset import _require_prepared_provider
from smithtune.providers.base import ModelSpec, PipelineError


def load_checkpoint_context(
    data_dir: Path, run_dir: Path, *, provider: str,
) -> tuple[ModelSpec, Any, Any]:
    """Read prepared model identity and run records before provider validation."""
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    model = _require_prepared_provider(manifest, provider)
    plan = _load_json(run_dir / "plan.json")
    result = _load_json(run_dir / "result.json")
    return model, plan, result


def validation_at_step(run_dir: Path, step: int) -> dict[str, Any]:
    """Return the saved validation at a global optimizer step for checkpoint selection."""
    if type(step) is not int or step < 1:
        raise PipelineError("--checkpoint-step must be a positive optimizer step")
    path = run_dir / "validations.json"
    validations = _load_json(path) if path.exists() else None
    if not isinstance(validations, list):
        raise PipelineError(f"--checkpoint-step needs the run's saved validations: {path}")
    matches = [row for row in validations if isinstance(row, dict) and row.get("step") == step]
    if len(matches) != 1:
        steps = ", ".join(str(row.get("step")) for row in validations if isinstance(row, dict))
        raise PipelineError(f"no validation at step {step} in {path}; validated steps: {steps or 'none'}")
    return matches[0]


def require_matching_base_model(model: ModelSpec, plan: dict, *, provider_label: str) -> None:
    if plan.get("base_model") != model.base_model:
        raise PipelineError(f"prepared data base model differs from the {provider_label} training run")

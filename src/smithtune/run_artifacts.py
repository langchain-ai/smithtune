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


def require_matching_base_model(model: ModelSpec, plan: dict, *, provider_label: str) -> None:
    if plan.get("base_model") != model.base_model:
        raise PipelineError(f"prepared data base model differs from the {provider_label} training run")

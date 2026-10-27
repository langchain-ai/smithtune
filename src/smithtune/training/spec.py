"""Internal descriptions shared by a training recipe and its provider adapter.

These contracts contain model and artifact identities, not framework or cloud
SDK objects. Describing a job does not qualify its model or allocate resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smithtune.providers.base import CommonSFTSettings, ModelSpec


@dataclass(frozen=True)
class SFTJobSpec:
    """Explicit inputs to a LoRA SFT recipe, independent of its execution host."""

    model: ModelSpec
    model_revision: str
    prepared_manifest_uri: str
    settings: CommonSFTSettings
    lora_rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    target_parameters: tuple[str, ...] = ()

    schema_version: int = field(default=1, init=False)
    method: str = field(default="sft", init=False)


@dataclass(frozen=True)
class TrainingArtifacts:
    """Inference weights and restart state have distinct compatibility rules."""

    selected_model_uri: str
    resume_checkpoint_uri: str | None
    metrics_uri: str
    result_manifest_uri: str

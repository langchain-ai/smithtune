"""Small internal contract for provider-specific SFT execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Protocol

from smithtune.inference_contract import InferenceContract


class PipelineError(ValueError):
    """A local input or state error shared across provider adapters."""


ReasoningPolicy = Literal["omit", "preserve"]


@dataclass(frozen=True)
class ModelSpec:
    """Provider model identity and rendering configuration."""

    name: str
    base_model: str
    tokenizer_model: str
    tokenizer_revision: str
    renderer: str
    max_seq_len: int
    thinking_trace_history_mode: str = ""
    trust_remote_code: bool = False
    default_lora_rank: int = 8
    requires_tool_declarations: bool = False
    provider: str = "fireworks"
    trainer_max_seq_len: int | None = None
    supports_reasoning_content: bool = False
    template_sha256: str = ""
    rendering_version: str = ""

    @property
    def training_context_limit(self) -> int:
        return self.max_seq_len if self.trainer_max_seq_len is None else self.trainer_max_seq_len

    def validate(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.name, self.base_model)):
            raise PipelineError("model name and provider base model are required")
        if not isinstance(self.provider, str) or not self.provider:
            raise PipelineError("model provider is required")
        if not all(
            isinstance(value, str) and value
            for value in (self.tokenizer_model, self.tokenizer_revision, self.renderer)
        ):
            raise PipelineError("tokenizer, pinned revision, and renderer are required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.max_seq_len, self.default_lora_rank)
        ):
            raise PipelineError(
                "context limit and default LoRA rank must be positive integers"
            )
        if self.trainer_max_seq_len is not None and (
            isinstance(self.trainer_max_seq_len, bool)
            or not isinstance(self.trainer_max_seq_len, int)
            or not 1 <= self.trainer_max_seq_len <= self.max_seq_len
        ):
            raise PipelineError("trainer context limit must be positive and no greater than the preparation limit")
        if not isinstance(self.requires_tool_declarations, bool) or not isinstance(self.trust_remote_code, bool):
            raise PipelineError("requires_tool_declarations and trust_remote_code must be boolean")
        if not isinstance(self.supports_reasoning_content, bool):
            raise PipelineError("supports_reasoning_content must be boolean")
        if not isinstance(self.thinking_trace_history_mode, str):
            raise PipelineError("thinking_trace_history_mode must be a string")
        if not isinstance(self.template_sha256, str) or (
            self.template_sha256 and not re.fullmatch(r"[0-9a-f]{64}", self.template_sha256)
        ):
            raise PipelineError("template_sha256 must be empty or a lowercase SHA-256 digest")
        if not isinstance(self.rendering_version, str):
            raise PipelineError("rendering_version must be a string")


@dataclass(frozen=True)
class CommonSFTSettings:
    """Training settings supported by every provider adapter."""

    max_epochs: int = 5
    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 32
    seed: int = 42

    def validate(self) -> None:
        if self.max_epochs < 1:
            raise PipelineError("max_epochs must be positive")
        if self.early_stopping_patience < 1:
            raise PipelineError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise PipelineError("early_stopping_min_delta cannot be negative")
        if self.learning_rate <= 0 or self.batch_size < 1:
            raise PipelineError("learning_rate and batch_size must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PipelineError("seed must be a non-negative integer")


@dataclass(frozen=True)
class ModelOptions:
    """Select a supported provider model and optionally lower its context limit."""

    model_profile: str | None = None
    max_seq_len: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class TrainingOptions(CommonSFTSettings):
    """Parsed training options before provider defaults and validation."""

    lora_rank: int | None = None
    lora_alpha: int | None = None
    pipeline_depth: int | None = None
    microbatch_token_budget: int | None = None
    max_spend_usd: float | None = None
    hourly_rate_usd: float | None = None
    replicas: int | None = None
    spend_reserve_fraction: float | None = None
    max_dropped_training_rows: int | None = None


class TrainingProvider(Protocol):
    """Plan and execute one provider's training lifecycle."""

    name: str

    def model_from_options(self, options: ModelOptions) -> ModelSpec:
        """Resolve the provider's model profile and reject unsupported options."""

    def settings_from_options(self, options: TrainingOptions) -> CommonSFTSettings:
        """Resolve and validate the provider's training configuration."""

    def prepare(
        self,
        workspace_id: str,
        dataset_id: str,
        data_dir: Path,
        *,
        model_options: ModelOptions,
        inference_contract: InferenceContract | None = None,
        reasoning_policy: ReasoningPolicy = "omit",
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fetch: bool = True,
        check_render: bool = True,
    ) -> dict[str, Any]:
        """Apply provider preparation policy to shared trajectory preparation."""

    def plan(self, data_dir: Path, run_id: str, settings: Any) -> dict[str, Any]:
        """Return a reviewable plan without provisioning resources."""

    def train(
        self,
        data_dir: Path,
        run_dir: Path,
        run_id: str,
        settings: Any,
        *,
        confirm: bool,
        init_from_checkpoint: str | None,
    ) -> dict[str, Any]:
        """Execute training and return provider-neutral result metadata."""

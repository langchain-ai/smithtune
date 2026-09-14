"""Resolve complete model configurations from named profiles or explicit options."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Mapping

from smithtune.dataset import _require_prepared_provider
from smithtune.providers.base import ModelOptions, ModelSpec, PipelineError


def resolve_model_options(
    options: ModelOptions,
    profiles: Mapping[str, ModelSpec],
    *,
    provider: str,
) -> ModelSpec:
    if options.model_profile != "custom":
        custom = {
            field.name: getattr(options, field.name)
            for field in fields(options)
            if field.name not in {
                "model_profile", "trust_remote_code", "requires_tool_declarations",
                "supports_reasoning_content",
            }
        }
        if any(value is not None for value in custom.values()) or (
            options.trust_remote_code or options.requires_tool_declarations
            or options.supports_reasoning_content
        ):
            raise PipelineError("custom model fields require --model-profile custom")
        try:
            model = profiles[options.model_profile]
        except KeyError as exc:
            raise PipelineError(f"unknown {provider} model profile: {options.model_profile}") from exc
    else:
        required = {
            "base model": options.base_model,
            "tokenizer model": options.tokenizer_model,
            "tokenizer revision": options.tokenizer_revision,
            "renderer": options.renderer,
            "max sequence length": options.max_seq_len,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            raise PipelineError(f"custom model is missing: {', '.join(missing)}")
        model = ModelSpec(
            name=options.base_model.rsplit("/", 1)[-1],
            base_model=options.base_model,
            tokenizer_model=options.tokenizer_model,
            tokenizer_revision=options.tokenizer_revision,
            renderer=options.renderer,
            max_seq_len=options.max_seq_len,
            trainer_max_seq_len=options.trainer_max_seq_len,
            thinking_trace_history_mode=options.thinking_trace_history_mode or "",
            trust_remote_code=options.trust_remote_code,
            requires_tool_declarations=options.requires_tool_declarations,
            supports_reasoning_content=options.supports_reasoning_content,
            default_lora_rank=8 if options.default_lora_rank is None else options.default_lora_rank,
            provider=provider,
        )
    model.validate()
    if model.provider != provider:
        raise PipelineError(f"model profile belongs to {model.provider}, expected {provider}")
    return model


def resolve_prepared_model(
    manifest: dict[str, Any],
    profiles: Mapping[str, ModelSpec],
    *,
    provider: str,
    allow_legacy: bool = False,
) -> ModelSpec:
    model = _require_prepared_provider(manifest, provider, allow_legacy=allow_legacy)
    # Older built-in manifests predate a distinct trainer limit. Only restore
    # that default when the full recorded profile still matches the preset.
    if "trainer_max_seq_len" not in manifest["model"]:
        preset = profiles.get(model.name)
        if preset is not None and replace(
            preset,
            trainer_max_seq_len=None,
            supports_reasoning_content=model.supports_reasoning_content,
        ) == model:
            model = replace(model, trainer_max_seq_len=preset.trainer_max_seq_len)
    return model

"""Resolve local rendering compatibility separately from provider availability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from smithtune.dataset import _require_prepared_provider
from smithtune.providers.base import ModelOptions, ModelSpec, PipelineError


def resolve_model_options(
    options: ModelOptions,
    profiles: Mapping[str, ModelSpec],
    *,
    provider: str,
) -> ModelSpec:
    if options.model is not None and options.model_profile is not None:
        raise PipelineError("choose either --model or --model-profile")
    # Preserve the programmatic API default; the CLI requires an explicit choice.
    profile = "qwen3p8-27b" if options.model_profile is None else options.model_profile
    if profile != "custom":
        custom = {
            field.name: getattr(options, field.name)
            for field in fields(options)
            if field.name not in {
                "model_profile", "model", "max_seq_len", "trust_remote_code", "requires_tool_declarations",
                "supports_reasoning_content",
            }
        }
        if any(value is not None for value in custom.values()) or (
            options.trust_remote_code or options.requires_tool_declarations
            or options.supports_reasoning_content
        ):
            raise PipelineError("custom model fields require --model-profile custom")
        if options.model is not None:
            matches = [
                model for alias, model in profiles.items()
                if options.model in {alias, model.base_model, model.tokenizer_model}
            ]
            if len(matches) != 1:
                raise PipelineError(
                    f"no supported {provider} rendering configuration for the selected model; "
                    f"supported aliases: {', '.join(sorted(profiles))}. "
                    "Provider training availability is checked separately."
                )
            model = matches[0]
        else:
            try:
                model = profiles[profile]
            except KeyError as exc:
                raise PipelineError(f"unknown {provider} model profile: {profile}") from exc
        if options.max_seq_len is not None:
            if (
                isinstance(options.max_seq_len, bool)
                or not isinstance(options.max_seq_len, int)
                or not 1 <= options.max_seq_len <= model.training_context_limit
            ):
                raise PipelineError(
                    f"--max-seq-len must be a positive integer no greater than "
                    f"the supported training context ({model.training_context_limit:,})"
                )
            model = replace(
                model, max_seq_len=options.max_seq_len,
                trainer_max_seq_len=(
                    options.max_seq_len if model.trainer_max_seq_len is not None else None
                ),
            )
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

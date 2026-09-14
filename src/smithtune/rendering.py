"""Resolve provider rendering and validate training and replay contexts."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from importlib import metadata
from typing import Any

from smithtune.providers.base import ModelSpec, PipelineError


SFT_TARGET_POLICY = "all_assistant_messages"


DEFAULT_REPLAY_MAX_TOKENS = 4_096


def rendering_version(model: ModelSpec) -> str:
    """Identify the installed formatting implementation, not just its alias."""
    try:
        transformers_version = metadata.version("transformers")
        if model.provider == "baseten":
            from smithtune.hf_rendering import HF_RENDERING_VERSION

            implementation = f"{HF_RENDERING_VERSION};trl={metadata.version('trl')}"
        else:
            distribution = metadata.distribution("fireworks-training-cookbook")
            source = json.loads(distribution.read_text("direct_url.json") or "{}")
            commit = source.get("vcs_info", {}).get("commit_id")
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise PipelineError("the Fireworks cookbook installation has no pinned source revision")
            implementation = f"fireworks-cookbook@{commit}"
    except metadata.PackageNotFoundError as exc:
        raise PipelineError("training dependencies are missing; run smithtune doctor") from exc
    return f"{implementation};transformers={transformers_version}"


def load_training_renderer(model: ModelSpec) -> Any:
    renderer_name = resolved_renderer_name(model)
    if model.rendering_version and model.rendering_version != rendering_version(model):
        raise PipelineError("rendering implementation differs from preparation; restore dependencies or prepare again")
    if model.provider == "baseten":
        from smithtune.hf_rendering import HFRenderer, load_tokenizer

        return HFRenderer(model, load_tokenizer(model))
    try:
        from training.renderer import get_renderer
        from training.utils.tokenizers import load_tokenizer
    except ImportError as exc:
        raise PipelineError("Fireworks training dependencies are missing; run smithtune doctor") from exc
    tokenizer = load_tokenizer(
        model.tokenizer_model, model.tokenizer_revision,
        trust_remote_code=model.trust_remote_code,
    )
    if model.template_sha256 and _tokenizer_template_hash(tokenizer) != model.template_sha256:
        raise PipelineError("tokenizer chat template differs from the prepared manifest; prepare again")
    return get_renderer(renderer_name, tokenizer)


def _tokenizer_template_hash(tokenizer: Any, *, allow_missing: bool = False) -> str:
    from smithtune.hf_rendering import template_sha256

    if allow_missing and getattr(tokenizer, "chat_template", None) is None:
        # Python-backed formatters are identified by the tokenizer commit and
        # cookbook revision; there is no Jinja template to fingerprint.
        return ""
    try:
        template = tokenizer.get_chat_template()
    except (AttributeError, ValueError) as exc:
        raise PipelineError("the selected tokenizer has no usable official chat template") from exc
    if not isinstance(template, str) or not template:
        raise PipelineError("the selected tokenizer has no usable official chat template")
    return template_sha256(template)


def resolve_rendering_model(model: ModelSpec) -> ModelSpec:
    """Download the selected assets and save immutable identities before data access."""
    resolved_renderer_name(model)
    if re.fullmatch(r"[0-9a-f]{40}", model.tokenizer_revision) is None:
        try:
            from huggingface_hub import HfApi

            revision = HfApi().model_info(model.tokenizer_model, revision=model.tokenizer_revision).sha
        except Exception as exc:
            raise PipelineError(
                "could not resolve the tokenizer revision; check Hub access/cache and "
                "HF_TOKEN for gated or private models"
            ) from exc
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise PipelineError("Hugging Face returned no immutable tokenizer revision")
        model = replace(model, tokenizer_revision=revision)
    renderer = load_training_renderer(model)
    return replace(
        model,
        template_sha256=_tokenizer_template_hash(
            renderer.tokenizer,
            allow_missing=model.provider == "fireworks" and model.renderer == "kimi_k3",
        ),
        rendering_version=rendering_version(model),
    )


def render_row_tokens(
    row: dict[str, Any], model: ModelSpec, *, renderer: Any,
    include_loss_mask: bool = False, reduction: str = "mean",
) -> list[Any]:
    """Preserve the all-assistant policy while selecting provider-specific rendering."""
    if model.provider == "baseten":
        return renderer.render(row["messages"], tools=row.get("tools"))
    from training.utils import parse_train_on_what, render_messages_to_datums

    result = render_messages_to_datums(
        row["messages"], renderer=renderer,
        train_on_what=parse_train_on_what(SFT_TARGET_POLICY),
        tools=row.get("tools"), include_loss_mask=include_loss_mask, reduction=reduction,
    )
    return result if isinstance(result, list) else [result]


def validate_reasoning_support(model: ModelSpec) -> None:
    """Require a target and renderer with verified structured-reasoning support."""
    if not model.supports_reasoning_content:
        raise PipelineError(
            f"model {model.name} does not support reasoning content; use reasoning_policy='omit'"
        )
    if resolved_renderer_name(model) not in {"qwen3_8_preserved", "kimi_k3", "hf_assistant"}:
        raise PipelineError(
            f"renderer {model.renderer} has no verified reasoning-content adapter; "
            "use reasoning_policy='omit'"
        )


def resolved_renderer_name(model: ModelSpec) -> str:
    """Validate an explicit thinking-history mode against the renderer registry."""
    if model.provider == "baseten":
        from smithtune.hf_rendering import validate_hf_model

        validate_hf_model(model)
        return model.renderer
    mode = getattr(model, "thinking_trace_history_mode", "")
    if not mode:
        return model.renderer
    try:
        from training.utils.supervised import resolve_renderer_plan
    except ImportError as exc:
        raise PipelineError("training dependencies are missing; reinstall using the GitHub installation command in the README, then run smithtune doctor") from exc
    try:
        return resolve_renderer_plan(
            model.tokenizer_model,
            model.renderer,
            thinking_trace_history_mode=mode,
        ).renderer_name
    except ValueError as exc:
        raise PipelineError(f"invalid renderer/history configuration: {exc}") from exc


def validate_model_context(
    rows: list[dict[str, Any]],
    model: ModelSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Render all rows and reject complete examples above the model limit."""
    renderer = load_training_renderer(model)
    has_tools = any(row.get("tools") for row in rows)
    if has_tools and not model.requires_tool_declarations:
        raise PipelineError(
            f"model profile {model.name} does not require tool declarations for a tool-enabled dataset"
        )
    if has_tools and model.provider == "fireworks":
        from training.utils.supervised import renderer_declares_tools

        if not renderer_declares_tools(renderer):
            raise PipelineError(f"renderer {model.renderer} cannot declare tools required by model profile {model.name}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rendered_count = context_tokens = target_tokens = max_context = 0
    for row in rows:
        rendered_items = render_row_tokens(row, model, renderer=renderer)
        if not rendered_items:
            raise PipelineError(f"example {row['_source']['example_id']} rendered no training datum")
        row_context = row_targets = 0
        row_max_context = 0
        for item in rendered_items:
            count = len(item.token_ids)
            targets = sum(float(weight) > 0 for weight in item.token_weights)
            if targets == 0:
                raise PipelineError(f"example {row['_source']['example_id']} has no target tokens")
            row_context += count
            row_targets += targets
            row_max_context = max(row_max_context, count)
        if row_max_context > model.max_seq_len:
            rejected.append(
                {
                    "example_id": row["_source"]["example_id"],
                    "source_thread_id": row["_source"]["source_thread_id"],
                    "source_trace_id": row["_source"].get("source_trace_id"),
                    "rendered_tokens": row_max_context,
                    "context_limit": model.max_seq_len,
                    "reason": "rendered example exceeds model context limit",
                }
            )
            continue
        accepted.append(row)
        rendered_count += len(rendered_items)
        context_tokens += row_context
        target_tokens += row_targets
        max_context = max(max_context, row_max_context)
    return accepted, rejected, {
        "rendered_datums": rendered_count,
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "max_context_tokens": max_context,
        "rejected_examples": len(rejected),
    }


def validate_replay_context(
    cases: list[dict[str, Any]],
    model: ModelSpec,
    max_output_tokens: int = DEFAULT_REPLAY_MAX_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject complete replay cases that cannot fit the model context."""
    if max_output_tokens < 1:
        raise PipelineError("max output tokens must be positive")
    renderer = load_training_renderer(model)
    has_tools = any(case.get("tools") for case in cases)
    if has_tools and not model.requires_tool_declarations:
        raise PipelineError(
            f"model profile {model.name} does not require tool declarations for tool-enabled replay"
        )
    if model.provider == "fireworks":
        from training.utils.supervised import build_tool_prefixed_messages, renderer_declares_tools

        if has_tools and not renderer_declares_tools(renderer):
            raise PipelineError(f"renderer {model.renderer} cannot declare tools required by model profile {model.name}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for case in cases:
        if model.provider == "baseten":
            prompt_tokens = len(renderer.prompt_tokens(case["messages"], tools=case.get("tools")))
        else:
            normalized = build_tool_prefixed_messages(
                case["messages"], renderer=renderer, tools=case.get("tools"),
            )
            prompt_tokens = len(renderer.build_generation_prompt(normalized).to_ints())
        if prompt_tokens + max_output_tokens > model.max_seq_len:
            rejected.append(
                {
                    "id": case["id"],
                    "example_id": case["example_id"],
                    "source_thread_id": case.get("source_thread_id"),
                    "source_trace_id": case.get("source_trace_id"),
                    "prompt_tokens": prompt_tokens,
                    "max_output_tokens": max_output_tokens,
                    "context_limit": model.max_seq_len,
                    "reason": "replay prompt and output budget exceed model context limit",
                }
            )
            continue
        accepted.append({**case, "prompt_tokens": prompt_tokens})
    return accepted, rejected

"""Validate training and replay contexts with the shared renderer stack."""

from __future__ import annotations

from typing import Any

from smithtune.providers.base import ModelSpec, PipelineError


SFT_TARGET_POLICY = "all_assistant_messages"


DEFAULT_REPLAY_MAX_TOKENS = 4_096


def validate_reasoning_support(model: ModelSpec) -> None:
    """Require a target and renderer with verified structured-reasoning support."""
    if not model.supports_reasoning_content:
        raise PipelineError(
            f"model {model.name} does not support reasoning content; use reasoning_policy='omit'"
        )
    if resolved_renderer_name(model) not in {"qwen3_8_preserved", "kimi_k3"}:
        raise PipelineError(
            f"renderer {model.renderer} has no verified reasoning-content adapter; "
            "use reasoning_policy='omit'"
        )


def resolved_renderer_name(model: ModelSpec) -> str:
    """Validate an explicit thinking-history mode against the renderer registry."""
    mode = getattr(model, "thinking_trace_history_mode", "")
    if not mode:
        return model.renderer
    try:
        from training.utils.supervised import resolve_renderer_plan
    except ImportError as exc:
        raise PipelineError("training dependencies are missing; reinstall smithtune with uv tool install --reinstall smithtune and run smithtune doctor") from exc
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
    try:
        from training.renderer import get_renderer
        from training.utils import parse_train_on_what, render_messages_to_datums
        from training.utils.supervised import renderer_declares_tools
        from training.utils.tokenizers import load_tokenizer
    except ImportError as exc:
        raise PipelineError("training dependencies are missing; reinstall smithtune with uv tool install --reinstall smithtune and run smithtune doctor") from exc

    renderer_name = resolved_renderer_name(model)
    tokenizer = load_tokenizer(
        model.tokenizer_model,
        model.tokenizer_revision,
        trust_remote_code=model.trust_remote_code,
    )
    renderer = get_renderer(renderer_name, tokenizer)
    has_tools = any(row.get("tools") for row in rows)
    if has_tools and not model.requires_tool_declarations:
        raise PipelineError(
            f"model profile {model.name} does not require tool declarations for a tool-enabled dataset"
        )
    if has_tools and not renderer_declares_tools(renderer):
        raise PipelineError(f"renderer {model.renderer} cannot declare tools required by model profile {model.name}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rendered_count = context_tokens = target_tokens = max_context = 0
    for row in rows:
        rendered = render_messages_to_datums(
            row["messages"],
            renderer=renderer,
            train_on_what=parse_train_on_what(SFT_TARGET_POLICY),
            tools=row.get("tools"),
            reduction="mean",
        )
        rendered_items = rendered if isinstance(rendered, list) else [rendered]
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
    try:
        from training.renderer import get_renderer
        from training.utils.supervised import (
            build_tool_prefixed_messages,
            renderer_declares_tools,
        )
        from training.utils.tokenizers import load_tokenizer
    except ImportError as exc:
        raise PipelineError("training dependencies are missing; reinstall smithtune with uv tool install --reinstall smithtune and run smithtune doctor") from exc
    renderer_name = resolved_renderer_name(model)
    tokenizer = load_tokenizer(
        model.tokenizer_model,
        model.tokenizer_revision,
        trust_remote_code=model.trust_remote_code,
    )
    renderer = get_renderer(renderer_name, tokenizer)
    has_tools = any(case.get("tools") for case in cases)
    if has_tools and not model.requires_tool_declarations:
        raise PipelineError(
            f"model profile {model.name} does not require tool declarations for tool-enabled replay"
        )
    if has_tools and not renderer_declares_tools(renderer):
        raise PipelineError(f"renderer {model.renderer} cannot declare tools required by model profile {model.name}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for case in cases:
        normalized = build_tool_prefixed_messages(
            case["messages"],
            renderer=renderer,
            tools=case.get("tools"),
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

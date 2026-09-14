"""Model-native Hugging Face rendering for Baseten's token-level Loops API."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from smithtune.providers.base import ModelSpec, PipelineError


HF_QWEN_RENDERER = "hf_qwen3_8_preserved"
HF_RENDERING_VERSION = "smithtune-hf-qwen3.8-v1"

# These are template expressions, not searches through user messages. Refuse a
# different template structure instead of guessing where assistant loss belongs.
_ASSISTANT_BODY = r"""        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}"""

_MASKED_ASSISTANT_BODY = r"""        {%- set assistant_header = '<|im_start|>' + message.role + '\n' %}
        {%- set assistant_body = content %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {%- if reasoning_content %}
                {%- set assistant_header = assistant_header + '<think>\n' %}
                {%- set assistant_body = reasoning_content + '\n</think>\n\n' + content %}
            {%- else %}
                {%- set assistant_header = assistant_header + '<think>\n\n</think>\n\n' %}
            {%- endif %}
        {%- endif %}
        {{- assistant_header }}
        {%- generation %}
        {{- assistant_body }}"""

_ASSISTANT_END = r"""        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}"""
_MASKED_ASSISTANT_END = r"""        {{- '<|im_end|>' }}
        {%- endgeneration %}
        {{- '\n' }}
    {%- elif message.role == "tool" %}"""


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def assistant_mask_template(template: str) -> str:
    """Annotate Qwen's native template without changing its emitted text."""
    for original, replacement in (
        (_ASSISTANT_BODY, _MASKED_ASSISTANT_BODY),
        (_ASSISTANT_END, _MASKED_ASSISTANT_END),
    ):
        if template.count(original) != 1:
            raise PipelineError(
                "the selected Qwen chat template has no verified assistant-mask adapter; "
                "use a supported tokenizer revision"
            )
        template = template.replace(original, replacement, 1)
    return template


@dataclass(frozen=True)
class TokenDatum:
    """Unshifted token IDs and binary loss weights, independent of a provider SDK."""

    token_ids: list[int]
    token_weights: list[float]


def load_tokenizer(model: ModelSpec) -> Any:
    """Load only tokenizer assets; never provision a trainer to obtain them."""
    try:
        from transformers import AutoTokenizer, PreTrainedConfig
    except ImportError as exc:
        raise PipelineError("Transformers is required for Baseten rendering; run smithtune doctor") from exc
    try:
        # An explicit generic config avoids model-only RoPE validation for models
        # newer than Transformers. Tokenizer selection still uses tokenizer_config.
        return AutoTokenizer.from_pretrained(
            model.tokenizer_model,
            revision=model.tokenizer_revision,
            trust_remote_code=model.trust_remote_code,
            config=PreTrainedConfig(),
        )
    except Exception as exc:
        raise PipelineError(
            f"could not load tokenizer {model.tokenizer_model} at {model.tokenizer_revision}; "
            "check Hub access/cache and HF_TOKEN for gated or private models"
        ) from exc


def validate_hf_model(model: ModelSpec) -> None:
    if model.renderer != HF_QWEN_RENDERER:
        raise PipelineError(
            f"Baseten renderer {model.renderer} is not supported by the native HF path; "
            "prepare again with a supported Baseten model"
        )
    if model.thinking_trace_history_mode not in {"", "preserved"}:
        raise PipelineError(
            f"renderer {model.renderer} conflicts with thinking_trace_history_mode="
            f"{model.thinking_trace_history_mode}; use preserved"
        )


class HFRenderer:
    """Use the official template and expose all-assistant loss via generation spans."""

    def __init__(self, model: ModelSpec, tokenizer: Any) -> None:
        validate_hf_model(model)
        self.tokenizer = tokenizer
        try:
            self.template = tokenizer.get_chat_template()
        except (AttributeError, ValueError) as exc:
            raise PipelineError("the selected tokenizer has no usable chat template") from exc
        if not isinstance(self.template, str) or not self.template:
            raise PipelineError("the selected tokenizer has no usable chat template")
        if model.template_sha256 and template_sha256(self.template) != model.template_sha256:
            raise PipelineError("tokenizer chat template differs from the prepared manifest; prepare again")
        self.mask_template = assistant_mask_template(self.template)

    def render(self, messages: list[dict[str, Any]], tools: Any = None) -> list[TokenDatum]:
        normalized = _normalize_messages(messages)
        kwargs = {
            "tools": tools,
            "add_generation_prompt": False,
            "preserve_thinking": True,
        }
        try:
            original = self.tokenizer.apply_chat_template(
                normalized, chat_template=self.template, tokenize=False, **kwargs,
            )
            annotated = self.tokenizer.apply_chat_template(
                normalized, chat_template=self.mask_template, tokenize=False, **kwargs,
            )
            if original != annotated:
                raise PipelineError("assistant-mask annotations changed the native chat template output")
            encoded = self.tokenizer.apply_chat_template(
                normalized, chat_template=self.mask_template,
                tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                **kwargs,
            )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError("the selected tokenizer could not render this trajectory with assistant masks") from exc
        tokens = encoded.get("input_ids")
        masks = encoded.get("assistant_masks")
        if (
            not isinstance(tokens, list) or not isinstance(masks, list)
            or len(tokens) != len(masks) or not tokens
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
            or any(mask not in (0, 1) for mask in masks)
            or not any(masks)
        ):
            raise PipelineError("the tokenizer returned invalid or empty assistant loss masks")
        return [TokenDatum(tokens, [float(mask) for mask in masks])]

    def prompt_tokens(self, messages: list[dict[str, Any]], tools: Any = None) -> list[int]:
        try:
            return self.tokenizer.apply_chat_template(
                _normalize_messages(messages), tools=tools, chat_template=self.template,
                tokenize=True, add_generation_prompt=True, preserve_thinking=True,
            )
        except Exception as exc:
            raise PipelineError("the selected tokenizer could not render the replay prompt") from exc


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        # Canonical OpenAI tool-call arguments are JSON strings; HF templates
        # iterate the parsed argument object when formatting function parameters.
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as exc:
                    raise PipelineError("tool-call arguments must be a JSON object") from exc
            if not isinstance(arguments, dict):
                raise PipelineError("tool-call arguments must be a JSON object")
            function["arguments"] = arguments
    return normalized

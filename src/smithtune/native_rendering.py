"""Assistant loss over verified native inference prefixes for selected models."""

from __future__ import annotations

from typing import Any

from smithtune.hf_rendering import TokenDatum, _normalize_messages, template_sha256
from smithtune.providers.base import ModelSpec, PipelineError


NATIVE_RENDERING_VERSION = "smithtune-native-prefix-v1"
NATIVE_MODELS = {
    "hf_prefix_kimi_k3": ("moonshotai/Kimi-K3", "preserved"),
    "hf_prefix_qwen3_5": ("Qwen/Qwen3.5-9B", "interleaved"),
    "hf_prefix_glm53_flash": ("zai-org/GLM-5.3-Flash", "preserved"),
}


def validate_native_model(model: ModelSpec) -> None:
    expected = NATIVE_MODELS.get(model.renderer)
    if expected is None or expected[0] != model.tokenizer_model:
        raise PipelineError("no verified native prefix adapter for this tokenizer")
    if model.thinking_trace_history_mode not in {"", expected[1]}:
        raise PipelineError("native renderer conflicts with thinking_trace_history_mode")


class NativePrefixRenderer:
    """Keep native formatting and train each assistant against its own prompt.

    A later user turn can change historical reasoning or message boundaries.
    Coalesce targets only when their token prefixes remain identical; otherwise
    retain the earlier datum and mask its history in the next datum.
    """

    def __init__(self, model: ModelSpec, tokenizer: Any) -> None:
        validate_native_model(model)
        self.model = model
        self.tokenizer = tokenizer
        if model.template_sha256:
            if template_sha256(tokenizer.get_chat_template()) != model.template_sha256:
                raise PipelineError("tokenizer chat template differs from the prepared manifest; prepare again")

    def _tokens(self, messages: list[dict[str, Any]], tools: Any, *, prompt: bool) -> list[int]:
        try:
            tokens = self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=False,
                add_generation_prompt=prompt,
            )
        except Exception as exc:
            raise PipelineError("the official tokenizer could not render this trajectory") from exc
        if not isinstance(tokens, list) or not tokens or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
        ):
            raise PipelineError("the native formatter returned invalid token IDs")
        return tokens

    def _special_token(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != 1:
            raise PipelineError("native response boundary is not a single special token")
        return tokens[0]

    def render(self, messages: list[dict[str, Any]], tools: Any = None) -> list[TokenDatum]:
        messages = _normalize_messages(messages)
        datums: list[TokenDatum] = []
        pending: TokenDatum | None = None
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = self._tokens(messages[:index], tools, prompt=True)
            tokens = self._tokens(messages[:index + 1], tools, prompt=False)
            if tokens[:len(prompt)] != prompt:
                # BPE can merge the prompt's trailing newline with a response's
                # leading newline (notably Qwen's empty thinking block). Keep
                # the actual inference tokens and encode only the completion.
                prompt_text = self.tokenizer.decode(prompt)
                full_text = self.tokenizer.decode(tokens)
                if not full_text.startswith(prompt_text):
                    raise PipelineError("native assistant text does not extend its inference prompt")
                tokens = prompt + self.tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False)
                if self.tokenizer.decode(tokens) != full_text:
                    raise PipelineError("encoding the native completion changed its rendered text")
            end = len(tokens)
            if self.model.renderer == "hf_prefix_kimi_k3":
                # Kimi emits the message close as its stop sequence, then the
                # harness appends this history delimiter outside generation.
                if tokens[-1] != self._special_token("<|end_of_msg|>"):
                    raise PipelineError("Kimi response is missing its native history delimiter")
                end -= 1
            elif self.model.renderer == "hf_prefix_glm53_flash":
                # GLM stops by generating the next role tag; its template only
                # emits that tag when a subsequent message is present.
                stop = "<|observation|>" if message.get("tool_calls") else "<|user|>"
                tokens = [*tokens, self._special_token(stop)]
                end += 1
            if end <= len(prompt):
                raise PipelineError("native assistant response has no target tokens")
            weights = [0.0] * len(prompt) + [1.0] * (end - len(prompt)) + [0.0] * (len(tokens) - end)
            if pending is not None:
                length = len(pending.token_ids)
                if length <= len(prompt) and tokens[:length] == pending.token_ids:
                    weights[:length] = pending.token_weights
                else:
                    datums.append(pending)
            pending = TokenDatum(tokens, weights)
        if pending is not None:
            datums.append(pending)
        if not datums:
            raise PipelineError("the trajectory contains no assistant training targets")
        return datums

    def prompt_tokens(self, messages: list[dict[str, Any]], tools: Any = None) -> list[int]:
        return self._tokens(_normalize_messages(messages), tools, prompt=True)

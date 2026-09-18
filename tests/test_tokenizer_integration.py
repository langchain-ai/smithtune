"""Opt-in checks using public, pinned tokenizer assets; no provider resources."""

from dataclasses import replace
import copy
import os

import pytest

from binding_fixtures import bound_row
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from smithtune.hf_rendering import _normalize_messages
from smithtune.rendering import load_training_renderer, render_row_tokens, resolve_rendering_model


pytestmark = pytest.mark.skipif(
    os.environ.get("SMITHTUNE_TOKENIZER_TESTS") != "1",
    reason="set SMITHTUNE_TOKENIZER_TESTS=1 to download the pinned public tokenizers",
)


@pytest.mark.parametrize("stop_reason,closed_body,valid", [("stop", True, True), ("length", True, False), ("stop", False, False)])
def test_kimi_sampler_stop_framing_preserves_truncation_checks(stop_reason, closed_body, valid):
    from smithtune.providers.fireworks_sampling import restore_stop_suffix

    renderer = load_training_renderer(fireworks.MODEL_SPECS["kimi-k3"])
    text = "reason<|close|>think<|sep|><|open|>response<|sep|>ready"
    if closed_body:
        text += "<|close|>response<|sep|>"
    text += "<|close|>"
    tokens = list(renderer.tokenizer._encode_text_piece(text, allow_special_tokens=True))
    normalized = restore_stop_suffix(tokens, renderer, stop_reason)
    assert normalized[:len(tokens)] == tokens
    _, termination = renderer.parse_response(normalized)
    assert termination.is_clean is valid
    if stop_reason == "length":
        assert normalized == tokens


@pytest.mark.parametrize("model", [
    *baseten.MODEL_SPECS.values(), *fireworks.MODEL_SPECS.values(),
], ids=lambda model: f"{model.provider}-{model.name}")
def test_supported_tokenizers_render_all_assistant_targets(model):
    model = resolve_rendering_model(model)
    renderer = load_training_renderer(model)
    messages = [
        {"role": "system", "content": "SYSTEM_CONTEXT_SENTINEL"},
        {"role": "user", "content": "USER_CONTEXT_SENTINEL"},
        {"role": "assistant", "content": "", "reasoning_content": "REASONING_SENTINEL",
         "tool_calls": [{"id": "call-1", "type": "function", "function": {
             "name": "weather", "arguments": '{"city":"Paris"}',
         }}]},
        {"role": "tool", "tool_call_id": "call-1", "name": "weather", "content": "TOOL_RESULT_SENTINEL"},
        {"role": "assistant", "content": "ASSISTANT_ONE_SENTINEL"},
        {"role": "user", "content": "USER_TWO_SENTINEL"},
        {"role": "assistant", "content": "ASSISTANT_TWO_SENTINEL café 🐢 東京", "reasoning_content": ""},
    ]
    tools = [{"type": "function", "function": {
        "name": "weather", "description": "Look up weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}]
    if model.renderer != "muse_glimmer":
        messages[2]["content"] = [{"type": "text", "text": "LEADING_TEXT_SENTINEL"}]
    original_messages = copy.deepcopy(messages)
    rows = render_row_tokens(bound_row({"messages": messages, "tools": tools, "_source": {"example_id": "tokenizer-check"}}), model, renderer=renderer, include_loss_mask=True)
    assert messages == original_messages
    assert len(rows) == 3
    target = ""
    for datum in rows:
        assert len(datum.token_ids) == len(datum.token_weights)
        target += renderer.tokenizer.decode([
            token for token, weight in zip(datum.token_ids, datum.token_weights, strict=True) if weight
        ])
    for expected in ("REASONING_SENTINEL", "ASSISTANT_ONE_SENTINEL", "ASSISTANT_TWO_SENTINEL", "weather", "Paris", "café 🐢 東京"):
        assert expected in target
    for expected in ("REASONING_SENTINEL", "ASSISTANT_ONE_SENTINEL", "ASSISTANT_TWO_SENTINEL"):
        assert target.count(expected) == 1
    if model.renderer != "muse_glimmer":
        assert target.count("LEADING_TEXT_SENTINEL") == 1
        assert target.index("LEADING_TEXT_SENTINEL") < target.index("weather")
    for context in ("SYSTEM_CONTEXT_SENTINEL", "USER_CONTEXT_SENTINEL", "TOOL_RESULT_SENTINEL", "USER_TWO_SENTINEL"):
        assert context not in target
    # Reloading the saved identities must accept the same assets and versions.
    assert resolve_rendering_model(model) == model
    if model.renderer == "hf_assistant":
        assert target.count("<think>") == 3
        assert target.count("</think>") == 3
        assert target.count("<|im_end|>") == 3
        assert "trl=1.13.0" in model.rendering_version
    if model.provider == "baseten":
        with pytest.raises(PipelineError, match="implementation differs"):
            load_training_renderer(replace(model, rendering_version="previous-renderer"))
    if model.renderer == "hf_prefix_glm53_flash":
        assert target.count("<|observation|>") == 1
        assert target.count("<|user|>") == 2
    if model.renderer == "hf_prefix_kimi_k3":
        assert "<|end_of_msg|>" not in target
        assert target.count("<|close|>message<|sep|>") == 3


@pytest.mark.parametrize("model", [model for model in baseten.MODEL_SPECS.values() if model.renderer.startswith("hf_prefix_")], ids=lambda model: model.name)
def test_native_masks_preserve_actual_generation_prefix_and_native_text(model):
    renderer = load_training_renderer(model)
    messages = [
        {"role": "user", "content": "FIRST_USER"},
        {"role": "assistant", "reasoning_content": "FIRST_REASON", "content": "FIRST_ANSWER"},
        {"role": "assistant", "reasoning_content": "", "content": "FOLLOWUP_ANSWER"},
        {"role": "user", "content": "SECOND_USER"},
        {"role": "assistant", "reasoning_content": "LAST_REASON", "content": ""},
    ]
    for index in (1, 2, 4):
        prefix = renderer.prompt_tokens(messages[:index])
        datum = renderer.render(messages[:index + 1], final_target=True)[0]
        assert datum.token_ids[:len(prefix)] == prefix
        expected = renderer.tokenizer.apply_chat_template(
            _normalize_messages(messages[:index + 1]), tokenize=False, add_generation_prompt=False,
        )
        if model.renderer == "hf_prefix_glm53_flash":
            expected += "<|user|>"
        assert renderer.tokenizer.decode(datum.token_ids) == expected
        assert datum.token_weights[len(prefix)] == 1
        assert not any(datum.token_weights[:len(prefix)])
    rows = renderer.render(messages)
    target = "".join(renderer.tokenizer.decode([
        token for token, weight in zip(row.token_ids, row.token_weights, strict=True) if weight
    ]) for row in rows)
    for sentinel in ("FIRST_REASON", "FIRST_ANSWER", "FOLLOWUP_ANSWER", "LAST_REASON"):
        assert target.count(sentinel) == 1
    assert "FIRST_USER" not in target and "SECOND_USER" not in target

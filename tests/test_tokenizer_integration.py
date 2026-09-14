"""Opt-in checks using public, pinned tokenizer assets; no provider resources."""

from dataclasses import replace
import os

import pytest

from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from smithtune.rendering import load_training_renderer, render_row_tokens, resolve_rendering_model


pytestmark = pytest.mark.skipif(
    os.environ.get("SMITHTUNE_TOKENIZER_TESTS") != "1",
    reason="set SMITHTUNE_TOKENIZER_TESTS=1 to download the pinned public tokenizers",
)


@pytest.mark.parametrize("model", [
    *baseten.MODEL_SPECS.values(), *fireworks.MODEL_SPECS.values(),
], ids=lambda model: f"{model.provider}-{model.name}")
def test_supported_tokenizers_render_all_assistant_targets(model):
    model = resolve_rendering_model(model)
    renderer = load_training_renderer(model)
    messages = [
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
    rows = render_row_tokens({"messages": messages, "tools": tools}, model, renderer=renderer, include_loss_mask=True)
    assert len(rows) == 1
    datum = rows[0]
    assert len(datum.token_ids) == len(datum.token_weights)
    target = renderer.tokenizer.decode([
        token for token, weight in zip(datum.token_ids, datum.token_weights, strict=True) if weight
    ])
    for expected in ("REASONING_SENTINEL", "ASSISTANT_ONE_SENTINEL", "ASSISTANT_TWO_SENTINEL", "weather", "Paris", "café 🐢 東京"):
        assert expected in target
    for context in ("USER_CONTEXT_SENTINEL", "TOOL_RESULT_SENTINEL", "USER_TWO_SENTINEL"):
        assert context not in target
    # Reloading the saved identities must accept the same assets and versions.
    assert resolve_rendering_model(model) == model
    if model.provider == "baseten":
        assert target.count("<think>") == 3
        assert target.count("</think>") == 3
        assert target.count("<|im_end|>") == 3
        assert "trl=1.13.0" in model.rendering_version
        with pytest.raises(PipelineError, match="implementation differs"):
            load_training_renderer(replace(model, rendering_version="previous-renderer"))

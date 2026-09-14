"""Exercise the real packaged renderer without downloading tokenizers or models."""

import hashlib
from importlib import metadata, resources
import json
from pathlib import Path

import pytest


class CharacterTokenizer:
    """Deterministic tokenizer to isolate renderer and loss-mask equivalence."""

    name_or_path = "smithtune-test-tokenizer"

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(int(token)) for token in tokens)


MESSAGES = [
    {"role": "system", "content": "Be precise."},
    {"role": "user", "content": "What is the weather?"},
    {"role": "assistant", "content": "", "reasoning_content": "Look it up.",
     "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}]},
    {"role": "tool", "tool_call_id": "call-1", "name": "weather", "content": "Sunny"},
    {"role": "assistant", "content": "It is sunny."},
    {"role": "user", "content": "Thanks. And tomorrow?"},
    {"role": "assistant", "content": "I would need another forecast."},
]
TOOLS = [{"type": "function", "function": {
    "name": "weather", "description": "Look up weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}}]


def renderer_snapshot():
    from training.renderer import get_renderer
    from training.utils import render_messages_to_datums
    from training.utils.supervised import build_tool_prefixed_messages

    renderer = get_renderer("qwen3_8_preserved", CharacterTokenizer())
    rows = render_messages_to_datums(MESSAGES, renderer=renderer, tools=TOOLS, include_loss_mask=True, reduction="none")
    prompt = renderer.build_generation_prompt(build_tool_prefixed_messages(MESSAGES[:-1], renderer=renderer, tools=TOOLS))
    return {
        "rows": [{"tokens": list(row.token_ids), "weights": list(row.token_weights),
                  "inputs": row.datum.model_input.to_ints(),
                  "targets": list(row.datum.loss_fn_inputs["target_tokens"].data),
                  "loss_weights": list(row.datum.loss_fn_inputs["weights"].data)} for row in rows],
        "prompt": prompt.to_ints(),
    }


def test_renderer_matches_pinned_upstream_snapshot():
    actual = renderer_snapshot()
    expected = json.loads((Path(__file__).parent / "fixtures/runtime-renderer.json").read_text())
    assert actual == expected


def test_real_renderer_context_boundaries(monkeypatch):
    from dataclasses import replace
    import training.utils.tokenizers
    from smithtune.providers.baseten import DEFAULT_MODEL
    from smithtune.rendering import validate_model_context, validate_replay_context

    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: CharacterTokenizer())
    snapshot = renderer_snapshot()
    count = max(len(row["tokens"]) for row in snapshot["rows"])
    row = {"messages": MESSAGES, "tools": TOOLS, "_source": {"example_id": "test", "source_thread_id": "thread"}}
    model = replace(DEFAULT_MODEL, max_seq_len=count)
    assert validate_model_context([row], model)[0] == [row]
    assert validate_model_context([row], replace(model, max_seq_len=count - 1))[0] == []
    case = {"id": "test", "example_id": "test", "messages": MESSAGES[:-1], "tools": TOOLS}
    prompt_limit = len(snapshot["prompt"]) + 32
    assert len(validate_replay_context([case], replace(model, max_seq_len=prompt_limit), 32)[0]) == 1
    assert validate_replay_context([case], replace(model, max_seq_len=prompt_limit - 1), 32)[0] == []


def test_runtime_source_and_resources_match_reviewed_snapshot():
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "packages/training-runtime/upstream.json").read_text())
    training = resources.files("training")
    for name, digest in {**manifest["files"], **manifest.get("patched_files", {})}.items():
        if name.startswith("src/training/"):
            content = training.joinpath(name.removeprefix("src/training/")).read_bytes()
            assert hashlib.sha256(content).hexdigest() == digest, name
    assert "Apache" in training.joinpath("_vendor/tinker_cookbook_0_4_3/LICENSE").read_text()


def test_training_dependencies_are_importable_without_cookbook_distribution():
    from training.recipes import sft_loop
    from fireworks.training.sdk import FireworksClient
    import baseten.loops

    assert callable(sft_loop.main)
    assert FireworksClient is not None
    assert baseten.loops is not None
    for package in ("tinker-cookbook", "fireworks-training-cookbook"):
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.version(package)

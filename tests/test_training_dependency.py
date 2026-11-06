"""Verify the installed upstream cookbook and its renderer without network calls."""

from tests.binding_fixtures import bound_row

from importlib import metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
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
    from smithtune.providers.fireworks.provider import DEFAULT_MODEL
    from smithtune.rendering import validate_model_context, validate_replay_context

    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: CharacterTokenizer())
    snapshot = renderer_snapshot()
    count = max(len(row["tokens"]) for row in snapshot["rows"])
    row = bound_row({"messages": MESSAGES, "tools": TOOLS, "_source": {"example_id": "test", "source_scope": "thread", "source_scope_id": "thread"}})
    model = replace(DEFAULT_MODEL, max_seq_len=count)
    assert validate_model_context([row], model)[0] == [row]
    assert validate_model_context([row], replace(model, max_seq_len=count - 1))[0] == []
    case = {"id": "test", "example_id": "test", "messages": MESSAGES[:-1], "tools": TOOLS}
    prompt_limit = len(snapshot["prompt"]) + 32
    assert len(validate_replay_context([case], replace(model, max_seq_len=prompt_limit), 32)[0]) == 1
    assert validate_replay_context([case], replace(model, max_seq_len=prompt_limit - 1), 32)[0] == []


def test_cookbook_is_installed_from_the_declared_pypi_version():
    requirements = [Requirement(value) for value in metadata.requires("smithtune")]
    assert all(requirement.url is None for requirement in requirements)
    cookbook = next(r for r in requirements if canonicalize_name(r.name) == "fireworks-training-cookbook")
    assert str(cookbook.specifier) == "==2026.11.5.2"
    assert cookbook.specifier.contains(metadata.version(cookbook.name))


def test_rendering_identity_tracks_installed_cookbook_version(monkeypatch):
    from dataclasses import replace
    from smithtune.providers.fireworks.provider import DEFAULT_MODEL
    from smithtune.rendering import load_training_renderer, rendering_version
    from smithtune.providers.base import PipelineError

    identity = rendering_version(DEFAULT_MODEL)
    assert f"fireworks-cookbook=={metadata.version('fireworks-training-cookbook')}" in identity
    saved = replace(DEFAULT_MODEL, rendering_version=identity)
    original_version = metadata.version
    monkeypatch.setattr(metadata, "version", lambda name: "2026.11.5.3" if name == "fireworks-training-cookbook" else original_version(name))
    with pytest.raises(PipelineError, match="rendering implementation differs from preparation"):
        load_training_renderer(saved)


def test_upstream_training_dependencies_are_importable():
    from training.recipes import sft_loop
    from fireworks.training.sdk import FireworksClient
    import baseten.loops
    from trl.chat_template_utils import get_training_chat_template

    assert callable(sft_loop.main)
    assert FireworksClient is not None
    assert baseten.loops is not None
    assert callable(get_training_chat_template)
    assert metadata.version("transformers") == "5.10.4"
    assert metadata.version("trl") == "1.13.0"


def test_transformers_rejects_chat_template_path_traversal(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = {"../escaped": "{{ messages }}"}
    with pytest.raises(ValueError, match="Invalid chat template name"):
        tokenizer.save_pretrained(tmp_path / "tokenizer")
    assert not (tmp_path / "escaped.jinja").exists()


def test_installed_dependencies_are_compatible():
    conflicts = []
    for distribution in metadata.distributions():
        owner = canonicalize_name(distribution.metadata["Name"])
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            installed = metadata.version(requirement.name)
            if not requirement.specifier.contains(installed, prereleases=True):
                conflicts.append(f"{owner} requires {requirement}; installed {installed}")
    assert not conflicts, "\n".join(conflicts)

"""Native template parity, loss boundaries, and provider-independent rendering."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from smithtune import hf_rendering, rendering
from smithtune.hf_rendering import HFRenderer, assistant_mask_template, template_sha256
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError


TEMPLATE = (Path(__file__).parent / "fixtures/qwen3p8-chat-template.jinja").read_text()
TOOLS = [{"type": "function", "function": {
    "name": "weather", "description": "Look up weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]
MESSAGES = [
    {"role": "system", "content": "Be precise."},
    {"role": "user", "content": "What is the weather?"},
    {"role": "assistant", "content": "", "reasoning_content": "Look it up.",
     "tool_calls": [{"id": "call-1", "type": "function", "function": {
         "name": "weather", "arguments": '{"city":"Paris"}',
     }}]},
    {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
    {"role": "assistant", "content": "It is sunny."},
    {"role": "user", "content": "Thanks. And tomorrow?"},
    {"role": "assistant", "content": "I would need another forecast."},
]


class GenerationMarkers(Extension):
    """Expose Jinja generation spans for a deterministic character-token oracle."""

    tags: ClassVar[set[str]] = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("mark"), [], [], body).set_lineno(lineno)

    def mark(self, caller):
        return "\ue000" + caller() + "\ue001"


class CharacterChatTokenizer:
    """Render the actual Jinja template, with one token per character for assertions."""

    def get_chat_template(self):
        return TEMPLATE

    def apply_chat_template(self, messages, *, chat_template, tokenize, **kwargs):
        def fail(message):
            raise ValueError(message)

        env = ImmutableSandboxedEnvironment(extensions=[GenerationMarkers])
        env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
        env.globals["raise_exception"] = fail
        marked = env.from_string(chat_template).render(messages=messages, **kwargs)
        chars, masks = [], []
        active = 0
        for char in marked:
            if char == "\ue000":
                active = 1
            elif char == "\ue001":
                active = 0
            else:
                chars.append(char)
                masks.append(active)
        text = "".join(chars)
        if not tokenize:
            return text
        ids = list(map(ord, text))
        return {"input_ids": ids, "assistant_masks": masks} if kwargs.get("return_dict") else ids


def _loss_spans(datum):
    spans = []
    text = "".join(map(chr, datum.token_ids))
    start = None
    for index, weight in enumerate([*datum.token_weights, 0]):
        if weight and start is None:
            start = index
        elif not weight and start is not None:
            spans.append(text[start:index])
            start = None
    return spans


def test_tools_reasoning_and_all_assistant_turns_keep_existing_loss_policy():
    original = copy.deepcopy(MESSAGES)
    renderer = HFRenderer(baseten.DEFAULT_MODEL, CharacterChatTokenizer())
    datums = renderer.render(MESSAGES, TOOLS)
    assert len(datums) == 1
    assert _loss_spans(datums[0]) == [
        "Look it up.\n</think>\n\n<tool_call>\n<function=weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>",
        "It is sunny.<|im_end|>",
        "I would need another forecast.<|im_end|>",
    ]
    rendered = "".join(map(chr, datums[0].token_ids))
    assert "# Tools" in rendered and "Sunny" in rendered and "Be precise." in rendered
    assert original == MESSAGES


@pytest.mark.parametrize("content", [
    "", " café 🐢 東京 ", "<|im_start|>assistant\nquoted header",
    [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
])
@pytest.mark.parametrize("reasoning", [None, "", " thinking\ncarefully "])
def test_native_text_parity_and_masks_for_message_variants(content, reasoning):
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": content}]
    if reasoning is not None:
        messages[-1]["reasoning_content"] = reasoning
    tokenizer = CharacterChatTokenizer()
    datum = HFRenderer(baseten.DEFAULT_MODEL, tokenizer).render(messages)[0]
    expected = tokenizer.apply_chat_template(
        messages, chat_template=TEMPLATE, tokenize=False,
        add_generation_prompt=False, preserve_thinking=True,
    )
    assert "".join(map(chr, datum.token_ids)) == expected
    assert len(_loss_spans(datum)) == 1
    assert _loss_spans(datum)[0].endswith("<|im_end|>")


def test_parallel_tool_calls_and_results_remain_one_trajectory():
    messages = copy.deepcopy(MESSAGES[:4])
    messages[2]["tool_calls"].append({"id": "call-2", "type": "function", "function": {
        "name": "weather", "arguments": {"city": "Tokyo", "details": {"days": 2}, "unit": None},
    }})
    messages.append({"role": "tool", "tool_call_id": "call-2", "content": "Rain"})
    messages.append({"role": "assistant", "content": "Forecasts retrieved."})
    datums = HFRenderer(baseten.DEFAULT_MODEL, CharacterChatTokenizer()).render(messages, TOOLS)
    assert len(datums) == 1
    spans = _loss_spans(datums[0])
    assert len(spans) == 2
    assert spans[0].count("<function=weather>") == 2
    assert "Tokyo" in spans[0] and '{"days": 2}' in spans[0]
    assert not any("Sunny" in span or "Rain" in span for span in spans)


def test_template_structure_changes_fail_without_guessing_loss_boundaries():
    with pytest.raises(PipelineError, match="no verified assistant-mask adapter"):
        assistant_mask_template(TEMPLATE.replace("message.role", "message['role']"))


def test_annotated_template_cannot_change_the_training_text():
    renderer = HFRenderer(baseten.DEFAULT_MODEL, CharacterChatTokenizer())
    renderer.mask_template += "different"
    with pytest.raises(PipelineError, match="changed the native"):
        renderer.render(MESSAGES, TOOLS)


def test_template_fingerprint_mismatch_fails_before_rendering():
    with pytest.raises(PipelineError, match="differs from the prepared manifest"):
        HFRenderer(replace(baseten.DEFAULT_MODEL, template_sha256="0" * 64), CharacterChatTokenizer())


@pytest.mark.parametrize("arguments", ["invalid-json", "[]", "null", 3])
def test_invalid_tool_arguments_are_rejected(arguments):
    messages = copy.deepcopy(MESSAGES)
    messages[2]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(PipelineError, match="JSON object"):
        HFRenderer(baseten.DEFAULT_MODEL, CharacterChatTokenizer()).render(messages, TOOLS)


def test_baseten_preparation_and_datum_conversion_share_native_renderer(monkeypatch):
    native = HFRenderer(baseten.DEFAULT_MODEL, CharacterChatTokenizer())
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: native)
    row = {"messages": MESSAGES, "tools": TOOLS, "_source": {"example_id": "example", "source_thread_id": "thread"}}
    token_datum = native.render(MESSAGES, TOOLS)[0]
    limit = len(token_datum.token_ids)
    model = replace(baseten.DEFAULT_MODEL, max_seq_len=limit, trainer_max_seq_len=limit)
    accepted, rejected, audit = rendering.validate_model_context([row], model)
    assert accepted == [row] and rejected == []
    assert audit["target_tokens"] == sum(token_datum.token_weights)
    assert rendering.validate_model_context([row], replace(model, max_seq_len=limit - 1))[0] == []
    loops_types = SimpleNamespace(
        ModelInput=SimpleNamespace(from_ints=lambda ids: ids),
        TensorData=lambda **kwargs: SimpleNamespace(**kwargs),
        Datum=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    datum = baseten.render_row(row, model, loops_types=loops_types, renderer=native)[0]
    assert datum.model_input == token_datum.token_ids[:-1]
    assert datum.loss_fn_inputs["weights"].data == token_datum.token_weights[1:]
    assert datum.loss_fn_inputs["target_tokens"].data == [
        token if weight else -100
        for token, weight in zip(token_datum.token_ids[1:], token_datum.token_weights[1:], strict=True)
    ]


def test_native_loading_has_no_fireworks_dependency(monkeypatch):
    calls = []
    tokenizer = CharacterChatTokenizer()
    monkeypatch.setitem(sys.modules, "training", None)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)) or tokenizer),
        PreTrainedConfig=lambda: "generic-config",
    ))
    model = baseten.DEFAULT_MODEL
    renderer = rendering.load_training_renderer(model)
    assert renderer.render(MESSAGES, TOOLS)
    assert calls == [((model.tokenizer_model,), {
        "revision": model.tokenizer_revision, "trust_remote_code": False, "config": "generic-config",
    })]


def test_resolution_pins_branch_once_and_records_template_identity(monkeypatch):
    calls = []
    sha = "a" * 40
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(
        model_info=lambda repo, **kwargs: calls.append((repo, kwargs)) or SimpleNamespace(sha=sha),
    )))
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: SimpleNamespace(tokenizer=CharacterChatTokenizer()))
    monkeypatch.setattr(rendering, "rendering_version", lambda model: "native-version")
    model = replace(baseten.DEFAULT_MODEL, tokenizer_revision="main")
    resolved = rendering.resolve_rendering_model(model)
    assert calls == [(model.tokenizer_model, {"revision": "main"})]
    assert resolved.tokenizer_revision == sha
    assert resolved.template_sha256 == template_sha256(TEMPLATE)
    assert resolved.rendering_version == "native-version"
    rendering.resolve_rendering_model(resolved)
    assert len(calls) == 1


def test_native_prepare_persists_resolved_identity_and_whole_trajectories(monkeypatch, tmp_path):
    from test_pipeline import example, write_raw
    from smithtune import dataset
    from smithtune.providers.base import ModelOptions

    examples = [example(index) for index in range(10)]
    write_raw(tmp_path, examples)
    calls = []

    def capability(model, context):
        calls.append("capability")
        return baseten.BasetenModelCapability(model, context, False)

    def tokenizer(model):
        assert calls and calls[0] == "capability"
        calls.append("tokenizer")
        return CharacterChatTokenizer()

    monkeypatch.setattr(hf_rendering, "load_tokenizer", tokenizer)
    monkeypatch.setattr(rendering, "rendering_version", lambda model: "native-version")
    manifest = baseten.BasetenProvider(capability_resolver=capability).prepare(
        "workspace-id", "dataset-id", tmp_path, model_options=ModelOptions(model="qwen3p8-27b"), fetch=False,
    )
    saved = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert saved == manifest
    model = dataset._model_from_manifest(saved)
    assert model.template_sha256 == template_sha256(TEMPLATE)
    assert model.rendering_version == "native-version"
    assert model.tokenizer_revision == baseten.DEFAULT_MODEL.tokenizer_revision
    rows = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (tmp_path / f"prepared/{split}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(examples)
    assert all(len(row["messages"]) == 2 for row in rows)
    assert {row["_source"]["example_id"] for row in rows} == {source["id"] for source in examples}


def test_implementation_drift_fails_before_loading_tokenizer(monkeypatch):
    monkeypatch.setattr(rendering, "rendering_version", lambda model: "new-version")
    monkeypatch.setattr(hf_rendering, "load_tokenizer", lambda model: pytest.fail("must reject before loading"))
    with pytest.raises(PipelineError, match="implementation differs"):
        rendering.load_training_renderer(replace(baseten.DEFAULT_MODEL, rendering_version="prepared-version"))


@pytest.mark.parametrize("provider", [baseten, fireworks])
def test_provider_preflight_precedes_tokenizer_and_dataset_access(monkeypatch, tmp_path, provider):
    def unsupported(*args, **kwargs):
        raise PipelineError("unsupported training model")

    monkeypatch.setattr(provider, "preflight_model", unsupported)
    monkeypatch.setattr(provider, "resolve_rendering_model", lambda model: pytest.fail("tokenizer must not load"))
    from smithtune.providers.base import ModelOptions

    adapter = baseten.BasetenProvider() if provider is baseten else fireworks.FireworksProvider()
    with pytest.raises(PipelineError, match="unsupported training"):
        adapter.prepare("workspace", "dataset", tmp_path, model_options=ModelOptions())
    assert list(tmp_path.iterdir()) == []


def test_real_transformers_assistant_masks_cover_unicode_and_special_tokens():
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    backend = Tokenizer(models.BPE(vocab={char: i for i, char in enumerate(alphabet + specials)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, additional_special_tokens=specials, chat_template=TEMPLATE,
    )
    messages = copy.deepcopy(MESSAGES)
    messages[-1]["content"] = "café 🐢 東京"
    datum = HFRenderer(baseten.DEFAULT_MODEL, tokenizer).render(messages, TOOLS)[0]
    target_text = tokenizer.decode([token for token, mask in zip(datum.token_ids, datum.token_weights, strict=True) if mask])
    assert "café 🐢 東京" in target_text
    assert target_text.count("<|im_end|>") == 3
    assert "<|im_start|>" not in target_text
    assert "Sunny" not in target_text

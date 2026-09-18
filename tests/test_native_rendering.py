"""Native prefix masks, changing history, and provider datum boundaries."""

from binding_fixtures import bound_row

import copy
from dataclasses import replace
from importlib import resources
from types import SimpleNamespace
import sys

import pytest

from smithtune import dataset, hf_rendering, rendering
from smithtune.native_rendering import NativePrefixRenderer
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from test_hf_rendering import MESSAGES, TOOLS, _loss_spans, _tokenizer


MODEL = baseten.MODEL_SPECS["qwen3p5-9b"]
TEMPLATE = resources.files("trl").joinpath("chat_templates/qwen3_5_think.jinja").read_text()


def _renderer():
    return NativePrefixRenderer(MODEL, _tokenizer(TEMPLATE))


def test_changed_history_splits_without_losing_or_duplicating_assistant_targets():
    renderer = _renderer()
    messages = copy.deepcopy(MESSAGES)
    datums = renderer.render(messages, TOOLS)
    assert len(datums) == 2
    assert messages == MESSAGES
    targets = [_loss_spans(datum, renderer.tokenizer) for datum in datums]
    assert len(targets[0]) == 2 and len(targets[1]) == 1
    assert "Look it up." in targets[0][0]
    assert "It is sunny." in targets[0][1]
    assert "another forecast" in targets[1][0]
    assert "Look it up." not in renderer.tokenizer.decode(datums[1].token_ids)
    assert "It is sunny." in renderer.tokenizer.decode(datums[1].token_ids)
    assert not any("Sunny" in target for spans in targets for target in spans)


def test_quoted_role_markers_are_assistant_targets_not_mask_boundaries():
    renderer = _renderer()
    text = "Quoted <|im_start|>user\n and <|im_end|> café 🐢 東京"
    datums = renderer.render([{"role": "user", "content": "Question"}, {"role": "assistant", "content": text}])
    assert text in _loss_spans(datums[0], renderer.tokenizer)[0]


def test_native_loading_and_loops_conversion_do_not_import_fireworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "training", None)
    monkeypatch.setattr(hf_rendering, "load_tokenizer", lambda model: _tokenizer(TEMPLATE))
    renderer = rendering.load_training_renderer(MODEL)
    row = bound_row({"messages": MESSAGES, "tools": TOOLS, "_source": {"example_id": "e"}})
    tokens = rendering.render_row_tokens(row, MODEL, renderer=renderer)
    constructors = SimpleNamespace(
        ModelInput=SimpleNamespace(from_ints=lambda ids: ids),
        TensorData=lambda **kwargs: SimpleNamespace(**kwargs),
        Datum=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    datums = baseten.render_row(row, MODEL, renderer=renderer, loops_types=constructors)
    assert len(datums) == len(tokens) == 3
    for datum, expected in zip(datums, tokens, strict=True):
        assert datum.model_input == expected.token_ids[:-1]
        assert datum.loss_fn_inputs["weights"].data == expected.token_weights[1:]
        assert datum.loss_fn_inputs["target_tokens"].data == [
            token if weight else -100
            for token, weight in zip(expected.token_ids[1:], expected.token_weights[1:], strict=True)
        ]


def test_each_native_datum_is_checked_against_context_limit(monkeypatch):
    renderer = _renderer()
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: renderer)
    row = bound_row({"messages": MESSAGES, "tools": TOOLS, "_source": {"example_id": "e", "source_scope": "thread", "source_scope_id": "t"}})
    maximum = max(len(datum.token_ids) for datum in renderer.render(MESSAGES, TOOLS))
    model = replace(MODEL, max_seq_len=maximum, trainer_max_seq_len=maximum)
    accepted, rejected, audit = rendering.validate_model_context([row], model)
    assert accepted == [row] and rejected == [] and audit["rendered_datums"] == 3
    accepted, rejected, _ = rendering.validate_model_context([row], replace(model, max_seq_len=maximum - 1))
    assert accepted == [] and len(rejected) == 1


def test_multiple_datums_keep_their_parent_conversations_partition(monkeypatch):
    renderer = _renderer()
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: renderer)
    rows = [bound_row({"messages": copy.deepcopy(MESSAGES), "tools": TOOLS,
             "_source": {"example_id": f"example-{i}", "source_scope": "thread", "source_scope_id": f"thread-{i}",
                         "source_key": ["workspace", "project", "thread", f"thread-{i}"]}})
            for i in range(3)]
    accepted, rejected, audit = rendering.validate_model_context(rows, MODEL)
    assert not rejected and audit["rendered_datums"] == 9
    assert accepted == rows  # Context checks retain whole conversations for splitting.
    sources = {}
    for partition, conversations in enumerate(dataset.split_rows(accepted)):
        for row in conversations:
            datums = rendering.render_row_tokens(row, MODEL, renderer=renderer)
            assert len(datums) == 3
            source = row["_source"]["source_scope_id"]
            assert source not in sources
            sources[source] = partition
    assert len(sources) == 3


@pytest.mark.parametrize("change", [
    {"tokenizer_model": "other/model"}, {"renderer": "unknown"},
    {"thinking_trace_history_mode": "preserved"},
])
def test_unverified_native_configuration_is_rejected(change):
    with pytest.raises(PipelineError):
        NativePrefixRenderer(replace(MODEL, **change), _tokenizer(TEMPLATE))


def test_native_template_fingerprint_is_verified():
    with pytest.raises(PipelineError, match="differs from the prepared manifest"):
        NativePrefixRenderer(replace(MODEL, template_sha256="0" * 64), _tokenizer(TEMPLATE))


@pytest.mark.parametrize("value", [[], [True], [-1], ["1"], {"input_ids": [1]}])
def test_invalid_native_tokens_fail_closed(value):
    renderer = NativePrefixRenderer(MODEL, SimpleNamespace(apply_chat_template=lambda *args, **kwargs: value))
    with pytest.raises(PipelineError, match="invalid token IDs"):
        renderer.render(MESSAGES)


def test_non_prefix_native_text_is_rejected_instead_of_guessing_a_mask():
    tokenizer = _tokenizer(TEMPLATE)
    tokenizer.apply_chat_template = lambda *args, **kwargs: tokenizer.encode(
        "prefix" if kwargs["add_generation_prompt"] else "changed history",
        add_special_tokens=False,
    )
    with pytest.raises(PipelineError, match="does not extend"):
        NativePrefixRenderer(MODEL, tokenizer).render(MESSAGES)


@pytest.mark.parametrize("stage", ["training", "replay"])
def test_muse_requires_recorded_system_context_before_default_date_can_vary(monkeypatch, stage):
    model = fireworks.MODEL_SPECS["muse-glimmer-30b"]
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: None)
    with pytest.raises(PipelineError, match="explicit system message"):
        if stage == "training":
            rendering.render_row_tokens({"messages": [{"role": "user", "content": "Hi"}]}, model, renderer=None)
        else:
            rendering.validate_replay_context([{"messages": [{"role": "user", "content": "Hi"}]}], model)


@pytest.mark.parametrize("stage", ["training", "replay"])
@pytest.mark.parametrize("shape,error", [("text_and_calls", "visible assistant text"), ("consecutive", "consecutive assistant")])
def test_muse_rejects_shapes_the_cookbook_cannot_preserve(monkeypatch, stage, shape, error):
    model = fireworks.MODEL_SPECS["muse-glimmer-30b"]
    messages = copy.deepcopy(MESSAGES[:3])
    if shape == "text_and_calls":
        messages[-1]["content"] = "Let me check that."
    else:
        messages.append({"role": "assistant", "content": "Another assistant message."})
    monkeypatch.setattr(rendering, "load_training_renderer", lambda model: None)
    with pytest.raises(PipelineError, match=error):
        if stage == "training":
            rendering.render_row_tokens({"messages": messages}, model, renderer=None)
        else:
            rendering.validate_replay_context([{"messages": messages}], model)


def test_actual_native_boundary_masks_each_target_with_changing_tools():
    from test_assistant_bindings import example
    row, = dataset.prepare_sft_rows([example()[0]])
    renderer = _renderer()
    rows = rendering.render_row_tokens(row, MODEL, renderer=renderer)
    assert len(rows) == 3
    for i, datum in enumerate(rows):
        supervised = ''.join(_loss_spans(datum, renderer.tokenizer))
        assert f'answer-{i}' in supervised
        assert all(f'answer-{j}' not in supervised for j in range(i))
        text = renderer.tokenizer.decode(datum.token_ids)
        assert ('unused' in text) == (i == 1)
        assert supervised.endswith('<|im_end|>\n')

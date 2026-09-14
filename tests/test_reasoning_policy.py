from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest

from smithtune import capabilities, dataset, rendering
from smithtune import evaluation
from smithtune import inference
from smithtune import cli as pipeline
from smithtune.models import resolve_prepared_model
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from test_pipeline import example, loaded_contract, message, write_raw


def reasoning(text="recorded reasoning"):
    return {"type": "reasoning", "reasoning": text,
            "signature": "opaque-signature", "encrypted_content": "opaque-state"}


def trajectory():
    return example(0, [
        message("system", "policy", "s"),
        message("human", "look up x", "u"),
        message("ai", [reasoning("reasoning only")], "r"),
        message("ai", [reasoning(),
                       {"type": "tool_call", "id": "call", "name": "lookup", "args": {"query": "x"}}], "a"),
        message("tool", "result", "t", tool_call_id="call"),
        message("ai", [reasoning("answer reasoning"), {"type": "text", "text": "answer"}], "f"),
    ])


def test_default_omits_reasoning_without_changing_source_or_tool_interactions(tmp_path):
    source = trajectory()
    before = copy.deepcopy(source)
    audit = dataset.validate_trajectories([source], 1)
    rows = dataset.prepare_sft_rows([source], loaded_contract(tmp_path))
    messages = rows[0]["messages"]

    assert source == before
    assert audit.reasoning_blocks == 3
    assert audit.messages == 6
    assert [item["id"] for item in messages] == ["s", "u", "a", "t", "f"]
    assert messages[0]["content"] == "policy"
    assert messages[2]["tool_calls"][0] == {
        "id": "call", "type": "function",
        "function": {"name": "lookup", "arguments": '{"query":"x"}'},
    }
    assert messages[3]["tool_call_id"] == "call"
    assert messages[3]["content"] == "result"
    assert messages[4]["content"] == [{"type": "text", "text": "answer"}]
    serialized = json.dumps(messages)
    assert "reasoning" not in serialized
    assert "opaque" not in serialized


@pytest.mark.parametrize("model", [fireworks.DEFAULT_MODEL, baseten.DEFAULT_MODEL, fireworks.MODEL_SPECS["kimi-k3"]])
def test_preserve_requires_opt_in_and_keeps_readable_text_separate(model):
    source = trajectory()
    preserved = dataset.prepare_sft_rows([source], reasoning_policy="preserve", model=model)[0]["messages"]
    omitted = dataset.prepare_sft_rows([source], model=model)[0]["messages"]
    assert len(preserved) == 6
    assert preserved[2]["reasoning_content"] == "reasoning only"
    assert preserved[3]["reasoning_content"] == "recorded reasoning"
    assert preserved[3]["tool_calls"] == omitted[2]["tool_calls"]
    assert preserved[5]["reasoning_content"] == "answer reasoning"
    assert preserved[5]["content"] == omitted[4]["content"]
    assert "opaque" not in json.dumps(preserved)


@pytest.mark.parametrize("model,error", [
    (None, "requires a target ModelSpec"),
    (replace(fireworks.DEFAULT_MODEL, supports_reasoning_content=False), "does not support reasoning"),
    (replace(fireworks.DEFAULT_MODEL, renderer="unverified", thinking_trace_history_mode=""), "no verified reasoning"),
])
def test_preserve_rejects_incompatible_target_even_without_render_check(model, error):
    with pytest.raises(PipelineError, match=error):
        dataset.prepare_sft_rows([trajectory()], reasoning_policy="preserve", model=model)


def test_no_reasoning_data_is_identical_under_either_policy():
    source = [example(i) for i in range(3)]
    model = replace(fireworks.DEFAULT_MODEL, supports_reasoning_content=False)
    assert dataset.prepare_sft_rows(source) == dataset.prepare_sft_rows(
        source, reasoning_policy="preserve", model=model,
    )


def test_omit_rejects_an_example_left_without_a_training_target():
    source = example(0, [message("human", "question", "u"), message("ai", [reasoning()], "a")])
    dataset.validate_trajectories([source], 1)
    with pytest.raises(PipelineError, match="no assistant training target after reasoning"):
        dataset.prepare_sft_rows([source])


def test_removing_reasoning_does_not_hide_unmatched_tool_results():
    source = trajectory()
    source["inputs"]["messages"][4]["tool_call_id"] = "missing"
    with pytest.raises(PipelineError, match="unmatched tool calls or results"):
        dataset.prepare_sft_rows([source])


def test_preserve_rejects_reordering_reasoning_after_text():
    source = message("ai", [{"type": "text", "text": "answer"}, reasoning()], "a")
    assert dataset.convert_message(source)["content"] == [{"type": "text", "text": "answer"}]
    with pytest.raises(PipelineError, match="without reordering"):
        dataset.convert_message(source, reasoning_policy="preserve", model=fireworks.DEFAULT_MODEL)


@pytest.mark.parametrize("policy", ["omit", "preserve"])
def test_prepare_manifest_and_replay_use_same_policy(tmp_path, monkeypatch, policy):
    source = trajectory()
    # Include an opaque-only reasoning message to exercise omission under preserve.
    source["inputs"]["messages"].insert(2, message("ai", [reasoning("")], "opaque-only"))
    write_raw(tmp_path, [source], workspace_id="workspace")
    original_bytes = (tmp_path / "raw" / "examples.json").read_bytes()
    manifest = dataset.prepare_dataset(
        "workspace", "dataset-id", fireworks.DEFAULT_MODEL, tmp_path,
        inference_contract=loaded_contract(tmp_path), reasoning_policy=policy,
        validation_fraction=0, test_fraction=1, fetch=False, check_render=False,
    )
    assert (tmp_path / "raw" / "examples.json").read_bytes() == original_bytes
    assert manifest["conversion"]["reasoning_policy"] == policy
    assert manifest["conversion"]["reasoning_blocks"] == {
        "source": 4, "preserved": 3 if policy == "preserve" else 0,
        "omitted": 1 if policy == "preserve" else 4,
    }
    assert manifest["conversion"]["messages_removed"] == (1 if policy == "preserve" else 2)
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, *args: (
        [{**case, "prompt_tokens": 10} for case in cases], [],
    ))
    plan = evaluation.prepare_replay_evaluation(tmp_path, tmp_path / "eval")
    assert plan["reasoning_policy"] == policy
    cases = [json.loads(line) for line in (tmp_path / "eval" / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 2
    assert [case["reference"]["id"] for case in cases] == ["a", "f"]
    last_prefix = inference._inference_messages(cases[-1]["messages"])
    assert any(item.get("reasoning_content") for item in last_prefix) == (policy == "preserve")
    assert "opaque" not in json.dumps(last_prefix)
    assert ("reasoning_content" in cases[-1]["reference"]) == (policy == "preserve")
    evidence = json.loads(evaluation._judge_input(cases[-1], cases[-1]["reference"])[1]["content"])
    assert "reasoning_content" not in evidence["candidate_next_action"]
    assert "reasoning_content" not in evidence["reference_next_action"]
    assert evidence["trajectory_prefix_visible_to_candidate"] == last_prefix
    if policy == "preserve":
        manifest["conversion"]["reasoning_policy"] = "omit"
        (tmp_path / "prepared" / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(PipelineError, match="contain reasoning despite"):
            evaluation.prepare_replay_evaluation(tmp_path, tmp_path / "eval")


@pytest.mark.parametrize("provider", [fireworks.FireworksProvider(), baseten.BasetenProvider()])
@pytest.mark.parametrize("policy", ["omit", "preserve"])
def test_cli_and_provider_pass_preparation_policy(tmp_path, monkeypatch, provider, policy):
    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    capability_calls = []
    rendering_calls = []

    def fireworks_capability(model, context):
        capability_calls.append((model, context))
        return capabilities.FireworksModelCapability(model, "Qwen/Qwen3.8-27B", 131_072, True)

    def baseten_capability(model, context):
        capability_calls.append((model, context))
        return baseten.BasetenModelCapability(model, 262_144)

    def resolve_rendering(model):
        rendering_calls.append(model)
        return model

    monkeypatch.setattr(capabilities, "fetch_fireworks_model_capability", fireworks_capability)
    monkeypatch.setattr(baseten, "fetch_model_capability", baseten_capability)
    adapter_module = fireworks if provider.name == "fireworks" else baseten
    monkeypatch.setattr(adapter_module, "resolve_rendering_model", resolve_rendering)
    monkeypatch.setattr(rendering, "resolved_renderer_name", lambda model: model.renderer)
    source = example(0, [message("human", "question", "u"),
                         message("ai", [reasoning(), {"type": "text", "text": "answer"}], "a")])
    write_raw(tmp_path, [source], workspace_id="workspace")
    argv = ["pipeline.py", "prepare", "--provider", provider.name,
            "--model", "qwen3p8-27b", "--workspace-id", "workspace", "--dataset-id", "dataset-id",
            "--data-dir", str(tmp_path), "--validation-fraction", "0", "--test-fraction", "0",
            "--no-fetch", "--skip-render-check"]
    if policy == "preserve":
        argv += ["--reasoning-policy", policy]
    monkeypatch.setattr("sys.argv", argv)
    pipeline.main()
    manifest = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "prepared" / "train.jsonl").read_text().splitlines()]
    assert capability_calls == [(adapter_module.DEFAULT_MODEL.base_model, adapter_module.DEFAULT_MODEL.training_context_limit)]
    assert rendering_calls == [adapter_module.DEFAULT_MODEL]
    assert manifest["conversion"]["reasoning_policy"] == policy
    assert ("reasoning_content" in rows[0]["messages"][1]) == (policy == "preserve")


def test_legacy_manifest_keeps_conservative_capability_and_trainer_limit():
    model = asdict(baseten.DEFAULT_MODEL)
    del model["supports_reasoning_content"]
    del model["trainer_max_seq_len"]
    manifest = {"model": model, "provider": {
        "name": "baseten", "renderer": model["renderer"], "tokenizer_revision": model["tokenizer_revision"],
    }}
    resolved = resolve_prepared_model(manifest, baseten.MODEL_SPECS, provider="baseten")
    assert resolved.supports_reasoning_content is False
    assert resolved.trainer_max_seq_len == baseten.DEFAULT_MODEL.trainer_max_seq_len


def test_reasoning_capability_must_be_boolean():
    with pytest.raises(PipelineError, match="must be boolean"):
        replace(baseten.DEFAULT_MODEL, supports_reasoning_content="yes").validate()


def test_invalid_policy_fails_even_without_source_reasoning():
    with pytest.raises(PipelineError, match="reasoning_policy"):
        dataset.prepare_sft_rows([example(0)], reasoning_policy="auto")


def test_reasoning_only_trajectory_has_no_action_replay_cases():
    source = example(0, [message("human", "question", "u"), message("ai", [reasoning()], "a")])
    rows = dataset.prepare_sft_rows([source], model=fireworks.DEFAULT_MODEL, reasoning_policy="preserve")
    assert evaluation.build_replay_cases(rows) == []


@pytest.mark.parametrize("role,block", [
    ("human", reasoning()), ("ai", {"type": "reasoning", "reasoning": 123}),
    ("ai", {"type": "image", "data": "not text"}),
])
def test_native_validation_still_rejects_invalid_blocks(role, block):
    with pytest.raises(PipelineError):
        dataset.validate_trajectories([example(0, [message(role, [block], "m")])], 1)


class CharacterTokenizer:
    """Expose the actual renderer's text and loss masks without a model download."""

    name_or_path = "test-tokenizer"
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(int(token)) for token in tokens)


@pytest.mark.parametrize("policy", ["omit", "preserve"])
def test_real_qwen_renderer_applies_policy_to_history_and_supervised_tokens(tmp_path, policy):
    from training.renderer import get_renderer
    from training.utils import parse_train_on_what, render_messages_to_datums
    from training.utils.supervised import build_tool_prefixed_messages
    from smithtune.rendering import SFT_TARGET_POLICY

    tokenizer = CharacterTokenizer()
    renderer = get_renderer("qwen3_8_preserved", tokenizer)
    row = dataset.prepare_sft_rows(
        [trajectory()], loaded_contract(tmp_path),
        model=fireworks.DEFAULT_MODEL, reasoning_policy=policy,
    )[0]
    datum = render_messages_to_datums(
        row["messages"], renderer=renderer, tools=row["tools"],
        train_on_what=parse_train_on_what(SFT_TARGET_POLICY), reduction="mean",
    )
    datums = datum if isinstance(datum, list) else [datum]
    rendered_text = "".join(tokenizer.decode(item.token_ids) for item in datums)
    target_text = "".join(tokenizer.decode([
        token for token, weight in zip(item.token_ids, item.token_weights, strict=True) if weight > 0
    ]) for item in datums)
    assert ("recorded reasoning" in rendered_text) == (policy == "preserve")
    assert ("recorded reasoning" in target_text) == (policy == "preserve")
    assert ("answer reasoning" in target_text) == (policy == "preserve")
    assert "lookup" in target_text
    assert "answer" in target_text
    assert "opaque" not in rendered_text

    cases = evaluation.build_replay_cases([row])
    normalized = build_tool_prefixed_messages(
        cases[-1]["messages"], renderer=renderer, tools=row["tools"],
    )
    prefix = tokenizer.decode(renderer.build_generation_prompt(normalized).to_ints())
    assert ("recorded reasoning" in prefix) == (policy == "preserve")
    assert "answer reasoning" not in prefix

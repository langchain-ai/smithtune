"""Saved GLM effort stays consistent across preparation, training, and replay."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from smithtune import capabilities, cli, rendering
from smithtune.dataset import _model_from_manifest
from smithtune.models import resolve_model_options
from smithtune.providers import baseten, fireworks, fireworks_sampling, fireworks_training
from smithtune.providers.base import ModelOptions, PipelineError
from test_hf_rendering import _tokenizer
from test_pipeline import example, write_raw


@pytest.fixture
def glm_tokenizer(monkeypatch):
    import training.utils.tokenizers
    import training.utils.supervised

    tokenizer = _tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "[gMASK]", "[sop]", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
    ]})
    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(training.utils.supervised, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    return tokenizer


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_cli_selects_glm_effort(effort):
    args = cli._parser().parse_args([
        "prepare", "--model", "glm-5p3-flash", "--workspace-id", "workspace",
        "--dataset-id", "dataset", "--reasoning-effort", effort,
    ])
    model = fireworks.FireworksProvider().model_from_options(cli._model_options(args))
    assert model.reasoning_effort == effort


@pytest.mark.parametrize("provider,alias", [(fireworks, "qwen3p8-27b"), (baseten, "glm-5p3-flash")])
def test_effort_is_rejected_for_unqualified_provider_models(provider, alias):
    with pytest.raises(PipelineError, match="only for Fireworks GLM-5.3-Flash"):
        resolve_model_options(
            ModelOptions(model=alias, reasoning_effort="low"),
            provider.MODEL_SPECS, provider=provider.DEFAULT_MODEL.provider,
        )


@pytest.mark.parametrize("value", ["none", "medium", "", False, 3, []])
def test_invalid_effort_is_rejected_in_saved_model_profiles(value):
    with pytest.raises(PipelineError, match="reasoning_effort must be"):
        replace(fireworks.MODEL_SPECS["glm-5p3-flash"], reasoning_effort=value).validate()


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_saved_effort_controls_preparation_plan_training_workers_and_replay(tmp_path, monkeypatch, glm_tokenizer, effort):
    from training.recipes import sft_loop
    from training.utils import resource_autosizing

    monkeypatch.setattr(capabilities, "fetch_fireworks_model_capability", lambda model, context: (
        capabilities.FireworksModelCapability(model, "zai-org/GLM-5.3-Flash", context, True)
    ))
    write_raw(tmp_path, [example(i) for i in range(10)], dataset_id="dataset", workspace_id="workspace")
    provider = fireworks.FireworksProvider()
    manifest = provider.prepare(
        "workspace", "dataset", tmp_path, fetch=False,
        model_options=ModelOptions(model="glm-5p3-flash", reasoning_effort=effort),
    )
    saved = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert saved == manifest
    assert saved["model"]["reasoning_effort"] == effort
    plan = provider.plan(tmp_path, "test-run", fireworks.SFTSettings())
    assert plan["config"]["reasoning_effort"] == effort
    model = _model_from_manifest(saved)
    renderer = rendering.load_training_renderer(model)
    rows = [json.loads(line) for line in (tmp_path / "prepared/train.jsonl").read_text().splitlines()]
    expected = rendering.render_row_tokens(rows[0], model, renderer=renderer)[0]
    assert f"Reasoning Effort: {effort.capitalize()}" in glm_tokenizer.decode(expected.token_ids)

    monkeypatch.setattr(resource_autosizing, "select_render_worker_count", lambda **kwargs: 0)
    config = SimpleNamespace(
        tokenizer_model=model.tokenizer_model, renderer_name=model.renderer,
        train_on_what=rendering.SFT_TARGET_POLICY, max_seq_len=model.max_seq_len,
        tokenizer_revision=model.tokenizer_revision, thinking_trace_history_mode=model.thinking_trace_history_mode,
        tokenizer_trust_remote_code=False, dataset=str(tmp_path / "prepared/train.jsonl"),
        evaluation_dataset=str(tmp_path / "prepared/train.jsonl"), max_examples=None,
        batch_size=1, group_by_length=False, length_group_factor=50, epochs=1, seed=42,
    )
    session = fireworks_training.ServerlessTraining(config, tmp_path, reasoning_effort=model.reasoning_effort)
    assert session.validation[0].model_input.to_ints() == expected.datum.model_input.to_ints()
    for batch in session.loader:
        for datum in sft_loop._flatten_rendered_batch(batch):
            assert f"Reasoning Effort: {effort.capitalize()}" in glm_tokenizer.decode(datum.model_input.to_ints())
    # Exercise the initializer supplied for multiprocessing without starting processes.
    monkeypatch.setattr(resource_autosizing, "select_render_worker_count", lambda **kwargs: 2)
    worker_session = fireworks_training.ServerlessTraining(config, tmp_path, reasoning_effort=model.reasoning_effort)
    worker_session.loader.worker_init_fn(1)
    worker_prompt = rendering.replay_prompt(rows[0]["messages"][:-1], [], model, sft_loop._worker_state["renderer"])
    replay_prompt = rendering.replay_prompt(rows[0]["messages"][:-1], [], model, renderer)
    assert worker_prompt.to_ints() == replay_prompt.to_ints()


def test_older_glm_profiles_keep_the_default_max_effort(glm_tokenizer):
    model = fireworks.MODEL_SPECS["glm-5p3-flash"]
    legacy = asdict(model)
    del legacy["reasoning_effort"]
    restored = _model_from_manifest({"model": legacy})
    restored.validate()
    messages = [{"role": "user", "content": "Question"}]
    old_prompt = rendering.replay_prompt(messages, [], restored, rendering.load_training_renderer(restored))
    current_prompt = rendering.replay_prompt(messages, [], model, rendering.load_training_renderer(model))
    assert old_prompt.to_ints() == current_prompt.to_ints()


def test_standalone_replay_rejects_effort_different_from_the_training_run(tmp_path):
    model = replace(fireworks.MODEL_SPECS["glm-5p3-flash"], reasoning_effort="low")
    data = tmp_path / "data/prepared"
    run = tmp_path / "run"
    data.mkdir(parents=True)
    run.mkdir()
    (data / "manifest.json").write_text(json.dumps({"model": asdict(model), "provider": {
        "name": "fireworks", "renderer": model.renderer, "tokenizer_revision": model.tokenizer_revision,
    }}))
    (run / "plan.json").write_text(json.dumps({"base_model": model.base_model, "config": {"reasoning_effort": "max"}}))
    (run / "result.json").write_text(json.dumps({"best": {"resume_checkpoint": "account/run-" + "a" * 32 + "/epoch-1"}}))
    with pytest.raises(PipelineError, match="reasoning effort differs"):
        fireworks_sampling.checkpoint_from_run(data.parent, run)


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_replay_identity_preserves_legacy_default_and_distinguishes_effort(tmp_path, monkeypatch, effort):
    monkeypatch.setattr(fireworks_sampling, "load_training_renderer", lambda model: object())
    model = replace(fireworks.MODEL_SPECS["glm-5p3-flash"], reasoning_effort=effort)
    sampler = fireworks_sampling.FireworksReplaySampler(model, "checkpoint", tmp_path)
    legacy_config = {
        "provider": "fireworks", "serving_mode": "serverless", "checkpoint": "checkpoint",
        "base_model": model.base_model, "lora_rank": model.default_lora_rank, "lora_alpha": 32,
    }
    if effort in (None, "max"):
        assert sampler.config == legacy_config
    else:
        assert sampler.config == {**legacy_config, "reasoning_effort": effort}

"""Tiny, randomly initialized CPU models; no downloads or paid training."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from smithtune.providers.base import CommonSFTSettings
from smithtune.training.hf_sft import Example, _loss, configure_lora, train_loop

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


def tiny_qwen(*, layer_types=None, max_position_embeddings=64):
    torch.manual_seed(42)
    layer_types = layer_types or ["linear_attention", "full_attention"]
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": len(layer_types), "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": max_position_embeddings, "layer_types": layer_types,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16},
        image_token_id=28, video_token_id=29, vision_start_token_id=30, vision_end_token_id=31,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen3_5ForConditionalGeneration(config)


def test_real_qwen_peft_cpu_smoke_updates_only_adapters_and_roundtrips(tmp_path):
    base = tiny_qwen()
    initial = deepcopy(base)
    model = configure_lora(base, 2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    adapters = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    item = Example([1, 2, 3, 4, 5], [-100, -100, 3, -100, 5])
    settings = CommonSFTSettings(max_epochs=3, early_stopping_patience=3, learning_rate=0.01, batch_size=2)
    records = []
    result = train_loop(model, [[item], [item], [item]], [[item]], settings, device="cpu",
                        on_epoch=lambda record, state: records.append((record, state)))
    assert result["optimizer_steps"] == 6  # includes the final partial batch
    assert result["epochs"][-1]["validation_loss"] < result["epochs"][0]["validation_loss"]
    assert all(torch.equal(value, frozen[name]) for name, value in model.named_parameters() if name in frozen)
    assert any(not torch.equal(value, adapters[name]) for name, value in model.named_parameters() if name in adapters)
    assert all("lora_" in name and "language_model" in name for name in adapters)
    assert records[-1][1]["optimizer"]["state"]
    model.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    reloaded = peft.PeftModel.from_pretrained(initial, tmp_path / "adapter")
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(_loss(model, item, "cpu"), _loss(reloaded, item, "cpu"))


def test_loss_masks_and_causal_shift_match_manual_cross_entropy():
    model = configure_lora(tiny_qwen(), 2).eval()
    item = Example([1, 2, 3, 4, 5], [-100, -100, 3, -100, 5])
    with torch.no_grad():
        ids = torch.tensor([item.input_ids])
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        manual = torch.nn.functional.cross_entropy(logits[0, [1, 3]].float(), torch.tensor([3, 5]))
        torch.testing.assert_close(_loss(model, item, "cpu"), manual)


def test_bf16_sdpa_checkpointed_training_matches_gpu_recipe_without_downloads():
    # Linear attention after a trainable attention layer must also backpropagate;
    # test multiple 64-token fallback chunks rather than only a tiny first chunk.
    base = tiny_qwen(layer_types=["full_attention", "linear_attention", "full_attention"],
                     max_position_embeddings=512).to(dtype=torch.bfloat16)
    base.set_attn_implementation("sdpa")
    model = configure_lora(base, 2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = [index % 15 + 1 for index in range(437)]
    item = Example(tokens, [-100, *tokens[1:]])
    result = train_loop(model, [[item]], [[item]], CommonSFTSettings(max_epochs=1, batch_size=1), device="cpu")
    assert result["optimizer_steps"] == 1
    assert torch.isfinite(torch.tensor(result["best_validation_loss"]))


@pytest.mark.parametrize(("message", "kind"), [
    ("Cannot copy out of meta tensor", "unmaterialized_model_tensor"),
    ("Failed to find C compiler", "runtime_compiler_required"),
])
def test_worker_failure_diagnostics_keep_exception_text_and_locals_private(tmp_path, monkeypatch, capsys, message, kind):
    import smithtune.training.hf_sft as worker

    marker = "PRIVATE_TRAINING_CONTENT"

    def fail(*args, **kwargs):
        worker.report_stage("moving_model")
        raise RuntimeError(f"{message}; {marker}")

    monkeypatch.setattr(worker, "run", fail)
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda: {"cuda_available": True})
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(tmp_path / "output/data"))
    with pytest.raises(SystemExit) as caught:
        from smithtune.providers.sagemaker.worker import main
        main(["train"])
    assert caught.value.code == 1
    diagnostic = (tmp_path / "output/data/worker-failure.json").read_text()
    failure = (tmp_path / "output/failure").read_text()
    logs = capsys.readouterr()
    assert marker not in diagnostic + failure + logs.out + logs.err
    assert kind in diagnostic
    assert "moving_model" in failure


def test_gpu_inspection_failure_does_not_leak_runtime_error(monkeypatch):
    from smithtune.training.hf_sft import gpu_diagnostics

    def fail():
        raise RuntimeError("PRIVATE_DRIVER_DETAILS")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    assert gpu_diagnostics() == {"inspection_error_type": "RuntimeError"}


class ScalarModel(torch.nn.Module):
    """Analytic mean token loss to observe gradient accumulation exactly."""

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, labels, **kwargs):
        selected = labels[:, 1:]
        targets = selected[selected != -100].float() / 100
        return SimpleNamespace(loss=((self.value - targets) ** 2).mean())


def test_gradient_accumulation_weights_active_tokens_including_partial_batch(monkeypatch):
    actual = []
    original_step = torch.optim.AdamW.step

    def step(optimizer, *args, **kwargs):
        actual.append(optimizer.param_groups[0]["params"][0].grad.clone())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    rows = [
        [Example([1, 20, 30], [-100, 20, 30])],
        [Example([1, 40, 50], [-100, -100, 50])],
    ]
    model = ScalarModel()
    train_loop(model, rows, rows, CommonSFTSettings(max_epochs=1, batch_size=8), device="cpu")
    # 2 * (0.1 - mean(0.2, 0.3, 0.5)); not the mean of two example losses.
    torch.testing.assert_close(actual[0], torch.tensor(-0.46666667))


def test_best_checkpoint_is_independent_of_early_stopping_min_delta(monkeypatch):
    import smithtune.training.hf_sft as worker
    model = ScalarModel()
    values = iter([1.0, 0.95, 0.96])

    def loss(model, example, device):
        return model.value.square() if model.training else torch.tensor(next(values))

    monkeypatch.setattr(worker, "_loss", loss)
    rows = [[Example([1, 2], [-100, 2])]]
    result = train_loop(model, rows, rows, CommonSFTSettings(
        max_epochs=5, early_stopping_min_delta=0.1, early_stopping_patience=2,
    ), device="cpu")
    assert result["best_epoch"] == 2
    assert len(result["epochs"]) == 3


def test_staged_bundle_to_best_adapter_artifact_on_cpu(tmp_path, monkeypatch):
    import smithtune.training.hf_sft as worker
    from smithtune.artifacts import _load_json
    from smithtune.providers import sagemaker
    from tests.test_sagemaker import FakeAWS, config, prepared

    prepared(tmp_path)
    remote = FakeAWS()
    settings = sagemaker.SageMakerSFTSettings(aws=config(), max_epochs=2, batch_size=1, learning_rate=0.01)
    sagemaker.SageMakerProvider(session_factory=lambda _: remote).train(
        tmp_path, tmp_path / "run", "sft-test", settings, confirm=True,
    )
    calls = []

    def load_model(name, **kwargs):
        calls.append((name, kwargs))
        return tiny_qwen()

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", load_model)

    class Tokenizer:
        def save_pretrained(self, path):
            path.mkdir()
            (path / "tokenizer_config.json").write_text('{}')

    renderer = SimpleNamespace(
        tokenizer=Tokenizer(),
        render=lambda messages, tools=None, final_target=False: [SimpleNamespace(token_ids=[1, 2, 3, 4], token_weights=[0, 1, 0, 1])],
    )
    monkeypatch.setattr(worker, "load_training_renderer", lambda _: renderer)
    result = worker.run(tmp_path / "run/input", tmp_path / "model", tmp_path / "checkpoints", device="cpu")
    assert calls[0][0] == "Qwen/Qwen3.5-9B"
    assert calls[0][1]["revision"] == sagemaker.MODEL_SPECS["qwen3p5-9b"].tokenizer_revision
    assert result["artifact_type"] == "peft_lora_adapter"
    assert result["max_sequence_tokens"] == 4
    for epoch in (1, 2):
        folder = tmp_path / f"checkpoints/epoch-{epoch}"
        assert (folder / "complete.json").exists()
        state = torch.load(folder / "training-state.pt", map_location="cpu", weights_only=True)
        assert state["completed_epoch"] == epoch
        assert state["optimizer"]["state"]
    selected = tmp_path / f"checkpoints/epoch-{result['best_epoch']}/adapter/adapter_model.safetensors"
    assert selected.read_bytes() == (tmp_path / "model/adapter/adapter_model.safetensors").read_bytes()
    assert _load_json(tmp_path / "model/result.json")["bundle"]["files"]

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker as tinker_sdk

from smithtune import cli
from smithtune.providers import lora_sft
from smithtune.providers import tinker as tinker_package
from smithtune.providers.base import ModelOptions, PipelineError, TrainingOptions
from smithtune.providers.tinker import provider as tinker
from tests.binding_fixtures import bound_row


class FakeFuture:
    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


class FakeHolder:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _weights(datum) -> list[float]:
    return lora_sft.datum_weights(datum)


class FakeTrainer:
    model_id = "tinker-run-1"

    def __init__(self, validation_losses=(1.0, 0.8, 0.9), *, fail_training: BaseException | None = None) -> None:
        self.validation_losses = iter(validation_losses)
        self.fail_training = fail_training
        self.forward_backward_calls: list[list] = []
        self.forward_calls: list[list] = []
        self.learning_rates: list[float] = []
        self.saved_states: list[str] = []
        self.saved_samplers: list[str] = []
        self.loaded: list[str] = []

    def forward_backward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        if self.fail_training is not None:
            raise self.fail_training
        self.forward_backward_calls.append(data)
        # Tinker sums a per-token loss of 0.5 over the weighted target tokens.
        return FakeFuture(SimpleNamespace(metrics={"loss:sum": 0.5 * sum(sum(_weights(d)) for d in data)}))

    def forward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.forward_calls.append(data)
        per_token = next(self.validation_losses)
        return FakeFuture(SimpleNamespace(metrics={"loss:sum": per_token * sum(sum(_weights(d)) for d in data)}))

    def optim_step(self, params):
        self.learning_rates.append(params.learning_rate)
        return FakeFuture(SimpleNamespace(metrics={"grad_norm": 1.0}))

    def save_state(self, name):
        self.saved_states.append(name)
        return FakeFuture(SimpleNamespace(path=f"tinker://{self.model_id}/weights/{name}"))

    def save_weights_for_sampler(self, name):
        self.saved_samplers.append(name)
        return FakeFuture(SimpleNamespace(path=f"tinker://{self.model_id}/sampler_weights/{name}"))

    def load_state_with_optimizer(self, path):
        self.loaded.append(path)
        return FakeFuture(SimpleNamespace(path=path))


class FakeService:
    def __init__(self, trainer: FakeTrainer) -> None:
        self.trainer = trainer
        self.holder = FakeHolder()
        self.factory_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def create_lora_training_client(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.trainer


def _row(index: int = 1) -> dict:
    return bound_row({
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        "tools": [],
        "_source": {"example_id": f"example-{index}", "source_scope": "thread", "source_scope_id": f"thread-{index}"},
    })


def _write_prepared_dataset(root: Path, *, model=None, max_context_tokens: int = 4_096) -> None:
    prepared = root / "prepared"
    prepared.mkdir()
    model = model or tinker.DEFAULT_MODEL
    manifest = {
        "schema_version": 2, "target_policy": "each_assistant_once",
        "langsmith": {"workspace_id": "workspace-id", "dataset_id": "dataset-id", "examples": 100},
        "split": {"train": 90, "validation": 10, "test": 0},
        "model": model.__dict__,
        "provider": {"name": "tinker", "renderer": model.renderer, "tokenizer_revision": model.tokenizer_revision},
        "audit": {"max_context_tokens": max_context_tokens},
        "source_examples_sha256": "a" * 64,
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [json.dumps(_row(index)) for index in range(100)]
    (prepared / "train.jsonl").write_text("\n".join(rows[:90]) + "\n", encoding="utf-8")
    (prepared / "validation.jsonl").write_text("\n".join(rows[90:]) + "\n", encoding="utf-8")
    (prepared / "test.jsonl").write_text("", encoding="utf-8")


def _rendered(row, model, **kwargs):
    # Two target tokens per datum: [prompt, prompt, target, target] shifts to [0, 1, 1].
    target = SimpleNamespace(token_ids=[1, 2, 3, 4], token_weights=[0, 0, 1, 1])
    return [lora_sft.to_datum(target, kwargs["tinker_types"], model.max_seq_len, ignore_index=None)]


def _provider(service: FakeService, **overrides) -> tinker.TinkerProvider:
    def service_factory(**kwargs):
        service.factory_calls.append(kwargs)
        return service

    arguments = {
        "capability_resolver": lambda model, length: tinker.TinkerModelCapability(model, 65_536),
        "service_factory": service_factory,
        "tinker_types": tinker_sdk,
        "render_fn": _rendered,
        "renderer_factory": lambda model: object(),
        "sleeper": lambda seconds: None,
    }
    arguments.update(overrides)
    return tinker.TinkerProvider(**arguments)


def _train(provider, root, settings=None, *, run_id="approved-run", init_from_checkpoint=None):
    return provider.train(
        root, root / "run", run_id, settings or tinker.TinkerSFTSettings(max_epochs=2, batch_size=4),
        confirm=True, init_from_checkpoint=init_from_checkpoint,
    )


def test_tinker_profiles_render_with_hugging_face_templates():
    for alias, model in tinker.MODEL_SPECS.items():
        assert model.provider == "tinker"
        resolved = tinker.TinkerProvider().model_from_options(ModelOptions(model=model.base_model))
        assert resolved == tinker.TinkerProvider().model_from_options(ModelOptions(model=alias))
    assert tinker.DEFAULT_MODEL.base_model == "Qwen/Qwen3.8-27B"
    assert tinker.MODEL_SPECS["qwen3p5-9b"].renderer == "hf_prefix_qwen3_5"
    with pytest.raises(PipelineError, match="supported aliases"):
        tinker.TinkerProvider().model_from_options(ModelOptions(model="kimi-k3"))


def test_settings_accept_shared_and_microbatch_options_and_reject_other_providers():
    provider = tinker.TinkerProvider()
    settings = provider.settings_from_options(TrainingOptions(
        lora_rank=16, microbatch_token_budget=8_192, max_dropped_training_rows=0, batch_size=8,
    ))
    assert (settings.lora_rank, settings.microbatch_token_budget, settings.max_dropped_training_rows) == (16, 8_192, 0)
    with pytest.raises(PipelineError, match="Fireworks-only"):
        provider.settings_from_options(TrainingOptions(lora_alpha=32))
    with pytest.raises(PipelineError, match="Baseten-only"):
        provider.settings_from_options(TrainingOptions(max_spend_usd=10.0, hourly_rate_usd=5.0))
    with pytest.raises(PipelineError, match="Baseten-only"):
        provider.settings_from_options(TrainingOptions(replicas=2))


def test_plan_reports_token_mean_loss_and_needs_no_remote_calls(tmp_path):
    _write_prepared_dataset(tmp_path)
    plan = tinker.TinkerProvider(capability_resolver=pytest.fail).plan(
        tmp_path, "preview", tinker.TinkerSFTSettings(),
    )
    assert plan["provider"] == "tinker"
    assert plan["execution"] == {"requires_active_client": True}
    assert plan["config"]["max_seq_len"] == 4_096
    assert plan["config"]["lora_rank"] == tinker.DEFAULT_MODEL.default_lora_rank
    assert plan["config"]["loss_normalization"] == tinker.LOSS_NORMALIZATION
    assert plan["training_schedule"]["estimate"] is True
    assert plan["budget"] == {"enabled": False}


def test_plan_rejects_data_prepared_for_another_provider(tmp_path):
    from smithtune.providers.baseten.provider import DEFAULT_MODEL as BASETEN_MODEL

    _write_prepared_dataset(tmp_path, model=BASETEN_MODEL)
    with pytest.raises(PipelineError, match="prepared provider mismatch"):
        tinker.TinkerProvider().plan(tmp_path, "preview", tinker.TinkerSFTSettings())


def test_render_row_keeps_real_targets_with_zero_weight(monkeypatch):
    monkeypatch.setattr(tinker, "render_row_tokens", lambda *args, **kwargs: [
        SimpleNamespace(token_ids=[10, 20, 30, 40], token_weights=[0, 0, 1, 1]),
    ])
    [datum] = tinker.render_row(_row(), tinker.DEFAULT_MODEL, tinker_types=tinker_sdk, renderer=object())
    assert datum.model_input.to_ints() == [10, 20, 30]
    # Tinker indexes target tokens directly, so excluded positions keep a real
    # token instead of Loops' -100 sentinel.
    assert datum.loss_fn_inputs["target_tokens"].data == [20, 30, 40]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0]


def test_train_scales_each_step_to_a_token_mean_selects_the_best_checkpoint_and_closes_the_session(tmp_path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=(1.0, 0.8))
    service = FakeService(trainer)

    result = _train(_provider(service), tmp_path)

    assert service.factory_calls == [{"user_metadata": {"smithtune_run_id": "approved-run"}}]
    assert service.create_calls == [{
        "base_model": "Qwen/Qwen3.8-27B", "rank": tinker.DEFAULT_MODEL.default_lora_rank, "seed": 42,
        "user_metadata": {"smithtune_run_id": "approved-run"},
    }]
    # 90 one-datum rows in batches of four: 22 full steps and one of two datums.
    assert len(trainer.forward_backward_calls) == 2 * 23
    assert [len(batch) for batch in trainer.forward_backward_calls[:23]] == [4] * 22 + [2]
    for batch in trainer.forward_backward_calls:
        weights = [weight for datum in batch for weight in _weights(datum)]
        assert sum(weights) == pytest.approx(1.0)
    assert _weights(trainer.forward_backward_calls[0][0]) == pytest.approx([0.0, 0.125, 0.125])
    # Validation uses the unscaled binary mask and divides the summed loss.
    assert all(_weights(datum) == [0.0, 1.0, 1.0] for batch in trainer.forward_calls for datum in batch)
    assert trainer.learning_rates == [1e-4] * 46

    training_losses = [entry["loss"] for entry in result["step_losses"] if entry["operation"] == "forward_backward"]
    assert training_losses == pytest.approx([0.5] * 46)
    assert [epoch["validation_loss"] for epoch in result["epochs"]] == pytest.approx([1.0, 0.8])
    assert [epoch["train_loss"] for epoch in result["epochs"]] == pytest.approx([0.5, 0.5])
    assert result["status"] == "completed"
    assert result["tinker_run_id"] == "tinker-run-1"
    assert result["best_epoch"] == 2
    assert result["best_sampler_weights_uri"] == "tinker://tinker-run-1/sampler_weights/approved-run-sampler-epoch-2"
    assert result["best_resumable_state_uri"] == "tinker://tinker-run-1/weights/approved-run-state-epoch-2"
    assert trainer.saved_states == ["approved-run-state-epoch-1", "approved-run-state-epoch-2"]
    assert result["cleanup"] == {"session_closed": True, "error": None}
    assert service.holder.closed == 1

    run = tmp_path / "run"
    state = json.loads((run / "run-state.json").read_text())
    assert state["provider"] == "tinker"
    assert state["status"] == "completed"
    assert state["best_sampler_weights_uri"] == result["best_sampler_weights_uri"]
    assert json.loads((run / "result.json").read_text())["status"] == "completed"
    assert json.loads((run / "plan.json").read_text())["training_schedule"]["estimate"] is False


def test_train_restores_optimizer_state_from_an_explicit_checkpoint(tmp_path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    _train(_provider(FakeService(trainer)), tmp_path, tinker.TinkerSFTSettings(max_epochs=1, batch_size=32),
           init_from_checkpoint="tinker://previous/weights/state-1")
    assert trainer.loaded == ["tinker://previous/weights/state-1"]


def test_training_failure_closes_the_session_and_records_a_credential_safe_error(tmp_path):
    _write_prepared_dataset(tmp_path)
    service = FakeService(FakeTrainer(fail_training=RuntimeError("secret-bearing provider detail")))

    with pytest.raises(RuntimeError, match="secret-bearing"):
        _train(_provider(service), tmp_path)

    assert service.holder.closed == 1
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["primary_error"]["type"] == "RuntimeError"
    assert "secret" not in json.dumps(result)


def test_unconfirmed_training_and_oversized_rows_stop_before_creating_a_session(tmp_path):
    _write_prepared_dataset(tmp_path)
    service = FakeService(FakeTrainer())
    with pytest.raises(PipelineError, match="--confirm"):
        _provider(service).train(tmp_path, tmp_path / "run", "id", tinker.TinkerSFTSettings(),
                                 confirm=False, init_from_checkpoint=None)

    def oversized(row, model, **kwargs):
        target = SimpleNamespace(token_ids=list(range(5_000)), token_weights=[0] * 4_999 + [1])
        return [lora_sft.to_datum(target, kwargs["tinker_types"], model.max_seq_len, ignore_index=None)]

    with pytest.raises(tinker.TinkerDataError, match="exceed the Tinker trainer limit"):
        _train(_provider(service, render_fn=oversized), tmp_path)
    assert service.factory_calls == []
    assert service.create_calls == []


class CapabilityService:
    def __init__(self, models=None, failure=None):
        self.models = models or []
        self.failure = failure
        self.holder = FakeHolder()

    def get_server_capabilities(self):
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(supported_models=[SimpleNamespace(model_name=name, max_context_length=limit)
                                                 for name, limit in self.models])


def test_capability_preflight_checks_the_advertised_model_and_context_then_closes_the_session():
    service = CapabilityService([("Qwen/Qwen3.8-27B", 65_536), ("Qwen/Qwen3.5-9B", None)])
    capability = tinker.fetch_model_capability("Qwen/Qwen3.8-27B", 65_536, service_factory=lambda: service)
    assert capability == tinker.TinkerModelCapability("Qwen/Qwen3.8-27B", 65_536)
    assert service.holder.closed == 1
    # A model without an advertised limit is checked again by the trainer.
    assert tinker.fetch_model_capability("Qwen/Qwen3.5-9B", 65_536, service_factory=lambda: service).max_context_length is None
    with pytest.raises(tinker.TinkerRuntimeError, match="--max-seq-len 65536"):
        tinker.fetch_model_capability("Qwen/Qwen3.8-27B", 131_072, service_factory=lambda: service)
    with pytest.raises(tinker.TinkerRuntimeError, match="does not advertise"):
        tinker.fetch_model_capability("Qwen/Qwen3.8-32B", 1_024, service_factory=lambda: service)


def test_capability_failures_hide_provider_details_and_close_the_session():
    service = CapabilityService(failure=RuntimeError("tml-secret"))
    with pytest.raises(tinker.TinkerRuntimeError) as failure:
        tinker.fetch_model_capability("Qwen/Qwen3.8-27B", 1_024, service_factory=lambda: service)
    assert "tml-secret" not in str(failure.value)
    assert service.holder.closed == 1


@pytest.mark.parametrize("value", ["", "  "])
def test_credentials_must_be_set(monkeypatch, value):
    monkeypatch.setenv("TINKER_API_KEY", value)
    with pytest.raises(PipelineError, match="TINKER_API_KEY is not set"):
        tinker_package.require_api_key()


def test_credentials_are_sent_only_to_the_standard_tinker_api(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    with pytest.raises(PipelineError, match="unset TINKER_BASE_URL"):
        tinker_package.require_api_key()
    monkeypatch.setenv("TINKER_BASE_URL", tinker_package.TINKER_BASE_URL + "/")
    assert tinker_package.require_api_key() == "tml-test"


@pytest.mark.parametrize("command", [
    ["deploy", "--provider", "tinker", "--run-dir", "run"],
    ["undeploy", "--provider", "tinker", "--run-dir", "run"],
])
def test_tinker_deployment_is_refused_with_guidance(command, capsys):
    with pytest.raises(SystemExit) as failure:
        cli.main(command)
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "cannot" in error and "evaluate" in error


def test_fireworks_rejects_microbatch_options_and_names_their_providers():
    from smithtune.providers.fireworks.provider import FireworksProvider

    with pytest.raises(PipelineError, match="require Baseten or Tinker"):
        FireworksProvider().settings_from_options(TrainingOptions(microbatch_token_budget=1_024))


def test_tinker_shares_the_hugging_face_rendering_implementation():
    from smithtune.providers.baseten.provider import DEFAULT_MODEL as BASETEN_MODEL
    from smithtune.rendering import rendering_version

    assert rendering_version(tinker.DEFAULT_MODEL) == rendering_version(replace(BASETEN_MODEL, provider="baseten"))

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from smithtune import dataset
from smithtune.providers import baseten
from smithtune.providers.base import CommonSFTSettings, ModelOptions, ModelSpec, PipelineError, TrainingOptions


class FakeModelInput:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    @classmethod
    def from_ints(cls, values: list[int]) -> FakeModelInput:
        return cls(values)

    def to_ints(self) -> list[int]:
        return self._values


@dataclass
class FakeTensorData:
    data: list[float] | list[int]
    dtype: str
    shape: list[int]


@dataclass
class FakeDatum:
    model_input: FakeModelInput
    loss_fn_inputs: dict[str, FakeTensorData]
    source_id: int | None = None


FAKE_LOOPS_TYPES = SimpleNamespace(
    Datum=FakeDatum,
    ModelInput=FakeModelInput,
    TensorData=FakeTensorData,
)


class FakeManagement:
    def __init__(self, *, session_runs=None, inactive=None) -> None:
        self.session_runs = session_runs or []
        self.inactive = iter(inactive or [])
        self.session_lookups: list[str] = []
        self.deactivated: list[str] = []
        self.polled: list[str] = []

    def session_run_ids(self, session_id: str) -> list[str]:
        self.session_lookups.append(session_id)
        return self.session_runs

    def deactivate_run(self, run_id: str) -> None:
        self.deactivated.append(run_id)

    def run_is_inactive(self, run_id: str) -> bool:
        self.polled.append(run_id)
        return next(self.inactive)


class FakeFuture:
    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


class FakeAdamParams:
    def __init__(self, **values) -> None:
        self.values = values


class FakeTrainer:
    run_id = "baseten-run-1"

    def __init__(self, validation_losses=(1.0, 0.8, 0.9)) -> None:
        self.validation_losses = iter(validation_losses)
        self.forward_backward_sizes: list[int] = []
        self.forward_sizes: list[int] = []
        self.optimizer_params: list[FakeAdamParams] = []
        self.save_state_names: list[str] = []
        self.save_sampler_names: list[str] = []
        self.loaded: list[str] = []
        self.closed = False

    def forward_backward(self, batch):
        self.forward_backward_sizes.append(len(batch))
        return FakeFuture(SimpleNamespace(loss=0.5, metrics={"active_tokens": len(batch)}))

    def optim_step(self, params):
        self.optimizer_params.append(params)
        return FakeFuture(SimpleNamespace(metrics={"grad_norm": 1.0}))

    def forward(self, batch, loss):
        assert loss == "cross_entropy"
        self.forward_sizes.append(len(batch))
        return FakeFuture(
            SimpleNamespace(
                loss=next(self.validation_losses),
                metrics={"active_tokens": len(batch)},
            )
        )

    def save_state(self, name):
        self.save_state_names.append(name)
        return FakeFuture(SimpleNamespace(path=f"state://{name}"))

    def save_weights_for_sampler(self, name):
        self.save_sampler_names.append(name)
        return FakeFuture(SimpleNamespace(path=f"sampler://{name}"))

    def load_state_with_optimizer(self, path):
        self.loaded.append(path)
        return FakeFuture(SimpleNamespace(path=path))

    def close(self):
        self.closed = True


class FakeService:
    session_id = "session-1"

    def __init__(self, trainer: FakeTrainer) -> None:
        self.trainer = trainer
        self.service_client_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def create_lora_training_client(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.trainer


FAKE_LIFECYCLE_TYPES = SimpleNamespace(
    Datum=FakeDatum,
    ModelInput=FakeModelInput,
    TensorData=FakeTensorData,
    AdamParams=FakeAdamParams,
)


def test_plan_validates_the_approved_prepared_dataset_without_provider_calls(
    tmp_path: Path,
):
    _write_prepared_dataset(tmp_path)
    provider = baseten.BasetenProvider()

    plan = provider.plan(tmp_path, "approved-run", baseten.BasetenSFTSettings())

    assert plan["run_id"] == "approved-run"
    assert plan["provider"] == "baseten"
    assert plan["base_model"] == "Qwen/Qwen3.8-27B"
    assert plan["model"] == asdict(replace(baseten.DEFAULT_MODEL, trainer_max_seq_len=131_072))
    assert plan["dataset"] == {
        "source_rows": 100,
        "train_rows": 90,
        "validation_rows": 10,
        "test_rows": 0,
    }
    assert plan["config"] == {
        "tokenizer_model": "Qwen/Qwen3.8-27B",
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "renderer": "hf_assistant",
        "thinking_trace_history_mode": "preserved",
        "max_seq_len": 131_072,
        "prepared_max_seq_len": 135_590,
        "lora_rank": 8,
        "learning_rate": 1e-4,
        "microbatch_token_budget": 262_144,
        "max_dropped_training_rows": 1,
        "effective_batch_size": 32,
        "max_epochs": 5,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0,
        "seed": 42,
        "replicas": 1,
        "capacity": "dedicated",
        "with_sampler": False,
    }


def test_baseten_provider_owns_its_model_profile():
    model = baseten.BasetenProvider().model_from_options(ModelOptions())

    assert model == baseten.MODEL_SPECS["qwen3p8-27b"]
    assert model.base_model == "Qwen/Qwen3.8-27B"
    assert model.provider == "baseten"
    assert model.max_seq_len == 262_144
    assert model.training_context_limit == 262_144


@pytest.mark.parametrize("context_tokens", [180_000, 262_144])
def test_qwen_training_passes_larger_context_to_loops(tmp_path, context_tokens):
    _write_prepared_dataset(tmp_path, model=baseten.DEFAULT_MODEL, max_context_tokens=context_tokens)
    trainer = FakeTrainer(validation_losses=[1.0])
    service = FakeService(trainer)

    def render(row, model, **kwargs):
        return [_datum(context_tokens if row["_source"]["example_id"] == "example-0" else 1)]

    provider = _provider(service, FakeManagement(inactive=[True]), render_fn=render)
    result = _train(provider, tmp_path, baseten.BasetenSFTSettings(max_epochs=1))

    plan = json.loads((tmp_path / "run/plan.json").read_text())
    assert result["status"] == "completed"
    assert plan["config"]["max_seq_len"] == context_tokens
    assert service.create_calls[0]["max_seq_len"] == context_tokens
    assert plan["dataset"]["train_rows"] == 90
    assert plan["dataset"].get("dropped_train_rows", []) == []


def test_model_resolution_rejects_unsupported_models():
    with pytest.raises(PipelineError, match="no supported baseten rendering configuration"):
        baseten.BasetenProvider().model_from_options(ModelOptions(model="unknown"))


def test_settings_resolution_uses_baseten_defaults_and_retains_supported_options():
    provider = baseten.BasetenProvider()

    assert provider.settings_from_options(TrainingOptions()) == baseten.BasetenSFTSettings()
    options = TrainingOptions(
        max_epochs=2,
        learning_rate=2e-4,
        early_stopping_min_delta=0.1,
        lora_rank=8,
        microbatch_token_budget=131_072,
        max_spend_usd=75.0,
        hourly_rate_usd=30.0,
        seed=7,
        replicas=2,
        spend_reserve_fraction=0.2,
        max_dropped_training_rows=0,
    )
    assert provider.settings_from_options(options) == baseten.BasetenSFTSettings(
        max_epochs=2,
        learning_rate=2e-4,
        early_stopping_min_delta=0.1,
        lora_rank=8,
        microbatch_token_budget=131_072,
        max_spend_usd=75.0,
        hourly_rate_usd=30.0,
        seed=7,
        replicas=2,
        spend_reserve_fraction=0.2,
        max_dropped_training_rows=0,
    )


@pytest.mark.parametrize("field", ["lora_alpha", "pipeline_depth"])
@pytest.mark.parametrize("value", [0, 4])
def test_settings_resolution_rejects_fireworks_options(field: str, value: int):
    options = replace(TrainingOptions(), **{field: value})

    with pytest.raises(PipelineError, match="Fireworks-only"):
        baseten.BasetenProvider().settings_from_options(options)


@pytest.mark.parametrize(
    ("field", "value"),
    [("lora_rank", 0), ("batch_size", 0), ("microbatch_token_budget", 0)],
)
def test_settings_resolution_validates_baseten_options(field: str, value: int):
    options = replace(TrainingOptions(), **{field: value})

    with pytest.raises(PipelineError):
        baseten.BasetenProvider().settings_from_options(options)


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction", "expected_validation", "expected_test"),
    [(None, None, 0.1, 0.1), (0.0, 0.2, 0.0, 0.2), (0.2, 0.0, 0.2, 0.0)],
)
def test_preparation_delegates_with_baseten_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_fraction: float | None,
    test_fraction: float | None,
    expected_validation: float,
    expected_test: float,
):
    calls = []
    contract = object()
    manifest = {"prepared": True}

    def prepare_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return manifest

    monkeypatch.setattr(dataset, "prepare_dataset", prepare_dataset)
    monkeypatch.setattr(baseten, "resolve_rendering_model", lambda model: model)
    capability_calls = []

    def capability_resolver(model, length):
        capability_calls.append((model, length))
        return baseten.BasetenModelCapability(model, 262_144)

    result = baseten.BasetenProvider(capability_resolver=capability_resolver).prepare(
        "workspace-id",
        "dataset-id",
        tmp_path,
        model_options=ModelOptions(),
        inference_contract=contract,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        fetch=False,
        check_render=False,
    )

    assert result is manifest
    assert capability_calls == [("Qwen/Qwen3.8-27B", 262_144)]
    assert calls == [
        (
            (
                "workspace-id",
                "dataset-id",
                baseten.MODEL_SPECS["qwen3p8-27b"],
                tmp_path,
            ),
            {
                "inference_contract": contract,
                "source_workspace_id": None,
                "reasoning_policy": "omit",
                "validation_fraction": expected_validation,
                "test_fraction": expected_test,
                "fetch": False,
                "check_render": False,
            },
        )
    ]


def test_preparation_rejects_invalid_model_before_fetching_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("unsupported models must be rejected before data access")

    monkeypatch.setattr(dataset, "prepare_dataset", unexpected_prepare)

    with pytest.raises(PipelineError, match="no supported baseten rendering configuration"):
        baseten.BasetenProvider().prepare(
            "workspace-id",
            "dataset-id",
            tmp_path,
            model_options=ModelOptions(model="unknown"),
        )


@pytest.mark.parametrize(
    "settings",
    [
        CommonSFTSettings(),
        SimpleNamespace(**vars(baseten.BasetenSFTSettings()), validate=lambda: None),
    ],
)
def test_planning_requires_real_baseten_settings(tmp_path: Path, settings):
    _write_prepared_dataset(tmp_path)

    with pytest.raises(PipelineError, match="requires Baseten SFT settings"):
        baseten.BasetenProvider().plan(tmp_path, "approved-run", settings)


def test_capability_preflight_accepts_the_pinned_model_and_context():
    response = {
        "supported_models": [
            {
                "model_name": "Qwen/Qwen3.8-27B",
                "max_context_length": 262_144,
                "supports_vision_language": False,
            }
        ]
    }

    capability = baseten.fetch_model_capability(
        "Qwen/Qwen3.8-27B",
        131_072,
        api_key="fake-key",
        opener=lambda request, timeout: io.BytesIO(json.dumps(response).encode()),
    )

    assert capability.model_name == "Qwen/Qwen3.8-27B"
    assert capability.max_context_length == 262_144


def test_capability_preflight_refuses_a_missing_model():
    response = {"supported_models": []}

    with pytest.raises(baseten.BasetenRuntimeError, match="does not advertise"):
        baseten.fetch_model_capability(
            "Qwen/Qwen3.8-27B",
            131_072,
            api_key="fake-key",
            opener=lambda request, timeout: io.BytesIO(json.dumps(response).encode()),
        )


def test_capability_read_retries_a_transient_response_with_bounded_backoff():
    response = {
        "supported_models": [
            {"model_name": "Qwen/Qwen3.8-27B", "max_context_length": 131_072}
        ]
    }
    calls = 0
    sleeps: list[float] = []

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("fake", 429, "busy", {}, None)
        return io.BytesIO(json.dumps(response).encode())

    capability = baseten.fetch_model_capability(
        "Qwen/Qwen3.8-27B",
        131_072,
        api_key="fake-key",
        opener=opener,
        sleeper=sleeps.append,
    )

    assert capability.max_context_length == 131_072
    assert calls == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    ("field", "value", "config_field"),
    [
        ("max_epochs", 6, "max_epochs"),
        ("lora_rank", 16, "lora_rank"),
        ("batch_size", 16, "effective_batch_size"),
        ("early_stopping_patience", 2, "early_stopping_patience"),
        ("seed", 7, "seed"),
        ("replicas", 2, "replicas"),
        ("max_dropped_training_rows", 0, "max_dropped_training_rows"),
    ],
)
def test_baseten_training_policy_is_configurable(
    tmp_path: Path, field: str, value: int, config_field: str
):
    _write_prepared_dataset(tmp_path)
    settings = baseten.BasetenSFTSettings(**{field: value})

    plan = baseten.BasetenProvider().plan(tmp_path, "approved-run", settings)

    assert plan["config"][config_field] == value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_epochs", 0, None),
        ("early_stopping_patience", 0, None),
        ("lora_rank", 0, None),
        ("lora_rank", 1.5, None),
        ("batch_size", True, None),
        ("seed", -1, None),
        ("seed", 1.5, None),
        ("replicas", 0, None),
        ("replicas", 1.5, None),
        ("max_dropped_training_rows", -1, None),
        ("max_dropped_training_rows", 1.5, None),
        ("spend_reserve_fraction", -0.1, None),
        ("spend_reserve_fraction", 1.0, None),
        ("spend_reserve_fraction", float("nan"), None),
        ("learning_rate", float("nan"), "finite"),
        ("early_stopping_min_delta", float("inf"), "finite"),
        ("max_epochs", float("nan"), "finite"),
        ("microbatch_token_budget", float("inf"), "finite"),
    ],
)
def test_invalid_settings_are_rejected_before_provisioning(
    tmp_path: Path, field: str, value, message: str | None
):
    _write_prepared_dataset(tmp_path)
    service = FakeService(FakeTrainer())
    provider = _provider(service, FakeManagement())

    with pytest.raises(PipelineError, match=message):
        _train(provider, tmp_path, baseten.BasetenSFTSettings(**{field: value}))

    assert service.service_client_calls == []
    assert service.create_calls == []


@pytest.mark.parametrize(
    "invalid_count", ["train-mismatch", "test-mismatch", "source-undercount", "empty-train", "empty-validation"]
)
def test_prepared_dataset_counts_are_validated_before_provisioning(
    tmp_path: Path, invalid_count: str
):
    _write_prepared_dataset(tmp_path)
    prepared = tmp_path / "prepared"
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if invalid_count == "source-undercount":
        manifest["langsmith"]["examples"] = 99
    elif invalid_count.startswith("empty-"):
        split = invalid_count.removeprefix("empty-")
        (prepared / f"{split}.jsonl").write_text("")
        manifest["split"][split] = 0
    else:
        split = invalid_count.removesuffix("-mismatch")
        manifest["split"][split] += 1
    manifest_path.write_text(json.dumps(manifest))
    service = FakeService(FakeTrainer())
    provider = _provider(service, FakeManagement())

    with pytest.raises(PipelineError):
        _train(provider, tmp_path, run_id="invalid-dataset-run")

    assert service.service_client_calls == []
    assert service.create_calls == []


def test_capability_failure_happens_before_paid_provisioning(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    provider = _provider(
        FakeService(FakeTrainer()),
        FakeManagement(inactive=[]),
        capability_resolver=lambda model, length: (_ for _ in ()).throw(
            baseten.BasetenRuntimeError("context limit is too low")
        ),
        service_factory=lambda: (_ for _ in ()).throw(
            AssertionError("paid provisioning must not start")
        ),
    )

    with pytest.raises(baseten.BasetenRuntimeError, match="context limit"):
        _train(provider, tmp_path)


@pytest.mark.parametrize(
    "override_name",
    ["LOOPS_REUSE_FROM_RUN_ID", "LOOPS_REUSE_FROM_SESSION_ID"],
)
def test_loops_reuse_overrides_are_rejected_before_capability_or_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override_name: str
):
    _write_prepared_dataset(tmp_path)
    monkeypatch.setenv(override_name, "value-must-not-be-reported")
    service = FakeService(FakeTrainer())
    capability_calls: list[str] = []
    provider = _provider(
        service,
        FakeManagement(inactive=[]),
        capability_resolver=lambda model, length: capability_calls.append(model),
    )

    with pytest.raises(PipelineError, match=override_name) as captured:
        _train(provider, tmp_path)

    assert "value-must-not-be-reported" not in str(captured.value)
    assert capability_calls == []
    assert service.service_client_calls == []
    assert service.create_calls == []


def test_budget_omission_never_reads_the_clock():
    guard = baseten.BudgetGuard(
        None,
        None,
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock must stay unused")),
    )

    guard.start()
    guard.check("training")

    assert guard.maximum_active_seconds is None
    assert guard.elapsed_seconds == 0.0


def test_budget_uses_the_ninety_percent_active_time_limit():
    times = iter([100.0, 8_199.9, 8_200.1])
    guard = baseten.BudgetGuard(75.0, 30.0, clock=lambda: next(times))

    guard.start()
    guard.check("optimizer step")
    with pytest.raises(baseten.BudgetExceeded, match="optimizer step"):
        guard.check("optimizer step")

    assert guard.maximum_active_seconds == 8_100.0


@pytest.mark.parametrize(
    ("max_spend_usd", "hourly_rate_usd", "reserve", "expected"),
    [
        (None, None, 0.1, None),
        (75.0, 30.0, 0.1, 8_100.0),
        (75.0, 30.0, 0.2, 7_200.0),
        (75.0, 30.0, 0.0, 9_000.0),
    ],
)
def test_settings_plan_and_runtime_guard_agree_on_budget_duration(
    tmp_path: Path, max_spend_usd, hourly_rate_usd, reserve, expected
):
    _write_prepared_dataset(tmp_path)
    settings = baseten.BasetenSFTSettings(
        max_spend_usd=max_spend_usd, hourly_rate_usd=hourly_rate_usd,
        spend_reserve_fraction=reserve,
    )

    plan = baseten.BasetenProvider().plan(tmp_path, "approved-run", settings)
    guard = baseten.BudgetGuard(
        max_spend_usd, hourly_rate_usd, spend_reserve_fraction=reserve
    )

    assert settings.maximum_active_seconds() == expected
    assert plan["budget"]["maximum_active_seconds"] == expected
    assert plan["budget"]["spend_reserve_fraction"] == reserve
    assert guard.maximum_active_seconds == expected


@pytest.mark.parametrize(
    ("max_spend_usd", "hourly_rate_usd"),
    [
        (75.0, None),
        (None, 30.0),
        (0.0, 30.0),
        (75.0, -1.0),
        (float("inf"), 30.0),
        (75.0, float("nan")),
        (True, 30.0),
        (75.0, "30"),
    ],
)
def test_settings_and_runtime_guard_reject_the_same_invalid_budget(
    max_spend_usd, hourly_rate_usd
):
    settings = baseten.BasetenSFTSettings(
        max_spend_usd=max_spend_usd, hourly_rate_usd=hourly_rate_usd
    )

    with pytest.raises(PipelineError) as settings_error:
        settings.validate()
    with pytest.raises(PipelineError) as guard_error:
        baseten.BudgetGuard(max_spend_usd, hourly_rate_usd)

    assert str(settings_error.value) == str(guard_error.value)


def test_cleanup_deactivates_the_exact_run_and_polls_until_inactive():
    management = FakeManagement(inactive=[False, True])
    sleeps: list[float] = []

    targets = baseten.deactivate_identity(
        run_id="baseten-run-1",
        session_id="session-1",
        management=management,
        sleeper=sleeps.append,
    )

    assert targets == ["baseten-run-1"]
    assert management.session_lookups == []
    assert management.deactivated == ["baseten-run-1"]
    assert management.polled == ["baseten-run-1", "baseten-run-1"]
    assert sleeps == [1.0]


def test_cleanup_falls_back_to_every_run_in_the_recorded_session():
    management = FakeManagement(session_runs=["run-2", "run-1"], inactive=[True, True])

    targets = baseten.deactivate_identity(
        run_id=None,
        session_id="session-1",
        management=management,
        sleeper=lambda seconds: None,
    )

    assert targets == ["run-1", "run-2"]
    assert management.deactivated == ["run-1", "run-2"]


def test_idempotent_remote_calls_retry_only_transient_statuses():
    attempts = 0
    sleeps: list[float] = []

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.HTTPError("fake", 503, "unavailable", {}, None)
        return "ready"

    assert baseten.retry_idempotent(operation, sleeper=sleeps.append) == "ready"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]

    with pytest.raises(urllib.error.HTTPError):
        baseten.retry_idempotent(
            lambda: (_ for _ in ()).throw(
                urllib.error.HTTPError("fake", 401, "unauthorized", {}, None)
            ),
            sleeper=lambda seconds: (_ for _ in ()).throw(
                AssertionError("authentication failures must not retry")
            ),
        )


@pytest.mark.parametrize("failure_kind", ["connect", "http-503", "server-shutdown"])
def test_idempotent_remote_calls_retry_installed_transient_exception_types(
    failure_kind: str,
):
    import httpx
    from baseten.loops import ServerShutdownError

    request = httpx.Request("GET", "https://example.invalid/status")

    def failure():
        if failure_kind == "connect":
            return httpx.ConnectError("disconnected", request=request)
        if failure_kind == "http-503":
            response = httpx.Response(503, request=request)
            return httpx.HTTPStatusError(
                "service unavailable", request=request, response=response
            )
        return ServerShutdownError("server is shutting down")

    calls = 0
    sleeps: list[float] = []

    def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure()
        return "ready"

    assert baseten.retry_idempotent(operation, sleeper=sleeps.append) == "ready"
    assert calls == 2
    assert sleeps == [1.0]


def test_train_runs_forward_only_validation_selects_the_best_checkpoint_and_cleans_up(
    tmp_path: Path,
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    service = FakeService(trainer)
    management = FakeManagement(inactive=[True])
    provider = _provider(
        service,
        management,
        monotonic=lambda: (_ for _ in ()).throw(
            AssertionError("unbudgeted training must not read the clock")
        ),
    )
    run_dir = tmp_path / "run"

    result = _train(provider, tmp_path)

    assert result["status"] == "completed"
    assert result["best_epoch"] == 2
    assert result["best_sampler_weights_uri"] == "sampler://approved-run-sampler-epoch-2"
    assert result["last_resumable_state_uri"] == "state://approved-run-state-epoch-3"
    assert result["stopped_early"] is True
    assert trainer.forward_backward_sizes == [32, 32, 26] * 3
    assert trainer.forward_sizes == [10, 10, 10]
    assert len(trainer.optimizer_params) == 9
    assert trainer.save_state_names == [
        "approved-run-state-epoch-1",
        "approved-run-state-epoch-2",
        "approved-run-state-epoch-3",
    ]
    assert trainer.save_sampler_names == [
        "approved-run-sampler-epoch-1",
        "approved-run-sampler-epoch-2",
    ]
    assert trainer.closed is True
    assert management.deactivated == ["baseten-run-1"]
    assert service.service_client_calls == [{"availability_model": "dedicated"}]
    assert service.create_calls == [
        {
            "base_model": "Qwen/Qwen3.8-27B",
            "rank": 8,
            "replicas": 1,
            "seed": 42,
            "max_seq_len": 131_072,
            "with_sampler": False,
            "name": "approved-run",
        }
    ]
    assert json.loads((run_dir / "run-state.json").read_text())["status"] == "completed"
    assert len(json.loads((run_dir / "epochs.json").read_text())) == 3
    assert json.loads((run_dir / "result.json").read_text()) == result


def test_audit_hashes_distinguish_base_models_with_identical_rendering_and_settings(
    tmp_path: Path,
):
    _write_prepared_dataset(tmp_path)
    provider = baseten.BasetenProvider()
    settings = baseten.BasetenSFTSettings()
    manifest_path = tmp_path / "prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_plan = provider.plan(tmp_path, "audit-run", settings)
    original_audit = baseten._audit_identity(tmp_path, manifest, original_plan)

    manifest["model"]["base_model"] = "other/model-with-the-same-tokenizer"
    manifest_path.write_text(json.dumps(manifest))
    changed_plan = provider.plan(tmp_path, "audit-run", settings)
    changed_audit = baseten._audit_identity(tmp_path, manifest, changed_plan)

    assert changed_plan["model"] == {
        **original_plan["model"], "base_model": "other/model-with-the-same-tokenizer",
    }
    assert original_plan["config"] == changed_plan["config"]
    assert original_plan["budget"] == changed_plan["budget"]
    for key in ("source_dataset_identity", "split_files_sha256", "renderer_sha256"):
        assert original_audit[key] == changed_audit[key]
    assert original_audit["model_identity"] == original_plan["model"]
    assert changed_audit["model_identity"] == changed_plan["model"]
    assert original_audit["model_sha256"] != changed_audit["model_sha256"]
    assert original_audit["settings_sha256"] != changed_audit["settings_sha256"]


def test_strict_lowest_loss_checkpoint_is_separate_from_min_delta_patience(
    tmp_path: Path,
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0, 0.95])
    provider = _provider(
        FakeService(trainer), FakeManagement(inactive=[True])
    )
    settings = replace(
        baseten.BasetenSFTSettings(), early_stopping_min_delta=0.1
    )

    result = _train(provider, tmp_path, settings)

    assert result["best_epoch"] == 2
    assert result["best_sampler_weights_uri"] == "sampler://approved-run-sampler-epoch-2"
    assert result["stopped_early"] is True
    assert len(result["epochs"]) == 2
    assert trainer.save_sampler_names == [
        "approved-run-sampler-epoch-1",
        "approved-run-sampler-epoch-2",
    ]


@pytest.mark.parametrize(
    ("oversized_indices", "maximum_drops"), [({7}, 1), ({7, 9}, 2)],
    ids=["default-one-row", "configured-two-rows"],
)
def test_trainer_sequence_policy_drops_complete_rows_within_configured_allowance(
    tmp_path: Path, oversized_indices: set[int], maximum_drops: int,
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0])
    service = FakeService(trainer)
    trained_datums: list[tuple[int, int]] = []
    original_forward_backward = trainer.forward_backward

    def forward_backward(batch):
        trained_datums.extend(
            (datum.source_id, len(datum.model_input.to_ints())) for datum in batch
        )
        return original_forward_backward(batch)

    trainer.forward_backward = forward_backward

    def render(row, model, **kwargs):
        index = int(row["_source"]["example_id"].split("-")[-1])
        if index in oversized_indices:
            return [_datum(1, index), _datum(131_073, index)]
        return [_datum(131_072 if index == 8 else 1, index)]

    provider = _provider(
        service, FakeManagement(inactive=[True]), render_fn=render
    )
    run_dir = tmp_path / "run"

    result = _train(
        provider,
        tmp_path,
        baseten.BasetenSFTSettings(max_epochs=1, max_dropped_training_rows=maximum_drops),
    )

    plan = json.loads((run_dir / "plan.json").read_text())
    assert result["status"] == "completed"
    assert plan["config"]["prepared_max_seq_len"] == 135_590
    assert plan["config"]["max_seq_len"] == 131_072
    assert plan["dataset"] == {
        "source_rows": 100,
        "prepared_train_rows": 90,
        "train_rows": 90 - len(oversized_indices),
        "validation_rows": 10,
        "test_rows": 0,
        "dropped_train_rows": [
            {
                "prepared_index": index,
                "model_input_tokens": 131_073,
                "reason": "exceeds trainer limit of 131,072 tokens",
            }
            for index in sorted(oversized_indices)
        ],
    }
    assert result["split"] == plan["dataset"]
    assert len(trained_datums) == 90 - len(oversized_indices)
    assert {index for index, _tokens in trained_datums} == set(range(90)) - oversized_indices
    assert (8, 131_072) in trained_datums
    assert trainer.forward_backward_sizes == [32, 32, 26 - len(oversized_indices)]
    assert trainer.forward_sizes == [10]
    assert service.create_calls[0]["max_seq_len"] == 131_072


@pytest.mark.parametrize(
    ("oversized_indices", "maximum_drops", "message"),
    [
        ({7, 8}, 1, "configured maximum dropped training rows is 1"),
        ({7}, 0, "configured maximum dropped training rows is 0"),
        ({90}, 1, "validation data exceeds"),
    ],
    ids=["multiple-training-rows", "no-drops-allowed", "validation-row"],
)
def test_trainer_sequence_policy_rejects_oversized_rows_before_provisioning(
    tmp_path: Path, oversized_indices: set[int], maximum_drops: int, message: str
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    service = FakeService(trainer)

    def render(row, model, **kwargs):
        index = int(row["_source"]["example_id"].split("-")[-1])
        return [_datum(131_073 if index in oversized_indices else 1, index)]

    provider = _provider(service, FakeManagement(), render_fn=render)

    with pytest.raises(baseten.BasetenDataError, match=message):
        _train(
            provider,
            tmp_path,
            baseten.BasetenSFTSettings(max_epochs=1, max_dropped_training_rows=maximum_drops),
        )

    assert service.service_client_calls == []
    assert service.create_calls == []
    assert trainer.forward_backward_sizes == []
    assert trainer.forward_sizes == []


def test_audit_identity_is_durable_before_service_creation(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    run_dir = tmp_path / "run"
    trainer = FakeTrainer(validation_losses=[1.0])
    service = FakeService(trainer)
    state_at_service_creation: dict = {}

    def service_factory(**kwargs):
        service.service_client_calls.append(kwargs)
        state_at_service_creation.update(
            json.loads((run_dir / "run-state.json").read_text())
        )
        return service

    provider = _provider(
        service,
        FakeManagement(inactive=[True]),
        service_factory=service_factory,
    )

    _train(provider, tmp_path, baseten.BasetenSFTSettings(max_epochs=1))

    assert state_at_service_creation["started_at_utc"]
    assert state_at_service_creation["source_dataset_identity"] == {
        "workspace_id": "workspace-id",
        "dataset_id": "dataset-id",
        "examples": 100,
        "source_examples_sha256": "a" * 64,
    }
    split_hashes = state_at_service_creation["split_files_sha256"]
    for split in ("train", "validation", "test"):
        expected = hashlib.sha256(
            (tmp_path / "prepared" / f"{split}.jsonl").read_bytes()
        ).hexdigest()
        assert split_hashes[split] == expected
    assert len(state_at_service_creation["renderer_sha256"]) == 64
    assert len(state_at_service_creation["settings_sha256"]) == 64


def test_sigterm_handler_cleans_up_and_restores_the_previous_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    management = FakeManagement(inactive=[True])
    previous_handler = object()
    signal_calls: list[tuple[int, object]] = []
    signal_boundary = SimpleNamespace(
        SIGTERM=15,
        getsignal=lambda signal_number: previous_handler,
        signal=lambda signal_number, handler: signal_calls.append(
            (signal_number, handler)
        ),
    )
    monkeypatch.setattr(baseten, "signal", signal_boundary, raising=False)

    def terminate(batch):
        signal_calls[0][1](signal_boundary.SIGTERM, None)

    trainer.forward_backward = terminate
    provider = _provider(FakeService(trainer), management)

    with pytest.raises(baseten.TerminationRequested):
        _train(provider, tmp_path)

    assert trainer.closed is True
    assert management.deactivated == ["baseten-run-1"]
    assert signal_calls[-1] == (signal_boundary.SIGTERM, previous_handler)


def test_train_initializes_optimizer_state_from_the_explicit_checkpoint(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0])
    service = FakeService(trainer)
    provider = _provider(service, FakeManagement(inactive=[True]))

    _train(
        provider,
        tmp_path,
        baseten.BasetenSFTSettings(max_epochs=1),
        init_from_checkpoint="state://approved-external-checkpoint",
    )

    assert trainer.loaded == ["state://approved-external-checkpoint"]
    assert trainer.forward_backward_sizes


@pytest.mark.parametrize(
    ("checks_before_stop", "completed_epochs"),
    [
        pytest.param(8, 0, id="between-state-and-sampler"),
        pytest.param(10, 1, id="after-coherent-checkpoints"),
        pytest.param(11, 1, id="before-next-epoch"),
    ],
)
def test_budget_stop_only_publishes_coherent_epoch_checkpoints(
    tmp_path: Path, checks_before_stop: int, completed_epochs: int
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0])
    management = FakeManagement(inactive=[True])
    clock = iter([0.0, *([1.0] * checks_before_stop), 8_100.0, 8_100.0])
    provider = _provider(
        FakeService(trainer), management, monotonic=lambda: next(clock)
    )
    settings = baseten.BasetenSFTSettings(max_spend_usd=75.0, hourly_rate_usd=30.0)

    result = _train(provider, tmp_path, settings)

    assert result["status"] == "budget_stopped"
    assert result["budget"] == {
        "enabled": True,
        "max_spend_usd": 75.0,
        "hourly_rate_usd": 30.0,
        "spend_reserve_fraction": 0.1,
        "maximum_active_seconds": 8_100.0,
        "elapsed_seconds": 8_100.0,
        "stopped": True,
    }
    state_uri = "state://approved-run-state-epoch-1" if completed_epochs else None
    sampler_uri = "sampler://approved-run-sampler-epoch-1" if completed_epochs else None
    assert trainer.save_state_names == ["approved-run-state-epoch-1"]
    assert trainer.save_sampler_names == (["approved-run-sampler-epoch-1"] if completed_epochs else [])
    assert len(result["epochs"]) == completed_epochs
    assert result["last_resumable_state_uri"] == state_uri
    assert result["best_sampler_weights_uri"] == sampler_uri
    assert result["best_epoch"] == (1 if completed_epochs else None)
    state = json.loads((tmp_path / "run" / "run-state.json").read_text())
    assert state["completed_epoch"] == completed_epochs
    assert state["last_resumable_state_uri"] == state_uri
    assert state["best_sampler_weights_uri"] == sampler_uri
    assert trainer.closed is True
    assert management.deactivated == ["baseten-run-1"]


@pytest.mark.parametrize(
    ("times", "expected_loads"),
    [
        ([0.0, 8_100.0, 8_100.0], []),
        (
            [0.0, 8_099.0, 8_100.0, 8_100.0],
            ["state://approved-external-checkpoint"],
        ),
    ],
)
def test_budget_checks_before_and_after_checkpoint_initialization(
    tmp_path: Path, times: list[float], expected_loads: list[str]
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    clock = iter(times)
    provider = _provider(
        FakeService(trainer),
        FakeManagement(inactive=[True]),
        monotonic=lambda: next(clock),
    )
    settings = replace(
        baseten.BasetenSFTSettings(), max_spend_usd=75.0, hourly_rate_usd=30.0
    )

    result = _train(
        provider,
        tmp_path,
        settings,
        init_from_checkpoint="state://approved-external-checkpoint",
    )

    assert result["status"] == "budget_stopped"
    assert trainer.loaded == expected_loads
    assert trainer.forward_backward_sizes == []


def test_enabled_budget_takes_a_non_raising_snapshot_after_cleanup(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0])
    management = FakeManagement(inactive=[True])
    values = iter([0.0, *([1.0] * 11), 9_000.0])

    def clock():
        value = next(values)
        if value == 9_000.0:
            assert trainer.closed is True
            assert management.deactivated == ["baseten-run-1"]
        return value

    provider = _provider(
        FakeService(trainer), management, monotonic=clock
    )
    settings = replace(
        baseten.BasetenSFTSettings(),
        max_epochs=1,
        max_spend_usd=75.0,
        hourly_rate_usd=30.0,
    )

    result = _train(provider, tmp_path, settings)

    assert result["status"] == "completed"
    assert result["budget"]["elapsed_seconds"] == 9_000.0
    assert result["budget"]["stopped"] is False


@pytest.mark.parametrize("failure", [RuntimeError("remote mutation failed"), KeyboardInterrupt()])
def test_training_failure_and_baseexception_both_clean_up(tmp_path: Path, failure):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    calls = 0

    def explode(batch):
        nonlocal calls
        calls += 1
        raise failure

    trainer.forward_backward = explode
    management = FakeManagement(inactive=[True])
    provider = _provider(FakeService(trainer), management)
    run_dir = tmp_path / "run"

    with pytest.raises(type(failure)):
        _train(provider, tmp_path)

    assert calls == 1
    assert trainer.closed is True
    assert management.deactivated == ["baseten-run-1"]
    assert json.loads((run_dir / "result.json").read_text())["status"] == "failed"


def test_failure_state_records_a_credential_safe_primary_error_summary(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.forward_backward = lambda batch: (_ for _ in ()).throw(
        RuntimeError("credential-material-must-not-be-recorded")
    )
    provider = _provider(
        FakeService(trainer), FakeManagement(inactive=[True])
    )
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="credential-material"):
        _train(provider, tmp_path)

    state_text = (run_dir / "run-state.json").read_text()
    state = json.loads(state_text)
    assert "credential-material-must-not-be-recorded" not in state_text
    assert state["primary_error"] == {
        "type": "RuntimeError",
        "message": "Baseten lifecycle failed; exception details omitted for credential safety",
    }


def test_provisioning_failure_cleans_every_run_under_the_recorded_session(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    service = FakeService(FakeTrainer())

    def fail_create(**kwargs):
        raise RuntimeError("trainer creation failed")

    service.create_lora_training_client = fail_create
    management = FakeManagement(
        session_runs=["session-run-2", "session-run-1"], inactive=[True, True]
    )
    provider = _provider(service, management)

    with pytest.raises(RuntimeError, match="trainer creation failed"):
        _train(provider, tmp_path)

    assert management.session_lookups == ["session-1"]
    assert management.deactivated == ["session-run-1", "session-run-2"]


def test_primary_and_cleanup_failures_remain_visible_together(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    trainer.forward_backward = lambda batch: (_ for _ in ()).throw(
        ValueError("primary failed")
    )
    management = FakeManagement(inactive=[True])
    management.deactivate_run = lambda run_id: (_ for _ in ()).throw(
        RuntimeError("cleanup failed")
    )
    provider = _provider(FakeService(trainer), management)

    with pytest.raises(BaseExceptionGroup) as captured:
        _train(provider, tmp_path)

    assert any(isinstance(error, ValueError) for error in captured.value.exceptions)
    assert any(isinstance(error, RuntimeError) for error in captured.value.exceptions)
    assert (
        "baseten loops run deactivate --run-id baseten-run-1 --yes"
        in str(captured.value)
    )


def test_unconfirmed_deactivation_surfaces_the_exact_manual_command():
    management = FakeManagement(inactive=[False] * 12)

    with pytest.raises(
        baseten.BasetenRuntimeError,
        match=r"baseten loops run deactivate --run-id stuck-run --yes",
    ):
        baseten.deactivate_identity(
            run_id="stuck-run",
            session_id="session-1",
            management=management,
            sleeper=lambda seconds: None,
        )


@pytest.mark.parametrize("failure", [TimeoutError("poll timed out"), RuntimeError("poll failed")])
def test_raised_poll_failures_surface_the_exact_manual_command(failure):
    management = FakeManagement(inactive=[])
    management.run_is_inactive = lambda run_id: (_ for _ in ()).throw(failure)

    with pytest.raises(
        baseten.BasetenRuntimeError,
        match=r"baseten loops run deactivate --run-id stuck-run --yes",
    ):
        baseten.wait_for_run_inactive(
            "stuck-run",
            management=management,
            sleeper=lambda seconds: None,
        )


def test_raised_deactivation_failure_surfaces_the_exact_manual_command():
    management = FakeManagement(inactive=[True])
    management.deactivate_run = lambda run_id: (_ for _ in ()).throw(
        RuntimeError("management failed")
    )

    with pytest.raises(
        baseten.BasetenRuntimeError,
        match=r"baseten loops run deactivate --run-id stuck-run --yes",
    ):
        baseten.deactivate_identity(
            run_id="stuck-run",
            session_id="session-1",
            management=management,
            sleeper=lambda seconds: None,
        )


def test_cleanup_auth_failure_for_a_known_run_surfaces_the_manual_command(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    with pytest.raises(
        baseten.BasetenRuntimeError,
        match=r"baseten loops run deactivate --run-id stuck-run --yes",
    ):
        baseten.deactivate_identity(
            run_id="stuck-run",
            session_id="session-1",
            sleeper=lambda seconds: None,
        )


@pytest.mark.parametrize("failure_kind", ["http-503", "server-shutdown"])
def test_forward_only_validation_retries_transient_failure(tmp_path: Path, failure_kind: str):
    from baseten.loops import ServerShutdownError

    failure = (
        urllib.error.HTTPError("fake", 503, "unavailable", {}, None)
        if failure_kind == "http-503" else ServerShutdownError("server is shutting down")
    )
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0])
    original_forward = trainer.forward
    calls = 0
    sleeps: list[float] = []

    def transient_forward(batch, loss):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original_forward(batch, loss)

    trainer.forward = transient_forward
    provider = _provider(
        FakeService(trainer), FakeManagement(inactive=[True]), sleeper=sleeps.append,
    )

    _train(provider, tmp_path, baseten.BasetenSFTSettings(max_epochs=1))

    assert calls == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize("operation", ["forward_backward", "optim_step"])
def test_gradient_mutations_are_never_retried_after_transient_errors(
    tmp_path: Path, operation: str
):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer()
    calls = 0

    def transient_failure(*args):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("fake", 503, "unavailable", {}, None)

    setattr(trainer, operation, transient_failure)
    provider = _provider(FakeService(trainer), FakeManagement(inactive=[True]))

    with pytest.raises(urllib.error.HTTPError):
        _train(provider, tmp_path, baseten.BasetenSFTSettings(max_epochs=1))

    assert calls == 1


def test_validation_loss_is_weighted_by_active_target_tokens(tmp_path: Path):
    _write_prepared_dataset(tmp_path)
    trainer = FakeTrainer(validation_losses=[1.0, 3.0])

    def forward(batch, loss):
        value = next(trainer.validation_losses)
        active_tokens = sum(len(datum.model_input.to_ints()) for datum in batch)
        return FakeFuture(
            SimpleNamespace(loss=value, metrics={"active_tokens": active_tokens})
        )

    trainer.forward = forward

    def render(row, model, **kwargs):
        index = int(row["_source"]["example_id"].split("-")[-1])
        return [_datum(10 if index == 99 else 1)]

    provider = _provider(
        FakeService(trainer),
        FakeManagement(inactive=[True]),
        render_fn=render,
    )
    settings = replace(
        baseten.BasetenSFTSettings(), max_epochs=1, microbatch_token_budget=10
    )

    result = _train(provider, tmp_path, settings)

    assert result["epochs"][0]["validation_loss"] == pytest.approx(39 / 19)


def test_render_row_shifts_assistant_loss_tokens(monkeypatch: pytest.MonkeyPatch):
    _install_renderer_output(monkeypatch, [10, 20, 30, 40, 50], [0, 0, 1, 1, 0])

    datum = baseten.render_row(
        _row(), _model(), loops_types=FAKE_LOOPS_TYPES, renderer=object()
    )[0]

    assert datum.model_input.to_ints() == [10, 20, 30, 40]
    assert datum.loss_fn_inputs["target_tokens"].data == [-100, 30, 40, -100]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0]


def test_render_row_loads_the_selected_native_renderer(
    monkeypatch: pytest.MonkeyPatch,
):
    rendering_calls = []

    def render(row, model, **kwargs):
        rendering_calls.append((row, model, kwargs))
        return SimpleNamespace(token_ids=[10, 20, 30], token_weights=[0, 0, 1])

    _install_renderer_output(monkeypatch, render=render)
    calls = []
    renderer = object()
    monkeypatch.setattr(
        baseten, "load_training_renderer",
        lambda model: calls.append(model) or renderer,
    )
    model = _model()
    row = _row()
    baseten.render_row(row, model, loops_types=FAKE_LOOPS_TYPES)

    assert calls == [model]
    assert rendering_calls == [(row, model, {
        "renderer": renderer, "include_loss_mask": True, "reduction": "none",
    })]


def test_render_row_enforces_the_selected_model_context_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_renderer_output(monkeypatch, [10, 20, 30], [0, 0, 1])
    model = replace(
        _model(), renderer="hf_assistant", max_seq_len=3,
        trainer_max_seq_len=None, thinking_trace_history_mode="",
    )

    accepted = baseten.render_row(
        _row(), model, loops_types=FAKE_LOOPS_TYPES, renderer=object()
    )

    assert accepted[0].model_input.to_ints() == [10, 20]
    assert accepted[0].loss_fn_inputs["target_tokens"].data == [-100, 30]
    _install_renderer_output(monkeypatch, [10, 20, 30, 40], [0, 0, 1, 1])
    with pytest.raises(baseten.BasetenDataError, match="3.*token"):
        baseten.render_row(
            _row(), model, loops_types=FAKE_LOOPS_TYPES, renderer=object()
        )


def test_render_row_preserves_tool_declarations_and_masks_role_boundaries():
    calls: list[dict] = []

    def render(messages, *, tools):
        calls.append({"messages": messages, "tools": tools})
        # system | user | assistant tool call | tool result | assistant text
        return [SimpleNamespace(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            token_weights=[0, 0, 1, 1, 0, 0, 0, 1, 1],
        )]

    renderer = SimpleNamespace(render=render)
    row = _row()
    row["messages"] = [
        {"role": "system", "content": "You may use tools."},
        {"role": "user", "content": "Find the temperature."},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "72"},
        {"role": "assistant", "content": "It is 72 degrees."},
    ]

    datum = baseten.render_row(
        row, _model(), loops_types=FAKE_LOOPS_TYPES, renderer=renderer
    )[0]

    assert calls == [{"messages": row["messages"], "tools": row["tools"]}]
    assert datum.loss_fn_inputs["target_tokens"].data == [
        -100,
        3,
        4,
        -100,
        -100,
        -100,
        8,
        9,
    ]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("limit", [131_072, 262_144])
def test_render_row_enforces_selected_token_ceiling_without_truncation(
    monkeypatch: pytest.MonkeyPatch, limit: int,
):
    model = replace(_model(), max_seq_len=limit, trainer_max_seq_len=limit)
    _install_renderer_output(monkeypatch, list(range(limit)), [0] * (limit - 1) + [1])

    accepted = baseten.render_row(
        _row(), model, loops_types=FAKE_LOOPS_TYPES, renderer=object()
    )

    assert len(accepted) == 1
    assert len(accepted[0].model_input.to_ints()) == limit - 1

    _install_renderer_output(monkeypatch, list(range(limit + 1)), [0] * limit + [1])
    with pytest.raises(baseten.BasetenDataError, match=f"{limit:,}"):
        baseten.render_row(
            _row(), model, loops_types=FAKE_LOOPS_TYPES, renderer=object()
        )


def test_pack_microbatches_is_deterministic_and_never_exceeds_its_budget():
    datums = [_datum(length) for length in [70, 40, 30, 20]]

    batches = baseten.pack_microbatches(datums, token_budget=100)

    assert [[len(item.model_input.to_ints()) for item in batch] for batch in batches] == [
        [70],
        [40, 30, 20],
    ]


def test_pack_microbatches_rejects_a_datum_over_the_token_budget():
    with pytest.raises(baseten.BasetenDataError, match="exceeds.*token budget"):
        baseten.pack_microbatches([_datum(101)], token_budget=100)


def test_accumulation_groups_target_32_examples_and_flush_final_partial_group():
    batches = [[_datum(10) for _ in range(10)], [_datum(10) for _ in range(10)], [_datum(10) for _ in range(12)], [_datum(10) for _ in range(3)]]

    groups = baseten.plan_accumulation_groups(batches, effective_batch_size=32)

    assert [[len(batch) for batch in group] for group in groups] == [[10, 10, 12], [3]]


def test_accumulation_groups_split_microbatches_to_honor_the_effective_batch_cap():
    batches = [[_datum(10) for _ in range(20)], [_datum(10) for _ in range(20)]]

    groups = baseten.plan_accumulation_groups(batches, effective_batch_size=32)

    assert [[len(batch) for batch in group] for group in groups] == [[20, 12], [8]]
    assert all(sum(len(batch) for batch in group) <= 32 for group in groups)


def test_accumulation_groups_split_one_oversized_microbatch_in_order():
    input_batch = [_datum(10, source_id=index) for index in range(35)]

    groups = baseten.plan_accumulation_groups([input_batch], effective_batch_size=32)

    assert [[len(batch) for batch in group] for group in groups] == [[32], [3]]
    assert [[item.source_id for batch in group for item in batch] for group in groups] == [
        list(range(32)),
        [32, 33, 34],
    ]
    assert all(sum(len(batch) for batch in group) <= 32 for group in groups)


def test_project_metadata_and_readme_describe_baseten_training_artifacts():
    project = Path(__file__).resolve().parents[1]
    metadata = (project / "pyproject.toml").read_text(encoding="utf-8")
    readme = (project / "README.md").read_text(encoding="utf-8")

    assert "Baseten Loops training" in metadata
    for artifact in ("plan.json", "run-state.json", "epochs.json", "result.json"):
        assert artifact in readme


def _install_renderer_output(monkeypatch: pytest.MonkeyPatch, tokens=None, weights=None, render=None) -> None:
    if render is None:
        def render(*args, **kwargs):
            return SimpleNamespace(token_ids=tokens, token_weights=weights)

    def render_row_tokens(*args, **kwargs):
        result = render(*args, **kwargs)
        return result if isinstance(result, list) else [result]

    monkeypatch.setattr(baseten, "render_row_tokens", render_row_tokens)


def _row() -> dict:
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        "_source": {"example_id": "example-1", "source_scope": "thread", "source_scope_id": "thread-1"},
    }


def _model() -> ModelSpec:
    return replace(baseten.DEFAULT_MODEL, tokenizer_revision="revision")


def _datum(length: int, source_id: int | None = None) -> FakeDatum:
    return FakeDatum(
        model_input=FakeModelInput.from_ints(list(range(length))),
        loss_fn_inputs={},
        source_id=source_id,
    )


def _write_raw_dataset(root: Path, *, count: int) -> None:
    raw = root / "raw"
    raw.mkdir()
    examples = [
        {
            "id": f"example-{index}",
            "inputs": {"messages": [
                {"role": "human", "content": f"question {index}"},
                {"role": "ai", "content": f"answer {index}"},
            ]},
            "outputs": None,
            "metadata": {
                "source_scope": "thread", "source_scope_id": f"thread-{index}",
                "trajectory_format": "messages",
                "conversation_scope": "root",
            },
        }
        for index in range(count)
    ]
    (raw / "examples.json").write_text(json.dumps(examples))
    (raw / "dataset-export.json").write_text(json.dumps([
        {"inputs": item["inputs"], "outputs": item["outputs"]} for item in examples
    ]))
    (raw / "dataset.json").write_text(json.dumps({
        "id": "dataset-id", "name": "alternate-model-test", "example_count": count,
    }))

    from test_example_tools import write_empty_tool_snapshot
    write_empty_tool_snapshot(root, examples)


def _write_prepared_dataset(root: Path, *, model: ModelSpec | None = None, max_context_tokens: int = 135_590) -> None:
    prepared = root / "prepared"
    prepared.mkdir()
    # Older prepared artifacts retain their recorded 131K trainer limit.
    model = model or replace(baseten.DEFAULT_MODEL, trainer_max_seq_len=131_072)
    manifest = {
        "langsmith": {
            "workspace_id": "workspace-id",
            "dataset_id": "dataset-id",
            "examples": 100,
        },
        "split": {"train": 90, "validation": 10, "test": 0},
        "model": model.__dict__,
        "provider": {
            "name": "baseten",
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        },
        "audit": {"max_context_tokens": max_context_tokens},
        "source_examples_sha256": "a" * 64,
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for index in range(100):
        row = _row()
        row["_source"] = {
            "example_id": f"example-{index}",
            "source_scope": "thread", "source_scope_id": f"thread-{index}",
        }
        rows.append(json.dumps(row))
    (prepared / "train.jsonl").write_text("\n".join(rows[:90]) + "\n", encoding="utf-8")
    (prepared / "validation.jsonl").write_text(
        "\n".join(rows[90:]) + "\n", encoding="utf-8"
    )
    (prepared / "test.jsonl").write_text("", encoding="utf-8")


def _provider(service: FakeService, management: FakeManagement, **overrides):
    def service_factory(**kwargs):
        service.service_client_calls.append(kwargs)
        return service

    arguments = {
        "capability_resolver": lambda model, length: baseten.BasetenModelCapability(
            model, length
        ),
        "service_factory": service_factory,
        "management": management,
        "loops_types": FAKE_LIFECYCLE_TYPES,
        "render_fn": lambda row, model, **kwargs: [_datum(1)],
        "renderer_factory": lambda model: object(),
        "sleeper": lambda seconds: None,
    }
    arguments.update(overrides)
    return baseten.BasetenProvider(**arguments)


def _train(provider, root, settings=None, *, run_id="approved-run", init_from_checkpoint=None):
    return provider.train(
        root,
        root / "run",
        run_id,
        baseten.BasetenSFTSettings() if settings is None else settings,
        confirm=True,
        init_from_checkpoint=init_from_checkpoint,
    )

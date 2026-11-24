"""Tinker model configuration, rendering, and client-driven LoRA SFT."""

from __future__ import annotations

import math
import signal
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from smithtune.capabilities import preflight_model
from smithtune.inference_contract import InferenceContract
from smithtune.models import resolve_model_options, resolve_prepared_model
from smithtune.providers import lora_sft
from smithtune.providers.base import (
    CommonSFTSettings,
    ModelOptions,
    ModelSpec,
    PipelineError,
    ReasoningPolicy,
    TerminationRequested,
    TrainingOptions,
)
from smithtune.providers.tinker import CAPACITY, close_service, require_api_key, resolved, rest_client, service_client
from smithtune.rendering import load_training_renderer, render_row_tokens, resolve_rendering_model


# Tinker's default context variants. Its ":peft:" variants with longer
# contexts are separate model IDs with separate prices.
MODEL_SPECS = {
    "qwen3p8-27b": ModelSpec(
        name="qwen3p8-27b",
        base_model="Qwen/Qwen3.8-27B",
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        renderer="hf_assistant",
        max_seq_len=65_536,
        trainer_max_seq_len=65_536,
        thinking_trace_history_mode="preserved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="tinker",
    ),
    "qwen3p5-9b": ModelSpec(
        name="qwen3p5-9b",
        base_model="Qwen/Qwen3.5-9B",
        tokenizer_model="Qwen/Qwen3.5-9B",
        tokenizer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        renderer="hf_prefix_qwen3_5",
        max_seq_len=65_536,
        trainer_max_seq_len=65_536,
        thinking_trace_history_mode="interleaved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="tinker",
    ),
    "nemotron-3-super-120b": ModelSpec(
        name="nemotron-3-super-120b",
        base_model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        tokenizer_model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        tokenizer_revision="2dc98e2afe4face0e4ce40972a915c45368bd34a",
        renderer="hf_prefix_nemotron3_super",
        max_seq_len=65_536,
        trainer_max_seq_len=65_536,
        thinking_trace_history_mode="interleaved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="tinker",
    ),
    "nemotron-3p5-lightning-30b": ModelSpec(
        name="nemotron-3p5-lightning-30b",
        base_model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        tokenizer_model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        tokenizer_revision="a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
        renderer="hf_prefix_nemotron3_5_lightning",
        max_seq_len=65_536,
        trainer_max_seq_len=65_536,
        thinking_trace_history_mode="interleaved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="tinker",
    ),
}
DEFAULT_MODEL = MODEL_SPECS["qwen3p8-27b"]
LOSS_FUNCTION = "cross_entropy"
# Tinker sums cross-entropy over weighted tokens. Scaling each step's weights by
# 1 / its active target tokens makes every optimizer update a token mean, so the
# learning rate does not depend on how many tokens a batch happens to contain.
LOSS_NORMALIZATION = "mean over each optimizer step's active target tokens"


class TinkerDataError(PipelineError):
    """A canonical trajectory cannot safely be used for Tinker training."""


class TinkerRuntimeError(RuntimeError):
    """A credential-free failure at the Tinker service boundary."""


ERRORS = lora_sft.ProviderErrors("Tinker", TinkerDataError, TinkerRuntimeError)


@dataclass(frozen=True)
class TinkerSFTSettings(CommonSFTSettings):
    """Configuration accepted by the Tinker training lifecycle."""

    lora_rank: int | None = None
    microbatch_token_budget: int | None = None
    max_dropped_training_rows: int = 1
    sampler_checkpoints: str = "best"

    def validate(self) -> None:
        super().validate()
        if self.sampler_checkpoints not in lora_sft.SAMPLER_CHECKPOINT_POLICIES:
            raise PipelineError("sampler_checkpoints must be 'best' or 'all'")
        for name in ("lora_rank", "microbatch_token_budget"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise PipelineError(f"Tinker {name} must be a positive integer")
        if type(self.max_dropped_training_rows) is not int or self.max_dropped_training_rows < 0:
            raise PipelineError("Tinker max_dropped_training_rows must be a non-negative integer")


@dataclass(frozen=True)
class TinkerModelCapability:
    model_name: str
    # Tinker may omit the limit; the trainer then enforces it at request time.
    max_context_length: int | None = None


def fetch_model_capability(
    model: str, required_context: int, *, client_factory: Any = rest_client,
) -> TinkerModelCapability:
    """Read Tinker's supported models without opening a session or creating a trainer.

    REST clients share process-wide connections that Tinker keeps open and that
    must not be closed, unlike session clients.
    """
    try:
        response = resolved(client_factory().get_server_capabilities())
    except PipelineError:
        raise
    except Exception as exc:
        raise TinkerRuntimeError(f"Tinker capabilities request failed ({type(exc).__name__})") from None
    matches = [
        item for item in getattr(response, "supported_models", None) or []
        if getattr(item, "model_name", None) == model
    ]
    if len(matches) != 1:
        raise TinkerRuntimeError(f"Tinker does not advertise {model}")
    capability = TinkerModelCapability(model, getattr(matches[0], "max_context_length", None))
    validate_capability(capability, model, required_context)
    return capability


def validate_capability(capability: Any, expected_model: str, required_context: int) -> None:
    if getattr(capability, "model_name", None) != expected_model:
        raise TinkerRuntimeError(f"Tinker does not advertise {expected_model}")
    maximum = getattr(capability, "max_context_length", None)
    if maximum is None:
        return
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise TinkerRuntimeError("Tinker returned an invalid context limit")
    if maximum < required_context:
        raise TinkerRuntimeError(
            f"Tinker supports {maximum:,} tokens of {expected_model}, below the required "
            f"{required_context:,}; pass --max-seq-len {maximum} to prepare within that limit"
        )


def _is_transient(failure: BaseException) -> bool:
    if isinstance(failure, (TimeoutError, ConnectionError)):
        return True
    try:
        import tinker
    except ImportError:
        return False
    return isinstance(failure, (tinker.APIConnectionError, tinker.RateLimitError, tinker.InternalServerError))


def retry_transient(operation: Any, *, sleeper: Any = time.sleep, attempts: int = 3) -> Any:
    """Retry a transient, idempotent Tinker read with bounded backoff."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient(exc) or attempt + 1 >= attempts:
                raise
        sleeper(float(2**attempt))
    raise AssertionError("unreachable")


class TinkerProvider:
    """Train canonical LangSmith trajectories with Tinker."""

    name = "tinker"

    def __init__(
        self,
        *,
        capability_resolver: Any | None = None,
        service_factory: Any | None = None,
        tinker_types: Any | None = None,
        render_fn: Any | None = None,
        renderer_factory: Any | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        self._capability_resolver = capability_resolver or fetch_model_capability
        self._service_factory = service_factory
        self._tinker_types = tinker_types
        self._render_fn = render_fn
        self._renderer_factory = renderer_factory
        self._sleeper = sleeper

    def model_from_options(self, options: ModelOptions) -> ModelSpec:
        return resolve_model_options(options, MODEL_SPECS, provider=self.name)

    def settings_from_options(self, options: TrainingOptions) -> TinkerSFTSettings:
        """Resolve Tinker defaults and reject options for other providers."""
        if any(getattr(options, name) is not None for name in (
            "lora_alpha", "pipeline_depth", "training_mode", "training_shape", "provisioning_timeout",
        )):
            raise PipelineError("--lora-alpha, --pipeline-depth, --training-mode, --training-shape, and --provisioning-timeout are Fireworks-only")
        if any(getattr(options, name) is not None for name in (
            "max_spend_usd", "hourly_rate_usd", "replicas", "spend_reserve_fraction",
        )):
            raise PipelineError(
                "--max-spend-usd, --hourly-rate-usd, --replicas, and --spend-reserve-fraction are Baseten-only; "
                "Tinker bills per token and provisions no replicas"
            )
        optional = {
            name: getattr(options, name)
            for name in ("max_dropped_training_rows", "sampler_checkpoints")
            if getattr(options, name) is not None
        }
        settings = TinkerSFTSettings(
            max_epochs=options.max_epochs,
            early_stopping_patience=options.early_stopping_patience,
            early_stopping_min_delta=options.early_stopping_min_delta,
            learning_rate=options.learning_rate,
            batch_size=options.batch_size,
            seed=options.seed,
            validation_every_steps=options.validation_every_steps,
            lora_rank=options.lora_rank,
            microbatch_token_budget=options.microbatch_token_budget,
            **optional,
        )
        settings.validate()
        return settings

    def prepare(
        self,
        workspace_id: str,
        dataset_id: str,
        data_dir: Path,
        *,
        model_options: ModelOptions,
        inference_contract: InferenceContract | None = None,
        source_workspace_id: str | None = None,
        split_from: Path | None = None,
        reasoning_policy: ReasoningPolicy = "omit",
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fetch: bool = True,
        check_render: bool = True,
        sync_splits: bool = True,
    ) -> dict[str, Any]:
        """Prepare canonical rows with the Tinker model and shared split defaults."""
        from smithtune.dataset import DEFAULT_TEST_FRACTION, DEFAULT_VALIDATION_FRACTION, prepare_dataset

        model = preflight_model(
            self.model_from_options(model_options), capability_resolver=self._capability_resolver,
        )
        model = resolve_rendering_model(model)
        return prepare_dataset(
            workspace_id,
            dataset_id,
            model,
            data_dir,
            inference_contract=inference_contract,
            source_workspace_id=source_workspace_id,
            split_from=split_from,
            reasoning_policy=reasoning_policy,
            validation_fraction=(
                DEFAULT_VALIDATION_FRACTION if validation_fraction is None else validation_fraction
            ),
            test_fraction=DEFAULT_TEST_FRACTION if test_fraction is None else test_fraction,
            fetch=fetch,
            check_render=check_render,
            sync_splits=sync_splits,
        )

    def plan(self, data_dir: Path, run_id: str, settings: Any, *,
             replay: dict[str, Any] | None = None) -> dict[str, Any]:
        """Resolve the prepared model and run settings without remote calls."""
        manifest, train_rows, validation_rows = lora_sft.load_prepared_data(data_dir, label="Tinker")
        model = _validate_manifest(manifest)
        settings = _resolved_settings(settings, model)
        lora_sft.validate_canonical_rows(train_rows, split="train")
        prepared_max_seq_len = lora_sft.prepared_max_sequence_tokens(manifest, model, label="Tinker")
        value = {
            "run_id": run_id,
            "provider": self.name,
            "method": "Tinker LoRA SFT",
            "execution": {"requires_active_client": True},
            "base_model": model.base_model,
            "model": asdict(model),
            "dataset": {
                "source_rows": manifest["langsmith"]["examples"],
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": manifest["split"]["test"],
            },
            "config": {
                "tokenizer_model": model.tokenizer_model,
                "tokenizer_revision": model.tokenizer_revision,
                "renderer": model.renderer,
                "thinking_trace_history_mode": model.thinking_trace_history_mode,
                "max_seq_len": min(prepared_max_seq_len, model.training_context_limit),
                "prepared_max_seq_len": prepared_max_seq_len,
                "lora_rank": settings.lora_rank,
                "learning_rate": settings.learning_rate,
                "microbatch_token_budget": settings.microbatch_token_budget,
                "effective_batch_size": settings.batch_size,
                "max_epochs": settings.max_epochs,
                "validation_every_steps": settings.validation_every_steps,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
                "seed": settings.seed,
                "loss": LOSS_FUNCTION,
                "loss_normalization": LOSS_NORMALIZATION,
                "capacity": CAPACITY,
                "max_dropped_training_rows": settings.max_dropped_training_rows,
                "sampler_checkpoints": settings.sampler_checkpoints,
            },
            "training_schedule": lora_sft.estimated_schedule(train_rows, settings),
            "evaluation": "forward-only cross-entropy on the held-out validation split",
            "checkpoint": (
                ("resumable state and sampler weights at every validation"
                 if settings.sampler_checkpoints == "all"
                 else "resumable state at every validation and sampler weights for each new best")
                + "; Tinker keeps saved checkpoints until you delete them"
            ),
            "billing": "per token at Tinker's training, prefill, and sampling prices; no capacity to release",
            # Tinker bills per token, so there is no active-time guard to configure.
            "budget": {"enabled": False},
        }
        if replay is not None:
            from smithtune.evaluation.replay import preview_replay_for_plan

            preview = preview_replay_for_plan(data_dir, replay)
            value["replay"] = {**preview, "serving_mode": "sampler", "evaluated_models": 2,
                               "capacity": CAPACITY, "timing": "after training",
                               "cleanup": "no owned serving resources; closes the sampling session",
                               "budget": "sampling and judge tokens are billed separately from training"}
        return value

    def train(
        self,
        data_dir: Path,
        run_dir: Path,
        run_id: str,
        settings: Any,
        *,
        confirm: bool,
        init_from_checkpoint: str | None,
        replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run client-driven Tinker SFT through injected remote seams."""
        if not confirm:
            raise PipelineError("Tinker training is billed per token; rerun with --confirm")
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise PipelineError(f"run directory must be new or empty: {run_dir}")

        plan = self.plan(Path(data_dir), run_id, settings, replay=replay)
        if replay is not None:
            from smithtune.evaluation.replay import validate_judge_credentials

            validate_judge_credentials(replay["judge_model"])
        if self._service_factory is None:
            require_api_key()
        manifest, train_rows, validation_rows = lora_sft.load_prepared_data(Path(data_dir), label="Tinker")
        model = _validate_manifest(manifest)
        settings = _resolved_settings(settings, model)
        lora_sft.validate_canonical_rows(train_rows, split="train")
        lora_sft.validate_canonical_rows(validation_rows, split="validation")
        max_seq_len = plan["config"]["max_seq_len"]
        capability = self._capability_resolver(model.base_model, max_seq_len)
        preflight_model(model, capability_resolver=lambda *_: capability, required_context=max_seq_len)

        tinker_types = self._tinker_types or _resolve_tinker_types()
        render_fn = self._render_fn or render_row
        renderer = (self._renderer_factory or load_training_renderer)(model)
        splits = lora_sft.render_splits(
            train_rows,
            validation_rows,
            render=lambda row: render_fn(row, model, tinker_types=tinker_types, renderer=renderer),
            max_seq_len=max_seq_len,
            microbatch_token_budget=settings.microbatch_token_budget,
            max_dropped_training_rows=settings.max_dropped_training_rows,
            errors=ERRORS,
        )
        plan["training_schedule"] = lora_sft.rendered_schedule(splits, settings)
        plan["dataset"].update({
            "prepared_train_rows": len(train_rows),
            "train_rows": len(splits.train),
            "dropped_train_rows": splits.dropped_train_rows,
        })

        run_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {}

        def update_state(**changes: Any) -> None:
            state.update(changes)
            lora_sft.atomic_json(run_dir / "run-state.json", state)

        result_path = run_dir / "result.json"
        renderer_identity = {"renderer": model.renderer, "tokenizer_revision": model.tokenizer_revision}
        audit_identity = lora_sft.audit_identity(Path(data_dir), manifest, plan)
        lora_sft.atomic_json(run_dir / "plan.json", plan)
        if replay is not None:
            from smithtune.artifacts import _load_jsonl
            from smithtune.evaluation.replay import ensure_judge_calibration, prepare_replay_evaluation, preflight_langsmith
            from smithtune.inference import _chat_completion

            preflight_langsmith(data_dir)
            prepare_replay_evaluation(
                data_dir, run_dir / "replay", replay["max_points_per_trajectory"], replay["max_output_tokens"],
            )
            ensure_judge_calibration(_load_jsonl(run_dir / "replay" / "cases.jsonl"),
                                     run_dir / "replay", replay["judge_model"], _chat_completion)
        lora_sft.atomic_json(run_dir / "epochs.json", [])
        lora_sft.atomic_json(run_dir / "validations.json", [])
        update_state(
            status="planned",
            provider=self.name,
            run_id=run_id,
            configuration=plan["config"],
            split=plan["dataset"],
            renderer_identity=renderer_identity,
            **audit_identity,
            completed_epoch=0,
            completed_step=0,
            last_resumable_state_uri=None,
            best_resumable_state_uri=None,
            best_sampler_weights_uri=None,
            best_epoch=None,
            best_step=None,
            cleanup_status="pending",
            primary_error=None,
        )

        loop = lora_sft.SFTLoop(
            trainer=None,  # bound once Tinker creates the training client
            settings=settings,
            run_id=run_id,
            splits=splits,
            run_dir=run_dir,
            update_state=update_state,
            errors=ERRORS,
            retry=lambda operation: retry_transient(operation, sleeper=self._sleeper),
            save_every_sampler=settings.sampler_checkpoints == "all",
        )
        result: dict[str, Any] = {
            "status": "training",
            "provider": self.name,
            "tinker_run_id": None,
            "run_id": run_id,
            "model_identity": plan["model"],
            "model_sha256": audit_identity["model_sha256"],
            "configuration": plan["config"],
            "split": plan["dataset"],
            "renderer_identity": renderer_identity,
            "capability": asdict(capability),
            **loop.summary(),
            "primary_error": None,
            "cleanup": {"session_closed": False, "error": None},
        }
        service: Any | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        previous_sigterm_handler: Any = None
        sigterm_handler_installed = False

        try:
            update_state(status="provisioning")
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def request_termination(_signal_number: int, _frame: Any) -> None:
                raise TerminationRequested("SIGTERM requested Tinker lifecycle shutdown")

            signal.signal(signal.SIGTERM, request_termination)
            sigterm_handler_installed = True
            metadata = {"smithtune_run_id": run_id}
            service = (self._service_factory or service_client)(user_metadata=metadata)
            trainer = resolved(service.create_lora_training_client(
                base_model=model.base_model,
                rank=settings.lora_rank,
                seed=settings.seed,
                user_metadata=metadata,
            ))
            tinker_run_id = getattr(trainer, "model_id", None)
            if not isinstance(tinker_run_id, str) or not tinker_run_id.strip():
                raise TinkerRuntimeError("Tinker training client did not return a run ID")
            result["tinker_run_id"] = tinker_run_id
            update_state(tinker_run_id=tinker_run_id)
            if init_from_checkpoint is not None:
                trainer.load_state_with_optimizer(init_from_checkpoint).result()
            update_state(status="training")
            loop.trainer = _TinkerOperations(trainer, tinker_types)
            loop.run()
        except BaseException as exc:
            primary_error = exc
        finally:
            if service is not None:
                try:
                    close_service(service)
                    result["cleanup"]["session_closed"] = True
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if sigterm_handler_installed:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm_handler)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                result["cleanup"]["error"] = "; ".join(type(exc).__name__ for exc in cleanup_errors)

        safe_primary_error = (
            {"type": type(primary_error).__name__,
             "message": "Tinker lifecycle failed; exception details omitted for credential safety"}
            if primary_error is not None else None
        )
        status = "failed" if primary_error is not None or cleanup_errors else "completed"
        result.update({**loop.summary(), "status": status, "primary_error": safe_primary_error})
        try:
            update_state(
                status=status,
                completed_epoch=len(loop.epochs),
                completed_step=loop.completed_step,
                last_resumable_state_uri=loop.last_state_uri,
                best_resumable_state_uri=loop.best_state_uri,
                best_sampler_weights_uri=loop.best_sampler_uri,
                best_epoch=loop.best_epoch,
                best_step=loop.best_step,
                cleanup_status="failed" if cleanup_errors else "completed",
                cleanup_error=result["cleanup"]["error"],
                primary_error=safe_primary_error,
            )
            lora_sft.atomic_json(run_dir / "epochs.json", loop.epochs)
            lora_sft.atomic_json(run_dir / "validations.json", loop.validations)
            lora_sft.atomic_json(result_path, result)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                cleanup_errors.append(exc)

        if primary_error is not None and cleanup_errors:
            raise BaseExceptionGroup("Tinker training and cleanup both failed", [primary_error, *cleanup_errors])
        if primary_error is not None:
            raise primary_error
        if cleanup_errors:
            raise cleanup_errors[0] if len(cleanup_errors) == 1 else BaseExceptionGroup("Tinker cleanup failed", cleanup_errors)
        if replay is not None:
            from smithtune.evaluation.replay import run_replay_evaluation, training_metadata
            from smithtune.providers.tinker.sampling import TinkerReplaySampler

            try:
                if loop.best_sampler_uri is None:
                    raise PipelineError("training produced no best checkpoint for replay")
                sampler = TinkerReplaySampler(model, loop.best_sampler_uri, run_dir / "replay")
                result["replay"] = run_replay_evaluation(
                    data_dir, run_dir / "replay", loop.best_sampler_uri,
                    base_model=model.base_model, replay_sampler=sampler, confirm=True,
                    training=training_metadata(run_dir), **replay,
                )
            except BaseException:
                result["replay_status"] = "incomplete"
                lora_sft.atomic_json(result_path, result)
                update_state(replay_status="incomplete")
                raise
            result["replay_status"] = "completed"
            lora_sft.atomic_json(result_path, result)
            update_state(replay_status="completed")
        return result


class _TinkerOperations:
    """Turn Tinker's summed losses into token means for the shared loop."""

    def __init__(self, trainer: Any, tinker_types: Any) -> None:
        self.trainer = trainer
        self.types = tinker_types

    def _loss_sum(self, output: Any, *, operation: str) -> tuple[float, dict[str, float]]:
        metrics = lora_sft.finite_metrics(output, operation=operation, error=TinkerRuntimeError)
        loss_sum = metrics.get("loss:sum")
        if loss_sum is None:
            raise TinkerRuntimeError(f"{operation} returned no finite loss")
        return loss_sum, metrics

    def forward_backward(self, microbatch: list[Any], *, group_active_tokens: float):
        active_tokens = _active_tokens(microbatch, operation="forward_backward")
        if not math.isfinite(group_active_tokens) or group_active_tokens <= 0:
            raise TinkerRuntimeError("optimizer step has no active target tokens")
        scale = 1.0 / group_active_tokens
        scaled = [_scaled_datum(datum, scale, self.types) for datum in microbatch]
        loss_sum, metrics = self._loss_sum(
            self.trainer.forward_backward(scaled, LOSS_FUNCTION).result(), operation="forward_backward",
        )
        # The scaled sum is this microbatch's share of the step's token mean.
        return loss_sum * group_active_tokens / active_tokens, active_tokens, metrics

    def forward(self, microbatch: list[Any]):
        active_tokens = _active_tokens(microbatch, operation="validation forward")
        loss_sum, metrics = self._loss_sum(
            self.trainer.forward(microbatch, LOSS_FUNCTION).result(), operation="validation forward",
        )
        return loss_sum / active_tokens, active_tokens, metrics

    def optim_step(self, learning_rate: float) -> dict[str, float]:
        return lora_sft.finite_metrics(
            self.trainer.optim_step(self.types.AdamParams(learning_rate=learning_rate)).result(),
            operation="optimizer step", error=TinkerRuntimeError,
        )

    def save_state(self, name: str) -> str:
        return lora_sft.checkpoint_path(
            self.trainer.save_state(name).result(), operation="save_state", error=TinkerRuntimeError,
        )

    def save_weights_for_sampler(self, name: str) -> str:
        return lora_sft.checkpoint_path(
            self.trainer.save_weights_for_sampler(name).result(),
            operation="save_weights_for_sampler", error=TinkerRuntimeError,
        )


def _active_tokens(microbatch: list[Any], *, operation: str) -> float:
    active_tokens = lora_sft.batch_active_tokens(microbatch)
    if not math.isfinite(active_tokens) or active_tokens <= 0:
        raise TinkerRuntimeError(f"{operation} has no active target tokens")
    return active_tokens


def _scaled_datum(datum: Any, scale: float, tinker_types: Any) -> Any:
    weights = lora_sft.datum_weights(datum)
    if weights is None:
        raise TinkerDataError("rendered datum has no loss weights")
    return tinker_types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            **datum.loss_fn_inputs,
            "weights": tinker_types.TensorData(
                data=[weight * scale for weight in weights], dtype="float32", shape=[len(weights)],
            ),
        },
    )


def render_row(
    row: dict[str, Any],
    model: ModelSpec,
    *,
    tinker_types: Any | None = None,
    renderer: Any | None = None,
) -> list[Any]:
    """Render one canonical row into shifted Tinker data with binary weights."""
    active_renderer = renderer if renderer is not None else load_training_renderer(model)
    rendered_items = render_row_tokens(
        row, model, renderer=active_renderer, include_loss_mask=True, reduction="none",
    )
    if not rendered_items:
        raise TinkerDataError("canonical row rendered no training datum")
    constructors = tinker_types if tinker_types is not None else _resolve_tinker_types()
    return [
        lora_sft.to_datum(item, constructors, model.max_seq_len, ignore_index=None, error=TinkerDataError)
        for item in rendered_items
    ]


def _resolve_tinker_types() -> Any:
    try:
        import tinker
    except ImportError as exc:
        raise TinkerDataError(
            "the Tinker SDK is unavailable; reinstall using the installation command in the README, then run smithtune doctor"
        ) from exc
    return tinker


def _resolved_settings(settings: Any, model: ModelSpec) -> TinkerSFTSettings:
    if not isinstance(settings, TinkerSFTSettings):
        raise PipelineError("Tinker training requires Tinker SFT settings")
    settings.validate()
    return replace(
        settings,
        lora_rank=model.default_lora_rank if settings.lora_rank is None else settings.lora_rank,
        microbatch_token_budget=(
            model.max_seq_len if settings.microbatch_token_budget is None else settings.microbatch_token_budget
        ),
    )


def _validate_manifest(manifest: dict[str, Any]) -> ModelSpec:
    model = resolve_prepared_model(manifest, MODEL_SPECS, provider="tinker")
    lora_sft.validate_dataset_counts(manifest, label="Tinker")
    return model

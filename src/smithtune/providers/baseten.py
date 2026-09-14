"""Baseten Loops model configuration, rendering, and guarded SFT."""

from __future__ import annotations

import math
import json
import hashlib
import os
import random
import re
import signal
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import urllib.error
import urllib.request

from smithtune.inference_contract import InferenceContract
from smithtune.models import resolve_model_options, resolve_prepared_model
from smithtune.rendering import SFT_TARGET_POLICY, resolved_renderer_name
from smithtune.providers.base import (
    CommonSFTSettings,
    ModelOptions,
    ModelSpec,
    PipelineError,
    ReasoningPolicy,
    TrainingOptions,
)


# Loops uses this sentinel for targets excluded from assistant-only loss.
IGNORE_INDEX = -100
MODEL_SPECS = {
    "qwen3p8-27b": ModelSpec(
        name="qwen3p8-27b",
        base_model="Qwen/Qwen3.8-27B",
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        renderer="qwen3_8_preserved",
        max_seq_len=262_144,
        trainer_max_seq_len=131_072,
        thinking_trace_history_mode="preserved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="baseten",
    )
}
DEFAULT_MODEL = MODEL_SPECS["qwen3p8-27b"]
BASETEN_CAPABILITIES_URL = "https://api.baseten.co/v1/loops/capabilities"
BASETEN_API_ROOT = "https://api.baseten.co/v1/loops"
MAX_CAPABILITIES_RESPONSE_BYTES = 1024 * 1024
MAX_MANAGEMENT_RESPONSE_BYTES = 4 * 1024 * 1024
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


class BasetenDataError(PipelineError):
    """A canonical trajectory cannot safely be used for Baseten training."""


class BasetenRuntimeError(RuntimeError):
    """A credential-free failure at a Baseten service boundary."""


class BudgetExceeded(BasetenRuntimeError):
    """The configured Baseten active-time ceiling has been reached."""


class TerminationRequested(BaseException):
    """A SIGTERM requested orderly Baseten lifecycle cleanup."""


def _maximum_active_seconds(
    max_spend_usd: float | None, hourly_rate_usd: float | None,
    spend_reserve_fraction: float,
) -> float | None:
    """Validate the optional spend guard and apply its cleanup reserve."""
    if (max_spend_usd is None) != (hourly_rate_usd is None):
        raise PipelineError(
            "Baseten maximum spend and hourly rate must be supplied together"
        )
    for name, value in (
        ("maximum spend", max_spend_usd),
        ("hourly rate", hourly_rate_usd),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise PipelineError(f"Baseten {name} must be finite and positive")
    if (
        isinstance(spend_reserve_fraction, bool)
        or not isinstance(spend_reserve_fraction, (int, float))
        or not math.isfinite(spend_reserve_fraction)
        or not 0 <= spend_reserve_fraction < 1
    ):
        raise PipelineError("Baseten spend reserve fraction must be in [0, 1)")
    if max_spend_usd is None:
        return None
    assert hourly_rate_usd is not None
    return (
        max_spend_usd
        * (1 - spend_reserve_fraction)
        / hourly_rate_usd
        * 3600
    )


@dataclass(frozen=True)
class BasetenSFTSettings(CommonSFTSettings):
    """Configuration accepted by the Baseten Loops training lifecycle."""

    lora_rank: int | None = None
    microbatch_token_budget: int | None = None
    max_spend_usd: float | None = None
    hourly_rate_usd: float | None = None
    replicas: int = 1
    spend_reserve_fraction: float = 0.1
    max_dropped_training_rows: int = 1

    def validate(self) -> None:
        numeric = {
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "replicas": self.replicas,
            "max_dropped_training_rows": self.max_dropped_training_rows,
        }
        for name in ("lora_rank", "microbatch_token_budget"):
            value = getattr(self, name)
            if value is not None:
                numeric[name] = value
        invalid = [
            name for name, value in numeric.items()
            if isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ]
        if invalid:
            raise PipelineError("Baseten numeric settings must be finite: " + ", ".join(invalid))
        counts = {
            name: value for name, value in numeric.items()
            if name not in {"early_stopping_min_delta", "learning_rate"}
        }
        non_integer = [name for name, value in counts.items() if not isinstance(value, int)]
        if non_integer:
            raise PipelineError("Baseten count settings must be integers: " + ", ".join(non_integer))
        invalid_counts = [
            name for name, value in counts.items()
            if value < (0 if name in {"seed", "max_dropped_training_rows"} else 1)
        ]
        if invalid_counts:
            raise PipelineError("invalid Baseten count settings: " + ", ".join(invalid_counts))
        super().validate()
        _maximum_active_seconds(self.max_spend_usd, self.hourly_rate_usd, self.spend_reserve_fraction)

    def maximum_active_seconds(self) -> float | None:
        self.validate()
        return _maximum_active_seconds(self.max_spend_usd, self.hourly_rate_usd, self.spend_reserve_fraction)


class BudgetGuard:
    """Optional wall-clock guard with a configurable spend reserve."""

    def __init__(
        self,
        max_spend_usd: float | None,
        hourly_rate_usd: float | None,
        *,
        spend_reserve_fraction: float | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.maximum_active_seconds = _maximum_active_seconds(
            max_spend_usd, hourly_rate_usd,
            BasetenSFTSettings().spend_reserve_fraction
            if spend_reserve_fraction is None else spend_reserve_fraction,
        )
        self._clock = clock
        self._started_at: float | None = None
        self._elapsed_seconds = 0.0

    def start(self) -> None:
        if self.maximum_active_seconds is not None and self._started_at is None:
            self._started_at = float(self._clock())

    @property
    def elapsed_seconds(self) -> float:
        if self.maximum_active_seconds is None:
            return 0.0
        return self._elapsed_seconds

    def check(self, operation: str) -> None:
        if self.maximum_active_seconds is None:
            return
        if self._started_at is None:
            raise RuntimeError("budget guard has not started")
        self.snapshot()
        if self._elapsed_seconds >= self.maximum_active_seconds:
            raise BudgetExceeded(f"Baseten budget reached before {operation}")

    def snapshot(self) -> float:
        """Record current active time without enforcing the budget ceiling."""
        if self.maximum_active_seconds is None:
            return 0.0
        if self._started_at is None:
            return self._elapsed_seconds
        self._elapsed_seconds = float(self._clock()) - self._started_at
        return self._elapsed_seconds


@dataclass(frozen=True)
class BasetenModelCapability:
    model_name: str
    max_context_length: int
    supports_vision_language: bool | None = None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BasetenRuntimeError(f"duplicate Baseten response key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise BasetenRuntimeError(f"non-finite Baseten response value: {value}")


def fetch_model_capability(
    model: str,
    max_sequence_length: int,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    opener: Any = urllib.request.urlopen,
    sleeper: Any = time.sleep,
) -> BasetenModelCapability:
    """Confirm model/context support without provisioning a paid trainer."""
    resolved_key = api_key if api_key is not None else os.environ.get("BASETEN_API_KEY")
    if not resolved_key or not resolved_key.strip():
        raise BasetenRuntimeError("BASETEN_API_KEY is required for preflight")
    request = urllib.request.Request(
        BASETEN_CAPABILITIES_URL,
        headers={"Accept": "application/json", "Authorization": f"Api-Key {resolved_key}"},
        method="GET",
    )
    def read_payload() -> bytes:
        with opener(request, timeout=timeout_seconds) as response:
            return response.read(MAX_CAPABILITIES_RESPONSE_BYTES + 1)

    try:
        payload = retry_idempotent(read_payload, sleeper=sleeper)
    except urllib.error.HTTPError as exc:
        raise BasetenRuntimeError(f"Baseten capabilities request returned HTTP {exc.code}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise BasetenRuntimeError("Baseten capabilities request failed") from exc
    if len(payload) > MAX_CAPABILITIES_RESPONSE_BYTES:
        raise BasetenRuntimeError("Baseten capabilities response exceeded size limit")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise BasetenRuntimeError("Baseten returned malformed capabilities JSON") from exc
    items = document.get("supported_models") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise BasetenRuntimeError("Baseten capabilities omitted supported_models")
    matches = [item for item in items if isinstance(item, dict) and item.get("model_name") == model]
    if len(matches) != 1:
        raise BasetenRuntimeError(f"Baseten workspace does not advertise {model}")
    maximum = matches[0].get("max_context_length")
    vision = matches[0].get("supports_vision_language")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise BasetenRuntimeError("Baseten returned an invalid context limit")
    if vision is not None and not isinstance(vision, bool):
        raise BasetenRuntimeError("Baseten returned an invalid vision support flag")
    if maximum < max_sequence_length:
        raise BasetenRuntimeError(
            f"Baseten workspace context limit is below {max_sequence_length:,}"
        )
    return BasetenModelCapability(model, maximum, vision)


def retry_idempotent(operation: Any, *, sleeper: Any = time.sleep, attempts: int = 3) -> Any:
    """Retry a transient, idempotent remote operation with bounded backoff."""
    if attempts < 1:
        raise ValueError("retry attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient_remote_failure(exc) or attempt + 1 >= attempts:
                raise
        sleeper(float(2**attempt))
    raise AssertionError("unreachable")


def _is_transient_remote_failure(failure: BaseException) -> bool:
    if isinstance(failure, urllib.error.HTTPError):
        return failure.code in _TRANSIENT_HTTP_STATUS
    if isinstance(
        failure,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
    ):
        return True

    try:
        import httpx
    except ImportError:
        pass
    else:
        if isinstance(failure, httpx.HTTPStatusError):
            response = getattr(failure, "response", None)
            return getattr(response, "status_code", None) in _TRANSIENT_HTTP_STATUS
        if isinstance(failure, httpx.TransportError):
            return True

    try:
        from baseten.loops import ServerShutdownError
    except ImportError:
        pass
    else:
        if isinstance(failure, ServerShutdownError):
            return True
    return False


def _reject_loops_reuse_overrides() -> None:
    overrides = sorted(name for name in os.environ if name.startswith("LOOPS_REUSE_"))
    if overrides:
        raise PipelineError(
            "Baseten training rejects Loops resource-reuse overrides: "
            + ", ".join(overrides)
        )


def validate_resource_id(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise ValueError(f"invalid Baseten {kind} ID")
    return value


def _manual_deactivate_command(run_id: str) -> str:
    return f"baseten loops run deactivate --run-id {run_id} --yes"


def _cleanup_failure(run_id: str, operation: str, failure: BaseException) -> BasetenRuntimeError:
    command = _manual_deactivate_command(run_id)
    if isinstance(failure, BasetenRuntimeError) and command in str(failure):
        return failure
    return BasetenRuntimeError(
        f"Baseten {operation} failed for run {run_id} "
        f"({type(failure).__name__}); manually run: {command}"
    )


class _BasetenManagement:
    def __init__(self, *, api_key: str | None = None, opener: Any = urllib.request.urlopen) -> None:
        resolved = api_key if api_key is not None else os.environ.get("BASETEN_API_KEY")
        if not resolved or not resolved.strip():
            raise BasetenRuntimeError("BASETEN_API_KEY is required for cleanup")
        self._api_key = resolved
        self._opener = opener

    def _request(self, path: str, method: str) -> dict[str, Any]:
        if path.startswith("/") or ".." in path:
            raise ValueError("Baseten management path must be relative and contained")
        request = urllib.request.Request(
            f"{BASETEN_API_ROOT}/{path}",
            data=b"{}" if method == "POST" else None,
            headers={
                "Accept": "application/json",
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with self._opener(request, timeout=30.0) as response:
                payload = response.read(MAX_MANAGEMENT_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if method == "GET" and exc.code == 404:
                return {"missing": True}
            raise
        if len(payload) > MAX_MANAGEMENT_RESPONSE_BYTES:
            raise BasetenRuntimeError("Baseten management response exceeded size limit")
        if not payload:
            return {}
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BasetenRuntimeError("Baseten returned malformed management JSON") from exc
        if not isinstance(value, dict):
            raise BasetenRuntimeError("Baseten management response is not an object")
        return value

    def session_run_ids(self, session_id: str) -> list[str]:
        session_id = validate_resource_id(session_id, kind="session")
        response = self._request("runs", "GET")
        runs = response.get("runs")
        if not isinstance(runs, list):
            raise BasetenRuntimeError("Baseten runs response omitted runs")
        return sorted(
            {
                validate_resource_id(run["id"], kind="run")
                for run in runs
                if isinstance(run, dict)
                and run.get("session_id") == session_id
                and isinstance(run.get("id"), str)
            }
        )

    def deactivate_run(self, run_id: str) -> None:
        run_id = validate_resource_id(run_id, kind="run")
        if not self.run_is_inactive(run_id):
            self._request(f"runs/{run_id}/deactivate", "POST")

    def run_is_inactive(self, run_id: str) -> bool:
        run_id = validate_resource_id(run_id, kind="run")
        current = self._request(f"runs/{run_id}", "GET")
        if current.get("missing"):
            return True
        run = current.get("run")
        if not isinstance(run, dict):
            raise BasetenRuntimeError("Baseten run lookup omitted run")
        status = run.get("status")
        name = status.get("name") if isinstance(status, dict) else status
        return str(name).upper() == "INACTIVE"


def wait_for_run_inactive(
    run_id: str,
    *,
    management: Any,
    sleeper: Any = time.sleep,
    attempts: int = 12,
    interval_seconds: float = 1.0,
) -> None:
    run_id = validate_resource_id(run_id, kind="run")
    if attempts < 1:
        raise ValueError("inactive polling attempts must be positive")
    for attempt in range(attempts):
        try:
            inactive = retry_idempotent(
                lambda: management.run_is_inactive(run_id), sleeper=sleeper
            )
        except BaseException as exc:
            raise _cleanup_failure(run_id, "inactive polling", exc)
        if inactive:
            return
        if attempt + 1 < attempts:
            sleeper(interval_seconds)
    raise BasetenRuntimeError(
        f"Baseten run {run_id} remained active; manually run: "
        f"{_manual_deactivate_command(run_id)}"
    )


def deactivate_identity(
    *,
    run_id: str | None,
    session_id: str | None,
    management: Any | None = None,
    sleeper: Any = time.sleep,
) -> list[str]:
    if run_id is not None:
        targets = [validate_resource_id(run_id, kind="run")]
    else:
        targets = []
    try:
        active_management = management if management is not None else _BasetenManagement()
    except BaseException as exc:
        if targets:
            raise _cleanup_failure(targets[0], "management authentication", exc)
        raise
    if not targets and session_id is not None:
        session_id = validate_resource_id(session_id, kind="session")
        targets = sorted(
            set(
                retry_idempotent(
                    lambda: active_management.session_run_ids(session_id),
                    sleeper=sleeper,
                )
            )
        )
    elif not targets:
        raise BasetenRuntimeError("cleanup requires a Baseten run or session ID")
    errors: list[BaseException] = []
    for target in targets:
        try:
            retry_idempotent(
                lambda target=target: active_management.deactivate_run(target),
                sleeper=sleeper,
            )
        except BaseException as exc:
            errors.append(_cleanup_failure(target, "deactivation", exc))
    for target in targets:
        try:
            wait_for_run_inactive(
                target, management=active_management, sleeper=sleeper
            )
        except BaseException as exc:
            errors.append(_cleanup_failure(target, "inactive polling", exc))
    failure = _combine_exceptions("Baseten deactivation failed", errors)
    if failure is not None:
        raise failure
    return targets


class BasetenProvider:
    """Train canonical LangSmith trajectories with Baseten Loops."""

    name = "baseten"

    def __init__(
        self,
        *,
        capability_resolver: Any | None = None,
        service_factory: Any | None = None,
        management: Any | None = None,
        loops_types: Any | None = None,
        render_fn: Any | None = None,
        renderer_factory: Any | None = None,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        self._capability_resolver = capability_resolver or fetch_model_capability
        self._service_factory = service_factory
        self._management = management
        self._loops_types = loops_types
        self._render_fn = render_fn
        self._renderer_factory = renderer_factory
        self._monotonic = monotonic
        self._sleeper = sleeper

    def model_from_options(self, options: ModelOptions) -> ModelSpec:
        return resolve_model_options(options, MODEL_SPECS, provider=self.name)

    def settings_from_options(self, options: TrainingOptions) -> BasetenSFTSettings:
        """Resolve Baseten defaults and reject Fireworks-only settings."""
        if options.lora_alpha is not None or options.pipeline_depth is not None:
            raise PipelineError("--lora-alpha and --pipeline-depth are Fireworks-only")
        optional = {
            name: getattr(options, name)
            for name in ("replicas", "spend_reserve_fraction", "max_dropped_training_rows")
            if getattr(options, name) is not None
        }
        settings = BasetenSFTSettings(
            max_epochs=options.max_epochs,
            early_stopping_patience=options.early_stopping_patience,
            early_stopping_min_delta=options.early_stopping_min_delta,
            learning_rate=options.learning_rate,
            batch_size=options.batch_size,
            seed=options.seed,
            lora_rank=options.lora_rank,
            microbatch_token_budget=options.microbatch_token_budget,
            max_spend_usd=options.max_spend_usd,
            hourly_rate_usd=options.hourly_rate_usd,
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
        reasoning_policy: ReasoningPolicy = "omit",
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fetch: bool = True,
        check_render: bool = True,
    ) -> dict[str, Any]:
        """Prepare canonical rows with the Baseten model and shared split defaults."""
        from smithtune.dataset import DEFAULT_TEST_FRACTION, DEFAULT_VALIDATION_FRACTION, prepare_dataset

        return prepare_dataset(
            workspace_id,
            dataset_id,
            self.model_from_options(model_options),
            data_dir,
            inference_contract=inference_contract,
            reasoning_policy=reasoning_policy,
            validation_fraction=(
                DEFAULT_VALIDATION_FRACTION if validation_fraction is None else validation_fraction
            ),
            test_fraction=DEFAULT_TEST_FRACTION if test_fraction is None else test_fraction,
            fetch=fetch,
            check_render=check_render,
        )

    def plan(self, data_dir: Path, run_id: str, settings: Any) -> dict[str, Any]:
        """Resolve the prepared model and run settings without remote calls."""
        _reject_loops_reuse_overrides()
        manifest, _train_rows, _validation_rows = _load_prepared_data(data_dir)
        model = _validate_manifest(manifest)
        settings = _resolved_settings(settings, model)
        prepared_max_seq_len = _prepared_max_sequence_tokens(manifest, model)
        max_seq_len = min(prepared_max_seq_len, model.training_context_limit)
        return {
            "run_id": run_id,
            "provider": self.name,
            "method": "Baseten Loops LoRA SFT",
            "base_model": model.base_model,
            "model": asdict(model),
            "dataset": {
                "source_rows": manifest["langsmith"]["examples"],
                "train_rows": len(_train_rows),
                "validation_rows": len(_validation_rows),
                "test_rows": manifest["split"]["test"],
            },
            "config": {
                "tokenizer_model": model.tokenizer_model,
                "tokenizer_revision": model.tokenizer_revision,
                "renderer": model.renderer,
                "thinking_trace_history_mode": model.thinking_trace_history_mode,
                "max_seq_len": max_seq_len,
                "prepared_max_seq_len": prepared_max_seq_len,
                "lora_rank": settings.lora_rank,
                "learning_rate": settings.learning_rate,
                "microbatch_token_budget": settings.microbatch_token_budget,
                "effective_batch_size": settings.batch_size,
                "max_epochs": settings.max_epochs,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
                "seed": settings.seed,
                "replicas": settings.replicas,
                "capacity": "dedicated",
                "with_sampler": False,
                "max_dropped_training_rows": settings.max_dropped_training_rows,
            },
            "evaluation": "forward-only cross-entropy on the held-out validation split",
            "checkpoint": "resumable state after every epoch and sampler weights for each new best",
            "budget": {
                "enabled": settings.max_spend_usd is not None,
                "max_spend_usd": settings.max_spend_usd,
                "hourly_rate_usd": settings.hourly_rate_usd,
                "spend_reserve_fraction": settings.spend_reserve_fraction,
                "maximum_active_seconds": settings.maximum_active_seconds(),
            },
        }

    def train(
        self,
        data_dir: Path,
        run_dir: Path,
        run_id: str,
        settings: Any,
        *,
        confirm: bool,
        init_from_checkpoint: str | None,
    ) -> dict[str, Any]:
        """Run the guarded Loops SFT lifecycle through injected remote seams."""
        if not confirm:
            raise PipelineError(
                "Baseten training changes paid resources; rerun with --confirm"
            )
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise PipelineError(f"run directory must be new or empty: {run_dir}")

        plan = self.plan(Path(data_dir), run_id, settings)
        manifest, train_rows, validation_rows = _load_prepared_data(Path(data_dir))
        model = _validate_manifest(manifest)
        settings = _resolved_settings(settings, model)
        _validate_canonical_rows(train_rows, split="train")
        _validate_canonical_rows(validation_rows, split="validation")
        max_seq_len = plan["config"]["max_seq_len"]
        capability = retry_idempotent(
            lambda: self._capability_resolver(model.base_model, max_seq_len),
            sleeper=self._sleeper,
        )
        _validate_capability(capability, model.base_model, max_seq_len)

        loops_types = self._loops_types or _resolve_loops_types()
        render_fn = self._render_fn or render_row
        renderer_factory = self._renderer_factory or _load_renderer
        renderer = renderer_factory(model)
        rendered_train = [
            render_fn(row, model, loops_types=loops_types, renderer=renderer)
            for row in train_rows
        ]
        rendered_validation = [
            render_fn(row, model, loops_types=loops_types, renderer=renderer)
            for row in validation_rows
        ]
        if any(not values for values in rendered_train + rendered_validation):
            raise BasetenDataError("every prepared row must render training data")

        dropped_train_rows = [
            {
                "prepared_index": index,
                "model_input_tokens": max(
                    _datum_token_count(datum) for datum in values
                ),
                "reason": f"exceeds trainer limit of {max_seq_len:,} tokens",
            }
            for index, values in enumerate(rendered_train)
            if any(_datum_token_count(datum) > max_seq_len for datum in values)
        ]
        if len(dropped_train_rows) > settings.max_dropped_training_rows:
            raise BasetenDataError(
                f"{len(dropped_train_rows)} training rows exceed the Baseten trainer limit; "
                f"configured maximum dropped training rows is {settings.max_dropped_training_rows}"
            )
        oversized_validation_rows = [
            index
            for index, values in enumerate(rendered_validation)
            if any(_datum_token_count(datum) > max_seq_len for datum in values)
        ]
        if oversized_validation_rows:
            raise BasetenDataError(
                "validation data exceeds the Baseten trainer sequence limit"
            )
        dropped_indices = {
            item["prepared_index"] for item in dropped_train_rows
        }
        rendered_train = [
            values
            for index, values in enumerate(rendered_train)
            if index not in dropped_indices
        ]
        if not rendered_train:
            raise BasetenDataError("no training rows remain within the trainer limit")
        plan["dataset"].update(
            {
                "prepared_train_rows": len(train_rows),
                "train_rows": len(rendered_train),
                "dropped_train_rows": dropped_train_rows,
            }
        )
        # Check the same packing constraints before paid provisioning.
        pack_microbatches(
            (datum for values in rendered_train for datum in values),
            settings.microbatch_token_budget,
        )
        validation_batches = pack_microbatches(
            (datum for values in rendered_validation for datum in values),
            settings.microbatch_token_budget,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "run-state.json"
        state: dict[str, Any] = {}

        def update_state(**changes: Any) -> None:
            state.update(changes)
            _atomic_json(state_path, state)

        epochs_path = run_dir / "epochs.json"
        result_path = run_dir / "result.json"
        renderer_identity = {
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        }
        audit_identity = _audit_identity(Path(data_dir), manifest, plan)
        _atomic_json(run_dir / "plan.json", plan)
        _atomic_json(epochs_path, [])
        update_state(
            status="planned",
            provider="baseten",
            run_id=run_id,
            configuration=plan["config"],
            split=plan["dataset"],
            renderer_identity=renderer_identity,
            **audit_identity,
            completed_epoch=0,
            last_resumable_state_uri=None,
            best_sampler_weights_uri=None,
            best_epoch=None,
            budget_status="enabled" if settings.max_spend_usd is not None else "disabled",
            cleanup_status="pending",
            primary_error=None,
        )

        budget = BudgetGuard(
            settings.max_spend_usd,
            settings.hourly_rate_usd,
            spend_reserve_fraction=settings.spend_reserve_fraction,
            clock=self._monotonic,
        )
        epochs: list[dict[str, Any]] = []
        step_losses: list[dict[str, Any]] = []
        lowest_loss = math.inf
        patience_loss = math.inf
        epochs_without_patience_improvement = 0
        best_epoch: int | None = None
        best_sampler_uri: str | None = None
        last_state_uri: str | None = None
        service: Any | None = None
        trainer: Any | None = None
        session_id: str | None = None
        baseten_run_id: str | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        previous_sigterm_handler: Any = None
        sigterm_handler_installed = False
        budget_stopped = False
        stopped_early = False
        deactivated: list[str] = []

        result: dict[str, Any] = {
            "status": "training",
            "provider": "baseten",
            "session_id": None,
            "baseten_run_id": None,
            "run_id": run_id,
            "model_identity": plan["model"],
            "model_sha256": audit_identity["model_sha256"],
            "configuration": plan["config"],
            "split": plan["dataset"],
            "renderer_identity": renderer_identity,
            "capability": asdict(capability),
            "epochs": epochs,
            "step_losses": step_losses,
            "last_resumable_state_uri": None,
            "best_sampler_weights_uri": None,
            "best_epoch": None,
            "stopped_early": False,
            "primary_error": None,
            "budget": {
                **plan["budget"],
                "elapsed_seconds": 0.0,
                "stopped": False,
            },
            "cleanup": {
                "client_closed": False,
                "resources_deactivated": False,
                "deactivated_run_ids": [],
                "error": None,
            },
        }

        try:
            update_state(status="provisioning")
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def request_termination(_signal_number: int, _frame: Any) -> None:
                raise TerminationRequested("SIGTERM requested Baseten lifecycle shutdown")

            signal.signal(signal.SIGTERM, request_termination)
            sigterm_handler_installed = True
            budget.start()
            service_factory = self._service_factory or loops_types.ServiceClient
            service = service_factory(availability_model="dedicated")
            session_id = validate_resource_id(service.session_id, kind="session")
            result["session_id"] = session_id
            update_state(baseten_session_id=session_id)
            trainer = service.create_lora_training_client(
                base_model=model.base_model,
                rank=settings.lora_rank,
                replicas=settings.replicas,
                seed=settings.seed,
                max_seq_len=max_seq_len,
                with_sampler=False,
                name=run_id,
            )
            raw_run_id = getattr(trainer, "run_id", None)
            if not raw_run_id:
                raise BasetenRuntimeError("Baseten trainer did not return a run ID")
            baseten_run_id = validate_resource_id(raw_run_id, kind="run")
            result["baseten_run_id"] = baseten_run_id
            update_state(baseten_run_id=baseten_run_id)
            if init_from_checkpoint is not None:
                budget.check("checkpoint initialization")
                trainer.load_state_with_optimizer(init_from_checkpoint).result()
                budget.check("checkpoint initialization")
            update_state(status="training")

            for epoch in range(1, settings.max_epochs + 1):
                epoch_rows = list(rendered_train)
                random.Random(settings.seed + epoch - 1).shuffle(epoch_rows)
                train_datums = [datum for values in epoch_rows for datum in values]
                microbatches = pack_microbatches(
                    train_datums, settings.microbatch_token_budget
                )
                groups = plan_accumulation_groups(
                    microbatches, effective_batch_size=settings.batch_size
                )
                epoch_weighted_loss = 0.0
                epoch_active_tokens = 0.0
                optimizer_steps = 0
                for group in groups:
                    for microbatch in group:
                        budget.check("training step")
                        training_result = trainer.forward_backward(microbatch).result()
                        loss, active_tokens, metrics = _finite_loss_result(
                            training_result,
                            operation="forward_backward",
                            fallback_active_tokens=_batch_active_tokens(microbatch),
                        )
                        epoch_weighted_loss += loss * active_tokens
                        epoch_active_tokens += active_tokens
                        step_losses.append(
                            {
                                "epoch": epoch,
                                "operation": "forward_backward",
                                "examples": len(microbatch),
                                "loss": loss,
                                "active_tokens": active_tokens,
                                "metrics": metrics,
                            }
                        )
                    budget.check("optimizer step")
                    optimizer_result = trainer.optim_step(
                        loops_types.AdamParams(learning_rate=settings.learning_rate)
                    ).result()
                    _finite_metrics(optimizer_result, operation="optimizer step")
                    optimizer_steps += 1

                validation_weighted_loss = 0.0
                validation_active_tokens = 0.0
                for microbatch in validation_batches:
                    budget.check("validation step")
                    validation_result = retry_idempotent(
                        lambda microbatch=microbatch: trainer.forward(
                            microbatch, "cross_entropy"
                        ).result(),
                        sleeper=self._sleeper,
                    )
                    loss, active_tokens, metrics = _finite_loss_result(
                        validation_result,
                        operation="validation forward",
                        fallback_active_tokens=_batch_active_tokens(microbatch),
                    )
                    validation_weighted_loss += loss * active_tokens
                    validation_active_tokens += active_tokens
                    step_losses.append(
                        {
                            "epoch": epoch,
                            "operation": "validation_forward",
                            "examples": len(microbatch),
                            "loss": loss,
                            "active_tokens": active_tokens,
                            "metrics": metrics,
                        }
                    )
                if epoch_active_tokens <= 0 or validation_active_tokens <= 0:
                    raise BasetenRuntimeError("epoch returned no active target tokens")
                train_loss = epoch_weighted_loss / epoch_active_tokens
                validation_loss = validation_weighted_loss / validation_active_tokens

                sampler_improved = validation_loss < lowest_loss
                patience_improved = validation_loss < (
                    patience_loss - settings.early_stopping_min_delta
                )
                post_checkpoint_budget_error: BudgetExceeded | None = None
                budget.check("epoch state checkpoint")
                candidate_state_uri = _checkpoint_path(
                    trainer.save_state(_checkpoint_name(run_id, f"state-epoch-{epoch}")).result(),
                    operation="save_state",
                )
                try:
                    budget.check("epoch state checkpoint")
                except BudgetExceeded as exc:
                    if sampler_improved:
                        raise
                    post_checkpoint_budget_error = exc
                candidate_sampler_uri: str | None = None
                if sampler_improved:
                    budget.check("best checkpoint")
                    candidate_sampler_uri = _checkpoint_path(
                        trainer.save_weights_for_sampler(
                            _checkpoint_name(run_id, f"sampler-epoch-{epoch}")
                        ).result(),
                        operation="save_weights_for_sampler",
                    )
                    try:
                        budget.check("best checkpoint")
                    except BudgetExceeded as exc:
                        post_checkpoint_budget_error = exc

                # Advance durable metadata only after this epoch's required
                # checkpoint set is coherent.
                last_state_uri = candidate_state_uri
                sampler_uri = candidate_sampler_uri
                if sampler_improved:
                    lowest_loss = validation_loss
                    best_epoch = epoch
                    best_sampler_uri = sampler_uri
                if patience_improved:
                    patience_loss = validation_loss
                    epochs_without_patience_improvement = 0
                else:
                    epochs_without_patience_improvement += 1
                epoch_result = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "optimizer_steps": optimizer_steps,
                    "resumable_state_uri": candidate_state_uri,
                    "sampler_weights_uri": sampler_uri,
                    "new_best": sampler_improved,
                }
                epochs.append(epoch_result)
                _atomic_json(epochs_path, epochs)
                update_state(
                    completed_epoch=epoch,
                    last_resumable_state_uri=last_state_uri,
                    best_sampler_weights_uri=best_sampler_uri,
                    best_epoch=best_epoch,
                    active_seconds=budget.elapsed_seconds,
                )
                if post_checkpoint_budget_error is not None:
                    raise post_checkpoint_budget_error
                if (
                    epochs_without_patience_improvement
                    >= settings.early_stopping_patience
                ):
                    stopped_early = True
                    break
        except BudgetExceeded:
            budget_stopped = True
        except BaseException as exc:
            primary_error = exc
        finally:
            if trainer is not None:
                try:
                    trainer.close()
                    result["cleanup"]["client_closed"] = True
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if session_id is not None or baseten_run_id is not None:
                try:
                    deactivated = deactivate_identity(
                        run_id=baseten_run_id,
                        session_id=session_id,
                        management=self._management,
                        sleeper=self._sleeper,
                    )
                    result["cleanup"]["resources_deactivated"] = True
                    result["cleanup"]["deactivated_run_ids"] = deactivated
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if sigterm_handler_installed:
                try:
                    signal.signal(signal.SIGTERM, previous_sigterm_handler)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                result["cleanup"]["error"] = "; ".join(
                    f"{type(exc).__name__}: {exc}" for exc in cleanup_errors
                )

        final_active_seconds = budget.snapshot()
        safe_primary_error = (
            _safe_primary_error(primary_error) if primary_error is not None else None
        )
        status = "budget_stopped" if budget_stopped else "completed"
        if primary_error is not None or cleanup_errors:
            status = "failed"
        result.update(
            {
                "status": status,
                "epochs": epochs,
                "step_losses": step_losses,
                "last_resumable_state_uri": last_state_uri,
                "best_sampler_weights_uri": best_sampler_uri,
                "best_epoch": best_epoch,
                "stopped_early": stopped_early,
                "primary_error": safe_primary_error,
            }
        )
        result["budget"].update(
            {
                "elapsed_seconds": final_active_seconds,
                "stopped": budget_stopped,
            }
        )
        try:
            update_state(
                status=status,
                completed_epoch=len(epochs),
                last_resumable_state_uri=last_state_uri,
                best_sampler_weights_uri=best_sampler_uri,
                best_epoch=best_epoch,
                active_seconds=final_active_seconds,
                budget_status=(
                    "reached"
                    if budget_stopped
                    else ("enabled" if settings.max_spend_usd is not None else "disabled")
                ),
                cleanup_status="failed" if cleanup_errors else "completed",
                cleanup_error=result["cleanup"]["error"],
                primary_error=safe_primary_error,
            )
            _atomic_json(epochs_path, epochs)
            _atomic_json(result_path, result)
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            else:
                cleanup_errors.append(exc)

        cleanup_error = _combine_exceptions(
            "Baseten cleanup failed", cleanup_errors
        )
        if primary_error is not None and cleanup_error is not None:
            combined_error = _combine_exceptions(
                "Baseten training and cleanup both failed",
                [primary_error, cleanup_error],
            )
            assert combined_error is not None
            raise combined_error
        if primary_error is not None:
            raise primary_error
        if cleanup_error is not None:
            raise cleanup_error
        return result


def render_row(
    row: dict[str, Any],
    model: ModelSpec,
    *,
    loops_types: Any | None = None,
    renderer: Any | None = None,
) -> list[Any]:
    """Render one canonical row and convert every result to shifted Loops data."""
    try:
        from training.utils import parse_train_on_what, render_messages_to_datums
    except ImportError as exc:
        raise BasetenDataError("training runtime is unavailable; reinstall using the GitHub installation command in the README, then run smithtune doctor") from exc

    if renderer is not None:
        resolved_renderer_name(model)
    active_renderer = renderer if renderer is not None else _load_renderer(model)
    rendered = render_messages_to_datums(
        row["messages"],
        renderer=active_renderer,
        train_on_what=parse_train_on_what(SFT_TARGET_POLICY),
        tools=row.get("tools"),
        include_loss_mask=True,
        reduction="none",
    )
    rendered_items = rendered if isinstance(rendered, list) else [rendered]
    if not rendered_items:
        raise BasetenDataError("canonical row rendered no training datum")
    constructors = loops_types if loops_types is not None else _resolve_loops_types()
    return [_to_loops_datum(item, constructors, model.max_seq_len) for item in rendered_items]


def pack_microbatches(datums: Iterable[Any], token_budget: int) -> list[list[Any]]:
    """Greedily pack in order and reject trajectories larger than the budget."""
    if token_budget < 1:
        raise BasetenDataError("microbatch token budget must be positive")

    batches: list[list[Any]] = []
    current: list[Any] = []
    current_tokens = 0
    for datum in datums:
        token_count = _datum_token_count(datum)
        if token_count > token_budget:
            raise BasetenDataError(
                "datum exceeds the configured microbatch token budget"
            )
        if current and current_tokens + token_count > token_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(datum)
        current_tokens += token_count
    if current:
        batches.append(current)
    return batches


def plan_accumulation_groups(
    microbatches: Iterable[Iterable[Any]],
    *,
    effective_batch_size: int,
) -> list[list[list[Any]]]:
    """Group ordered microbatches for example-counted gradient accumulation."""
    if effective_batch_size < 1:
        raise BasetenDataError("effective batch size must be positive")

    groups: list[list[list[Any]]] = []
    current: list[list[Any]] = []
    examples = 0
    for batch in microbatches:
        remaining = list(batch)
        if not remaining:
            raise BasetenDataError("microbatches must not be empty")
        while remaining:
            capacity = effective_batch_size - examples
            segment, remaining = remaining[:capacity], remaining[capacity:]
            current.append(segment)
            examples += len(segment)
            if examples == effective_batch_size:
                groups.append(current)
                current = []
                examples = 0
    if current:
        groups.append(current)
    return groups


def _to_loops_datum(rendered: Any, loops_types: Any, max_sequence_tokens: int) -> Any:
    try:
        token_ids = [int(token) for token in rendered.token_ids]
        token_weights = [float(weight) for weight in rendered.token_weights]
    except (AttributeError, TypeError, ValueError) as exc:
        raise BasetenDataError("renderer returned invalid token IDs or weights") from exc
    if len(token_ids) < 2:
        raise BasetenDataError("rendered trajectory needs at least two tokens")
    if len(token_ids) != len(token_weights):
        raise BasetenDataError("rendered token IDs and weights have different lengths")
    if len(token_ids) > max_sequence_tokens:
        raise BasetenDataError(
            f"rendered trajectory exceeds the {max_sequence_tokens:,}-token limit"
        )
    if any(not math.isfinite(weight) or weight not in (0.0, 1.0) for weight in token_weights):
        raise BasetenDataError("renderer weights must be finite binary assistant masks")

    inputs = token_ids[:-1]
    weights = token_weights[1:]
    if not any(weights):
        raise BasetenDataError("rendered trajectory has no trainable assistant tokens")
    targets = [token if weight else IGNORE_INDEX for token, weight in zip(token_ids[1:], weights, strict=True)]
    return loops_types.Datum(
        model_input=loops_types.ModelInput.from_ints(inputs),
        loss_fn_inputs={
            "target_tokens": loops_types.TensorData(
                data=targets,
                dtype="int64",
                shape=[len(inputs)],
            ),
            "weights": loops_types.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(inputs)],
            ),
        },
    )


def _load_renderer(model: Any) -> Any:
    try:
        from training.renderer import get_renderer
        from training.utils.tokenizers import load_tokenizer
    except ImportError as exc:
        raise BasetenDataError("training runtime is unavailable; reinstall using the GitHub installation command in the README, then run smithtune doctor") from exc
    renderer_name = resolved_renderer_name(model)
    tokenizer = load_tokenizer(
        model.tokenizer_model,
        model.tokenizer_revision,
        trust_remote_code=getattr(model, "trust_remote_code", False),
    )
    return get_renderer(renderer_name, tokenizer)


def _resolve_loops_types() -> Any:
    try:
        import baseten.loops
    except ImportError as exc:
        raise BasetenDataError(
            "Baseten Loops is unavailable; reinstall using the GitHub installation command in the README, then run smithtune doctor"
        ) from exc
    return baseten.loops


def _datum_token_count(datum: Any) -> int:
    try:
        count = len(datum.model_input.to_ints())
    except (AttributeError, TypeError) as exc:
        raise BasetenDataError("datum has no readable model input tokens") from exc
    if count < 1:
        raise BasetenDataError("datum has no model input tokens")
    return count


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _audit_identity(
    data_dir: Path, manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    prepared = data_dir / "prepared"
    source = manifest["langsmith"]
    renderer_identity = {
        "renderer": plan["config"]["renderer"],
        "tokenizer_revision": plan["config"]["tokenizer_revision"],
    }
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_identity": plan["model"],
        "model_sha256": _sha256_json(plan["model"]),
        "source_dataset_identity": {
            "workspace_id": source.get("workspace_id"),
            "dataset_id": source.get("dataset_id"),
            "examples": source.get("examples"),
            "source_examples_sha256": manifest.get("source_examples_sha256"),
        },
        "split_files_sha256": {
            split: _sha256_file(prepared / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
        "renderer_sha256": _sha256_json(renderer_identity),
        "settings_sha256": _sha256_json(
            {"model": plan["model"], "budget": plan["budget"], "config": plan["config"]}
        ),
    }


def _safe_primary_error(failure: BaseException) -> dict[str, str]:
    return {
        "type": type(failure).__name__,
        "message": (
            "Baseten lifecycle failed; exception details omitted for credential safety"
        ),
    }


def _validate_canonical_rows(rows: list[dict[str, Any]], *, split: str) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            raise PipelineError(f"{split} row {index} has no canonical messages")
        if not all(isinstance(message, dict) for message in messages):
            raise PipelineError(f"{split} row {index} has invalid canonical messages")
        if tools is not None and not isinstance(tools, list):
            raise PipelineError(f"{split} row {index} has invalid tool declarations")


def _validate_capability(capability: Any, expected_model: str, required_context: int) -> None:
    model_name = getattr(capability, "model_name", None)
    maximum = getattr(capability, "max_context_length", None)
    if model_name != expected_model:
        raise BasetenRuntimeError(f"Baseten workspace does not advertise {expected_model}")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < required_context:
        raise BasetenRuntimeError(
            f"Baseten workspace context limit is below {required_context:,}"
        )


def _finite_metrics(result: Any, *, operation: str) -> dict[str, float]:
    try:
        raw_metrics = getattr(result, "metrics", None) or {}
        metrics = {str(name): float(value) for name, value in dict(raw_metrics).items()}
        loss = getattr(result, "loss", None)
        if loss is not None:
            metrics.setdefault("loss", float(loss))
    except (TypeError, ValueError) as exc:
        raise BasetenRuntimeError(f"{operation} returned invalid metrics") from exc
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise BasetenRuntimeError(
            f"{operation} returned non-finite metrics: {', '.join(sorted(invalid))}"
        )
    return metrics


def _finite_loss_result(
    result: Any,
    *,
    operation: str,
    fallback_active_tokens: float,
) -> tuple[float, float, dict[str, float]]:
    metrics = _finite_metrics(result, operation=operation)
    loss = metrics.get("loss")
    if loss is None:
        raise BasetenRuntimeError(f"{operation} returned no finite loss")
    active_tokens = metrics.get("active_tokens", float(fallback_active_tokens))
    if not math.isfinite(active_tokens) or active_tokens <= 0:
        raise BasetenRuntimeError(f"{operation} returned no active target tokens")
    return loss, active_tokens, metrics


def _batch_active_tokens(batch: list[Any]) -> float:
    count = 0.0
    for datum in batch:
        weights = getattr(datum, "loss_fn_inputs", {}).get("weights")
        values = getattr(weights, "data", None)
        if values is None:
            count += _datum_token_count(datum)
        else:
            count += sum(float(value) for value in values)
    return count


def _checkpoint_path(value: Any, *, operation: str) -> str:
    path = getattr(value, "path", None)
    if not isinstance(path, str) or not path:
        raise BasetenRuntimeError(f"{operation} returned no checkpoint path")
    return path


def _checkpoint_name(run_id: str, suffix: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", f"{run_id}-{suffix}".lower()).strip("-")
    if len(candidate) <= 54:
        return candidate
    digest = hashlib.sha256(candidate.encode()).hexdigest()[:8]
    return f"{candidate[:45].rstrip('-')}-{digest}"


def _combine_exceptions(message: str, errors: list[BaseException]) -> BaseException | None:
    if not errors:
        return None
    if len(errors) == 1:
        return errors[0]
    commands = sorted(
        {
            command
            for error in errors
            for command in re.findall(
                r"baseten loops run deactivate --run-id "
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127} --yes",
                str(error),
            )
        }
    )
    if commands:
        message += "; manual cleanup: " + "; ".join(commands)
    return BaseExceptionGroup(message, errors)


def _load_prepared_data(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = Path(data_dir) / "prepared"
    try:
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        train_rows = _read_jsonl(prepared / "train.jsonl")
        validation_rows = _read_jsonl(prepared / "validation.jsonl")
        test_rows = _read_jsonl(prepared / "test.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read prepared Baseten data: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PipelineError("prepared Baseten manifest is not an object")
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise PipelineError("prepared Baseten manifest has no valid split counts")
    for partition, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        count = split.get(partition)
        if isinstance(count, bool) or not isinstance(count, int) or count != len(rows):
            raise PipelineError(f"prepared {partition} row count differs from manifest")
        if partition in {"train", "validation"} and not rows:
            raise PipelineError(f"prepared dataset has no {partition} rows")
    return manifest, train_rows, validation_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise PipelineError(f"prepared data rows in {path.name} must be objects")
    return rows


def _resolved_settings(settings: Any, model: ModelSpec) -> BasetenSFTSettings:
    if not isinstance(settings, BasetenSFTSettings):
        raise PipelineError("Baseten training requires Baseten SFT settings")
    settings.validate()
    return replace(
        settings,
        lora_rank=model.default_lora_rank if settings.lora_rank is None else settings.lora_rank,
        microbatch_token_budget=(
            model.max_seq_len if settings.microbatch_token_budget is None else settings.microbatch_token_budget
        ),
    )


def _validate_manifest(manifest: dict[str, Any]) -> ModelSpec:
    model = resolve_prepared_model(manifest, MODEL_SPECS, provider="baseten")
    source = manifest.get("langsmith")
    split = manifest.get("split")
    if not isinstance(source, dict) or not isinstance(split, dict):
        raise PipelineError("prepared Baseten manifest has no valid dataset counts")
    counts = [source.get("examples"), *(split.get(name) for name in ("train", "validation", "test"))]
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise PipelineError("prepared Baseten dataset counts must be non-negative integers")
    if counts[0] < sum(counts[1:]):
        raise PipelineError("prepared Baseten split counts exceed source examples")
    return model


def _prepared_max_sequence_tokens(manifest: dict[str, Any], model: ModelSpec) -> int:
    audit = manifest.get("audit")
    maximum = audit.get("max_context_tokens") if isinstance(audit, dict) else None
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 2
        or maximum > model.max_seq_len
    ):
        raise PipelineError(
            "prepared Baseten data has no valid measured maximum context length"
        )
    return maximum

"""Baseten Loops model configuration, rendering, and guarded SFT."""

from __future__ import annotations

import math
import json
import os
import re
import signal
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from smithtune.inference_contract import InferenceContract
from smithtune.capabilities import open_without_redirects, preflight_model
from smithtune.models import resolve_model_options, resolve_prepared_model
from smithtune.rendering import load_training_renderer, render_row_tokens, resolve_rendering_model
from smithtune.providers.baseten import CAPACITY
from smithtune.providers.base import (
    CommonSFTSettings,
    ModelOptions,
    ModelSpec,
    PipelineError,
    ReasoningPolicy,
    TerminationRequested,
    TrainingOptions,
)
from smithtune.providers import lora_sft


# Loops uses this sentinel for targets excluded from assistant-only loss.
IGNORE_INDEX = -100
MODEL_SPECS = {
    "qwen3p8-27b": ModelSpec(
        name="qwen3p8-27b",
        base_model="Qwen/Qwen3.8-27B",
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        renderer="hf_assistant",
        max_seq_len=262_144,
        trainer_max_seq_len=262_144,
        thinking_trace_history_mode="preserved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="baseten",
    ),
    "kimi-k3": ModelSpec(
        name="kimi-k3",
        base_model="moonshotai/Kimi-K3",
        tokenizer_model="moonshotai/Kimi-K3",
        tokenizer_revision="301be1b88c89c0d3a763da6301352cb8fe399e90",
        renderer="hf_prefix_kimi_k3",
        max_seq_len=131_072,
        trainer_max_seq_len=131_072,
        thinking_trace_history_mode="preserved",
        trust_remote_code=True,
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="baseten",
    ),
    "qwen3p5-9b": ModelSpec(
        name="qwen3p5-9b",
        base_model="Qwen/Qwen3.5-9B",
        tokenizer_model="Qwen/Qwen3.5-9B",
        tokenizer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        renderer="hf_prefix_qwen3_5",
        max_seq_len=131_072,
        trainer_max_seq_len=131_072,
        thinking_trace_history_mode="interleaved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="baseten",
    ),
    "glm-5p3-flash": ModelSpec(
        name="glm-5p3-flash",
        base_model="zai-org/GLM-5.3-Flash",
        tokenizer_model="zai-org/GLM-5.3-Flash",
        tokenizer_revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        renderer="hf_prefix_glm53_flash",
        # Keep the documented configuration; live larger contexts may use a
        # different parallelism/loss configuration.
        max_seq_len=131_072,
        trainer_max_seq_len=131_072,
        thinking_trace_history_mode="preserved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
        provider="baseten",
    ),
}
DEFAULT_MODEL = MODEL_SPECS["qwen3p8-27b"]
# Smithtune trains without a paired sampler, which Baseten calls the SFT use
# case. Baseten enables different sequence limits for it than for the default
# RL use case.
BASETEN_CAPABILITIES_URL = "https://api.baseten.co/v1/loops/capabilities?use_case=sft"
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


ERRORS = lora_sft.ProviderErrors("Baseten", BasetenDataError, BasetenRuntimeError)


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
    sampler_checkpoints: str = "best"

    def validate(self) -> None:
        numeric = {
            "max_epochs": self.max_epochs,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "replicas": self.replicas,
            "max_dropped_training_rows": self.max_dropped_training_rows,
        }
        for name in ("early_stopping_patience", "lora_rank", "microbatch_token_budget"):
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
        if self.sampler_checkpoints not in lora_sft.SAMPLER_CHECKPOINT_POLICIES:
            raise PipelineError("sampler_checkpoints must be 'best' or 'all'")
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
class BasetenEnablementDetails:
    """Why a supported Baseten model is unavailable to this workspace."""

    reason: str | None = None
    reason_detail: str | None = None
    remediation: str | None = None


@dataclass(frozen=True)
class BasetenModelCapability:
    model_name: str
    # Deprecated by Baseten in favour of `max_seq_len`; drop once that ships.
    max_context_length: int
    supports_vision_language: bool | None = None
    # The catalog ceiling, independent of the workspace.
    max_seq_len: int | None = None
    # What this workspace may actually reach; 0 when it cannot run the model.
    max_enabled_seq_len: int | None = None
    # None on an API that does not send the enablement fields yet.
    enabled: bool | None = None
    enablement_details: BasetenEnablementDetails | None = None
    # Baseten sends this only to workspaces with serverless Loops, so None means
    # the workspace cannot use serverless capacity for any model.
    serverless: bool | None = None


class BasetenModelNotEnabled(BasetenRuntimeError):
    """The workspace is not enabled to train an otherwise supported model."""

    def __init__(self, message: str, *, model: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.model = model
        self.reason = reason


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BasetenRuntimeError(f"duplicate Baseten response key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise BasetenRuntimeError(f"non-finite Baseten response value: {value}")


def _optional_sequence_limit(entry: dict[str, Any], field: str) -> int | None:
    """Read an optional non-negative ceiling, distinguishing absent from null."""
    if field not in entry:
        return None
    value = entry[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BasetenRuntimeError(f"Baseten returned an invalid {field}")
    return value


def _enablement(entry: dict[str, Any]) -> tuple[bool | None, BasetenEnablementDetails | None]:
    """Parse the per-model enablement fields; absence means an older API."""
    enabled: bool | None = None
    if "enabled" in entry:
        enabled = entry["enabled"]
        if not isinstance(enabled, bool):
            raise BasetenRuntimeError("Baseten returned an invalid model enablement flag")
    raw_details = entry.get("enablement_details")
    if raw_details is None:
        return enabled, None
    if not isinstance(raw_details, dict):
        raise BasetenRuntimeError("Baseten returned invalid model enablement details")
    fields: dict[str, str | None] = {}
    for name in ("reason", "reason_detail", "remediation"):
        value = raw_details.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise BasetenRuntimeError("Baseten returned invalid model enablement details")
        fields[name] = value
    return enabled, BasetenEnablementDetails(**fields)


def _not_enabled(model: str, details: BasetenEnablementDetails | None) -> BasetenModelNotEnabled:
    """Repeat Baseten's own explanation instead of reporting the model missing."""
    reason = details.reason if details is not None else None
    explanation = " ".join(
        part for part in (
            details.reason_detail if details is not None else None,
            details.remediation if details is not None else None,
        ) if part
    )
    if not explanation:
        explanation = "this workspace is not enabled for the model"
        if reason:
            explanation += f" ({reason})"
    return BasetenModelNotEnabled(
        f"Baseten workspace cannot train {model}: {explanation}",
        model=model,
        reason=reason,
    )


def fetch_model_capability(
    model: str,
    max_sequence_length: int,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    opener: Any = open_without_redirects,
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
    supported = _optional_sequence_limit(matches[0], "max_seq_len")
    enabled_limit = _optional_sequence_limit(matches[0], "max_enabled_seq_len")
    enabled, details = _enablement(matches[0])
    serverless = matches[0].get("serverless")
    if serverless is not None and not isinstance(serverless, bool):
        raise BasetenRuntimeError("Baseten returned an invalid serverless flag")
    capability = BasetenModelCapability(
        model, maximum, vision, supported, enabled_limit, enabled, details, serverless,
    )
    _validate_capability(capability, model, max_sequence_length)
    return capability


def _placement_note(capability: BasetenModelCapability, lora_rank: int, max_seq_len: int) -> str:
    """Say what decides the trainer's placement, because the run response does not report it."""
    model = capability.model_name
    if capability.serverless:
        return (
            f"Baseten offers serverless Loops for {model} in this workspace. Baseten chooses "
            f"the trainer's capacity for LoRA rank {lora_rank} at {max_seq_len:,} tokens, "
            "and may use serverless capacity."
        )
    return (
        f"Baseten does not offer serverless Loops for {model} in this workspace; "
        "Baseten chooses the trainer's capacity."
    )


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


def _explain_trainer_creation_failure(failure: BaseException) -> BaseException:
    # The Loops SDK raises its HTTP errors with the response attached; reading it
    # directly avoids depending on the SDK's HTTP client library.
    response = getattr(failure, "response", None)
    if isinstance(getattr(response, "status_code", None), int):
        detail = _response_detail(response)
        status = response.status_code
        if status == 429:
            message = "Baseten has no capacity to provision this trainer (HTTP 429)"
            retry_after = response.headers.get("Retry-After")
            suffix = f"; retry after {retry_after}s" if retry_after and retry_after.isdigit() else "; retry later"
        else:
            message = f"Baseten refused to create the trainer (HTTP {status})"
            suffix = ""
        return BasetenRuntimeError(f"{message}: {detail}{suffix}" if detail else message + suffix)
    # The Loops SDK raises plain RuntimeError when a trainer reaches a terminal
    # state before becoming ready; its message names the deployment and link.
    if type(failure) is RuntimeError:
        return BasetenRuntimeError(f"Baseten trainer provisioning failed: {failure}")
    return failure


def _response_detail(response: Any) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:500] or None
    if isinstance(payload, dict):
        for key in ("detail", "message", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
    return None


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
    def __init__(self, *, api_key: str | None = None, opener: Any = open_without_redirects) -> None:
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
            failure = _cleanup_failure(run_id, "inactive polling", exc)
            if failure is exc:
                raise
            raise failure from exc
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
        if not targets:
            raise
        failure = _cleanup_failure(targets[0], "management authentication", exc)
        if failure is exc:
            raise
        raise failure from exc
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
        if any(getattr(options, name) is not None for name in (
            "lora_alpha", "pipeline_depth", "training_mode", "training_shape", "provisioning_timeout",
        )):
            raise PipelineError("--lora-alpha, --pipeline-depth, --training-mode, --training-shape, and --provisioning-timeout are Fireworks-only")
        optional = {
            name: getattr(options, name)
            for name in ("replicas", "spend_reserve_fraction", "max_dropped_training_rows", "sampler_checkpoints")
            if getattr(options, name) is not None
        }
        settings = BasetenSFTSettings(
            max_epochs=options.max_epochs,
            early_stopping_patience=options.early_stopping_patience,
            early_stopping_min_delta=options.early_stopping_min_delta,
            learning_rate=options.learning_rate,
            batch_size=options.batch_size,
            seed=options.seed,
            validation_every_steps=options.validation_every_steps,
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
        source_workspace_id: str | None = None,
        split_from: Path | None = None,
        reasoning_policy: ReasoningPolicy = "omit",
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fetch: bool = True,
        check_render: bool = True,
        sync_splits: bool = True,
    ) -> dict[str, Any]:
        """Prepare canonical rows with the Baseten model and shared split defaults."""
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
        _reject_loops_reuse_overrides()
        manifest, _train_rows, _validation_rows = _load_prepared_data(data_dir)
        model = _validate_manifest(manifest)
        settings = _resolved_settings(settings, model)
        _validate_canonical_rows(_train_rows, split="train")
        prepared_max_seq_len = _prepared_max_sequence_tokens(manifest, model)
        max_seq_len = min(prepared_max_seq_len, model.training_context_limit)
        value = {
            "run_id": run_id,
            "provider": self.name,
            "method": "Baseten Loops LoRA SFT",
            "execution": {"requires_active_client": True},
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
                "validation_every_steps": settings.validation_every_steps,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
                "seed": settings.seed,
                "replicas": settings.replicas,
                "capacity": CAPACITY,
                "with_sampler": False,
                "max_dropped_training_rows": settings.max_dropped_training_rows,
                "sampler_checkpoints": settings.sampler_checkpoints,
            },
            "training_schedule": lora_sft.estimated_schedule(_train_rows, settings),
            "evaluation": "forward-only cross-entropy on the held-out validation split",
            "checkpoint": (
                "resumable state and sampler weights at every validation"
                if settings.sampler_checkpoints == "all"
                else "resumable state at every validation and sampler weights for each new best"
            ),
            "budget": {
                "enabled": settings.max_spend_usd is not None,
                "max_spend_usd": settings.max_spend_usd,
                "hourly_rate_usd": settings.hourly_rate_usd,
                "spend_reserve_fraction": settings.spend_reserve_fraction,
                "maximum_active_seconds": settings.maximum_active_seconds(),
            },
        }

        if replay is not None:
            from smithtune.evaluation.replay import preview_replay_for_plan

            preview = preview_replay_for_plan(data_dir, replay)
            value["replay"] = {**preview, "serving_mode": "sampler", "evaluated_models": 2,
                               "capacity": CAPACITY, "timing": "after trainer shutdown",
                               "cleanup": "deactivate both samplers on exit",
                               "budget": "sampler and judge costs are separate from the training budget"}
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
        """Run the guarded Loops SFT lifecycle through injected remote seams."""
        if not confirm:
            raise PipelineError(
                "Baseten training changes paid resources; rerun with --confirm"
            )
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise PipelineError(f"run directory must be new or empty: {run_dir}")

        plan = self.plan(Path(data_dir), run_id, settings, replay=replay)
        if replay is not None:
            from smithtune.evaluation.replay import validate_judge_credentials

            validate_judge_credentials(replay["judge_model"])
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
        preflight_model(
            model, capability_resolver=lambda *_: capability,
            required_context=max_seq_len,
        )
        print(_placement_note(capability, settings.lora_rank, max_seq_len), file=sys.stderr)

        loops_types = self._loops_types or _resolve_loops_types()
        render_fn = self._render_fn or render_row
        renderer_factory = self._renderer_factory or _load_renderer
        renderer = renderer_factory(model)
        splits = lora_sft.render_splits(
            train_rows,
            validation_rows,
            render=lambda row: render_fn(row, model, loops_types=loops_types, renderer=renderer),
            max_seq_len=max_seq_len,
            microbatch_token_budget=settings.microbatch_token_budget,
            max_dropped_training_rows=settings.max_dropped_training_rows,
            errors=ERRORS,
        )
        plan["training_schedule"] = lora_sft.rendered_schedule(splits, settings)
        plan["dataset"].update(
            {
                "prepared_train_rows": len(train_rows),
                "train_rows": len(splits.train),
                "dropped_train_rows": splits.dropped_train_rows,
            }
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "run-state.json"
        state: dict[str, Any] = {}

        def update_state(**changes: Any) -> None:
            state.update(changes)
            _atomic_json(state_path, state)

        epochs_path = run_dir / "epochs.json"
        validations_path = run_dir / "validations.json"
        result_path = run_dir / "result.json"
        renderer_identity = {
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        }
        audit_identity = _audit_identity(Path(data_dir), manifest, plan)
        _atomic_json(run_dir / "plan.json", plan)
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
        _atomic_json(epochs_path, [])
        _atomic_json(validations_path, [])
        update_state(
            status="planned",
            provider="baseten",
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
        loop = lora_sft.SFTLoop(
            trainer=None,  # bound once Baseten provisions the trainer
            settings=settings,
            run_id=run_id,
            splits=splits,
            run_dir=run_dir,
            update_state=update_state,
            errors=ERRORS,
            guard=budget,
            deferred=(BudgetExceeded,),
            retry=lambda operation: retry_idempotent(operation, sleeper=self._sleeper),
            state_extras=lambda: {"active_seconds": budget.elapsed_seconds},
            save_every_sampler=settings.sampler_checkpoints == "all",
        )
        service: Any | None = None
        trainer: Any | None = None
        session_id: str | None = None
        baseten_run_id: str | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        previous_sigterm_handler: Any = None
        sigterm_handler_installed = False
        budget_stopped = False
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
            "epochs": loop.epochs,
            "validations": loop.validations,
            "step_losses": loop.step_losses,
            "last_resumable_state_uri": None,
            "best_resumable_state_uri": None,
            "best_sampler_weights_uri": None,
            "best_epoch": None,
            "best_step": None,
            "completed_step": 0,
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
            # "dedicated" availability means capacity that is never preempted, in
            # contrast to spot; it does not choose dedicated over serverless.
            service = service_factory(availability_model="dedicated")
            session_id = validate_resource_id(service.session_id, kind="session")
            result["session_id"] = session_id
            update_state(baseten_session_id=session_id)
            try:
                # Send no scope, so Baseten chooses placement. The SDK sends an explicit
                # scope only through a route that only Baseten's own workspaces can use.
                trainer = service.create_lora_training_client(
                    base_model=model.base_model,
                    rank=settings.lora_rank,
                    replicas=settings.replicas,
                    seed=settings.seed,
                    max_seq_len=max_seq_len,
                    with_sampler=False,
                    name=run_id,
                )
            except Exception as exc:
                failure = _explain_trainer_creation_failure(exc)
                if failure is exc:
                    raise
                raise failure from exc
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
            loop.trainer = _LoopsOperations(trainer, loops_types)
            loop.run()
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
                    # The SDK returns the run only once its trainer is ready, so a
                    # provisioning failure leaves the session's single run unrecorded.
                    if baseten_run_id is None and len(deactivated) == 1:
                        baseten_run_id = deactivated[0]
                        result["baseten_run_id"] = baseten_run_id
                        update_state(baseten_run_id=baseten_run_id)
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
        result.update({**loop.summary(), "status": status, "primary_error": safe_primary_error})
        result["budget"].update(
            {
                "elapsed_seconds": final_active_seconds,
                "stopped": budget_stopped,
            }
        )
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
                # Active time is measured only by the optional spend guard.
                active_seconds=(
                    final_active_seconds if settings.max_spend_usd is not None else None
                ),
                budget_status=(
                    "reached"
                    if budget_stopped
                    else ("enabled" if settings.max_spend_usd is not None else "disabled")
                ),
                cleanup_status="failed" if cleanup_errors else "completed",
                cleanup_error=result["cleanup"]["error"],
                primary_error=safe_primary_error,
            )
            _atomic_json(epochs_path, loop.epochs)
            _atomic_json(validations_path, loop.validations)
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
        best_sampler_uri = loop.best_sampler_uri
        if replay is not None:
            from smithtune.providers.baseten.sampling import BasetenReplaySampler
            from smithtune.evaluation.replay import run_replay_evaluation, training_metadata

            try:
                if best_sampler_uri is None:
                    raise PipelineError("training produced no best checkpoint for replay")
                sampler = BasetenReplaySampler(model, best_sampler_uri, run_dir / "replay")
                result["replay"] = run_replay_evaluation(
                    data_dir, run_dir / "replay", best_sampler_uri,
                    base_model=model.base_model, replay_sampler=sampler, confirm=True,
                    training=training_metadata(run_dir), **replay,
                )
            except BaseException:
                result["replay_status"] = "incomplete"
                _atomic_json(result_path, result)
                update_state(replay_status="incomplete")
                raise
            result["replay_status"] = "completed"
            _atomic_json(result_path, result)
            update_state(replay_status="completed")
        return result


def render_row(
    row: dict[str, Any],
    model: ModelSpec,
    *,
    loops_types: Any | None = None,
    renderer: Any | None = None,
) -> list[Any]:
    """Render one canonical row and convert every result to shifted Loops data."""
    active_renderer = renderer if renderer is not None else _load_renderer(model)
    rendered_items = render_row_tokens(
        row, model, renderer=active_renderer, include_loss_mask=True, reduction="none",
    )
    if not rendered_items:
        raise BasetenDataError("canonical row rendered no training datum")
    constructors = loops_types if loops_types is not None else _resolve_loops_types()
    return [_to_loops_datum(item, constructors, model.max_seq_len) for item in rendered_items]


def pack_microbatches(datums: Iterable[Any], token_budget: int) -> list[list[Any]]:
    """Greedily pack in order and reject trajectories larger than the budget."""
    return lora_sft.pack_microbatches(datums, token_budget, error=BasetenDataError)


def plan_accumulation_groups(
    microbatches: Iterable[Iterable[Any]],
    *,
    effective_batch_size: int,
) -> list[list[list[Any]]]:
    """Group ordered microbatches for example-counted gradient accumulation."""
    return lora_sft.plan_accumulation_groups(
        microbatches, effective_batch_size=effective_batch_size, error=BasetenDataError,
    )


def _to_loops_datum(rendered: Any, loops_types: Any, max_sequence_tokens: int) -> Any:
    return lora_sft.to_datum(
        rendered, loops_types, max_sequence_tokens, ignore_index=IGNORE_INDEX, error=BasetenDataError,
    )


class _LoopsOperations:
    """Interpret Loops results; its losses are already token-weighted means."""

    def __init__(self, trainer: Any, loops_types: Any) -> None:
        self.trainer = trainer
        self.loops_types = loops_types

    def forward_backward(self, microbatch: list[Any], *, group_active_tokens: float):
        return _finite_loss_result(
            self.trainer.forward_backward(microbatch).result(),
            operation="forward_backward",
            fallback_active_tokens=_batch_active_tokens(microbatch),
        )

    def forward(self, microbatch: list[Any]):
        return _finite_loss_result(
            self.trainer.forward(microbatch, "cross_entropy").result(),
            operation="validation forward",
            fallback_active_tokens=_batch_active_tokens(microbatch),
        )

    def optim_step(self, learning_rate: float) -> dict[str, float]:
        return _finite_metrics(
            self.trainer.optim_step(self.loops_types.AdamParams(learning_rate=learning_rate)).result(),
            operation="optimizer step",
        )

    def save_state(self, name: str) -> str:
        return _checkpoint_path(self.trainer.save_state(name).result(), operation="save_state")

    def save_weights_for_sampler(self, name: str) -> str:
        return _checkpoint_path(
            self.trainer.save_weights_for_sampler(name).result(), operation="save_weights_for_sampler",
        )


def _load_renderer(model: Any) -> Any:
    return load_training_renderer(model)


def _resolve_loops_types() -> Any:
    try:
        import baseten.loops
    except ImportError as exc:
        raise BasetenDataError(
            "Baseten Loops is unavailable; reinstall using the installation command in the README, then run smithtune doctor"
        ) from exc
    return baseten.loops


def _datum_token_count(datum: Any) -> int:
    return lora_sft.datum_token_count(datum, error=BasetenDataError)


_atomic_json = lora_sft.atomic_json
_audit_identity = lora_sft.audit_identity
_validate_canonical_rows = lora_sft.validate_canonical_rows
_batch_active_tokens = lora_sft.batch_active_tokens
_checkpoint_name = lora_sft.checkpoint_name


def _safe_primary_error(failure: BaseException) -> dict[str, str]:
    return {
        "type": type(failure).__name__,
        "message": (
            "Baseten lifecycle failed; exception details omitted for credential safety"
        ),
    }


def _validate_capability(capability: Any, expected_model: str, required_context: int) -> None:
    model_name = getattr(capability, "model_name", None)
    maximum = getattr(capability, "max_context_length", None)
    enabled = getattr(capability, "enabled", None)
    enabled_limit = getattr(capability, "max_enabled_seq_len", None)
    details = getattr(capability, "enablement_details", None)
    if model_name != expected_model:
        raise BasetenRuntimeError(f"Baseten workspace does not advertise {expected_model}")
    if enabled is not None and not isinstance(enabled, bool):
        raise BasetenRuntimeError("Baseten returned an invalid model enablement flag")
    if details is not None and not isinstance(details, BasetenEnablementDetails):
        raise BasetenRuntimeError("Baseten returned invalid model enablement details")
    # A missing flag means an API that predates per-model enablement, not a
    # disabled model; only an explicit false stops the run.
    if enabled is False:
        raise _not_enabled(expected_model, details)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < required_context:
        raise BasetenRuntimeError(
            f"Baseten workspace context limit is below {required_context:,}"
        )
    if enabled_limit is not None:
        if isinstance(enabled_limit, bool) or not isinstance(enabled_limit, int) or enabled_limit < 0:
            raise BasetenRuntimeError("Baseten returned an invalid max_enabled_seq_len")
        # The catalog ceiling above says what Baseten supports; this one says
        # what the workspace is approved to reach, and it is the binding limit.
        if enabled_limit < required_context:
            raise BasetenRuntimeError(
                f"Baseten workspace is enabled for {enabled_limit:,} tokens of "
                f"{expected_model}, below the required {required_context:,}; "
                f"pass --max-seq-len {enabled_limit} to prepare within the enabled limit"
            )


def _finite_metrics(result: Any, *, operation: str) -> dict[str, float]:
    return lora_sft.finite_metrics(result, operation=operation, error=BasetenRuntimeError)


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


def _checkpoint_path(value: Any, *, operation: str) -> str:
    return lora_sft.checkpoint_path(value, operation=operation, error=BasetenRuntimeError)


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
    return lora_sft.load_prepared_data(data_dir, label="Baseten")


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
    lora_sft.validate_dataset_counts(manifest, label="Baseten")
    return model


def _prepared_max_sequence_tokens(manifest: dict[str, Any], model: ModelSpec) -> int:
    return lora_sft.prepared_max_sequence_tokens(manifest, model, label="Baseten")

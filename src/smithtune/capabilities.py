"""Read-only training checks, independent of the local rendering registry."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from smithtune.providers.base import ModelSpec, PipelineError


FIREWORKS_MODEL_API = "https://api.fireworks.ai/v1/"
FIREWORKS_TRAINING_CATALOG = "https://docs.fireworks.ai/fine-tuning/models"
# The training catalog generated 2026-09-12 lists these shared pools. There is
# no documented read-only pool API; inference supportsServerless is unrelated.
FIREWORKS_SERVERLESS_CONTEXT_LIMITS = {
    "accounts/fireworks/models/qwen3p8-27b": 131_072,
    "accounts/fireworks/models/kimi-k3": 196_608,
}
# Loops does not expose supported losses in its capabilities response. Keep
# known cross-entropy compatibility separate from its live model catalog.
BASETEN_CROSS_ENTROPY_MODELS = frozenset({"Qwen/Qwen3.8-27B"})
MAX_METADATA_RESPONSE_BYTES = 1024 * 1024
_MODEL_RESOURCE = re.compile(r"accounts/[a-z0-9][a-z0-9-]{0,62}/models/[a-z0-9][a-z0-9-]{0,62}")
_HF_MODEL_PATH = re.compile(r"/([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)/?")


@dataclass(frozen=True)
class FireworksModelCapability:
    model_name: str
    tokenizer_model: str
    training_context_length: int
    supervised_lora_tunable: bool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_without_redirects(request: urllib.request.Request, *, timeout: float) -> Any:
    """Do not forward provider credentials to a redirected destination."""
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON value")


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _hugging_face_model(value: Any) -> str:
    if not isinstance(value, str) or any(character.isspace() for character in value):
        raise PipelineError("Fireworks returned an invalid Hugging Face model URL")
    try:
        url = urllib.parse.urlsplit(value)
        valid_origin = (
            url.scheme == "https" and url.netloc == "huggingface.co"
            and not url.query and not url.fragment
        )
        match = _HF_MODEL_PATH.fullmatch(url.path)
    except ValueError:
        raise PipelineError("Fireworks returned an invalid Hugging Face model URL") from None
    if not valid_origin or match is None:
        raise PipelineError("Fireworks returned an invalid Hugging Face model URL")
    return match.group(1)


def fetch_fireworks_model_capability(
    model: str,
    required_context: int,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    opener: Any = open_without_redirects,
) -> FireworksModelCapability:
    """Read model metadata without attaching to a paid training pool."""
    if not isinstance(model, str) or _MODEL_RESOURCE.fullmatch(model) is None:
        raise PipelineError("Fireworks model resource is invalid")
    _require_serverless_context(model, required_context)
    key = api_key if api_key is not None else os.environ.get("FIREWORKS_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise PipelineError("FIREWORKS_API_KEY is required for model preflight")
    # Each resource segment is validated and encoded; no caller-supplied origin.
    path = "/".join(urllib.parse.quote(part, safe="") for part in model.split("/"))
    request = urllib.request.Request(
        FIREWORKS_MODEL_API + path,
        headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            payload = response.read(MAX_METADATA_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"Fireworks model preflight returned HTTP {exc.code}") from None
    except (OSError, urllib.error.URLError, ValueError):
        raise PipelineError("Fireworks model preflight request failed") from None
    if len(payload) > MAX_METADATA_RESPONSE_BYTES:
        raise PipelineError("Fireworks model metadata exceeded the size limit")
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise PipelineError("Fireworks returned malformed model metadata") from None
    if not isinstance(document, dict) or document.get("name") != model:
        raise PipelineError("Fireworks returned metadata for an unexpected model")
    tunable = document.get("supervisedLoraTunable")
    if not isinstance(tunable, bool):
        raise PipelineError("Fireworks returned invalid LoRA training metadata")
    if not tunable:
        raise PipelineError("Fireworks does not advertise supervised LoRA training for this model")
    context = document.get("trainingContextLength")
    if not _positive_integer(context):
        raise PipelineError("Fireworks returned an invalid training context limit")
    tokenizer_model = _hugging_face_model(document.get("huggingFaceUrl"))
    return FireworksModelCapability(model, tokenizer_model, context, tunable)


def _require_serverless_context(model: str, required_context: int) -> None:
    if not _positive_integer(required_context):
        raise PipelineError("required training context must be a positive integer")
    limit = FIREWORKS_SERVERLESS_CONTEXT_LIMITS.get(model)
    if limit is None:
        raise PipelineError(
            "Fireworks serverless training availability is unverified for this model; "
            f"smithtune requires a documented shared-pool compatibility entry from {FIREWORKS_TRAINING_CATALOG}"
        )
    if required_context > limit:
        raise PipelineError(f"Fireworks documented serverless training context limit is {limit:,}")


def preflight_model(
    model: ModelSpec,
    *,
    capability_resolver: Any | None = None,
    required_context: int | None = None,
) -> ModelSpec:
    """Validate current support without changing a selected or prepared model."""
    model.validate()
    context = model.training_context_limit if required_context is None else required_context
    if not _positive_integer(context) or context > model.training_context_limit:
        raise PipelineError("required training context is outside the selected model limit")
    if model.provider == "baseten":
        from smithtune.providers.baseten import _validate_capability, fetch_model_capability

        if model.base_model not in BASETEN_CROSS_ENTROPY_MODELS:
            raise PipelineError("Baseten cross-entropy training compatibility is unverified for this model")
        if model.tokenizer_model != model.base_model:
            raise PipelineError("Baseten tokenizer must match the official base model identity")
        resolver = capability_resolver or fetch_model_capability
        capability = resolver(model.base_model, context)
        _validate_capability(capability, model.base_model, context)
        vision = getattr(capability, "supports_vision_language", None)
        if vision is not None and not isinstance(vision, bool):
            raise PipelineError("Baseten returned an invalid vision support flag")
    elif model.provider == "fireworks":
        _require_serverless_context(model.base_model, context)
        resolver = capability_resolver or fetch_fireworks_model_capability
        capability = resolver(model.base_model, context)
        if not isinstance(capability, FireworksModelCapability):
            raise PipelineError("Fireworks returned invalid model capability metadata")
        if capability.model_name != model.base_model:
            raise PipelineError("Fireworks returned metadata for an unexpected model")
        if capability.supervised_lora_tunable is not True:
            raise PipelineError("Fireworks does not advertise supervised LoRA training for this model")
        if not _positive_integer(capability.training_context_length):
            raise PipelineError("Fireworks returned an invalid training context limit")
        if capability.tokenizer_model != model.tokenizer_model:
            raise PipelineError("Fireworks model metadata does not match the selected tokenizer")
        # Generic trainingContextLength can differ from the shared pool's
        # documented limit (notably Kimi K3); the pool limit governs this API.
    else:
        raise PipelineError("unsupported training provider for model preflight")
    return model

"""Inference transports for provider replay and Anthropic judging."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from smithtune.inference_contract import ContractError, InferenceContract
from smithtune.capabilities import open_without_redirects
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, INFERENCE_URL


ANTHROPIC_ENDPOINTS = {
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "anthropic-gateway": ("https://gateway.smith.langchain.com/anthropic", "LANGSMITH_GATEWAY_API_KEY"),
}
# Baseten Model APIs (shared hosted models), used for judges; not dedicated deployments.
BASETEN_MODEL_API_URL = "https://inference.baseten.co/v1"


def judge_endpoint(judge_model: str) -> str | None:
    """Return the fixed endpoint for a judge route; None means Fireworks inference."""
    provider = judge_model.partition("/")[0]
    if provider in ANTHROPIC_ENDPOINTS:
        return ANTHROPIC_ENDPOINTS[provider][0]
    return BASETEN_MODEL_API_URL if provider == "baseten" else None


REQUEST_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class BasetenEndpoint:
    """An existing dedicated deployment and its configured serving context."""

    model_id: str
    deployment_id: str
    max_seq_len: int

    def validate(self) -> None:
        for value, label in ((self.model_id, "model ID"), (self.deployment_id, "deployment ID")):
            if not isinstance(value, str) or re.fullmatch(r"[a-z0-9]{1,64}", value) is None:
                raise PipelineError(f"Baseten {label} must contain lowercase letters and digits")
        if isinstance(self.max_seq_len, bool) or not isinstance(self.max_seq_len, int) or self.max_seq_len < 1:
            raise PipelineError("Baseten serving context (--max-seq-len) must be a positive integer")

    @property
    def url(self) -> str:
        self.validate()
        return f"https://model-{self.model_id}.api.baseten.co/deployment/{self.deployment_id}/sync/v1/chat/completions"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "url": self.url}


def _post_json(request: urllib.request.Request, label: str, *, opener=None) -> dict[str, Any]:
    """Send one request and return its JSON object without echoing bodies or headers."""
    try:
        with (opener or urllib.request.urlopen)(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"{label} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise PipelineError(f"{label} failed: {exc.reason}") from exc
    except OSError as exc:
        raise PipelineError(f"{label} failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"{label} returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise PipelineError(f"{label} returned an invalid response")
    return result


def _inference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
    return [{key: value for key, value in message.items() if key in allowed} for message in messages]


def _chat_request_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    request_contract: InferenceContract | None = None,
) -> dict[str, Any]:
    inference_messages = _inference_messages(messages)
    if request_contract is None:
        body: dict[str, Any] = {
            "model": model,
            "messages": inference_messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
    else:
        try:
            body = request_contract.build_chat_request(
                model=model,
                messages=inference_messages,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except ContractError as exc:
            raise PipelineError(f"cannot build inference request: {exc}") from exc
    body.setdefault("temperature", 0)
    return body


def _completion_message(result: dict[str, Any], model: str) -> dict[str, Any]:
    choices = result.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise PipelineError(f"inference returned no message for {model}")
    return message


def _fireworks_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    request_contract: InferenceContract | None = None,
) -> dict[str, Any]:
    body = _chat_request_body(model, messages, max_tokens, json_mode, request_contract)
    request = urllib.request.Request(
        INFERENCE_URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
            "Content-Type": "application/json",
            "X-Fireworks-Client-Source": CLIENT_SOURCE,
            "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"],
        },
    )
    return _completion_message(
        _post_json(request, f"inference for {model}", opener=open_without_redirects), model,
    )


def _baseten_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    request_contract: InferenceContract | None = None,
    *,
    endpoint: BasetenEndpoint,
) -> dict[str, Any]:
    endpoint.validate()
    key = os.environ.get("BASETEN_API_KEY")
    if not key or not key.strip():
        raise PipelineError("BASETEN_API_KEY is not set")
    body = _chat_request_body(model, messages, max_tokens, json_mode, request_contract)
    request = urllib.request.Request(
        endpoint.url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    result = _post_json(request, f"Baseten inference for {model}", opener=open_without_redirects)
    return _completion_message(result, model)


def _baseten_model_api_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
) -> dict[str, Any]:
    key = os.environ.get("BASETEN_API_KEY")
    if not key or not key.strip():
        raise PipelineError("BASETEN_API_KEY is not set for the judge")
    body = _chat_request_body(model, messages, max_tokens, json_mode, None)
    request = urllib.request.Request(
        BASETEN_MODEL_API_URL + "/chat/completions", data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    result = _post_json(request, f"Baseten inference for {model}", opener=open_without_redirects)
    return _completion_message(result, model)


def anthropic_connection(provider: str = "anthropic") -> tuple[str, str]:
    """Resolve an explicit endpoint and its own credential; never fall back across providers."""
    if provider not in ANTHROPIC_ENDPOINTS:
        raise PipelineError(f"unsupported Anthropic provider: {provider}")
    base_url, credential = ANTHROPIC_ENDPOINTS[provider]
    key = os.environ.get(credential)
    if not key or not key.strip():
        raise PipelineError(f"{credential} is not set for {provider}")
    return base_url, key


def _anthropic_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    *,
    provider: str = "anthropic",
) -> dict[str, Any]:
    base_url, api_key = anthropic_connection(provider)
    system = [message["content"] for message in messages if message.get("role") == "system"]
    conversation = [message for message in messages if message.get("role") != "system"]
    if not all(message.get("role") in {"user", "assistant"} for message in conversation):
        raise PipelineError("Anthropic judge messages must use system, user, or assistant roles")
    body = {
        "model": model,
        "system": "\n\n".join(system),
        "messages": _inference_messages(conversation),
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base_url + "/v1/messages",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    result = _post_json(request, f"{provider} inference for {model}")
    content = result.get("content")
    if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
        raise PipelineError(f"{provider} inference for {model} returned invalid content")
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    return {"role": "assistant", "content": text}


def _chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    request_contract: InferenceContract | None = None,
) -> dict[str, Any]:
    provider, _, model_id = model.partition("/")
    if provider in ANTHROPIC_ENDPOINTS:
        if request_contract is not None:
            raise PipelineError("agent inference contracts are supported only for Fireworks replay models")
        return _anthropic_chat_completion(
            model_id, messages, max_tokens, json_mode, provider=provider
        )
    if provider == "baseten":
        if request_contract is not None:
            raise PipelineError("agent inference contracts are supported only for Fireworks replay models")
        return _baseten_model_api_completion(model_id, messages, max_tokens, json_mode)
    return _fireworks_chat_completion(
        model,
        messages,
        max_tokens,
        json_mode,
        request_contract,
    )

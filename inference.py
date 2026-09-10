"""Inference transports for Fireworks replay and Anthropic judging."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from inference_contract import ContractError, InferenceContract
from providers.base import PipelineError
from providers.fireworks import CLIENT_SOURCE, INFERENCE_URL


ANTHROPIC_INFERENCE_URL = "https://gateway.smith.langchain.com/anthropic/v1/messages"


ANTHROPIC_MODEL_PREFIX = "anthropic/"


def _inference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"}
    return [{key: value for key, value in message.items() if key in allowed} for message in messages]


def _fireworks_chat_completion(
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
            body = request_contract.build_fireworks_request(
                model=model,
                messages=inference_messages,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except ContractError as exc:
            raise PipelineError(f"cannot build Fireworks inference request: {exc}") from exc
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
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"inference failed for {model} with HTTP {exc.code}") from exc
    choices = result.get("choices", [])
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        raise PipelineError(f"inference returned no message for {model}")
    return message


def _anthropic_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
) -> dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    custom_headers = os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
    if custom_headers:
        try:
            parsed_headers = json.loads(custom_headers)
        except json.JSONDecodeError:
            parsed_headers = dict(
                line.split(":", 1) for line in custom_headers.splitlines() if ":" in line
            )
        if isinstance(parsed_headers, dict):
            for name, value in parsed_headers.items():
                if name.strip().lower() == "x-api-key" and isinstance(value, str):
                    api_key = value.strip()
                    break
    if not api_key:
        raise PipelineError("ANTHROPIC_CUSTOM_HEADERS or ANTHROPIC_API_KEY is not set")
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
        ANTHROPIC_INFERENCE_URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"LangSmith gateway inference failed for {model} with HTTP {exc.code}") from exc
    content = result.get("content", [])
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    return {"role": "assistant", "content": text}


def _chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    json_mode: bool = False,
    request_contract: InferenceContract | None = None,
) -> dict[str, Any]:
    if model.startswith(ANTHROPIC_MODEL_PREFIX):
        if request_contract is not None:
            raise PipelineError("agent inference contracts are supported only for Fireworks replay models")
        return _anthropic_chat_completion(
            model.removeprefix(ANTHROPIC_MODEL_PREFIX), messages, max_tokens, json_mode
        )
    return _fireworks_chat_completion(
        model,
        messages,
        max_tokens,
        json_mode,
        request_contract,
    )

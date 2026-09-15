"""One model request per full trajectory and council member."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from importlib.resources import files

from jsonschema import Draft202012Validator

from smithtune.inference import _anthropic_chat_completion, _post_json
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, INFERENCE_URL, _set_skill_session


PROVIDERS = ("fireworks", "openai", "anthropic", "anthropic-gateway")
# GLM-5.3 rejects requests that disable reasoning.
FIREWORKS_REASONING = {"accounts/fireworks/models/glm-5p3-flash": "low"}
RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["keep", "reason"],
    "properties": {"keep": {"type": "integer", "enum": [0, 1]},
                   "reason": {"type": "string", "minLength": 1}},
}


def rubric_text() -> str:
    return files("smithtune").joinpath("skills/sft-trace-triage/judge.md").read_text(encoding="utf-8")


def validate_judgment(value: dict) -> dict:
    if list(Draft202012Validator(RESULT_SCHEMA).iter_errors(value)) or type(value.get("keep")) is not int:
        raise PipelineError("judge must return keep (0 or 1) and a reason")
    return value


def context_window_exceeded(exc: Exception) -> bool:
    """Use the provider's context rejection, never a guessed character limit."""
    error = exc.__cause__ if isinstance(exc.__cause__, urllib.error.HTTPError) else exc
    status = error.code if isinstance(error, urllib.error.HTTPError) else getattr(error, "status_code", None)
    if status not in {400, 413}:
        return False
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = json.loads(error.read())
        except (ValueError, OSError):
            return False
    else:
        body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return False
    if detail.get("code") in {"context_length_exceeded", "max_context_length_exceeded"}:
        return True
    message = str(detail.get("message", "")).lower()
    return any(phrase in message for phrase in (
        "maximum context length", "context window exceeded", "exceeds the context window",
        "prompt is too long", "input is too long", "exceeds the model's context",
    ))


def credential_name(provider: str) -> str:
    return {"fireworks": "FIREWORKS_API_KEY", "openai": "OPENAI_API_KEY",
            "anthropic": "SMITHTUNE_ANTHROPIC_API_KEY", "anthropic-gateway": "ANTHROPIC_API_KEY"}[provider]


def check_credentials(judges: list[dict]) -> None:
    for judge in judges:
        name = credential_name(judge["provider"])
        if not os.environ.get(name) and not (judge["provider"] == "anthropic-gateway" and os.environ.get("ANTHROPIC_CUSTOM_HEADERS")):
            raise PipelineError(f"{name} is not set for judge {judge['name']}")


def judge_messages(trajectory: dict, rubric: str, rules: list[str]) -> list[dict]:
    return [{"role": "system", "content": rubric + "\nRequired JSON schema:\n" + json.dumps(RESULT_SCHEMA)
             + "\nAdditional selection rules:\n" + json.dumps(rules)},
            {"role": "user", "content": json.dumps({"untrusted_trajectory": trajectory["messages"]}, ensure_ascii=False)}]


def api_judge(judge: dict, messages: list[dict], max_tokens: int) -> dict:
    provider, model = judge["provider"], judge["model"]
    if provider == "anthropic-gateway":
        response = _anthropic_chat_completion(model, messages, max_tokens, True)
        text = response["content"]
    else:
        headers = {"Content-Type": "application/json"}
        if provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers.update({"x-api-key": os.environ[credential_name(provider)], "anthropic-version": "2023-06-01"})
            body = {"model": model, "system": messages[0]["content"], "messages": messages[1:], "max_tokens": max_tokens}
        else:
            url = INFERENCE_URL if provider == "fireworks" else "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {os.environ[credential_name(provider)]}"
            body = {"model": model, "messages": messages, "response_format": {"type": "json_object"},
                    "max_tokens" if provider == "fireworks" else "max_completion_tokens": max_tokens}
            if provider == "fireworks" or model == "gpt-5.6-terra":
                body["reasoning_effort"] = FIREWORKS_REASONING.get(model, "none") if provider == "fireworks" else "none"
            if provider == "fireworks":
                _set_skill_session()
                headers.update({"X-Fireworks-Client-Source": CLIENT_SOURCE, "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]})
                body["temperature"] = 0
        response = _post_json(urllib.request.Request(url, method="POST", headers=headers, data=json.dumps(body).encode()), "triage judge")
        try:
            if provider == "anthropic":
                text = "".join(block["text"] for block in response["content"] if block.get("type") == "text")
            else:
                text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise PipelineError("judge returned no text response") from None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        raise PipelineError("judge returned invalid JSON") from None


def deepagent_judge(judge: dict, messages: list[dict], max_tokens: int, *, diagnostics=None) -> dict:
    """One model request per subagent, with full messages and no tool loop."""
    from smithtune.triage_agent import _model
    model = _model(judge, max_tokens)
    if judge["provider"] in {"fireworks", "openai"}:
        model = model.bind(response_format={"type": "json_object"})
    result = model.invoke(messages)
    if diagnostics is not None:
        diagnostics["finish_reason"] = result.response_metadata.get("finish_reason")
    content = result.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        raise PipelineError("judge returned invalid JSON") from None

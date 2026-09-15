"""One fresh model call (or constrained Deep Agent) per trace and judge slot."""

from __future__ import annotations

import json
import os
import urllib.request
from importlib.resources import files

from jsonschema import Draft202012Validator

from smithtune.inference import _anthropic_chat_completion, _post_json
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, INFERENCE_URL, _set_skill_session


PROVIDERS = ("fireworks", "openai", "anthropic", "anthropic-gateway")
RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["trace_id", "keep", "reason", "evidence"],
    "properties": {
        "trace_id": {"type": "string"}, "keep": {"type": "integer", "enum": [0, 1]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
        "evidence": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False, "required": ["quote"],
            "properties": {"message_index": {"type": "integer", "minimum": 0},
                           "run_id": {"type": "string"}, "quote": {"type": "string", "minLength": 1, "maxLength": 500}},
            "oneOf": [{"required": ["message_index"]}, {"required": ["run_id"]}],
        }},
    },
}
INCOMPLETE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["trace_id", "status", "reason"],
    "properties": {"trace_id": {"type": "string"}, "status": {"const": "incomplete"},
                   "reason": {"type": "string", "minLength": 1, "maxLength": 1000}},
}


class IncompleteJudgment(PipelineError):
    """A judge explicitly reports missing evidence instead of a quality vote."""


def rubric_text() -> str:
    return files("smithtune").joinpath("skills/sft-trace-triage/judge.md").read_text(encoding="utf-8")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def validate_judgment(value: dict, trace: dict) -> dict:
    if not list(Draft202012Validator(INCOMPLETE_SCHEMA).iter_errors(value)) and value["trace_id"] == trace["trace_id"]:
        raise IncompleteJudgment(value["reason"])
    if list(Draft202012Validator(RESULT_SCHEMA).iter_errors(value)):
        raise PipelineError("judge response does not match the result schema")
    if type(value["keep"]) is not int or value["trace_id"] != trace["trace_id"]:
        raise PipelineError("judge response has an invalid decision or trace identity")
    for evidence in value["evidence"]:
        if "message_index" in evidence:
            index = evidence["message_index"]
            if type(index) is not int or index >= len(trace["messages"]):
                raise PipelineError("judge evidence references an unknown message")
            source = trace["messages"][index]
        else:
            source = next((run for run in trace["runs"] if run["id"] == evidence["run_id"]), None)
            if source is None:
                raise PipelineError("judge evidence references an unknown run")
        if not any(evidence["quote"] in text for text in [json.dumps(source, ensure_ascii=False), *_strings(source)]):
            raise PipelineError("judge evidence quote does not occur in the source")
    return value


def credential_name(provider: str) -> str:
    return {"fireworks": "FIREWORKS_API_KEY", "openai": "OPENAI_API_KEY",
            "anthropic": "SMITHTUNE_ANTHROPIC_API_KEY", "anthropic-gateway": "ANTHROPIC_API_KEY"}[provider]


def check_credentials(judges: list[dict]) -> None:
    for judge in judges:
        name = credential_name(judge["provider"])
        if not os.environ.get(name) and not (judge["provider"] == "anthropic-gateway" and os.environ.get("ANTHROPIC_CUSTOM_HEADERS")):
            raise PipelineError(f"{name} is not set for judge {judge['name']}")


def judge_messages(trace: dict, rubric: str, rules: list[str]) -> list[dict]:
    evidence = {**trace, "messages": [{**message, "message_index": index}
                                     for index, message in enumerate(trace["messages"])]}
    return [{"role": "system", "content": rubric + "\nRequired JSON schema:\n" + json.dumps({"oneOf": [RESULT_SCHEMA, INCOMPLETE_SCHEMA]})
             + "\nAdditional reviewed selection rules:\n" + json.dumps(rules)},
            {"role": "user", "content": json.dumps({"untrusted_trace_evidence": evidence}, ensure_ascii=False)}]


def indexed_messages(messages: list[dict]) -> list[dict]:
    """Keep conversation text intact; expose repeated run payloads on demand."""
    trace = json.loads(messages[-1]["content"])["untrusted_trace_evidence"]
    index = {**trace, "runs": [
        {key: run[key] for key in ("id", "parent_run_id", "run_type", "name", "start_time", "end_time", "error") if key in run}
        for run in trace["runs"]
    ]}
    return [*messages[:-1], {"role": "user", "content": json.dumps({"untrusted_trace_evidence": index}, ensure_ascii=False)}]


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
    from smithtune.triage_agent import make_agent
    trace = json.loads(messages[-1]["content"])["untrusted_trace_evidence"]
    messages = indexed_messages(messages)
    agent, skill_files = make_agent(judge, messages[0]["content"], max_tokens, trace=trace, diagnostics=diagnostics)
    result = agent.invoke({"messages": messages[1:], "files": skill_files}, config={"recursion_limit": 24})
    try:
        content = result["messages"][-1].content
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        return json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError):
        raise PipelineError("Deep Agent returned invalid judge JSON") from None

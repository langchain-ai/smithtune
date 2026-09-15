"""Optional Deep Agents runner with a virtual, read-only skill filesystem."""

from __future__ import annotations

import os
from importlib.resources import files

from smithtune.providers.base import PipelineError
from smithtune.triage_judges import FIREWORKS_REASONING, credential_name


def check_installation() -> None:
    try:
        import deepagents  # noqa: F401
        import langchain_openai  # noqa: F401
        import pydantic_monty  # noqa: F401
    except ImportError:
        raise PipelineError("Deep Agents support is optional; install smithtune with the [deepagents] extra") from None


def _model(judge: dict, max_tokens: int):
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    from smithtune.inference import anthropic_connection
    from smithtune.providers.fireworks import CLIENT_SOURCE, _set_skill_session

    provider = judge["provider"]
    if provider in {"anthropic", "anthropic-gateway"}:
        base_url, key = anthropic_connection(provider)
        return ChatAnthropic(model=judge["model"], api_key=key, max_tokens=max_tokens, timeout=60, max_retries=0,
                             base_url=base_url)
    headers = {}
    if provider == "fireworks":
        _set_skill_session()
        headers = {"X-Fireworks-Client-Source": CLIENT_SOURCE, "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]}
    model_class = ChatOpenAI
    options = {"use_responses_api": False}
    if provider == "fireworks":
        options["reasoning_effort"] = FIREWORKS_REASONING.get(judge["model"], "none")
        class FireworksChat(ChatOpenAI):
            # ChatOpenAI intentionally drops provider-specific fields. Retain
            # Fireworks reasoning across tool calls without exposing it as text.
            def _create_chat_result(self, response, generation_info=None):
                result = super()._create_chat_result(response, generation_info)
                body = response if isinstance(response, dict) else response.model_dump()
                for generation, choice in zip(result.generations, body["choices"], strict=True):
                    reasoning = choice["message"].get("reasoning_content")
                    if isinstance(reasoning, str):
                        generation.message.additional_kwargs["reasoning_content"] = reasoning
                return result

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                for message, encoded in zip(self._convert_input(input_).to_messages(), payload["messages"], strict=True):
                    reasoning = message.additional_kwargs.get("reasoning_content")
                    if encoded["role"] == "assistant" and isinstance(reasoning, str):
                        encoded["reasoning_content"] = reasoning
                return payload

        model_class = FireworksChat
    elif judge["model"] == "gpt-5.6-terra":
        # Use Responses with reasoning disabled and no server-side storage.
        options = {"use_responses_api": True, "store": False,
                   "reasoning": {"effort": "none"}}
    return model_class(model=judge["model"], api_key=os.environ[credential_name(provider)],
                       base_url="https://api.fireworks.ai/inference/v1" if provider == "fireworks" else "https://api.openai.com/v1",
                       default_headers=headers, max_tokens=max_tokens, timeout=60, max_retries=0,
                       disable_streaming=True, **options)


def skill_files() -> dict:
    from deepagents.backends.utils import create_file_data

    skill = files("smithtune").joinpath("skills/sft-trace-triage")
    return {f"/skills/sft-trace-triage/{item.name}": create_file_data(item.read_text(encoding="utf-8"))
            for item in skill.iterdir() if item.is_file()}


def allowed_tools(names: set[str]):
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    class TriageTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            allowed = [tool for tool in request.tools if getattr(tool, "name", None) in names]
            return handler(request.override(tools=allowed))

        def wrap_tool_call(self, request, handler):
            if request.tool_call["name"] not in names:
                return ToolMessage(content="Only the supplied triage tools are allowed.", tool_call_id=request.tool_call["id"])
            return handler(request)

    return TriageTools()

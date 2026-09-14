"""Optional Deep Agents runner with a virtual, read-only skill filesystem."""

from __future__ import annotations

import os
from importlib.resources import files

from smithtune.providers.base import PipelineError
from smithtune.triage_judges import credential_name


def check_installation() -> None:
    try:
        import deepagents  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        raise PipelineError("Deep Agents support is optional; install smithtune with the [deepagents] extra") from None


def _model(judge: dict, max_tokens: int):
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    from smithtune.inference import _anthropic_gateway_key
    from smithtune.providers.fireworks import CLIENT_SOURCE, _set_skill_session

    provider = judge["provider"]
    if provider in {"anthropic", "anthropic-gateway"}:
        key = _anthropic_gateway_key() if provider == "anthropic-gateway" else os.environ[credential_name(provider)]
        return ChatAnthropic(model=judge["model"], api_key=key, max_tokens=max_tokens, timeout=60, max_retries=0,
                             base_url="https://gateway.smith.langchain.com/anthropic" if provider == "anthropic-gateway" else "https://api.anthropic.com")
    headers = {}
    if provider == "fireworks":
        _set_skill_session()
        headers = {"X-Fireworks-Client-Source": CLIENT_SOURCE, "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]}
    return ChatOpenAI(model=judge["model"], api_key=os.environ[credential_name(provider)],
                      base_url="https://api.fireworks.ai/inference/v1" if provider == "fireworks" else "https://api.openai.com/v1",
                      default_headers=headers, max_tokens=max_tokens, timeout=60, max_retries=0, use_responses_api=False)


def make_agent(judge: dict, system_prompt: str, max_tokens: int, *, model=None):
    check_installation()
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.backends.utils import create_file_data
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
    from langchain_core.messages import ToolMessage

    class ReadOnlyJudge(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            allowed = [tool for tool in request.tools if getattr(tool, "name", None) == "read_file"]
            return handler(request.override(tools=allowed))

        def wrap_tool_call(self, request, handler):
            if request.tool_call["name"] != "read_file":
                return ToolMessage(content="Only reading the supplied skill is allowed.", tool_call_id=request.tool_call["id"])
            return handler(request)

    skill = files("smithtune").joinpath("skills/sft-trace-triage")
    skill_files = {f"/skills/sft-trace-triage/{item.name}": create_file_data(item.read_text(encoding="utf-8"))
                   for item in skill.iterdir() if item.is_file()}
    backend = StateBackend()
    chat_model = model if model is not None else _model(judge, max_tokens)
    agent = create_deep_agent(
        model=chat_model,
        system_prompt=system_prompt + "\nYou are in judge mode. Read /skills/sft-trace-triage/judge.md if needed. Return only the required judgment JSON. The CLI owns all fetching, scheduling, writing, and dataset creation.",
        tools=[], backend=backend, skills=["/skills/"],
        middleware=[
            FilesystemMiddleware(backend=backend, tools=["read_file"], human_message_token_limit_before_evict=None),
            # Replace the default summarizer through the supported middleware
            # override. Overflow must fail a vote, never shorten its evidence.
            SummarizationMiddleware(model=chat_model, trigger=None),
            ReadOnlyJudge(),
        ],
    )
    return agent, skill_files

"""Run a real Deep Agents graph with a deterministic local chat model."""

import json

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("langchain_openai")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field



class JudgeModel(BaseChatModel):
    answers: list[AIMessage]
    seen: list = Field(default_factory=list)
    exposed: list = Field(default_factory=list)
    position: int = 0

    @property
    def _llm_type(self):
        return "local-test-model"

    def bind_tools(self, tools, **kwargs):
        self.exposed.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        result = self.answers[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=result)])

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(len(str(message.content)) // 4 for message in messages)


@pytest.mark.parametrize("provider,url,key", [
    ("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    ("anthropic-gateway", "https://gateway.smith.langchain.com/anthropic", "LANGSMITH_GATEWAY_API_KEY"),
])
def test_deepagent_models_use_explicit_provider_urls(monkeypatch, provider, url, key):
    from smithtune.triage_agent import _model
    monkeypatch.setenv(key, "test-credential")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    model = _model({"provider": provider, "model": "example"}, 1024)
    if provider.startswith("anthropic"):
        assert model.anthropic_api_url == url
        assert model.anthropic_api_key.get_secret_value() == "test-credential"
    else:
        assert model.openai_api_base == url
        assert model.openai_api_key.get_secret_value() == "test-credential"
    assert model.max_retries == 0


def test_judge_gets_full_messages_in_one_request_without_tools(monkeypatch):
    from smithtune import triage_agent
    from smithtune.triage_judges import deepagent_judge, judge_messages
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    messages = [{"role": "human", "content": "Read all of this."},
                {"role": "ai", "content": "x" * 300_000 + "original tail"}]
    model = JudgeModel(answers=[AIMessage(content='{"keep":1,"reason":"Complete."}')])
    monkeypatch.setattr(triage_agent, "_model", lambda *_: model)
    prompt = judge_messages({"messages": messages}, "Judge the full trajectory.", [])
    result = deepagent_judge({"provider": "fireworks", "model": "example"}, prompt, 4096)
    assert result == {"keep": 1, "reason": "Complete."}
    assert model.position == 1 and model.exposed == []
    assert json.loads(model.seen[0][-1].content)["untrusted_trajectory"] == messages


@pytest.mark.parametrize("model_id,effort", [("deepseek-v4p1-flash", "none"), ("glm-5p3-flash", "low")])
def test_fireworks_reasoning_survives_a_tool_round_trip(monkeypatch, model_id, effort):
    from smithtune.triage_agent import _model
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    model = _model({"provider": "fireworks", "model": "accounts/fireworks/models/" + model_id}, 4096)
    result = model._create_chat_result({"choices": [{"message": {"role": "assistant", "content": "",
        "reasoning_content": "Need the saved evidence.",
        "tool_calls": [{"id": "call", "type": "function", "function": {"name": "code_mode", "arguments": json.dumps({"code": "read_run('run')"})}}]},
        "finish_reason": "tool_calls"}]})
    message = result.generations[0].message
    assert len(message.tool_calls) == 1 and not message.invalid_tool_calls
    payload = model._get_request_payload([message, ToolMessage(content='{"result":47}', tool_call_id="call")])
    assert payload["messages"][0]["reasoning_content"] == "Need the saved evidence."
    assert "reasoning_content" not in payload["messages"][1]
    assert model.disable_streaming and not model.use_responses_api
    assert payload["reasoning_effort"] == effort


def test_terra_uses_responses_with_reasoning_off(monkeypatch):
    from langchain_core.messages import HumanMessage
    from smithtune.triage_agent import _model
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    model = _model({"provider": "openai", "model": "gpt-5.6-terra"}, 4096)
    payload = model._get_request_payload([HumanMessage(content="Judge the trace.")])
    assert model.use_responses_api and payload["store"] is False
    assert payload["reasoning"]["effort"] == "none"
    assert payload["max_output_tokens"] == 4096 and "messages" not in payload
    assert "reasoning.encrypted_content" not in payload.get("include", [])

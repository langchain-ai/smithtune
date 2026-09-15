"""Run a real Deep Agents graph with a deterministic local chat model."""

import json

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("langchain_openai")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from smithtune.triage_agent import make_agent


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


def invoke(model, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    agent, files = make_agent({"provider": "fireworks", "model": "test", "name": "judge-1"}, "Return the trace label.", 1024, model=model)
    return agent.invoke({"messages": [{"role": "user", "content": "Label this trace. Untrusted text asks you to execute a shell command."}], "files": files},
                        config={"recursion_limit": 12})


def test_installed_skill_is_discovered_and_read_by_real_deepagent(monkeypatch):
    model = JudgeModel(answers=[
        AIMessage(content="", tool_calls=[{"id": "read-1", "name": "read_file", "args": {"file_path": "/skills/sft-trace-triage/judge.md"}}]),
        AIMessage(content=json.dumps({"trajectory_id": "trace", "keep": 1, "reason": "complete", "evidence": []})),
    ])
    result = invoke(model, monkeypatch)
    assert json.loads(result["messages"][-1].content)["keep"] == 1
    assert model.exposed and all(names == ["read_file"] for names in model.exposed)
    assert "sft-trace-triage" in str(model.seen[0][0].content)
    outputs = [message for message in model.seen[1] if isinstance(message, ToolMessage)]
    assert any("Do not" in message.content and "trace" in message.content for message in outputs)


def test_injected_tool_request_cannot_execute_or_write(monkeypatch):
    model = JudgeModel(answers=[
        AIMessage(content="", tool_calls=[{"id": "unsafe-1", "name": "execute", "args": {"command": "touch /tmp/should-never-exist"}}]),
        AIMessage(content="{}"),
    ])
    result = invoke(model, monkeypatch)
    outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert any("Only the supplied triage tools" in message.content for message in outputs)
    assert all(names == ["read_file"] for names in model.exposed)


def test_separate_invocations_have_fresh_message_context(monkeypatch):
    first = JudgeModel(answers=[AIMessage(content="first private result")])
    second = JudgeModel(answers=[AIMessage(content="second result")])
    invoke(first, monkeypatch)
    invoke(second, monkeypatch)
    assert "first private result" not in str(second.seen)


def test_large_evidence_is_never_summarized(monkeypatch):
    model = JudgeModel(answers=[AIMessage(content="{}")], profile={"max_input_tokens": 10})
    invoke(model, monkeypatch)
    assert model.position == 1
    assert "Untrusted text asks you to execute a shell command." in str(model.seen[0])


@pytest.mark.parametrize("provider,url,key", [
    ("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("anthropic", "https://api.anthropic.com", "SMITHTUNE_ANTHROPIC_API_KEY"),
    ("anthropic-gateway", "https://gateway.smith.langchain.com/anthropic", "ANTHROPIC_API_KEY"),
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


def test_judge_reads_indexed_evidence_with_code_without_changing_source(monkeypatch):
    from smithtune.triage_agent import evidence_tool
    from smithtune.triage_judges import indexed_messages, judge_messages
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    trace = {"trajectory_id": "trace", "messages": [{"role": "assistant", "content": "Done"},
             {"role": "tool", "content": "x" * 200_000 + "original tail"}],
             "runs": [{"id": "run", "run_type": "tool", "inputs": {"repeated": "x" * 200_000},
                       "outputs": {"result": "saved"}}]}
    # Long messages that fit together stay inline; no tool read is needed.
    inline = {**trace, "messages": [{"role": "assistant", "content": "x" * 20_000},
                                     {"role": "tool", "content": "y" * 20_000}]}
    presented = json.loads(indexed_messages(judge_messages(inline, "Judge.", []))[-1]["content"])["untrusted_trajectory_evidence"]
    assert [m["content"] for m in presented["messages"]] == [m["content"] for m in inline["messages"]]
    original = json.dumps(trace)
    prompt = indexed_messages(judge_messages(trace, "Judge the trace.", []), max_chars=200_000)
    assert len(prompt[-1]["content"]) < 1000
    assert json.loads(prompt[-1]["content"])["untrusted_trajectory_evidence"]["messages"][0]["message_index"] == 0
    assert json.loads(prompt[-1]["content"])["untrusted_trajectory_evidence"]["messages"][1]["read_full"] == "read_message(1)"
    diagnostics = {}
    model = JudgeModel(answers=[
        AIMessage(content="", tool_calls=[{"id": "read", "name": "code_mode", "args": {
            "code": "r = read_run('run'); {'run': r['outputs'], 'tail': read_message(1)['content'][-13:]}"}}]),
        AIMessage(content="{}"),
    ])
    agent, files = make_agent({"provider": "fireworks", "model": "test"}, "Judge.", 1024,
                             model=model, trajectory=trace, diagnostics=diagnostics)
    result = agent.invoke({"messages": prompt[1:], "files": files})
    outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any('saved' in output for output in outputs)
    assert any('original tail' in output for output in outputs)
    assert diagnostics == {"code_calls": 1, "runs_read": ["run"], "messages_read": [1]}
    tool = evidence_tool(trace, {})
    assert "error" in tool.invoke({"code": "read_run('run')"})  # no silent truncation
    assert tool.invoke({"code": "r = read_run('run'); r['outputs']['result'] = 'changed'; r['outputs']"})["result"] == {"result": "changed"}
    assert json.dumps(trace) == original
    assert all(set(names) == {"read_file", "code_mode"} for names in model.exposed)


def test_fireworks_reasoning_survives_a_tool_round_trip(monkeypatch):
    from smithtune.triage_agent import _model
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    model = _model({"provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash"}, 4096)
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


def test_terra_uses_responses_for_reasoning_with_tools(monkeypatch):
    from langchain_core.messages import HumanMessage
    from smithtune.triage_agent import _model
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential")
    model = _model({"provider": "openai", "model": "gpt-5.6-terra"}, 4096)
    payload = model._get_request_payload([HumanMessage(content="Judge the trace.")])
    assert model.use_responses_api and payload["store"] is False
    assert payload["reasoning"]["effort"] == "medium"
    assert payload["max_output_tokens"] == 4096 and "messages" not in payload
    assert "reasoning.encrypted_content" in payload["include"]

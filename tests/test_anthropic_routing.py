
from binding_fixtures import bound_row
import json

import pytest

from smithtune import doctor, inference, triage_judges
from smithtune.evaluation import replay as evaluation
from smithtune.providers.base import PipelineError


@pytest.mark.parametrize("provider,origin,key", [
    ("anthropic", "https://api.anthropic.com", "direct-test-key"),
])
@pytest.mark.parametrize("caller", ["replay", "triage"])
def test_judging_keeps_provider_credentials_separate(monkeypatch, provider, origin, key, caller):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-test-key")
    monkeypatch.setenv("SMITHTUNE_ANTHROPIC_API_KEY", "obsolete-test-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", '{"X-Api-Key":"obsolete-header"}')
    captured = []

    def post(request, label):
        captured.append(request)
        return {"content": [{"type": "text", "text": '{"keep":1,"reason":"good"}'}]}

    monkeypatch.setattr(inference, "_post_json", post)
    messages = [{"role": "system", "content": "Judge accurately."}, {"role": "user", "content": "saved evidence"}]
    if caller == "replay":
        inference._chat_completion(f"{provider}/claude-example", messages, 123, True)
    else:
        triage_judges.api_judge({"provider": provider, "model": "claude-example"}, messages, 123)
    request, = captured
    assert request.full_url == origin + "/v1/messages"
    assert request.get_header("X-api-key") == key
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert json.loads(request.data) == {"model": "claude-example", "system": "Judge accurately.",
                                      "messages": messages[1:], "max_tokens": 123}


@pytest.mark.parametrize("provider,missing", [
    ("anthropic", "ANTHROPIC_API_KEY"),
])
def test_missing_key_never_falls_back_to_other_credentials(monkeypatch, provider, missing):
    for name in ("ANTHROPIC_API_KEY", "SMITHTUNE_ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "test-other-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", '{"x-api-key":"test-other-key"}')
    monkeypatch.delenv(missing)
    monkeypatch.setattr(inference, "_post_json", lambda *_: pytest.fail("must fail before any request"))
    with pytest.raises(PipelineError, match=missing):
        inference._chat_completion(f"{provider}/claude-example", [{"role": "user", "content": "case"}], 128)
    with pytest.raises(PipelineError, match=missing):
        triage_judges.check_credentials([{"provider": provider, "name": "judge"}])


def test_doctor_reports_only_credential_presence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-test-key")
    result = doctor.diagnose()
    assert result["credentials"]["ANTHROPIC_API_KEY"] == "set"
    assert set(result["credentials"]) == set(result["credentials_required_for"])
    assert "test-key" not in json.dumps(result)
    assert "SMITHTUNE_ANTHROPIC_API_KEY" not in result["credentials"]
    assert "ANTHROPIC_CUSTOM_HEADERS" not in result["credentials"]


@pytest.mark.parametrize("previous", ["missing_endpoint", "changed_endpoint", "missing_config"])
def test_replay_rejects_results_with_unverified_or_changed_judge_endpoint(tmp_path, monkeypatch, previous):
    from test_pipeline import write_manifest

    data_dir, output = tmp_path / "data", tmp_path / "evaluation"
    write_manifest(data_dir)
    row = bound_row({"messages": [{"role": "user", "content": "What is x?"}, {"role": "assistant", "content": "x is 1"}],
           "_source": {"example_id": "example-1", "source_scope": "thread", "source_scope_id": "thread-1"}})
    (data_dir / "prepared/test.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, model, max_output_tokens: ([{**case, "prompt_tokens": 10} for case in cases], []))

    def chat(model, messages, max_tokens, json_mode, request_contract=None):
        if json_mode:
            evidence = json.loads(messages[1]["content"])
            passed = evidence["candidate_next_action"]["content"] == evidence["untrusted_trajectory"]["reference_next_action"]["content"]
            return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "text check"})}
        return {"role": "assistant", "content": "x is 1"}

    options = dict(data_dir=data_dir, output_dir=output, tuned_model="tuned-model",
                   judge_model="anthropic/claude-example", confirm=True, chat=chat)
    evaluation.run_replay_evaluation(**options)
    config_path = output / "evaluation-config.json"
    config = json.loads(config_path.read_text())
    assert config["judge_endpoint"] == "https://api.anthropic.com"
    results_before = (output / "results.jsonl").read_bytes()
    options["chat"] = lambda *_args, **_kwargs: pytest.fail("completed replay must not repeat calls")
    evaluation.run_replay_evaluation(**options)
    if previous == "missing_config":
        config_path.unlink()
    else:
        if previous == "missing_endpoint":
            config.pop("judge_endpoint")
        else:
            config["judge_endpoint"] = "https://example.invalid/anthropic"
        config_path.write_text(json.dumps(config))
    with pytest.raises(PipelineError, match="evaluation settings|judge endpoint"):
        evaluation.run_replay_evaluation(**options)
    assert (output / "results.jsonl").read_bytes() == results_before


def test_baseten_judge_uses_model_api_with_its_own_key(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-test-key")
    captured = []

    def post(request, label, *, opener=None):
        captured.append(request)
        return {"choices": [{"message": {"role": "assistant", "content": '{"pass":true}'}}]}

    monkeypatch.setattr(inference, "_post_json", post)
    messages = [{"role": "system", "content": "Judge accurately."}, {"role": "user", "content": "saved evidence"}]
    message = inference._chat_completion("baseten/deepseek-ai/DeepSeek-V4.1-Flash", messages, 123, True)
    request, = captured
    assert message["content"] == '{"pass":true}'
    assert request.full_url == "https://inference.baseten.co/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer baseten-test-key"
    body = json.loads(request.data)
    assert body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash" and body["max_tokens"] == 123
    assert "fireworks-test-key" not in json.dumps(dict(request.header_items()))


def test_baseten_judge_requires_baseten_key_without_fallback(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fireworks-test-key")
    monkeypatch.setattr(inference, "_post_json", lambda *_, **__: pytest.fail("must fail before any request"))
    with pytest.raises(PipelineError, match="BASETEN_API_KEY is not set for the judge"):
        inference._chat_completion("baseten/zai-org/GLM-5.3-Flash", [{"role": "user", "content": "case"}], 128)
    with pytest.raises(PipelineError, match="BASETEN_API_KEY is not set for the judge"):
        evaluation.validate_judge_credentials("baseten/zai-org/GLM-5.3-Flash")


def test_baseten_judge_rejects_inference_contracts(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")
    monkeypatch.setattr(inference, "_post_json", lambda *_, **__: pytest.fail("must fail before any request"))
    with pytest.raises(PipelineError, match="supported only for Fireworks"):
        inference._chat_completion("baseten/zai-org/GLM-5.3-Flash", [{"role": "user", "content": "case"}], 128,
                                   request_contract=object())


@pytest.mark.parametrize("route,endpoint", [
    ("anthropic/claude-sonnet-5", "https://api.anthropic.com"),
    ("baseten/deepseek-ai/DeepSeek-V4.1-Flash", "https://inference.baseten.co/v1"),
    ("accounts/fireworks/models/deepseek-v4p1-flash", None),
])
def test_judge_endpoint_is_fixed_per_route(route, endpoint):
    assert inference.judge_endpoint(route) == endpoint


def test_default_replay_judge_needs_only_the_baseten_key(monkeypatch):
    assert evaluation.DEFAULT_JUDGE_MODEL == "baseten/zai-org/GLM-5.3-Flash"
    for name in ("ANTHROPIC_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")
    evaluation.validate_judge_credentials(evaluation.DEFAULT_JUDGE_MODEL)
    monkeypatch.delenv("BASETEN_API_KEY")
    with pytest.raises(PipelineError, match="BASETEN_API_KEY is not set for the judge"):
        evaluation.validate_judge_credentials(evaluation.DEFAULT_JUDGE_MODEL)

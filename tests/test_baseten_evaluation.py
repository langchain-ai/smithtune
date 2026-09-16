import copy
import io
import json
import urllib.error
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from smithtune import cli, evaluation, inference, rendering
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from test_pipeline import loaded_contract, tool_call, write_manifest


def endpoint():
    return inference.BasetenEndpoint("model123", "deploy123", 8192)


def response(value):
    return io.BytesIO(json.dumps(value).encode())


@pytest.mark.parametrize("change", [
    {"model_id": "https://attacker.invalid"}, {"model_id": "../model"},
    {"deployment_id": "deploy/other"}, {"deployment_id": "a?redirect=x"},
    {"model_id": "UPPER"}, {"deployment_id": ""},
    {"max_seq_len": 0}, {"max_seq_len": -1}, {"max_seq_len": True},
])
def test_endpoint_rejects_invalid_identity_and_context(change):
    with pytest.raises(PipelineError):
        replace(endpoint(), **change).validate()


def test_transport_preserves_recorded_history_tools_and_reasoning(tmp_path, monkeypatch):
    contract = loaded_contract(tmp_path, legacy=False)
    messages = [
        {"role": "system", "content": "recorded policy", "id": "system-1"},
        {"role": "user", "content": "look up x"},
        {**tool_call("lookup"), "reasoning_content": "Need to check x"},
        {"role": "tool", "content": "x is 1", "tool_call_id": "call-1"},
        {"role": "user", "content": "Explain that result"},
    ]
    original = copy.deepcopy((messages, contract.to_dict()))
    captured = []

    def open_request(request, **kwargs):
        captured.append(request)
        return response({"choices": [{"message": {"role": "assistant", "content": "x is 1"}}]})

    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(inference, "open_without_redirects", open_request)
    result = inference._baseten_chat_completion(
        "checkpoint-name", messages, 128, True, contract, endpoint=endpoint(),
    )
    assert result == {"role": "assistant", "content": "x is 1"}
    request, = captured
    assert request.full_url == "https://model-model123.api.baseten.co/deployment/deploy123/sync/v1/chat/completions"
    assert request.get_method() == "POST"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["authorization"] == "Bearer baseten-test-key"
    assert headers["content-type"] == "application/json"
    assert not any("fireworks" in key for key in headers)
    body = json.loads(request.data)
    assert body["messages"] == [{key: value for key, value in item.items() if key != "id"} for item in messages]
    assert body["model"] == "checkpoint-name"
    assert body["tools"] == list(contract.tools)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 128
    assert body["response_format"] == {"type": "json_object"}
    assert (messages, contract.to_dict()) == original


@pytest.mark.parametrize("failure,match", [
    ("http", "HTTP 403"), ("invalid_json", "invalid JSON"),
    ("no_message", "message"),
])
def test_transport_errors_are_actionable_and_do_not_echo_secrets(monkeypatch, failure, match):
    monkeypatch.setenv("BASETEN_API_KEY", "secret-test-key")

    def open_request(request, **kwargs):
        if failure == "http":
            raise urllib.error.HTTPError(request.full_url, 403, "secret-test-key", {}, io.BytesIO(b"private response"))
        if failure == "invalid_json":
            return io.BytesIO(b"not JSON")
        return response({"choices": []})

    monkeypatch.setattr(inference, "open_without_redirects", open_request)
    with pytest.raises(PipelineError, match=match) as error:
        inference._baseten_chat_completion("checkpoint-name", [], 128, endpoint=endpoint())
    assert "secret-test-key" not in str(error.value)
    assert "private response" not in str(error.value)


def test_transport_requires_baseten_credential_before_network(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(inference, "open_without_redirects", lambda *_a, **_k: pytest.fail("unexpected request"))
    with pytest.raises(PipelineError, match="BASETEN_API_KEY"):
        inference._baseten_chat_completion("checkpoint-name", [], 128, endpoint=endpoint())


@pytest.mark.parametrize("profile_limit,serving_limit", [(8, 5), (5, 8)])
def test_context_uses_lower_serving_or_profile_limit_without_truncation(monkeypatch, profile_limit, serving_limit):
    model = replace(baseten.DEFAULT_MODEL, max_seq_len=profile_limit)
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "complete question"}]
    case = {"id": "case", "example_id": "example", "messages": messages, "tools": []}
    original = copy.deepcopy(case)
    seen = []

    def prompt_tokens(messages, *, tools):
        seen.append(copy.deepcopy(messages))
        return [1, 2, 3]

    monkeypatch.setattr(rendering, "load_training_renderer", lambda _: SimpleNamespace(prompt_tokens=prompt_tokens))
    accepted, rejected = rendering.validate_replay_context([case], model, 3, max_seq_len=serving_limit)
    assert accepted == []
    assert rejected[0]["context_limit"] == 5
    assert rejected[0]["prompt_tokens"] == 3
    accepted, rejected = rendering.validate_replay_context([case], model, 2, max_seq_len=serving_limit)
    assert len(accepted) == 1
    assert rejected == []
    assert seen == [messages, messages]
    assert case == original


def replay_data(tmp_path, monkeypatch, model=baseten.DEFAULT_MODEL, *, case_type="text"):
    data = tmp_path / "data"
    contract = loaded_contract(tmp_path, legacy=False) if case_type == "tool_call" else None
    write_manifest(data, model=model, contract=contract)
    row = {
        "messages": [{"role": "system", "content": "policy"},
                     {"role": "user", "content": "What is x?"},
                     {"role": "assistant", "content": "x is 1"}],
        "_source": {"example_id": "example-1", "source_scope": "thread", "source_scope_id": "thread-1"},
    }
    if contract is not None:
        row["messages"][-1] = tool_call("lookup")
        row["messages"].append({"role": "tool", "content": "x is 1", "tool_call_id": "call-1"})
        row["tools"] = list(contract.tools)
        row["_source"]["contract_sha256"] = contract.contract_sha256
    (data / "prepared/test.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(rendering, "load_training_renderer", lambda _: SimpleNamespace(prompt_tokens=lambda *_a, **_k: [1, 2, 3]))
    return data


def arguments(command, data, output):
    return [command, "--provider", "baseten", "--data-dir", str(data), "--output-dir", str(output),
            "--model-id", "model123", "--deployment-id", "deploy123", "--max-seq-len", "8192",
            "--tuned-model", "checkpoint-name"]


@pytest.mark.parametrize("case_type", ["text", "tool_call"])
def test_cli_plans_evaluates_and_resumes_existing_baseten_endpoint(tmp_path, monkeypatch, capsys, case_type):
    data = replay_data(tmp_path, monkeypatch, case_type=case_type)
    output = tmp_path / "evaluation"
    requests = []
    monkeypatch.setenv("BASETEN_API_KEY", "baseten-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_SESSION_ID", raising=False)

    def open_request(request, **kwargs):
        requests.append(request)
        body = json.loads(request.data)
        if request.full_url == endpoint().url:
            assert request.get_header("Authorization") == "Bearer baseten-test-key"
            assert body["messages"] == [{"role": "system", "content": "policy"}, {"role": "user", "content": "What is x?"}]
            if case_type == "tool_call":
                assert body["tools"][0]["function"]["name"] == "lookup"
                candidate = tool_call("lookup" if body["model"] == "checkpoint-name" else "wrong")
                candidate.pop("id")
            else:
                answer = "x is 1" if body["model"] == "checkpoint-name" else "x is 2"
                candidate = {"role": "assistant", "content": answer}
            return response({"choices": [{"message": candidate}]})
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        assert request.get_header("X-api-key") == "anthropic-test-key"
        evidence = json.loads(body["messages"][-1]["content"])
        if case_type == "tool_call":
            assert evidence["reference_action_tool_results_not_visible_to_candidate"] == [
                {"role": "tool", "content": "x is 1", "tool_call_id": "call-1"},
            ]
        passed = evidence["candidate_next_action"] == evidence["reference_next_action"]
        return response({"content": [{"type": "text", "text": json.dumps({"pass": passed, "reason": "compare actions"})}]})

    monkeypatch.setattr(inference, "open_without_redirects", open_request)
    monkeypatch.setattr(inference.urllib.request, "urlopen", open_request)
    identity = {**asdict(endpoint()), "url": endpoint().url}
    cli.main(arguments("eval-plan", data, output))
    assert json.loads(capsys.readouterr().out)["baseten_endpoint"] == identity
    assert requests == []
    argv = [*arguments("evaluate", data, output), "--base-model", "base-checkpoint", "--confirm", "--concurrency", "1"]
    cli.main(argv)
    summary = json.loads(capsys.readouterr().out)
    assert summary["baseten_endpoint"] == identity
    assert summary["calibration_passed"] is True
    assert summary["tuned_pass_rate"] == 1
    assert summary["base_pass_rate"] == 0
    assert summary["paired_wins"] == 1
    assert summary["case_types"] == {case_type: 1}
    if case_type == "tool_call":
        assert summary["deterministic_metrics"]["tuned"]["tool_name_match"]["rate"] == 1
        assert summary["deterministic_metrics"]["base"]["tool_name_match"]["rate"] == 0
        assert summary["deterministic_metrics"]["tuned"]["arguments_schema_valid"]["rate"] == 1
    config = json.loads((output / "evaluation-config.json").read_text())
    assert config["baseten_endpoint"] == identity
    result, = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert result["tuned"]["model"] == "checkpoint-name"
    assert result["base"]["model"] == "base-checkpoint"
    assert result["tuned"]["serving_route"] == result["base"]["serving_route"] == endpoint().url
    saved = {name: (output / name).read_bytes() for name in (
        "plan.json", "cases.jsonl", "evaluation-config.json", "results.jsonl", "summary.json",
    )}
    count = len(requests)
    assert sum(request.full_url == endpoint().url for request in requests) == 2
    cli.main(argv)
    capsys.readouterr()
    assert len(requests) == count
    for flag, value in [("--deployment-id", "other123"), ("--max-seq-len", "6144")]:
        changed = list(argv)
        changed[changed.index(flag) + 1] = value
        with pytest.raises(SystemExit) as error:
            cli.main(changed)
        assert error.value.code == 2
        assert "different evaluation settings" in capsys.readouterr().err
        assert len(requests) == count
        assert {name: (output / name).read_bytes() for name in saved} == saved
    (output / "evaluation-config.json").unlink()
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert len(requests) == count


def test_missing_config_cannot_resume_baseten_results_through_fireworks(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    output = tmp_path / "evaluation"
    judge = "accounts/fireworks/models/judge"

    def chat(model, messages, max_tokens, json_mode=False, request_contract=None):
        if model == judge:
            evidence = json.loads(messages[-1]["content"])
            passed = evidence["candidate_next_action"] == evidence["reference_next_action"]
            return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "compare actions"})}
        return {"role": "assistant", "content": "x is 1"}

    summary = evaluation.run_replay_evaluation(
        data, output, "checkpoint-name", judge, baseten_endpoint=endpoint(), chat=chat, confirm=True,
    )
    assert summary["tuned_pass_rate"] == 1
    (output / "evaluation-config.json").unlink()
    saved = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(PipelineError, match="different serving routes"):
        evaluation.run_replay_evaluation(
            data, output, "checkpoint-name", judge, confirm=True,
            chat=lambda *_a, **_k: pytest.fail("resume must not request inference"),
        )
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == saved


def test_baseten_endpoint_rejects_fireworks_prepared_manifest(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch, model=fireworks.DEFAULT_MODEL)
    with pytest.raises(PipelineError, match="[Bb]aseten"):
        evaluation.prepare_replay_evaluation(data, tmp_path / "evaluation", baseten_endpoint=endpoint())


@pytest.mark.parametrize("extra", [
    ["--serving-mode", "preemptible"], ["--account-id", "acct"],
    ["--deployment-shape", "accounts/fireworks/deploymentShapes/test"],
    ["--deployment-timeout", "30"],
])
def test_cli_rejects_fireworks_serving_options_for_baseten(tmp_path, capsys, extra):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments("eval-plan", tmp_path / "data", tmp_path / "output") + extra)
    assert error.value.code == 2
    assert "[Errno 2]" not in capsys.readouterr().err
    assert not (tmp_path / "output/plan.json").exists()


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
@pytest.mark.parametrize("missing", ["--model-id", "--deployment-id", "--max-seq-len"])
def test_cli_requires_baseten_endpoint_fields(tmp_path, capsys, command, missing):
    argv = arguments(command, tmp_path / "data", tmp_path / "output")
    index = argv.index(missing)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert "Baseten evaluation requires" in capsys.readouterr().err
    assert not (tmp_path / "output/plan.json").exists()


def test_temporary_replay_validates_resume_and_skips_compute_when_complete(tmp_path, monkeypatch):
    from contextlib import contextmanager

    data = replay_data(tmp_path, monkeypatch)
    output = tmp_path / "replay"
    events = []
    monkeypatch.setattr(evaluation, "calibrate_judge", lambda cases, *_: [{"actual": True, "expected": True}] * (len(evaluation._calibration_cases(cases)) * 3))
    monkeypatch.setattr(evaluation, "judge_replay_candidate", lambda *_: {"pass": True, "reason": "ok"})

    @contextmanager
    def resources():
        events.append("activate")
        try:
            yield "checkpoint-name"
        finally:
            events.append("deactivate")

    def run(**overrides):
        return evaluation.run_replay_evaluation(
            data, output, "checkpoint-name", "anthropic/test", confirm=True,
            chat=lambda *_: {"role": "assistant", "content": "x is 1"},
            baseten_lifecycle=resources(), baseten_cleanup=lambda: events.append("cleanup_existing"),
            **{"baseten_endpoint": endpoint(), **overrides},
        )

    assert run()["serving_mode"] == "temporary"
    assert events == ["activate", "deactivate"]
    saved = (output / "results.jsonl").read_bytes()
    assert run()["serving_mode"] == "temporary"
    assert events == ["activate", "deactivate", "cleanup_existing"]
    with pytest.raises(PipelineError, match="different evaluation settings"):
        run(baseten_endpoint=replace(endpoint(), max_seq_len=4096))
    assert events == ["activate", "deactivate", "cleanup_existing"]
    assert (output / "results.jsonl").read_bytes() == saved


def test_run_bound_replay_rejects_another_models_prepared_data(tmp_path, monkeypatch):
    from smithtune.baseten_deployment import validate_evaluation_model

    data = replay_data(tmp_path, monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"provider": "baseten", "base_model": "different/base-model"}))
    (run / "result.json").write_text(json.dumps({"provider": "baseten", "baseten_run_id": "run123",
        "best_sampler_weights_uri": "bt://loops:run123/sampler_weights/step-1"}))
    with pytest.raises(PipelineError, match="different base model"):
        validate_evaluation_model(run, data)

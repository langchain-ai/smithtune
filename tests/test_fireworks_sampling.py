import copy
import json
from types import SimpleNamespace

import pytest

from smithtune import cli, evaluation
from smithtune.providers import fireworks_sampling as sampling
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL
from test_pipeline import write_manifest


CHECKPOINT = "account/run-" + "a" * 32 + "/epoch-2"


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def fake_renderer(monkeypatch):
    candidate = {"role": "assistant", "content": "", "reasoning_content": "reason",
                 "tool_calls": [{"id": "call-1", "type": "function", "function": {
                     "name": "lookup", "arguments": '{"query":"café"}',
                 }}]}
    renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda _: "raw generated text"),
        parse_response=lambda _: (copy.deepcopy(candidate), SimpleNamespace(is_clean=True)),
        to_openai_message=lambda message: message,
        get_stop_sequences=lambda: [123],
    )
    monkeypatch.setattr(sampling, "load_training_renderer", lambda _: renderer)
    return renderer


@pytest.mark.parametrize("live", [False, True])
def test_sampler_uses_official_session_and_preserves_tools_and_saved_lora(tmp_path, monkeypatch, live):
    fake_renderer(monkeypatch)
    calls = []

    class Sampler:
        def sample(self, **kwargs):
            calls.append(("sample", kwargs))
            return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[7, 8], stop_reason="stop")]))

        def close(self):
            calls.append("sampler-close")

    class Service:
        training_session_id = "ts-live"

        def create_lora_training_client(self, model, **kwargs):
            calls.append(("create", model, kwargs))
            return self

        def load_state(self, path):
            calls.append(("load", path))
            return Future(None)

        def save_weights_for_sampler(self, name):
            calls.append(("save", name))
            return Future(SimpleNamespace(path="snapshot"))

        def create_sampling_client(self, **kwargs):
            calls.append(("sampler", kwargs))
            return Sampler()

        def close(self):
            calls.append("service-close")

    service = Service()
    monkeypatch.setattr(sampling, "create_service", lambda: service)
    captured = []
    monkeypatch.setattr(sampling, "replay_prompt", lambda messages, tools, *_: (
        captured.append(copy.deepcopy((messages, tools))) or SimpleNamespace(to_ints=lambda: [1, 2, 3])
    ))
    options = {"service": service, "snapshot": "snapshot"} if live else {}
    adapter = sampling.FireworksReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path,
                                              lora_rank=4, lora_alpha=16, **options)
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "lookup"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with adapter:
        result = adapter.generate(CHECKPOINT, messages, 64, request_contract=SimpleNamespace(tools=tools))
        with pytest.raises(PipelineError, match="matching base model"):
            adapter.generate("different-model", messages, 64)
    assert result["tool_calls"][0]["function"]["arguments"] == '{"query":"café"}'
    assert result["reasoning_content"] == "reason"
    assert result["sampling"]["format_valid"] is True
    assert captured == [(messages, tools)]
    request = next(call[1] for call in calls if isinstance(call, tuple) and call[0] == "sample")
    assert request["num_samples"] == 1
    assert request["sampling_params"].temperature == 0
    assert request["sampling_params"].max_tokens == 64
    assert request["sampling_params"].stop == [123]
    assert calls.count("sampler-close") == 2
    assert ("service-close" in calls) is not live
    assert ("create", DEFAULT_MODEL.base_model, {"rank": 4, "alpha": 16}) in calls if not live else not any(
        isinstance(call, tuple) and call[0] == "create" for call in calls
    )
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["status"] == "closed"
    assert receipt["session_id"] == "ts-live"


def test_partial_sampler_startup_closes_resources(tmp_path, monkeypatch):
    fake_renderer(monkeypatch)
    events = []
    service = SimpleNamespace(training_session_id="session")

    def create(**kwargs):
        if "base_model" in kwargs:
            raise RuntimeError("failed base sampler")
        return SimpleNamespace(close=lambda: events.append("closed"))

    service.create_sampling_client = create
    adapter = sampling.FireworksReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service, snapshot="snapshot")
    with pytest.raises(RuntimeError, match="failed base sampler"), adapter:
        pass
    assert events == ["closed"]


def replay_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    write_manifest(data)
    row = {"messages": [{"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"}],
           "_source": {"example_id": "one", "source_scope": "thread", "source_scope_id": "thread"}}
    (data / "prepared/test.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, *_: ([{**case, "prompt_tokens": 2} for case in cases], []))
    return data


def test_saved_generations_resume_judging_without_reopening_session(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    calls = []

    class Sampler:
        checkpoint = CHECKPOINT

        def __init__(self):
            self.config = {"checkpoint": CHECKPOINT, "provider": "fireworks", "serving_mode": "serverless"}

        def __enter__(self):
            calls.append("open")
            return CHECKPOINT

        def __exit__(self, *args):
            calls.append("close")

        def generate(self, *args):
            calls.append("generate")
            return {"role": "assistant", "content": "answer"}

    fail = True

    def judge(model, messages, *_):
        evidence = json.loads(messages[1]["content"])
        candidate = evidence["candidate_next_action"]["content"]
        return {"content": json.dumps({"pass": candidate == "answer", "reason": "check"})}

    original_judge = evaluation.judge_replay_candidate

    def score(*args):
        nonlocal fail
        if calls and fail:
            raise PipelineError("judge unavailable")
        return original_judge(*args)

    monkeypatch.setattr(evaluation, "judge_replay_candidate", score)
    options = dict(data_dir=data, output_dir=tmp_path / "replay", tuned_model=CHECKPOINT,
                   judge_model="judge", chat=judge, replay_sampler=Sampler(), confirm=True)
    with pytest.raises(PipelineError, match="interrupted"):
        evaluation.run_replay_evaluation(**options)
    assert calls == ["open", "generate", "close"]
    assert len(json.loads((tmp_path / "replay/generations.jsonl").read_text())["candidate"]) == 2
    fail = False
    summary = evaluation.run_replay_evaluation(**options)
    assert summary["tuned_pass_rate"] == 1
    assert calls == ["open", "generate", "close"]
    evaluation.run_replay_evaluation(**options)
    assert calls == ["open", "generate", "close"]
    with pytest.raises(PipelineError, match="settings"):
        evaluation.run_replay_evaluation(**options, max_output_tokens=12)


def test_cli_saved_run_defaults_to_serverless_base_comparison(tmp_path, monkeypatch, capsys):
    data = replay_data(tmp_path, monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"base_model": DEFAULT_MODEL.base_model, "config": {"lora_rank": 4, "lora_alpha": 16}}))
    (run / "result.json").write_text(json.dumps({"best": {"resume_checkpoint": CHECKPOINT}}))
    fake_renderer(monkeypatch)
    calls = []
    monkeypatch.setattr(evaluation, "validate_judge_credentials", lambda _: None)
    monkeypatch.setattr(evaluation, "run_replay_evaluation", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    cli.main(["evaluate", "--data-dir", str(data), "--run-dir", str(run), "--confirm"])
    _, options = calls[0]
    assert calls[0][0][1] == run / "replay"
    assert options["base_model"] == DEFAULT_MODEL.base_model
    assert options["replay_sampler"].config["lora_alpha"] == 16
    assert options["replay_sampler"].config["lora_rank"] == 4
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--serving-mode", "preemptible"])
    assert "invalid choice" in capsys.readouterr().err


def test_fireworks_preview_requires_fireworks_data(tmp_path, capsys):
    from smithtune.providers.baseten import DEFAULT_MODEL as BASETEN_MODEL

    write_manifest(tmp_path / "data", model=BASETEN_MODEL)
    with pytest.raises(SystemExit):
        cli.main(["eval-plan", "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "replay")])
    assert "provider" in capsys.readouterr().err
    assert not (tmp_path / "replay" / "cases.jsonl").exists()


@pytest.mark.parametrize("failure", ["calibration", "replay"])
def test_judge_failure_prevents_training_or_preserves_completed_checkpoint(tmp_path, monkeypatch, failure):
    from smithtune.providers import fireworks_training as runtime
    from smithtune.providers import fireworks

    data = replay_data(tmp_path, monkeypatch)
    run = tmp_path / "run"
    events = []
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    monkeypatch.setattr(fireworks, "preflight_model", lambda _: None)
    monkeypatch.setattr(fireworks, "load_training_renderer", lambda _: None)
    monkeypatch.setattr(evaluation, "validate_judge_credentials", lambda _: None)

    def calibrate(*args):
        events.append("calibration")
        if failure == "calibration":
            raise PipelineError("judge unavailable")

    class Session:
        service = object()

        def __init__(self, *args):
            events.append("create")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def run_epoch(self, *args):
            return {"eval_loss": 0.5, "resume_checkpoint": CHECKPOINT}

        def complete(self):
            events.append("complete")

        def snapshot(self, checkpoint):
            assert checkpoint == CHECKPOINT
            return "snapshot"

    def replay(*args, **kwargs):
        assert json.loads((run / "result.json").read_text())["best"]["resume_checkpoint"] == CHECKPOINT
        raise PipelineError("judge unavailable")

    monkeypatch.setattr(runtime, "ServerlessTraining", Session)
    monkeypatch.setattr(sampling, "FireworksReplaySampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "ensure_judge_calibration", calibrate)
    monkeypatch.setattr(evaluation, "run_replay_evaluation", replay)
    with pytest.raises(PipelineError, match="judge unavailable"):
        fireworks.FireworksProvider().train(
            data, run, "run", fireworks.SFTSettings(max_epochs=1), confirm=True, init_from_checkpoint=None,
            replay={"judge_model": "judge", "concurrency": 4, "max_points_per_trajectory": None, "max_output_tokens": 256},
        )
    if failure == "calibration":
        assert events == ["calibration"]
        assert not (run / "result.json").exists()
    else:
        assert events == ["calibration", "create", "complete", "close"]
        assert "phase: replay_incomplete" in (run / "run.md").read_text()

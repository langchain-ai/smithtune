"""Sampler routing, training handoff, and durable replay integration."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smithtune import cli, evaluation, rendering
from smithtune.providers import baseten, baseten_sampling
from smithtune.providers.base import PipelineError
from test_baseten_evaluation import replay_data
from test_baseten_provider import FakeManagement, FakeService, FakeTrainer, _provider, _write_prepared_dataset


CHECKPOINT = "bt://loops:run123/sampler_weights/best-epoch-2"
JUDGE = "anthropic/test-judge"
REPLAY = {"judge_model": JUDGE, "concurrency": 1,
          "max_points_per_trajectory": None, "max_output_tokens": 128}


def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"run_id": "weather-sft", "provider": "baseten", "base_model": baseten.DEFAULT_MODEL.base_model}))
    (run / "result.json").write_text(json.dumps({"provider": "baseten", "status": "completed",
        "baseten_run_id": "run123", "best_epoch": 2, "best_sampler_weights_uri": CHECKPOINT,
        "last_resumable_state_uri": "bt://loops:run123/weights/last-epoch-3"}))
    return run


@pytest.mark.parametrize("with_run", [False, True])
def test_sampler_preview_is_offline_and_defaults_to_base_comparison(tmp_path, monkeypatch, capsys, with_run):
    data = replay_data(tmp_path, monkeypatch)
    constructor = Mock(side_effect=AssertionError("preview must not construct a sampler"))
    monkeypatch.setattr(baseten_sampling, "BasetenReplaySampler", constructor)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["eval-plan", "--provider", "baseten", "--data-dir", str(data)]
    if with_run:
        run = saved_run(tmp_path)
        args += ["--run-dir", str(run)]
    else:
        args += ["--output-dir", str(tmp_path / "replay")]
    cli.main(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["serving_mode"] == "sampler"
    assert plan["evaluated_models"] == 2
    if with_run:
        assert plan["checkpoint"] == CHECKPOINT
        assert (run / "replay/plan.json").exists()
    constructor.assert_not_called()


def test_default_standalone_evaluation_routes_best_checkpoint_to_sampler(tmp_path, monkeypatch, capsys):
    data = replay_data(tmp_path, monkeypatch)
    run = saved_run(tmp_path)
    sampler = object()
    constructor = Mock(return_value=sampler)
    evaluate = Mock(return_value={"serving_mode": "sampler"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setattr(baseten_sampling, "BasetenReplaySampler", constructor)
    monkeypatch.setattr(evaluation, "run_replay_evaluation", evaluate)
    monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", Mock(side_effect=AssertionError("unexpected endpoint")))
    cli.main(["evaluate", "--provider", "baseten", "--run-dir", str(run),
              "--data-dir", str(data), "--confirm"])
    assert json.loads(capsys.readouterr().out)["serving_mode"] == "sampler"
    model, checkpoint, output = constructor.call_args.args
    assert (model.base_model, checkpoint, output) == (baseten.DEFAULT_MODEL.base_model, CHECKPOINT, run / "replay")
    assert evaluate.call_args.args[2] == CHECKPOINT
    assert evaluate.call_args.kwargs["base_model"] == model.base_model
    assert evaluate.call_args.kwargs["replay_sampler"] is sampler
    assert evaluate.call_args.kwargs["training"] == {"parent_training_run_id": "weather-sft", "checkpoint_epoch": 2}


@pytest.mark.parametrize("failure", ["unconfirmed", "missing_judge"])
def test_standalone_preflight_never_constructs_sampler_on_failure(tmp_path, monkeypatch, failure):
    data = replay_data(tmp_path, monkeypatch)
    run = saved_run(tmp_path)
    constructor = Mock(side_effect=AssertionError("must fail before sampler construction"))
    monkeypatch.setattr(baseten_sampling, "BasetenReplaySampler", constructor)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["evaluate", "--provider", "baseten", "--run-dir", str(run), "--data-dir", str(data)]
    if failure == "missing_judge":
        args.append("--confirm")
    with pytest.raises(SystemExit):
        cli.main(args)
    constructor.assert_not_called()


@pytest.mark.parametrize("receipt", [False, True])
def test_existing_endpoint_selection_does_not_enter_sampler_route(tmp_path, monkeypatch, capsys, receipt):
    data = replay_data(tmp_path, monkeypatch)
    endpoint = cli.BasetenEndpoint("model123", "deploy123", 8192)
    evaluate = Mock(return_value={"serving_mode": "existing"})
    monkeypatch.setattr(evaluation, "run_replay_evaluation", evaluate)
    monkeypatch.setattr(baseten_sampling, "BasetenReplaySampler", Mock(side_effect=AssertionError("unexpected sampler")))
    args = ["evaluate", "--provider", "baseten", "--data-dir", str(data), "--output-dir", str(tmp_path / "replay")]
    if receipt:
        run = saved_run(tmp_path)
        monkeypatch.setattr(cli.baseten_deployment, "load_endpoint", Mock(return_value=(endpoint, "checkpoint-name")))
        args += ["--serving-mode", "existing", "--run-dir", str(run)]
    else:
        args += ["--model-id", endpoint.model_id, "--deployment-id", endpoint.deployment_id,
                 "--max-seq-len", "8192", "--tuned-model", "checkpoint-name"]
    cli.main(args)
    assert json.loads(capsys.readouterr().out)["serving_mode"] == "existing"
    assert evaluate.call_args.kwargs["baseten_endpoint"] == endpoint
    assert "replay_sampler" not in evaluate.call_args.kwargs


def training_data(tmp_path, monkeypatch):
    _write_prepared_dataset(tmp_path)
    prepared = tmp_path / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["split"]["test"] = 1
    manifest["langsmith"]["examples"] += 1
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    row = {"messages": [{"role": "user", "content": "What is x?"},
                        {"role": "assistant", "content": "x is 1"}],
           "_source": {"example_id": "held-out", "source_scope": "thread", "source_scope_id": "held-out-thread"}}
    (prepared / "test.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(rendering, "load_training_renderer", lambda _: SimpleNamespace(prompt_tokens=lambda *_a, **_k: [1, 2, 3]))
    return tmp_path


def test_training_replay_plan_is_offline_and_records_separate_sampler_cost(tmp_path, monkeypatch):
    data = training_data(tmp_path, monkeypatch)
    service = FakeService(FakeTrainer())
    provider = _provider(service, FakeManagement())
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = provider.plan(data, "preview", baseten.BasetenSFTSettings(), replay=REPLAY)
    assert plan["replay"]["serving_mode"] == "sampler"
    assert plan["replay"]["evaluated_models"] == 2
    assert plan["replay"]["timing"] == "after trainer shutdown"
    assert "separate" in plan["replay"]["budget"]
    assert service.service_client_calls == []


@pytest.mark.parametrize("command", ["plan", "train"])
def test_training_cli_forwards_requested_replay_settings(tmp_path, monkeypatch, capsys, command):
    provider = Mock()
    provider.plan.return_value = {"status": "planned"}
    provider.train.return_value = {"status": "completed"}
    monkeypatch.setattr(cli, "get_provider", lambda _: provider)
    args = [command, "--provider", "baseten", "--data-dir", str(tmp_path / "data"),
            "--run-id", "selected-run", "--evaluate", "--judge-model", JUDGE,
            "--concurrency", "1", "--max-output-tokens", "128"]
    if command == "train":
        args += ["--run-dir", str(tmp_path / "run")]
    cli.main(args)
    capsys.readouterr()
    called = provider.plan if command == "plan" else provider.train
    assert called.call_args.kwargs["replay"] == REPLAY


@pytest.mark.parametrize("replay_fails", [False, True])
def test_train_replays_best_checkpoint_only_after_trainer_cleanup(tmp_path, monkeypatch, replay_fails):
    data = training_data(tmp_path, monkeypatch)
    trainer = FakeTrainer(validation_losses=(1.0, 0.8, 0.9))
    service = FakeService(trainer)
    management = FakeManagement(inactive=[True])
    provider = _provider(service, management)
    run = tmp_path / "run"
    events = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")

    def calibrate(*_args):
        assert not service.create_calls
        events.append("calibrate")

    def sampler(model, checkpoint, output):
        assert trainer.closed and management.deactivated == [trainer.run_id]
        assert checkpoint == "sampler://approved-run-sampler-epoch-2"
        assert output == run / "replay"
        assert json.loads((run / "result.json").read_text())["status"] == "completed"
        events.append("sampler")
        return object()

    def replay(*args, **kwargs):
        assert args[2] == "sampler://approved-run-sampler-epoch-2"
        assert kwargs["base_model"] == baseten.DEFAULT_MODEL.base_model
        assert kwargs["confirm"] is True
        assert kwargs["training"] == {"parent_training_run_id": "approved-run", "checkpoint_epoch": 2}
        events.append("replay")
        if replay_fails:
            raise PipelineError("judge failed after training")
        return {"tuned_pass_rate": 1.0}

    monkeypatch.setattr(evaluation, "ensure_judge_calibration", calibrate)
    monkeypatch.setattr(baseten_sampling, "BasetenReplaySampler", sampler)
    monkeypatch.setattr(evaluation, "run_replay_evaluation", replay)
    options = dict(confirm=True, init_from_checkpoint=None, replay=REPLAY)
    if replay_fails:
        with pytest.raises(PipelineError, match="after training"):
            provider.train(data, run, "approved-run", baseten.BasetenSFTSettings(), **options)
    else:
        provider.train(data, run, "approved-run", baseten.BasetenSFTSettings(), **options)
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "completed"
    assert result["best_epoch"] == 2
    assert result["last_resumable_state_uri"].endswith("epoch-3")
    assert result["replay_status"] == ("incomplete" if replay_fails else "completed")
    assert json.loads((run / "run-state.json").read_text())["replay_status"] == result["replay_status"]
    assert events == ["calibrate", "sampler", "replay"]


@pytest.mark.parametrize("failure", ["credentials", "calibration", "langsmith"])
def test_training_judge_preflight_failure_never_allocates_trainer(tmp_path, monkeypatch, failure):
    data = training_data(tmp_path, monkeypatch)
    service = FakeService(FakeTrainer())
    provider = _provider(service, FakeManagement())
    if failure == "credentials":
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    elif failure == "calibration":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
        monkeypatch.setattr(evaluation, "calibrate_judge", lambda *_: [{"actual": False, "expected": True}])
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
        monkeypatch.setattr(evaluation, "preflight_langsmith", Mock(side_effect=PipelineError("LangSmith snapshot mismatch")))
        monkeypatch.setattr(evaluation, "ensure_judge_calibration", Mock(side_effect=AssertionError("must verify before judge")))
    with pytest.raises(PipelineError, match="ANTHROPIC_API_KEY|calibration|snapshot"):
        provider.train(data, tmp_path / "run", "approved-run", baseten.BasetenSFTSettings(),
                       confirm=True, init_from_checkpoint=None, replay=REPLAY)
    assert service.service_client_calls == []
    assert service.create_calls == []


def test_baseten_sampler_resume_reuses_both_generations_without_compute(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    events = []
    fail = True
    monkeypatch.setenv("BASETEN_API_KEY", "test-only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(evaluation, "calibrate_judge", lambda cases, *_: [{"actual": True, "expected": True}] * (len(evaluation._calibration_cases(cases)) * 3))

    class Sampler:
        checkpoint = CHECKPOINT

        def __init__(self):
            self.config = {"provider": "baseten", "serving_mode": "sampler", "checkpoint": CHECKPOINT}

        def __enter__(self):
            events.append("open")
            return CHECKPOINT

        def __exit__(self, *_):
            events.append("close")

        def generate(self, model, *_):
            events.append(model)
            return {"role": "assistant", "content": "x is 1"}

    def judge(_case, _candidate, *_):
        if fail and CHECKPOINT in events:
            raise PipelineError("judge unavailable")
        return {"pass": True, "reason": "matches"}

    monkeypatch.setattr(evaluation, "judge_replay_candidate", judge)
    options = dict(data_dir=data, output_dir=tmp_path / "replay", tuned_model=CHECKPOINT,
                   judge_model=JUDGE, base_model=baseten.DEFAULT_MODEL.base_model,
                   replay_sampler=Sampler(), confirm=True, concurrency=1)
    with pytest.raises(PipelineError, match="interrupted"):
        evaluation.run_replay_evaluation(**options)
    assert events == ["open", baseten.DEFAULT_MODEL.base_model, CHECKPOINT, "close"]
    generations = (tmp_path / "replay/generations.jsonl").read_bytes()
    assert len(generations.splitlines()) == 2
    fail = False
    summary = evaluation.run_replay_evaluation(**options)
    assert summary["serving_mode"] == "sampler"
    assert summary["base_pass_rate"] == summary["tuned_pass_rate"] == 1
    evaluation.run_replay_evaluation(**options)
    assert events == ["open", baseten.DEFAULT_MODEL.base_model, CHECKPOINT, "close"]
    assert (tmp_path / "replay/generations.jsonl").read_bytes() == generations
    with pytest.raises(PipelineError, match="different evaluation settings"):
        evaluation.run_replay_evaluation(**options, max_output_tokens=42)
    assert events == ["open", baseten.DEFAULT_MODEL.base_model, CHECKPOINT, "close"]

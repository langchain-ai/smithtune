import json
import sys
from types import SimpleNamespace

import pytest

from smithtune.artifacts import _json_dump
from smithtune.providers.base import PipelineError
from smithtune.providers.tinker import sampling
from smithtune.providers.tinker.provider import DEFAULT_MODEL
from tests.test_pipeline import write_manifest


CHECKPOINT = "tinker://tinker-run-1/sampler_weights/best-epoch-2"


@pytest.fixture
def renderer(monkeypatch):
    rendered = []

    def prompt(messages, tools):
        rendered.append((messages, tools))
        return [10, 11, 12]

    monkeypatch.setattr(sampling, "load_training_renderer", lambda _: SimpleNamespace(prompt_tokens=prompt))
    monkeypatch.setitem(sys.modules, "smithtune.providers.baseten.sampling_formats", SimpleNamespace(
        PARSING_VERSION="test-v1", stop_sequences=lambda *_: ["<end>"],
        parse_completion=lambda renderer, model, tokens, stop_reason, tools=None: {
            "role": "assistant", "content": "answer", "sampling": {"format_valid": stop_reason == "stop"},
        },
    ))
    return rendered


class FakeSamplingClient:
    def __init__(self, identity, requests):
        self.identity = identity
        self.requests = requests

    def sample(self, **request):
        self.requests.append((self.identity, request))
        sequence = SimpleNamespace(tokens=[7, 8], stop_reason="stop")
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))


class FakeService:
    def __init__(self, *, fail_base=False):
        self.fail_base = fail_base
        self.factory_calls = []
        self.created = []
        self.requests = []
        self.holder = SimpleNamespace(closed=0)
        self.holder.close = lambda: setattr(self.holder, "closed", self.holder.closed + 1)

    def __call__(self, **kwargs):
        self.factory_calls.append(kwargs)
        return self

    def create_sampling_client(self, *, model_path=None, base_model=None):
        self.created.append({"model_path": model_path, "base_model": base_model})
        if base_model is not None and self.fail_base:
            raise RuntimeError("base model unavailable")
        return FakeSamplingClient(model_path or base_model, self.requests)


def test_replay_samples_the_checkpoint_and_its_base_model_and_closes_the_session(tmp_path, renderer):
    service = FakeService()
    adapter = sampling.TinkerReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service_factory=service)
    messages = [{"role": "user", "content": "café"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with adapter as identity:
        assert identity == CHECKPOINT
        assert json.loads((tmp_path / "sampler.json").read_text())["status"] == "running"
        result = adapter.generate(identity, messages, 32, request_contract=SimpleNamespace(tools=tools))
        base = adapter.generate(DEFAULT_MODEL.base_model, messages, 32)
        with pytest.raises(PipelineError, match="matching base model"):
            adapter.generate("wrong-model", messages, 32)
        with pytest.raises(PipelineError, match="context limit"):
            adapter.generate(identity, messages, DEFAULT_MODEL.max_seq_len)

    assert service.created == [{"model_path": CHECKPOINT, "base_model": None},
                               {"model_path": None, "base_model": DEFAULT_MODEL.base_model}]
    assert renderer[:2] == [(messages, tools), (messages, [])]
    identity, request = service.requests[0]
    assert identity == CHECKPOINT
    assert request["prompt"].to_ints() == [10, 11, 12]
    assert request["num_samples"] == 1
    assert request["sampling_params"].temperature == 0
    assert request["sampling_params"].max_tokens == 32
    assert request["sampling_params"].stop == ["<end>"]
    assert result["sampling"] == {"format_valid": True, "checkpoint": CHECKPOINT, "prompt_tokens": 3, "output_tokens": 2}
    assert base["sampling"]["checkpoint"] is None
    assert service.holder.closed == 1
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["status"] == "closed"
    assert receipt["provider"] == "tinker"
    assert adapter.config["serving_mode"] == "sampler"


def test_a_failed_sampler_creation_still_closes_the_session(tmp_path, renderer):
    service = FakeService(fail_base=True)
    adapter = sampling.TinkerReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service_factory=service)
    with pytest.raises(RuntimeError, match="unavailable"), adapter:
        pass
    assert service.holder.closed == 1
    assert json.loads((tmp_path / "sampler.json").read_text())["status"] == "closed"


@pytest.mark.parametrize("checkpoint", ["tinker://tinker-run-1/weights/state-1", "bt://loops:run/sampler_weights/x"])
def test_replay_requires_saved_tinker_sampler_weights(tmp_path, renderer, checkpoint):
    with pytest.raises(PipelineError, match="saved sampler checkpoint"):
        sampling.TinkerReplaySampler(DEFAULT_MODEL, checkpoint, tmp_path, service_factory=FakeService())


def test_saved_run_accepts_only_best_sampler_weights_for_the_matching_base(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data, model=DEFAULT_MODEL)
    _json_dump(run / "plan.json", {"provider": "tinker", "base_model": DEFAULT_MODEL.base_model})
    result = {"provider": "tinker", "tinker_run_id": "tinker-run-1", "best_sampler_weights_uri": CHECKPOINT,
              "last_resumable_state_uri": "tinker://tinker-run-1/weights/last"}
    _json_dump(run / "result.json", result)
    model, checkpoint = sampling.checkpoint_from_run(data, run)
    assert model.base_model == DEFAULT_MODEL.base_model
    assert checkpoint == CHECKPOINT
    _json_dump(run / "result.json", {**result, "best_sampler_weights_uri": result["last_resumable_state_uri"]})
    with pytest.raises(PipelineError, match="best sampler checkpoint"):
        sampling.checkpoint_from_run(data, run)
    _json_dump(run / "result.json", {**result, "tinker_run_id": "other"})
    with pytest.raises(PipelineError, match="best sampler checkpoint"):
        sampling.checkpoint_from_run(data, run)
    _json_dump(run / "result.json", {**result, "provider": "baseten"})
    with pytest.raises(PipelineError, match="completed Tinker training run"):
        sampling.checkpoint_from_run(data, run)
    _json_dump(run / "result.json", result)
    _json_dump(run / "plan.json", {"provider": "tinker", "base_model": "different"})
    with pytest.raises(PipelineError, match="base model differs"):
        sampling.checkpoint_from_run(data, run)


def test_evaluate_routes_tinker_runs_to_the_tinker_sampler(tmp_path, monkeypatch, capsys):
    from smithtune import cli

    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data, model=DEFAULT_MODEL)
    _json_dump(run / "plan.json", {"provider": "tinker", "base_model": DEFAULT_MODEL.base_model})
    _json_dump(run / "result.json", {"provider": "tinker", "tinker_run_id": "tinker-run-1",
                                     "best_sampler_weights_uri": CHECKPOINT})
    calls = []
    monkeypatch.setattr(sampling, "TinkerReplaySampler", lambda *args, **kwargs: calls.append(("sampler", args)) or "sampler")
    monkeypatch.setattr(cli.replay_evaluation, "validate_judge_credentials", lambda judge: None)
    monkeypatch.setattr(cli.replay_evaluation, "run_replay_evaluation",
                        lambda *args, **kwargs: calls.append(("run", args, kwargs)) or {"status": "completed"})

    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--provider", "tinker", "--run-dir", str(run), "--data-dir", str(data)])
    assert "require --confirm" in capsys.readouterr().err

    cli.main(["evaluate", "--provider", "tinker", "--run-dir", str(run), "--data-dir", str(data), "--confirm"])
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}
    _, args, kwargs = calls[-1]
    assert args[2] == CHECKPOINT
    assert kwargs["base_model"] == DEFAULT_MODEL.base_model
    assert kwargs["replay_sampler"] == "sampler"

    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--provider", "tinker", "--run-dir", str(run), "--data-dir", str(data),
                  "--accelerator", "H200:1", "--confirm"])
    assert "omit endpoint and deployment options" in capsys.readouterr().err


def test_tinker_replay_requires_the_tinker_credential(tmp_path, monkeypatch):
    from smithtune.evaluation.replay import run_replay_evaluation

    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    sampler = SimpleNamespace(checkpoint=CHECKPOINT, config={"provider": "tinker", "serving_mode": "sampler"})
    with pytest.raises(PipelineError, match="TINKER_API_KEY is not set"):
        run_replay_evaluation(tmp_path, tmp_path / "replay", CHECKPOINT, "baseten/zai-org/GLM-5.3-Flash",
                              confirm=True, replay_sampler=sampler)


def test_a_validated_step_can_be_evaluated_when_its_sampler_weights_were_saved(tmp_path):
    from smithtune.evaluation.replay import training_metadata

    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data, model=DEFAULT_MODEL)
    _json_dump(run / "plan.json", {"provider": "tinker", "base_model": DEFAULT_MODEL.base_model, "run_id": "smoke"})
    _json_dump(run / "result.json", {"provider": "tinker", "tinker_run_id": "tinker-run-1",
                                     "best_sampler_weights_uri": CHECKPOINT, "best_epoch": 2, "best_step": 4})
    step_one = "tinker://tinker-run-1/sampler_weights/smoke-sampler-epoch-1"
    _json_dump(run / "validations.json", [
        {"epoch": 1, "step": 2, "sampler_weights_uri": step_one},
        {"epoch": 2, "step": 4, "sampler_weights_uri": CHECKPOINT},
        {"epoch": 3, "step": 6, "sampler_weights_uri": None},
    ])
    assert sampling.checkpoint_from_run(data, run, step=2)[1] == step_one
    assert training_metadata(run, 2) == {"parent_training_run_id": "smoke", "checkpoint_epoch": 1,
                                         "checkpoint_step": 2, "checkpoint_selection": "step"}
    with pytest.raises(PipelineError, match="--sampler-checkpoints all"):
        sampling.checkpoint_from_run(data, run, step=6)
    with pytest.raises(PipelineError, match="validated steps: 2, 4, 6"):
        sampling.checkpoint_from_run(data, run, step=3)
    _json_dump(run / "validations.json", [{"epoch": 1, "step": 2, "sampler_weights_uri": "tinker://other-run/sampler_weights/x"}])
    with pytest.raises(PipelineError, match="best sampler checkpoint"):
        sampling.checkpoint_from_run(data, run, step=2)


def test_evaluate_writes_a_selected_step_to_its_own_output(tmp_path, monkeypatch, capsys):
    from smithtune import cli

    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data, model=DEFAULT_MODEL)
    _json_dump(run / "plan.json", {"provider": "tinker", "base_model": DEFAULT_MODEL.base_model})
    step_one = "tinker://tinker-run-1/sampler_weights/smoke-sampler-epoch-1"
    _json_dump(run / "result.json", {"provider": "tinker", "tinker_run_id": "tinker-run-1", "best_sampler_weights_uri": CHECKPOINT})
    _json_dump(run / "validations.json", [{"epoch": 1, "step": 2, "sampler_weights_uri": step_one}])
    calls = []
    monkeypatch.setattr(sampling, "TinkerReplaySampler", lambda *args, **kwargs: calls.append(("sampler", args)) or "sampler")
    monkeypatch.setattr(cli.replay_evaluation, "validate_judge_credentials", lambda judge: None)
    monkeypatch.setattr(cli.replay_evaluation, "run_replay_evaluation",
                        lambda *args, **kwargs: calls.append(("run", args, kwargs)) or {"status": "completed"})

    cli.main(["evaluate", "--provider", "tinker", "--run-dir", str(run), "--data-dir", str(data),
              "--checkpoint-step", "2", "--confirm"])
    capsys.readouterr()
    _, args, kwargs = calls[-1]
    assert args[1] == run / "replay-step-2" and args[2] == step_one
    assert kwargs["training"]["checkpoint_step"] == 2
    assert calls[0] == ("sampler", (sampling.checkpoint_from_run(data, run, step=2)[0], step_one, run / "replay-step-2"))

"""Answers generated in the job feed the shared replay evaluation; no serving resources."""
from types import SimpleNamespace

import pytest

from smithtune import cli, rendering
from smithtune.artifacts import _json_dump, _load_json
from smithtune.evaluation import replay as evaluation
from smithtune.providers import sagemaker as aws
from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker.artifacts import collect, package_artifact
from smithtune.providers.sagemaker.provider import submit_evaluation_job
from smithtune.providers.sagemaker.sampling import PrecomputedSampler, checkpoint_from_run
from smithtune.training import hf_sft as worker
from smithtune.training.contracts import verify_artifact
from smithtune.training.evaluation import request_sha256, request_stop, reset_stop
from tests.test_sagemaker import config, prepared
from tests.test_sagemaker_end_to_end import Cloud
from tests.test_sagemaker_worker import tiny_qwen

REPLAY = {"judge_model": "anthropic/test-judge", "concurrency": 1, "max_points_per_trajectory": None, "max_output_tokens": 4}


@pytest.fixture(autouse=True)
def clean_stop_flag():
    reset_stop()
    yield
    reset_stop()
ARCHIVE_KEY = ("smithtune-test", "jobs/sft-test/output/sft-test/output/model.tar.gz")
CHECKPOINT_PREFIX = "jobs/sft-test/checkpoints/evaluation/"


def fake_renderer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<unk>": 0, "hello": 1, "<|im_end|>": 2, **{f"token{i}": i for i in range(3, 32)}}, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|im_end|>")
    return SimpleNamespace(
        tokenizer=tokenizer, prompt_tokens=lambda messages, tools=None: [1, 3, 4][:max(1, len(messages))],
        render=lambda messages, tools=None, final_target=False: [SimpleNamespace(token_ids=[1, 3, 4, 2], token_weights=[0, 1, 0, 1])],
    )


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Plan and submit with replay; fake the model, renderer and AWS."""
    from transformers import AutoModelForImageTextToText
    from smithtune.providers.baseten import sampling_formats

    prepared(tmp_path)
    renderer = fake_renderer()
    monkeypatch.setattr(rendering, "load_training_renderer", lambda _: renderer)
    monkeypatch.setattr(worker, "load_training_renderer", lambda _: renderer)
    def model_factory(name, **_kwargs):
        model = tiny_qwen()
        model.config._name_or_path = name
        model.name_or_path = name
        return model

    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", model_factory)
    monkeypatch.setattr(sampling_formats, "stop_sequences", lambda renderer, spec: ["<|im_end|>"])
    remote = Cloud()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    settings = aws.SageMakerSFTSettings(aws=config(), max_epochs=1, batch_size=2)
    plan = provider.plan(tmp_path, "sft-test", settings, replay=REPLAY)
    assert plan["evaluation"]["cases"] == 2 and plan["evaluation"]["answers_generated_in_job"] == 4
    assert plan["evaluation"]["judge_model"] == "anthropic/test-judge"
    assert set(plan["input_sha256"]) >= {"evaluation/cases.jsonl", "evaluation/settings.json"}
    assert plan["create_training_job"]["Environment"] == {"SMITHTUNE_MAX_RUNTIME_SECONDS": "600"}
    run = tmp_path / "run"
    provider.train(tmp_path, run, "sft-test", settings, confirm=True, replay=REPLAY)
    assert (run / "input/evaluation/cases.jsonl").exists()
    return SimpleNamespace(data=tmp_path, run=run, remote=remote, provider=provider, renderer=renderer)


def finish_job(staged, output):
    archive = staged.data / "model.tar.gz"
    package_artifact(output, archive)
    staged.remote.objects[ARCHIVE_KEY] = archive.read_bytes()
    staged.remote.completed = True


def stopping_generation(monkeypatch):
    """Deterministic stop plus a well-formed parse; the parser has its own tests."""
    from smithtune.providers.baseten import sampling_formats
    from smithtune.training import inference

    monkeypatch.setattr(inference, "generate_tokens", lambda model, tokens, **k: {
        "tokens": [5, 2], "prompt_tokens": len(tokens), "output_tokens": 2, "elapsed_seconds": 0.0, "stop_reason": "stop"})
    monkeypatch.setattr(sampling_formats, "parse_completion", lambda renderer, spec, tokens, stop_reason, tools=None: {
        "role": "assistant", "content": "token5", "sampling": {"format_valid": True, "stop_reason": stop_reason}})


def test_in_job_answers_feed_the_shared_replay_without_serving(staged, monkeypatch):
    stopping_generation(monkeypatch)
    output = staged.data / "model"
    result = worker.run(staged.run / "input", output, staged.data / "checkpoints", device="cpu")
    assert result["evaluation"] == {"status": "complete", "completed": 4, "total": 4}
    assert (staged.data / "checkpoints/evaluation/predictions.jsonl").exists()
    assert (output / "evaluation/predictions.jsonl").exists() and (output / "evaluation/cases.jsonl").exists()
    verify_artifact(output)
    finish_job(staged, output)
    collected = collect(staged.run, provider=staged.provider)
    assert collected["evaluation"]["status"] == "complete" and collected["training"]["status"] == "complete"
    assert _load_json(staged.run / "result.json")["best_epoch"] == 1
    model, checkpoint, artifact_dir, summary = checkpoint_from_run(staged.data, staged.run)
    assert checkpoint == "sagemaker://sft-test/adapter@epoch-1" and summary["status"] == "complete"
    sampler = PrecomputedSampler(model, checkpoint, artifact_dir)
    assert sampler.config["provider"] == "sagemaker" and sampler.config["serving_mode"] == "precomputed"
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    calls = []
    monkeypatch.setattr(evaluation, "calibrate_judge", lambda cases, *_: [{"actual": True, "expected": True}] * (len(evaluation._calibration_cases(cases)) * 3))
    monkeypatch.setattr(evaluation, "judge_replay_candidate", lambda case, candidate, *_: calls.append(candidate) or {"pass": True, "reason": "ok"})
    summary = evaluation.run_replay_evaluation(
        staged.data, staged.data / "replay", checkpoint, "anthropic/test-judge", base_model=model.base_model,
        max_output_tokens=4, replay_sampler=sampler, confirm=True, concurrency=1,
        training=evaluation.training_metadata(staged.run),
    )
    assert summary["serving_mode"] == "precomputed" and summary["tuned_pass_rate"] == 1
    assert len(calls) == 4 and all(candidate["sampling"]["precomputed"] for candidate in calls)
    generations = [line for line in (staged.data / "replay/generations.jsonl").read_text().splitlines() if line]
    assert len(generations) == 4
    # The CLI path uses the saved generation settings and refuses mismatched overrides.
    args = cli._parser().parse_args(["evaluate", "--provider", "sagemaker", "--data-dir", str(staged.data),
                                     "--run-dir", str(staged.run), "--max-output-tokens", "9", "--confirm"])
    args.output_dir = staged.data / "replay"
    with pytest.raises(PipelineError, match="max_output_tokens=4"):
        cli._run_sagemaker_evaluation(args)


def test_generation_failure_or_stop_never_fails_training(staged, monkeypatch):
    from smithtune.training import inference

    def fail(*_a, **_k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(inference, "generate_tokens", fail)
    output = staged.data / "model"
    result = worker.run(staged.run / "input", output, staged.data / "checkpoints", device="cpu")
    assert result["best_epoch"] == 1
    assert result["evaluation"]["status"] == "failed" and result["evaluation"]["error_kind"] == "out_of_memory"
    assert result["evaluation"]["completed"] == 0
    verify_artifact(output)
    finish_job(staged, output)
    collect(staged.run, provider=staged.provider)
    args = cli._parser().parse_args(["evaluate", "--provider", "sagemaker", "--data-dir", str(staged.data),
                                     "--run-dir", str(staged.run), "--confirm"])
    args.output_dir = staged.data / "replay"
    with pytest.raises(PipelineError, match="is failed .0 of 4 answers.*evaluate-job"):
        cli._run_sagemaker_evaluation(args)
    # A stop request ends generation after the current answer and records it as incomplete.
    monkeypatch.setattr(inference, "generate_tokens", lambda *a, **k: request_stop() or {
        "tokens": [5, 2], "prompt_tokens": 3, "output_tokens": 2, "elapsed_seconds": 0.0, "stop_reason": "stop"})
    result = worker.run(staged.run / "input", staged.data / "model2", staged.data / "checkpoints2", device="cpu")
    assert result["evaluation"] == {"status": "incomplete", "completed": 1, "total": 4, "reason": "stop_requested"}


def test_collect_prefers_more_complete_answers_from_the_checkpoint_prefix(staged, monkeypatch):
    output = staged.data / "model"
    result = worker.run(staged.run / "input", output, staged.data / "checkpoints", device="cpu")
    assert result["evaluation"]["status"] == "complete"
    predictions = (staged.data / "checkpoints/evaluation/predictions.jsonl").read_text()
    summary = (staged.data / "checkpoints/evaluation/evaluation.json").read_bytes()
    # Simulate an archive uploaded before generation finished: the archive carries 2 of 4 answers.
    partial = "".join(line + "\n" for line in predictions.splitlines()[:2])
    (output / "evaluation/predictions.jsonl").write_text(partial)
    _json_dump(output / "evaluation/evaluation.json", {"schema_version": 1, "status": "incomplete", "completed": 2, "total": 4,
                                                       "settings": {"max_new_tokens": 4, "max_cases": 50, "max_points_per_trajectory": None}})
    _json_dump(output / "result.json", {**_load_json(output / "result.json"), "evaluation": {"status": "incomplete", "completed": 2, "total": 4}})
    from smithtune.training.contracts import write_artifact_manifest
    write_artifact_manifest(output)
    finish_job(staged, output)
    staged.remote.objects[("smithtune-test", CHECKPOINT_PREFIX + "predictions.jsonl")] = predictions.encode()
    staged.remote.objects[("smithtune-test", CHECKPOINT_PREFIX + "evaluation.json")] = summary
    collected = collect(staged.run, provider=staged.provider)
    assert collected["evaluation"] == {"status": "complete", "completed": 4, "total": 4}
    assert (staged.run / "artifact/evaluation/predictions.jsonl").read_text() == predictions
    verify_artifact(staged.run / "artifact")
    assert _load_json(staged.run / "result.json")["evaluation"]["status"] == "complete"


def test_evaluation_only_job_resumes_the_run_checkpoint_prefix(staged, monkeypatch):
    output = staged.data / "model"
    worker.run(staged.run / "input", output, staged.data / "checkpoints", device="cpu")
    finish_job(staged, output)
    collect(staged.run, provider=staged.provider)
    with pytest.raises(PipelineError, match="--confirm"):
        submit_evaluation_job(staged.run, confirm=False, provider=staged.provider)
    receipt = submit_evaluation_job(staged.run, confirm=True, provider=staged.provider)
    job_dir = staged.run / "evaluation-jobs" / receipt["job_name"]
    request = staged.remote.calls[-1][1]
    assert staged.remote.calls[-1][0] == "create"
    assert request["CheckpointConfig"]["S3Uri"] == "s3://smithtune-test/jobs/sft-test/checkpoints"
    assert request["Environment"]["SMITHTUNE_WORKER_COMMAND"] == "evaluate"
    assert request["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"].endswith(f"/{receipt['job_name']}/input/")
    assert staged.provider.status(job_dir)["job_name"] == receipt["job_name"]
    verify_artifact(job_dir / "input/export")
    # The staged export and the restored checkpoint prefix let the worker finish without training.
    checkpoints = staged.data / "resumed-checkpoints"
    (checkpoints / "evaluation").mkdir(parents=True)
    lines = (staged.data / "checkpoints/evaluation/predictions.jsonl").read_text().splitlines()
    (checkpoints / "evaluation/predictions.jsonl").write_text("\n".join(lines[:3]) + "\n")
    result = worker.run_evaluation_only(job_dir / "input", staged.data / "model-eval", checkpoints, device="cpu")
    assert result["evaluation"] == {"status": "complete", "completed": 4, "total": 4}
    assert result["evaluation_job"]["source_run_id"] == "sft-test"
    verify_artifact(staged.data / "model-eval")
    assert (staged.data / "model-eval/adapter/adapter_model.safetensors").read_bytes() == (output / "adapter/adapter_model.safetensors").read_bytes()


def test_precomputed_sampler_serves_only_saved_answers(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "evaluation").mkdir(parents=True)
    messages = [{"role": "user", "content": "hello"}]
    records = [
        {"schema_version": 1, "case_id": "c1", "variant": "tuned", "request_sha256": request_sha256(messages, "tuned", 8),
         "candidate": {"role": "assistant", "content": "tuned answer", "sampling": {"format_valid": True}},
         "prompt_tokens": 3, "output_tokens": 2, "stop_reason": "stop"},
        {"schema_version": 1, "case_id": "c1", "variant": "base", "request_sha256": request_sha256(messages, "base", 8),
         "candidate": None, "skipped": "prompt plus output budget exceeds the model context", "prompt_tokens": 3},
    ]
    import json
    (artifact / "evaluation/predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    _json_dump(artifact / "evaluation/evaluation.json", {"schema_version": 1, "status": "complete", "completed": 2, "total": 2,
                                                         "settings": {"max_new_tokens": 8, "max_cases": 50, "max_points_per_trajectory": None}})
    model = aws.MODEL_SPECS["qwen3p5-9b"]
    sampler = PrecomputedSampler(model, "sagemaker://run/adapter@epoch-2", artifact)
    with sampler as route:
        assert route == "sagemaker://run/adapter@epoch-2"
    tuned = sampler.generate("sagemaker://run/adapter@epoch-2", messages, 8)
    assert tuned["content"] == "tuned answer" and tuned["sampling"]["checkpoint"] == "sagemaker://run/adapter@epoch-2"
    base = sampler.generate(model.base_model, messages, 8)
    assert base["sampling"]["format_valid"] is False and base["sampling"]["skipped"]
    for arguments, match in (
        (("other", messages, 8), "only its checkpoint"),
        ((model.base_model, messages, 9), "output budget differs"),
        ((model.base_model, [{"role": "user", "content": "bye"}], 8), "no saved answer"),
    ):
        with pytest.raises(PipelineError, match=match):
            sampler.generate(*arguments)
    with pytest.raises(PipelineError, match="JSON mode"):
        sampler.generate(model.base_model, messages, 8, True)

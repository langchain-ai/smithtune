"""Bounded context probes preserve the training objective; no downloads or AWS."""

from copy import deepcopy

import pytest

from smithtune.artifacts import _json_dump
from smithtune.providers.base import CommonSFTSettings, PipelineError
from smithtune.providers.sagemaker import sha256_file
from smithtune.providers.sagemaker import benchmark
from smithtune.training.hf_sft import Example, configure_lora, train_loop
from tests.test_sagemaker_worker import ScalarModel, tiny_qwen

torch = pytest.importorskip("torch")


def test_benchmark_optimizer_matches_real_worker_over_full_conversation():
    model = configure_lora(tiny_qwen(), 2)
    reference = deepcopy(model)
    for item in (model, reference):
        item.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    examples = [Example([1, 2, 3, 4], [-100, 2, -100, 4]),
                Example([1, 2, 3, 4, 5], [-100, -100, -100, -100, 5])]
    settings = CommonSFTSettings(max_epochs=1, batch_size=2, learning_rate=0.01)
    expected = train_loop(reference, [examples], [examples], settings, device="cpu")
    actual = benchmark.optimizer_group(model, examples, device="cpu", learning_rate=settings.learning_rate)
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 1
    assert actual["loss"] == pytest.approx(expected["epochs"][0]["train_loss"])
    for value, target in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(value, target)


def test_each_stage_starts_fresh_and_preserves_all_assistant_targets():
    model = ScalarModel()
    examples = [Example([1, 20, 30], [-100, 20, 30]), Example([1, 40, 50], [-100, -100, 50])]
    stages = benchmark.run_probes(model, [(4096, examples), (8192, examples)],
                                  CommonSFTSettings(learning_rate=0.01), device="cpu")
    assert all(stage["status"] == "passed" and stage["optimizer_steps"] == 1 for stage in stages)
    assert stages[0]["loss"] == stages[1]["loss"]
    assert stages[0]["gradient_norm"] == stages[1]["gradient_norm"]
    assert all(stage["assistant_target_tokens"] == 3 and stage["datum_tokens"] == [3, 3] for stage in stages)
    assert stages[0]["gradient_norm"] == pytest.approx(0.46666667)


@pytest.mark.parametrize(("error", "reason"), [
    (torch.OutOfMemoryError("PRIVATE SYNTHETIC CONTENT"), "out_of_memory"),
    (PipelineError("non-finite scalar loss PRIVATE SYNTHETIC CONTENT"), "non_finite_training"),
])
def test_failed_probe_records_safe_diagnostic_and_skips_larger_sizes(monkeypatch, error, reason):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(benchmark, "optimizer_group", fail)
    item = Example([1, 2], [-100, 2])
    snapshots = []
    stages = benchmark.run_probes(ScalarModel(), [(4096, [item]), (8192, [item]), (16384, [item])],
                                  CommonSFTSettings(), device="cpu", on_progress=lambda rows: snapshots.append(deepcopy(rows)))
    assert len(calls) == 1 and len(snapshots) == 3
    assert [stage["status"] for stage in stages] == ["failed", "skipped", "skipped"]
    assert all(stage["reason"] == reason and stage["optimizer_steps"] == 0 for stage in stages)
    assert "PRIVATE SYNTHETIC CONTENT" not in str(stages)


def test_exhausted_budget_does_not_run_any_step():
    item = Example([1, 2], [-100, 2])
    stages = benchmark.run_probes(ScalarModel(), [(4096, [item])], CommonSFTSettings(), device="cpu", max_seconds=0)
    assert stages[0]["status"] == "skipped"
    assert stages[0]["reason"] == "runtime_budget_exhausted"


def test_conditional_16k_requires_both_time_and_memory_headroom():
    gib = 1024**3
    previous = {"elapsed_seconds": 10, "peak_allocated_bytes": 25 * gib}
    args = {"baseline_bytes": 18 * gib, "total_bytes": 48 * gib, "remaining_seconds": 200}
    assert benchmark.conditional_skip(previous, **args) is None
    assert benchmark.conditional_skip(previous | {"peak_allocated_bytes": 35 * gib}, **args) == "insufficient_estimated_memory_headroom"
    assert benchmark.conditional_skip(previous, **(args | {"remaining_seconds": 30})) == "insufficient_runtime_headroom"


def test_rendered_cases_verify_lengths_targets_and_preserve_full_conversation(tmp_path, monkeypatch):
    from smithtune.providers.sagemaker import MODEL_SPECS

    cases = [{"ceiling": ceiling, "max_rendered_tokens": ceiling - 1, "datum_count": 2,
              "assistant_target_tokens": 3} for ceiling in (4096, 8192, 16384)]
    rows = []
    for case in cases:
        row = {"synthetic_only": True, "messages": [{"role": "tool", "content": "complete context"}],
               "tools": [], "ceiling": case["ceiling"]}
        _json_dump(tmp_path / f"trajectory-{case['ceiling']}.json", row)

    def render(conversations, spec, renderer):
        row = conversations[0]
        rows.append(row)
        length = row["ceiling"] - 1
        return [[Example([1, 2, 3], [-100, 2, 3]), Example([1] * length, [-100] * (length - 1) + [1])]]

    monkeypatch.setattr(benchmark, "render_conversations", render)
    result = benchmark.render_cases(tmp_path, {"cases": cases}, MODEL_SPECS["qwen3p5-9b"], None)
    assert len(result) == 3 and all(row["messages"][0]["content"] == "complete context" for row in rows)
    with pytest.raises(PipelineError, match="measured provenance"):
        benchmark.render_cases(tmp_path, {"cases": [cases[0] | {"assistant_target_tokens": 2}]}, None, None)


def test_manifest_checksums_are_verified_before_model_load(tmp_path):
    from tests.test_sagemaker import FakeAWS, config, prepared
    from smithtune.providers import sagemaker

    prepared(tmp_path)
    provider = sagemaker.SageMakerProvider(session_factory=lambda _: FakeAWS())
    provider.train(tmp_path, tmp_path / "run", "context-test", sagemaker.SageMakerSFTSettings(aws=config()), confirm=True)
    channel = tmp_path / "run/input"
    cases = [{"ceiling": value} for value in (4096, 8192, 16384)]
    for case in cases:
        _json_dump(channel / f"trajectory-{case['ceiling']}.json", {"synthetic_only": True})
    files = {path.relative_to(channel).as_posix(): sha256_file(path) for path in channel.rglob("*") if path.is_file()}
    _json_dump(channel / "benchmark.json", {"schema_version": 1, "synthetic_only": True, "cases": cases, "files": files})
    _, _, spec, _, _ = benchmark.read_cases(channel)
    assert spec.training_context_limit == 16384
    (channel / "trajectory-8192.json").write_text("tampered")
    with pytest.raises(PipelineError, match="checksum"):
        benchmark.read_cases(channel)


def test_no_weight_change_or_nonfinite_loss_fails_the_probe(monkeypatch):
    item = Example([1, 20], [-100, 20])
    model = ScalarModel()
    monkeypatch.setattr(benchmark, "_loss", lambda model, *args: model.value * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        benchmark.optimizer_group(model, [item], device="cpu", learning_rate=0.01)
    assert model.value.grad is None


def test_unexpected_failures_are_not_mislabeled_as_memory_limits(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected runtime error")

    monkeypatch.setattr(benchmark, "optimizer_group", fail)
    with pytest.raises(RuntimeError, match="unexpected"):
        benchmark.run_probes(ScalarModel(), [(4096, [Example([1, 2], [-100, 2])])], CommonSFTSettings(), device="cpu")

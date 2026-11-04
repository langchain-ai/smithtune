"""In-job replay generation: durable records, resume, bounded stops and failure isolation."""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from smithtune.providers.base import PipelineError
from smithtune.inference_contract import json_sha256
from smithtune.training import evaluation
from smithtune.training.evaluation import (
    EvaluationSettings, generate_predictions, load_predictions, request_sha256, request_stop, reset_stop,
)


class Model:
    def __init__(self):
        self.adapter_enabled = True
        self.calls = []

    def eval(self):
        return self

    @contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


def renderer():
    return SimpleNamespace(prompt_tokens=lambda messages, tools=None: [1] * len(messages),
                           tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=False: [31]))


SPEC = SimpleNamespace(max_seq_len=16, renderer="hf_prefix_qwen3_5")
CASES = [{"id": "a", "messages": [{"role": "user", "content": "one"}], "tools": []},
         {"id": "b", "messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "x"},
                                  {"role": "user", "content": "three"}], "tools": [{"type": "function", "function": {"name": "f"}}]}]


@pytest.fixture(autouse=True)
def clean_stop_flag():
    reset_stop()
    yield
    reset_stop()


def fakes(model):
    def generate(m, tokens, *, device, eos_token_id, max_new_tokens):
        model.calls.append(("base" if not model.adapter_enabled else "tuned", len(tokens)))
        return {"tokens": [7, 8, eos_token_id], "prompt_tokens": len(tokens), "output_tokens": 3,
                "elapsed_seconds": 0.0, "stop_reason": "stop"}

    def parse(r, spec, tokens, stop_reason, tools=None):
        return {"role": "assistant", "content": f"answer {len(tokens)}", "sampling": {"format_valid": True}}
    return generate, parse


def _stop_sequences(monkeypatch):
    from smithtune.providers.baseten import sampling_formats
    monkeypatch.setattr(sampling_formats, "stop_sequences", lambda renderer, spec: ["<stop>"])


@pytest.mark.parametrize("tail", ["torn", "missing_newline"])
def test_base_then_tuned_records_are_durable_and_resume_after_a_torn_line(tmp_path, monkeypatch, tail):
    _stop_sequences(monkeypatch)
    model = Model()
    generate, parse = fakes(model)
    output = tmp_path / "predictions.jsonl"
    settings = EvaluationSettings(max_new_tokens=4, max_cases=10)
    status = generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu", output_path=output,
                                  generate=generate, parse=parse)
    assert status == {"status": "complete", "completed": 4, "total": 4}
    assert [call[0] for call in model.calls] == ["base", "base", "tuned", "tuned"]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r["case_id"], r["variant"]) for r in records] == [("a", "base"), ("b", "base"), ("a", "tuned"), ("b", "tuned")]
    assert records[0]["request_sha256"] != records[2]["request_sha256"]
    assert records[0]["request_sha256"] == request_sha256(CASES[0]["messages"], "base", 4)
    assert records[0]["candidate"]["content"] == "answer 3" and records[0]["truncated"] is False
    # A hard stop mid-write leaves a torn final line; resume repairs and completes only what is missing.
    text = output.read_text().splitlines()
    output.write_text("\n".join(text[:3]) + ("\n" + text[3][:20] if tail == "torn" else ""))
    model.calls.clear()
    status = generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu", output_path=output,
                                  generate=generate, parse=parse)
    assert status["status"] == "complete" and [call[0] for call in model.calls] == ["tuned"]
    assert len(load_predictions(output)) == 4 and len(output.read_text().splitlines()) == 4


def test_corrupt_middle_line_is_an_error_not_a_silent_skip(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"schema_version": 1, "case_id": "a", "variant": "base", "request_sha256": "x"}\nnot json\n{"schema_version": 1, "case_id": "b", "variant": "base", "request_sha256": "y"}\n')
    with pytest.raises(PipelineError, match="corrupt"):
        load_predictions(path)


@pytest.mark.parametrize("change", ["messages", "tools", "budget", "cases"])
def test_resume_rejects_changed_requests_before_generating(tmp_path, monkeypatch, change):
    _stop_sequences(monkeypatch)
    model = Model()
    generate, parse = fakes(model)
    output = tmp_path / "predictions.jsonl"
    settings = EvaluationSettings(max_new_tokens=4, max_cases=10)
    generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu", output_path=output,
                         generate=generate, parse=parse)
    saved = output.read_bytes()
    model.calls.clear()
    cases = deepcopy(CASES)
    if change == "messages":
        cases[1]["messages"][0]["content"] = "changed"
    elif change == "tools":
        cases[1]["tools"][0]["function"]["name"] = "different_tool"
    elif change == "budget":
        settings = EvaluationSettings(max_new_tokens=5, max_cases=10)
    else:
        cases.pop()
    with pytest.raises(PipelineError, match="saved prediction"):
        generate_predictions(model, renderer(), SPEC, cases, settings, device="cpu", output_path=output,
                             generate=generate, parse=parse)
    assert not model.calls and output.read_bytes() == saved


def test_duplicate_prediction_records_cannot_silently_replace_an_answer(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = {"schema_version": 1, "case_id": "a", "variant": "base", "request_sha256": "x"}
    path.write_text(json.dumps(record) + "\n" + json.dumps(record | {"candidate": "different"}) + "\n")
    with pytest.raises(PipelineError, match="duplicate"):
        load_predictions(path)


@pytest.mark.parametrize("original_cases", [None, CASES])
def test_legacy_resume_needs_original_exported_tools(tmp_path, monkeypatch, original_cases):
    _stop_sequences(monkeypatch)
    model = Model()
    generate, parse = fakes(model)
    output = tmp_path / "predictions.jsonl"
    record = {"schema_version": 1, "case_id": "b", "variant": "base", "candidate": None,
              "request_sha256": json_sha256({"messages": CASES[1]["messages"], "variant": "base", "max_new_tokens": 4})}
    output.write_text(json.dumps(record) + "\n")
    saved = output.read_bytes()
    cases = deepcopy(CASES)
    if original_cases is not None:
        cases[1]["tools"][0]["function"]["name"] = "changed"
    with pytest.raises(PipelineError, match="legacy saved prediction"):
        generate_predictions(model, renderer(), SPEC, cases, EvaluationSettings(max_new_tokens=4, max_cases=10),
                             device="cpu", output_path=output, generate=generate, parse=parse, legacy_cases=original_cases)
    assert not model.calls and output.read_bytes() == saved


def test_deadline_and_stop_request_end_generation_as_incomplete(tmp_path, monkeypatch):
    _stop_sequences(monkeypatch)
    model = Model()
    generate, parse = fakes(model)
    settings = EvaluationSettings(max_new_tokens=4, max_cases=10)
    ticks = iter([0, 0, 100, 100, 100])
    status = generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu",
                                  output_path=tmp_path / "p.jsonl", deadline=50, clock=lambda: next(ticks),
                                  generate=generate, parse=parse)
    assert status == {"status": "incomplete", "completed": 2, "total": 4, "reason": "time_budget_exhausted"}
    request_stop()
    status = generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu",
                                  output_path=tmp_path / "p.jsonl", generate=generate, parse=parse)
    assert status == {"status": "incomplete", "completed": 2, "total": 4, "reason": "stop_requested"}
    assert len(load_predictions(tmp_path / "p.jsonl")) == 2


def test_prompts_that_cannot_fit_the_output_budget_are_recorded_as_skipped(tmp_path, monkeypatch):
    _stop_sequences(monkeypatch)
    model = Model()
    generate, parse = fakes(model)
    settings = EvaluationSettings(max_new_tokens=15, max_cases=10)
    status = generate_predictions(model, renderer(), SPEC, CASES, settings, device="cpu",
                                  output_path=tmp_path / "p.jsonl", generate=generate, parse=parse)
    assert status["status"] == "complete"
    records = load_predictions(tmp_path / "p.jsonl")
    assert records[("a", "base")]["candidate"]["content"] == "answer 3"
    assert records[("b", "base")]["candidate"] is None and "exceeds" in records[("b", "base")]["skipped"]
    assert [call[0] for call in model.calls] == ["base", "tuned"]


def test_generation_deadline_leaves_an_upload_margin():
    from smithtune.providers.sagemaker.worker import EVALUATION_MARGIN_SECONDS, generation_deadline

    assert generation_deadline(100.0, {}) is None
    assert generation_deadline(100.0, {"SMITHTUNE_MAX_RUNTIME_SECONDS": "3600"}) == 100.0 + 3600 - EVALUATION_MARGIN_SECONDS
    assert generation_deadline(100.0, {"SMITHTUNE_MAX_RUNTIME_SECONDS": "90"}) == 160.0
    assert generation_deadline(100.0, {"SMITHTUNE_MAX_RUNTIME_SECONDS": "soon"}) is None


def test_settings_reject_unbounded_or_invalid_limits():
    EvaluationSettings(max_new_tokens=128, max_cases=5).validate()
    for values in ({"max_new_tokens": 0, "max_cases": 5}, {"max_new_tokens": 128, "max_cases": 501},
                   {"max_new_tokens": True, "max_cases": 5}, {"max_new_tokens": 128, "max_cases": 5, "max_points_per_trajectory": 0}):
        with pytest.raises(PipelineError):
            EvaluationSettings(**values).validate()
    assert evaluation.evaluation_status(3, 4) == {"status": "incomplete", "completed": 3, "total": 4}
    assert evaluation.evaluation_status(1, 4, error_kind="out_of_memory")["status"] == "failed"

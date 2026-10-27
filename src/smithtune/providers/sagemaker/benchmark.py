"""Bounded synthetic context sizing using the actual single-GPU SFT objective.

Experimental worker, not a production training command. One optimizer group per
full conversation; no validation epochs, checkpoint export, endpoint or judges.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time

from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker import sha256_file
from smithtune.training.contracts import model_revision
from smithtune.training.hf_sft import (
    _loss, configure_lora, enable_chunked_loss, failure_diagnostics, gpu_diagnostics, read_bundle,
    render_conversations, report_stage,
)
from smithtune.rendering import load_training_renderer


def read_cases(input_dir: Path):
    metadata = _load_json(input_dir / "benchmark.json")
    if metadata.get("schema_version") != 1 or metadata.get("synthetic_only") is not True:
        raise PipelineError("context benchmark requires a synthetic manifest")
    cases = metadata.get("cases", [])
    if [case.get("ceiling") for case in cases] != [4096, 8192, 16384]:
        raise PipelineError("benchmark stages must be 4K, 8K and conditional 16K")
    files = metadata.get("files", {})
    expected_files = {"bundle.json", *(f"prepared/{name}" for name in (
        "manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")),
        *(f"trajectory-{case['ceiling']}.json" for case in cases)}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise PipelineError("benchmark must checksum every bundle and synthetic case file")
    for name, expected in files.items():
        if sha256_file(input_dir / name) != expected:
            raise PipelineError("benchmark input checksum mismatch")
    bundle, spec, _rows, settings, rank = read_bundle(input_dir)
    # Change only the explicit context ceiling, keeping the pinned model/template.
    spec = replace(spec, max_seq_len=16384, trainer_max_seq_len=16384)
    spec.validate()
    return metadata, bundle, spec, settings, rank


def _synthetic_row(row: dict) -> dict:
    """Synthetic cases have no per-assistant provenance; declare their tools globally."""
    if "_source" in row:
        return row
    return {**row, "tool_policy": "global_override", "tools": row.get("tools", []),
            "_source": {"example_id": "synthetic", "message_positions": list(range(len(row["messages"])))}}


def render_cases(input_dir, metadata, spec, renderer):
    cases = []
    for case in metadata["cases"]:
        row = _load_json(input_dir / f"trajectory-{case['ceiling']}.json")
        if row.get("synthetic_only") is not True:
            raise PipelineError("benchmark case is not marked synthetic")
        examples = render_conversations([_synthetic_row(row)], spec, renderer)[0]
        lengths = [len(item.input_ids) for item in examples]
        targets = sum(item.active_tokens for item in examples)
        if (max(lengths) != case["max_rendered_tokens"] or len(examples) != case["datum_count"]
                or targets != case["assistant_target_tokens"]
                or not case["ceiling"] - 64 <= max(lengths) <= case["ceiling"]):
            raise PipelineError("synthetic case rendering differs from measured provenance")
        cases.append((case["ceiling"], examples))
    return cases


def optimizer_group(model, examples, *, device, learning_rate):
    """Same loss, token normalization, AdamW and clipping as the training worker."""
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    denominator = sum(item.active_tokens for item in examples)
    if not parameters or not examples or not denominator:
        raise PipelineError("benchmark needs trainable parameters and assistant targets")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    try:
        for example in examples:
            loss = _loss(model, example, device)
            (loss * (example.active_tokens / denominator)).backward()
            loss_sum += float(loss.detach()) * example.active_tokens
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
            raise PipelineError("non-finite adapter weights after benchmark step")
        return {"loss": loss_sum / denominator, "gradient_norm": float(norm), "optimizer_steps": 1}
    finally:
        optimizer.zero_grad(set_to_none=True)


def conditional_skip(previous: dict, *, baseline_bytes, total_bytes, remaining_seconds):
    """Conservative estimate, not a claim that activation memory scales linearly."""
    if remaining_seconds < max(90, 4 * previous["elapsed_seconds"]):
        return "insufficient_runtime_headroom"
    projected = baseline_bytes + 2 * max(0, previous["peak_allocated_bytes"] - baseline_bytes)
    if projected + 2 * 1024**3 > 0.9 * total_bytes:
        return "insufficient_estimated_memory_headroom"
    return None


def run_probes(model, cases, settings, *, device, max_seconds=450, on_progress=None):
    import torch
    from transformers import set_seed

    gpu = str(device) != "cpu"
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    initial = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    baseline = torch.cuda.memory_allocated(device) if gpu else 0
    total = torch.cuda.get_device_properties(device).total_memory if gpu else 0
    started = time.monotonic()
    records = []
    stop_reason = None
    for index, (ceiling, examples) in enumerate(cases):
        record = {"ceiling": ceiling, "datum_tokens": [len(item.input_ids) for item in examples],
                  "assistant_target_tokens": sum(item.active_tokens for item in examples), "optimizer_steps": 0}
        if not stop_reason and time.monotonic() - started >= max_seconds:
            stop_reason = "runtime_budget_exhausted"
        if not stop_reason and index == 2 and gpu:
            stop_reason = conditional_skip(records[-1], baseline_bytes=baseline, total_bytes=total,
                                           remaining_seconds=max_seconds - (time.monotonic() - started))
        if stop_reason:
            records.append(record | {"status": "skipped", "reason": stop_reason})
            if on_progress:
                on_progress(records)
            continue
        # Each stage starts from identical adapters, seed and fresh optimizer state.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(initial[name])
                parameter.grad = None
        set_seed(settings.seed)
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        report_stage(f"benchmark_{ceiling}")
        step_started = time.monotonic()
        try:
            record.update(optimizer_group(model, examples, device=device, learning_rate=settings.learning_rate))
            changed = any(not torch.equal(parameter.detach().cpu(), initial[name]) for name, parameter in parameters.items())
            if not changed:
                raise PipelineError("benchmark step did not change adapter weights")
            if gpu:
                torch.cuda.synchronize(device)
            record.update(status="passed", adapter_changed=True)
        except Exception as exc:
            diagnostic = failure_diagnostics(exc)
            if isinstance(exc, torch.OutOfMemoryError):
                stop_reason = "out_of_memory"
            elif diagnostic["error_kind"] == "non_finite_gradient" or "non-finite" in str(exc).lower():
                stop_reason = "non_finite_training"
            else:
                raise
            record.update(status="failed", reason=stop_reason, diagnostic=diagnostic)
        record.update(elapsed_seconds=time.monotonic() - step_started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if gpu else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if gpu else 0)
        records.append(record)
        print(json.dumps({"smithtune_event": "benchmark_result", **record}), flush=True)
        if on_progress:
            on_progress(records)
        if stop_reason:
            for parameter in parameters.values():
                parameter.grad = None
            gc.collect()
            if gpu:
                torch.cuda.empty_cache()
    return records


def run(input_dir: Path, model_dir: Path, *, device="cuda:0"):
    import torch
    from transformers import AutoModelForImageTextToText, set_seed

    report_stage("verify_benchmark_inputs")
    metadata, bundle, spec, settings, rank = read_cases(input_dir)
    if device != "cpu" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise PipelineError("context benchmark requires CUDA bfloat16 support")
    report_stage("render_benchmark_cases")
    renderer = load_training_renderer(spec)
    cases = render_cases(input_dir, metadata, spec, renderer)
    set_seed(settings.seed)
    report_stage("loading_model")
    base = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, revision=model_revision(spec), trust_remote_code=False,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16, attn_implementation="sdpa",
    )
    base.config.use_cache = False
    model = configure_lora(base, rank, settings.lora_targets).to(device)
    enable_chunked_loss(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "operation": "synthetic_context_benchmark", "synthetic_only": True,
              "base_model": spec.base_model, "base_model_revision": spec.tokenizer_revision,
              "template_sha256": spec.template_sha256, "source_bundle": bundle,
              "input_manifest_sha256": sha256_file(input_dir / "benchmark.json"),
              "runtime": gpu_diagnostics(), "lora_rank": rank, "learning_rate": settings.learning_rate,
              "loss_implementation": "unchanged_training_worker_full_vocabulary",
              "loss_reduction": "mean_over_active_tokens_per_optimizer_group",
              "fresh_adapters_and_optimizer_per_stage": True,
              "no_endpoint": True, "no_checkpoint_export": True, "max_probe_seconds": 450}

    def save_progress(records):
        result["stages"] = records
        result["optimizer_steps"] = sum(record["optimizer_steps"] for record in records)
        result["initial_stages_passed"] = len(records) >= 2 and all(record["status"] == "passed" for record in records[:2])
        _json_dump(model_dir / "result.json", result)

    save_progress([])
    run_probes(model, cases, settings, device=device, on_progress=save_progress)
    report_stage("benchmark_complete")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")))
    args = parser.parse_args(argv)
    try:
        run(args.input_dir, args.model_dir)
    except Exception as exc:
        diagnostic = failure_diagnostics(exc)
        print(json.dumps(diagnostic), flush=True)
        output = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
        output.mkdir(parents=True, exist_ok=True)
        _json_dump(output / "worker-failure.json", diagnostic)
        (output.parent / "failure").write_text(f"SmithTune benchmark failed ({diagnostic['error_type']}; {diagnostic['error_kind']}).\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

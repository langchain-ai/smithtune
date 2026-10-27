"""Bounded, synthetic GPU reload/generation check; never runs an optimizer.

Experimental worker for a short-lived SageMaker job, not an endpoint or replay
implementation. Stage a checksummed export and original preparation bundle in
the training channel. Use the same locked image dependencies as training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers.base import PipelineError
from smithtune.providers.baseten.sampling_formats import parse_completion, stop_sequences
from smithtune.providers.sagemaker import sha256_file
from smithtune.training.contracts import model_revision
from smithtune.training.hf_sft import failure_diagnostics, gpu_diagnostics, read_bundle, report_stage
from smithtune.rendering import load_training_renderer
from smithtune.training.inference import generate_tokens, last_logits, verify_loaded_adapter


def synthetic_cases() -> list[dict]:
    tools = [{"type": "function", "function": {
        "name": "lookup_inventory", "description": "Read synthetic inventory for an item.",
        "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}},
                       "required": ["item_id"], "additionalProperties": False},
    }}]
    system = {"role": "system", "content": "You are a synthetic inventory assistant. Use lookup_inventory for stock questions. Report only quantities returned by the tool. Keep reasoning and answers brief."}
    request = {"role": "user", "content": "How many units of demo-item-42 are available?"}
    history = [system, request, {"role": "assistant", "content": "", "tool_calls": [{
        "id": "synthetic-reload-call", "type": "function", "function": {
            "name": "lookup_inventory", "arguments": '{"item_id":"demo-item-42"}',
        },
    }]}, {"role": "tool", "tool_call_id": "synthetic-reload-call",
          "content": '{"item_id":"demo-item-42","stock":17,"warehouse":"synthetic-west"}'}]
    return [
        {"id": "tool_request", "messages": [system, request], "tools": tools, "expected": "tool_call"},
        {"id": "tool_result", "messages": history, "tools": tools, "expected": "stock_reply"},
        {"id": "followup", "messages": [*history,
            {"role": "assistant", "content": "There are 17 units of demo-item-42 available."},
            {"role": "user", "content": "Would that cover an order of 20 units?"}],
         "tools": tools, "expected": "followup_reply"},
    ]


def synthetic_behavior(case: dict, candidate: dict) -> bool:
    if not candidate["sampling"]["format_valid"]:
        return False
    calls = candidate.get("tool_calls", [])
    if case["expected"] == "tool_call":
        return len(calls) == 1 and calls[0]["function"]["name"] == "lookup_inventory" and json.loads(
            calls[0]["function"]["arguments"]) == {"item_id": "demo-item-42"}
    content = candidate.get("content", "").lower()
    if case["expected"] == "stock_reply":
        return not calls and "17" in content
    return not calls and "no" in content and "17" in content


def run(input_dir: Path, model_dir: Path, *, device="cuda:0", max_new_tokens=96) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    report_stage("verify_reload_inputs")
    metadata = _load_json(input_dir / "reload.json")
    if metadata.get("schema_version") != 1 or not metadata.get("synthetic_only"):
        raise PipelineError("reload smoke requires an explicit synthetic manifest")
    for name, expected in metadata["files"].items():
        path = input_dir / name
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(path) != expected:
            raise PipelineError("reload input checksum/path validation failed")
    bundle, spec, _rows, _settings, _rank = read_bundle(input_dir)
    exported = _load_json(input_dir / "export/result.json")
    if (exported["bundle"] != bundle or exported["base_model"] != spec.base_model
            or exported["base_model_revision"] != model_revision(spec)):
        raise PipelineError("export does not match the original training provenance")
    if device != "cpu" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise PipelineError("reload smoke requires a CUDA GPU with bfloat16 support")
    print(json.dumps({"smithtune_event": "runtime", **gpu_diagnostics()}), flush=True)
    report_stage("reload_base_model")
    renderer = load_training_renderer(spec)
    base = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, revision=model_revision(spec), trust_remote_code=False,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    report_stage("reload_adapter")
    model = PeftModel.from_pretrained(base, input_dir / "export/adapter", is_trainable=False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PipelineError("reload worker must not expose trainable parameters")
    integrity = verify_loaded_adapter(model, input_dir / "export/adapter")
    cases = synthetic_cases()
    probe = renderer.prompt_tokens(cases[0]["messages"], tools=cases[0]["tools"])
    report_stage("verify_adapter_effect")
    with model.disable_adapter():
        base_logits = last_logits(model, probe, device)
    tuned_logits = last_logits(model, probe, device)
    delta = float((tuned_logits - base_logits).abs().max())
    if delta == 0:
        raise PipelineError("loaded adapter does not change probe logits")
    del base_logits, tuned_logits
    stop_ids = renderer.tokenizer.encode(stop_sequences(renderer, spec)[0], add_special_tokens=False)
    if len(stop_ids) != 1:
        raise PipelineError("reload smoke requires one native stop token")
    report_stage("synthetic_generation")
    generations = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("base", "tuned"):
        for case in cases:
            tokens = renderer.prompt_tokens(case["messages"], tools=case["tools"])
            with model.disable_adapter() if variant == "base" else nullcontext():
                generated = generate_tokens(model, tokens, device=device, eos_token_id=stop_ids[0],
                                            max_new_tokens=max_new_tokens)
            candidate = parse_completion(renderer, spec, generated.pop("tokens"), generated["stop_reason"], tools=case["tools"])
            record = {"variant": variant, "case_id": case["id"], **generated,
                      "candidate": candidate, "synthetic_behavior_check": synthetic_behavior(case, candidate)}
            generations.append(record)
            _json_dump(model_dir / "generations.json", generations)
            print(json.dumps({"smithtune_event": "generation", "variant": variant, "case_id": case["id"],
                              "output_tokens": generated["output_tokens"], "stop_reason": generated["stop_reason"],
                              "format_valid": candidate["sampling"]["format_valid"]}), flush=True)
    report_stage("exporting")
    result = {"schema_version": 1, "operation": "adapter_reload_smoke", "synthetic_only": True,
              "source_run_id": bundle["run_id"], "base_model": spec.base_model,
              "base_model_revision": spec.tokenizer_revision, "source_export_sha256": sha256_file(input_dir / "export/result.json"),
              "adapter": integrity, "max_abs_probe_logit_delta": delta, "optimizer_steps": 0,
              "generation_count": len(generations), "max_new_tokens": max_new_tokens,
              "format_valid_count": sum(row["candidate"]["sampling"]["format_valid"] for row in generations),
              "synthetic_behavior_pass_count": sum(row["synthetic_behavior_check"] for row in generations),
              "peak_cuda_memory_bytes": 0 if device == "cpu" else torch.cuda.max_memory_allocated(),
              "runtime": gpu_diagnostics(), "notes": ["Synthetic checks do not establish task quality.",
                                                       "Length-limited or malformed samples are retained as failed format checks."]}
    _json_dump(model_dir / "result.json", result)
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
        (output.parent / "failure").write_text(f"Reload smoke failed ({diagnostic['error_kind']}); inspect worker diagnostics.\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

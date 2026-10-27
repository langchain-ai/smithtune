"""Replay through answers generated inside a SageMaker job; no serving resources."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from smithtune.artifacts import _load_json
from smithtune.providers.base import PipelineError
from smithtune.run_artifacts import load_checkpoint_context, require_matching_base_model
from smithtune.training.contracts import sha256_file, verify_artifact
from smithtune.training.evaluation import EvaluationSettings, load_predictions, request_sha256

SERVING_MODE = "precomputed"


def checkpoint_identity(run_id: str, best_epoch: int) -> str:
    return f"sagemaker://{run_id}/adapter@epoch-{best_epoch}"


def checkpoint_from_run(data_dir: Path, run_dir: Path) -> tuple[Any, str, Path, dict]:
    """Return the prepared model, tuned identity, verified artifact and evaluation summary."""
    model, plan, result = load_checkpoint_context(data_dir, run_dir, provider="sagemaker")
    if not isinstance(plan, dict) or not isinstance(result, dict) or any(
        item.get("provider") != "sagemaker" for item in (plan, result)
    ):
        raise PipelineError("SageMaker evaluation requires a collected SageMaker training run (run smithtune aws collect)")
    require_matching_base_model(model, plan, provider_label="SageMaker")
    artifact_dir = Path(run_dir) / "artifact"
    exported, _spec = verify_artifact(artifact_dir)
    if exported.get("run_id") != result.get("run_id") or exported.get("best_epoch") != result.get("best_epoch"):
        raise PipelineError("collected artifact differs from the run result; run smithtune aws collect again")
    summary = read_evaluation_summary(artifact_dir)
    return model, checkpoint_identity(result["run_id"], result["best_epoch"]), artifact_dir, summary


def read_evaluation_summary(artifact_dir: Path) -> dict:
    """Evaluation outcome recorded by the job, separate from training success."""
    path = Path(artifact_dir) / "evaluation/evaluation.json"
    if not path.exists():
        return {"status": "not_requested"}
    value = _load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("status") not in {
        "complete", "incomplete", "failed",
    }:
        raise PipelineError("unsupported evaluation summary in the collected artifact")
    raw = value.get("settings")
    if not isinstance(raw, dict):
        raise PipelineError("evaluation summary has no generation settings")
    value["settings"] = EvaluationSettings(**{k: raw[k] for k in ("max_new_tokens", "max_cases", "max_points_per_trajectory")})
    value["settings"].validate()
    return value


class PrecomputedSampler:
    """Serve saved answers by request identity; a missing answer is an error, never a fallback."""

    def __init__(self, model, checkpoint: str, artifact_dir: Path):
        from smithtune.providers.baseten.sampling_formats import PARSING_VERSION

        self.model = model
        self.checkpoint = checkpoint
        self.artifact_dir = Path(artifact_dir)
        summary = read_evaluation_summary(self.artifact_dir)
        if summary["status"] == "not_requested":
            raise PipelineError("this run generated no evaluation answers; train with --evaluate or run smithtune aws evaluate-job")
        self.settings: EvaluationSettings = summary["settings"]
        predictions = self.artifact_dir / "evaluation/predictions.jsonl"
        self.records = {record["request_sha256"]: record for record in load_predictions(predictions).values()}
        self.config = {
            "parsing_version": PARSING_VERSION, "provider": "sagemaker", "serving_mode": SERVING_MODE,
            "checkpoint": checkpoint, "base_model": model.base_model, "max_seq_len": model.max_seq_len,
            "tokenizer_revision": model.tokenizer_revision, "template_sha256": model.template_sha256,
            "rendering_version": model.rendering_version, "renderer": model.renderer,
            "max_new_tokens": self.settings.max_new_tokens,
            "predictions_sha256": sha256_file(predictions),
        }

    def __enter__(self) -> str:
        return self.checkpoint

    def __exit__(self, *_exc) -> None:
        return None

    def generate(self, model, messages, max_tokens, json_mode=False, request_contract=None):
        if json_mode:
            raise PipelineError("precomputed replay answers cannot be regenerated in JSON mode")
        if model == self.checkpoint:
            variant = "tuned"
        elif model == self.model.base_model:
            variant = "base"
        else:
            raise PipelineError("SageMaker replay serves only its checkpoint and matching base model")
        if max_tokens != self.settings.max_new_tokens:
            raise PipelineError("replay output budget differs from the answers generated in the job")
        record = self.records.get(request_sha256(messages, variant, max_tokens))
        if record is None:
            raise PipelineError("no saved answer for this replay case; run smithtune aws evaluate-job to generate the missing answers")
        if record.get("candidate") is None:
            return {"role": "assistant", "content": "",
                    "sampling": {"format_valid": False, "skipped": record.get("skipped"), "precomputed": True}}
        candidate = deepcopy(record["candidate"])
        candidate.setdefault("sampling", {}).update(
            precomputed=True, checkpoint=self.checkpoint if variant == "tuned" else None,
            prompt_tokens=record.get("prompt_tokens"), output_tokens=record.get("output_tokens"),
            stop_reason=record.get("stop_reason"),
        )
        return candidate

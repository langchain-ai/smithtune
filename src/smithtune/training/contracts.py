"""Versioned local-file contract shared by cloud training and serving workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

from smithtune.artifacts import _json_dump, _load_json, _load_jsonl
from smithtune.models import resolve_prepared_model
from smithtune.providers.base import CommonSFTSettings, ModelSpec, PipelineError
from smithtune.providers.baseten.provider import MODEL_SPECS as BASETEN_MODELS

QWEN3P5_9B = BASETEN_MODELS["qwen3p5-9b"]
# The 0.8B checkpoint shares the 9B's tokenizer, chat template and vocabulary;
# only the weights differ. It exists for CPU pipeline tests, not as a target.
WORKER_MODELS = {
    "qwen3p5-9b": QWEN3P5_9B,
    "qwen3p5-0p8b-test": replace(QWEN3P5_9B, name="qwen3p5-0p8b-test", base_model="Qwen/Qwen3.5-0.8B"),
}
# Pinned base-weight commits. The tokenizer revision pins rendering; the weights
# of a test profile come from another repository and commit.
MODEL_REVISIONS = {
    "qwen3p5-9b": QWEN3P5_9B.tokenizer_revision,
    "qwen3p5-0p8b-test": "2fc06364715b967f1860aea9cf38778875588b17",
}
CPU_TRAINING_PROFILES = frozenset({"qwen3p5-0p8b-test"})
WORKER_DEVICES = ("cpu", "cuda:0")
PREPARED_FILES = ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")
OPTIONAL_BUNDLE_FILES = ("evaluation/cases.jsonl", "evaluation/settings.json")
# Module-name suffixes inside the language model's decoder layers. The vision
# encoder, embeddings and lm_head are never adapted: the chunked loss reads the
# head weight directly and would silently drop an adapter on it.
LORA_TARGET_SETS = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all-linear": ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                   "out_proj", "gate_proj", "up_proj", "down_proj"),
}
LORA_TARGET_MODULES = LORA_TARGET_SETS["attention"]
LOSS_IMPLEMENTATION = "trl-chunked-nll;chunk=256;ignored-positions-skipped"


@dataclass(frozen=True)
class WorkerSettings(CommonSFTSettings):
    trainer_backend: str = "trl"
    device: str = "cuda:0"
    lora_targets: str = "attention"

    def validate(self):
        for name in ("max_epochs", "early_stopping_patience", "batch_size", "seed"):
            value = getattr(self, name)
            if name == "early_stopping_patience" and value is None:
                continue
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise PipelineError(f"{name} must be an integer >= {minimum}")
        for name in ("learning_rate", "early_stopping_min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PipelineError(f"{name} must be finite")
        super().validate()
        if self.validation_every_steps is not None:
            raise PipelineError("validation_every_steps is not supported by the open-training worker; it validates at epoch end")
        if self.trainer_backend not in {"torch", "trl"}:
            raise PipelineError("trainer_backend must be torch or trl")
        if self.device not in WORKER_DEVICES:
            raise PipelineError("device must be cpu or cuda:0")
        if self.lora_targets not in LORA_TARGET_SETS:
            raise PipelineError("lora_targets must be attention or all-linear")


def model_revision(spec: ModelSpec) -> str:
    try:
        return MODEL_REVISIONS[spec.name]
    except KeyError as exc:
        raise PipelineError("unsupported open-training worker profile") from exc


def describe_sft_job(model: ModelSpec, settings: CommonSFTSettings, *, lora_rank: int,
                     prepared_manifest_uri: str) -> dict:
    """Describe the training intent the worker checks against its implemented recipe.

    The target names are suffixes inside the language model's decoder layers,
    never the vision encoder, embeddings or lm_head. Backend, device and cloud
    execution settings stay outside this description.
    """
    model.validate()
    settings.validate()
    if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or lora_rank < 1:
        raise PipelineError("invalid LoRA rank")
    if not isinstance(prepared_manifest_uri, str) or not prepared_manifest_uri:
        raise PipelineError("training specification requires a prepared manifest URI")
    return {
        "schema_version": 1, "method": "sft",
        "model": asdict(model), "model_revision": model.tokenizer_revision,
        "prepared_manifest_uri": prepared_manifest_uri,
        "settings": {item.name: getattr(settings, item.name) for item in fields(CommonSFTSettings)},
        "lora_rank": lora_rank, "lora_alpha": 2 * lora_rank,
        "target_modules": list(LORA_TARGET_SETS[getattr(settings, "lora_targets", "attention")]), "target_parameters": [],
        "loss": LOSS_IMPLEMENTATION,
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_prepared(data_dir: Path, *, expected_provider: str | None = None) -> tuple[dict, dict[str, list[dict]]]:
    from smithtune.dataset import _split_key

    prepared = Path(data_dir) / "prepared"
    manifest = _load_json(prepared / "manifest.json")
    if not isinstance(manifest, dict):
        raise PipelineError("prepared manifest must be an object")
    provider = manifest.get("model", {}).get("provider")
    if provider != "sagemaker" or (expected_provider and provider != expected_provider):
        raise PipelineError("prepared provider does not match this training worker")
    profiles = {name: replace(spec, provider=provider) for name, spec in WORKER_MODELS.items()}
    model = resolve_prepared_model(manifest, profiles, provider=provider)
    preset = profiles.get(model.name)
    if preset is None or replace(
        model, max_seq_len=preset.max_seq_len, trainer_max_seq_len=preset.trainer_max_seq_len,
        template_sha256="", rendering_version="",
    ) != preset:
        raise PipelineError("prepared model is not a supported open-training worker profile")
    if not model.template_sha256 or not model.rendering_version:
        raise PipelineError("prepare again with a pinned rendering identity")
    if model.training_context_limit > preset.training_context_limit:
        raise PipelineError("prepared context exceeds the model's supported context")
    rows = {split: _load_jsonl(prepared / f"{split}.jsonl") for split in ("train", "validation", "test")}
    seen: dict[str, str] = {}
    for split, values in rows.items():
        if manifest.get("split", {}).get(split) != len(values) or (split != "test" and not values):
            raise PipelineError(f"prepared {split} counts are invalid")
        for row in values:
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list) or not row["messages"]:
                raise PipelineError(f"invalid canonical row in {split}")
            if not isinstance(row.get("_source"), dict):
                raise PipelineError("canonical rows require source conversation identity")
            identity = _split_key(row)
            if identity in seen and seen[identity] != split:
                raise PipelineError("a source conversation occurs in multiple splits")
            seen[identity] = split
    maximum = manifest.get("audit", {}).get("max_context_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 2 <= maximum <= model.training_context_limit:
        raise PipelineError("prepared data needs a valid measured context length; prepare with render checks")
    return manifest, rows


def artifact_files(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PipelineError("model artifacts must not contain symbolic links")
        if path.is_file() and path.relative_to(directory).as_posix() != "artifact.json":
            files[path.relative_to(directory).as_posix()] = sha256_file(path)
    return files


def read_result(directory: Path) -> tuple[dict, ModelSpec]:
    result = _load_json(directory / "result.json")
    if not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("artifact_type") != "peft_lora_adapter":
        raise PipelineError("expected a version-1 SmithTune LoRA export")
    try:
        spec = ModelSpec(**result["model"])
        spec.validate()
        baseline = WORKER_MODELS[spec.name]
    except (KeyError, TypeError) as exc:
        raise PipelineError("unsupported model artifact profile") from exc
    if (spec.provider != "sagemaker"
            or replace(spec, provider=baseline.provider, max_seq_len=baseline.max_seq_len,
                       trainer_max_seq_len=baseline.trainer_max_seq_len,
                       template_sha256="", rendering_version="") != baseline
            or result.get("base_model") != spec.base_model
            or result.get("base_model_revision") != model_revision(spec)
            or result.get("adapter_path") != "adapter"
            or not spec.template_sha256 or not spec.rendering_version):
        raise PipelineError("artifact model/revision/renderer does not match the supported profile")
    if not (directory / "adapter/adapter_model.safetensors").is_file():
        raise PipelineError("artifact has no safetensors adapter")
    config = _load_json(directory / "adapter/adapter_config.json")
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != spec.base_model:
        raise PipelineError("adapter configuration does not match its recorded base model")
    return result, spec


def write_artifact_manifest(directory: Path) -> dict:
    manifest = {"schema_version": 1, "files": artifact_files(directory)}
    _json_dump(directory / "artifact.json", manifest)
    return manifest


def verify_artifact(directory: Path) -> tuple[dict, ModelSpec]:
    manifest = _load_json(directory / "artifact.json")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("files") != artifact_files(directory)):
        raise PipelineError("model artifact file checksums do not match")
    return read_result(directory)

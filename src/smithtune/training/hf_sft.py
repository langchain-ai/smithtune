"""Cloud-independent HF/PEFT SFT recipe with explicit SmithTune token masks.

The optional TRL backend and reference Torch loop share rendering, LoRA,
validation and exports. Cloud adapters only stage files and run this recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
import random
import re
import shutil

from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers.base import ModelSpec, PipelineError
from smithtune.training.contracts import LORA_TARGET_MODULES, PREPARED_FILES, describe_sft_job, load_prepared, model_revision, sha256_file, write_artifact_manifest, WorkerSettings
from smithtune.rendering import load_training_renderer, render_row_tokens


if TYPE_CHECKING:
    from smithtune.hf_rendering import TokenDatum


def to_trainer_example(datum: TokenDatum, *, max_seq_len: int) -> dict[str, list[int]]:
    """Preserve rendered tokens and target masks; the causal model shifts labels.

    Padding and sequence packing are not performed here. A collator must mask
    any padding it adds, and must preserve these explicit labels.
    """
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise PipelineError("max_seq_len must be a positive integer")
    tokens = datum.token_ids
    weights = datum.token_weights
    if not tokens or len(tokens) != len(weights):
        raise PipelineError("rendered tokens and loss weights must have equal nonzero length")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise PipelineError("rendered token IDs must be nonnegative integers")
    if any(type(weight) not in (int, float) or weight not in (0, 1) for weight in weights):
        raise PipelineError("HF SFT requires binary rendered loss weights")
    if len(tokens) > max_seq_len:
        raise PipelineError("rendered target exceeds max_seq_len; reject its whole trajectory")
    if not any(weights[1:]):
        raise PipelineError("rendered target has no supervised tokens after causal shifting")
    return {
        "input_ids": list(tokens),
        "attention_mask": [1] * len(tokens),
        "labels": [token if weight else -100 for token, weight in zip(tokens, weights, strict=True)],
    }


IGNORE_INDEX = -100
_STAGE = "startup"


def report_stage(stage: str) -> None:
    """Only controlled stage names, never data or arbitrary exception text."""
    global _STAGE
    _STAGE = stage
    print(json.dumps({"smithtune_event": "stage", "stage": stage}), flush=True)


def gpu_diagnostics() -> dict:
    try:
        import torch

        result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available()}
        if result["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            result.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                          compute_capability=list(torch.cuda.get_device_capability(0)),
                          compiled_architectures=torch.cuda.get_arch_list(),
                          allocated_bytes=torch.cuda.memory_allocated(0),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
        return result
    except Exception as exc:
        return {"inspection_error_type": type(exc).__name__}


def failure_diagnostics(exc: Exception) -> dict:
    # Match messages internally to a fixed vocabulary. Raw exception strings,
    # source lines, frame locals, URLs and credential-bearing values stay private.
    message = str(exc).lower()
    kind = "unclassified"
    patterns = (
        ("out of memory", "out_of_memory"),
        ("cannot copy out of meta tensor", "unmaterialized_model_tensor"),
        ("no kernel image", "cuda_kernel_compatibility"),
        ("invalid device function", "cuda_kernel_compatibility"),
        ("driver on your system is too old", "cuda_driver_compatibility"),
        ("failed to find c compiler", "runtime_compiler_required"),
        ("same dtype", "mixed_tensor_dtypes"),
        ("expected scalar type", "mixed_tensor_dtypes"),
        ("same device", "mixed_tensor_devices"),
        ("does not require grad", "missing_gradient_graph"),
        ("doesn't require grad", "missing_gradient_graph"),
        ("recomputed tensors", "checkpoint_recomputation_mismatch"),
        ("non-finite", "non_finite_gradient"),
        ("cudnn", "cudnn_runtime"),
    )
    for needle, value in patterns:
        if needle in message:
            kind = value
            break
    frames = []
    current = exc.__traceback__
    while current is not None:
        module = current.tb_frame.f_globals.get("__name__", "")
        function = current.tb_frame.f_code.co_name
        if (module == "__main__" or module.startswith(("smithtune.", "torch.", "triton.", "transformers.", "peft.", "accelerate."))):
            if re.fullmatch(r"[A-Za-z0-9_.]+", module) and re.fullmatch(r"[A-Za-z0-9_<>]+", function):
                frames.append({"module": module, "function": function, "line": current.tb_lineno})
        current = current.tb_next
    own = [frame for frame in frames if frame["module"].startswith("smithtune.") or frame["module"] == "__main__"][:4]
    stack = own + [frame for frame in frames[-12:] if frame not in own]
    return {"smithtune_event": "failure", "stage": _STAGE, "error_type": type(exc).__name__,
            "error_kind": kind, "stack": stack, "runtime": gpu_diagnostics()}


@dataclass(frozen=True)
class Example:
    input_ids: list[int]
    labels: list[int]

    @property
    def active_tokens(self) -> int:
        # HF causal-LM losses shift labels internally. Position zero has no
        # preceding prediction, and must not enter the normalization denominator.
        return sum(label != IGNORE_INDEX for label in self.labels[1:])


def as_example(datum, max_seq_len: int) -> Example:
    value = to_trainer_example(datum, max_seq_len=max_seq_len)
    return Example(value["input_ids"], value["labels"])


def render_conversations(rows, model: ModelSpec, renderer) -> list[list[Example]]:
    result = []
    for row in rows:
        try:
            examples = [as_example(item, model.training_context_limit) for item in render_row_tokens(
                row, model, renderer=renderer, include_loss_mask=True, reduction="none",
            )]
            if not examples:
                raise PipelineError("renderer produced no datums")
        except PipelineError as exc:
            raise PipelineError(f"example {row.get('_source', {}).get('example_id', 'unknown')}: {exc}") from exc
        result.append(examples)
    return result


def read_bundle(input_dir: Path):
    bundle = _load_json(input_dir / "bundle.json")
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise PipelineError("unsupported training bundle version")
    if not isinstance(bundle.get("files"), dict) or set(bundle["files"]) != set(PREPARED_FILES):
        raise PipelineError("bundle must identify every prepared file")
    for name in PREPARED_FILES:
        if sha256_file(input_dir / "prepared" / name) != bundle["files"][name]:
            raise PipelineError(f"training bundle checksum mismatch: {name}")
    manifest, rows = load_prepared(input_dir)
    raw = dict(bundle["settings"])
    rank = raw.pop("lora_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise PipelineError("invalid LoRA rank")
    settings = WorkerSettings(**raw)
    settings.validate()
    spec = ModelSpec(**manifest["model"])
    # Optional for old pilot bundles. New descriptions must match the exact
    # implemented recipe; a descriptive contract cannot silently change it.
    if "training" in bundle:
        training = bundle["training"]
        if not isinstance(training, dict) or training != describe_sft_job(
            spec, settings, lora_rank=rank, prepared_manifest_uri=training.get("prepared_manifest_uri"),
        ):
            raise PipelineError("training specification does not match the worker's model/settings/LoRA recipe")
    return bundle, spec, rows, settings, rank


def configure_lora(model, rank: int):
    from peft import LoraConfig, get_peft_model

    # Only language attention projections, not the multimodal vision encoder.
    # This initial recipe intentionally excludes linear-attention/MLP adapters.
    targets = [name for name, _ in model.named_modules()
               if ".language_model." in name and name.rsplit(".", 1)[-1] in LORA_TARGET_MODULES]
    if not targets:
        raise PipelineError("Qwen language attention projections not found; refusing a different LoRA recipe")
    return get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))


def _loss(model, example: Example, device):
    import torch

    ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([example.labels], dtype=torch.long, device=device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    loss = result.loss
    if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
        raise PipelineError("training returned a missing or non-finite scalar loss")
    return loss


def train_loop(model, train_rows: list[list[Example]], validation_rows: list[list[Example]],
               settings: WorkerSettings, *, device, on_epoch=None) -> dict:
    """Accumulate token-normalized gradients without concatenating conversations.

    A microbatch is one rendered datum. The final partial group is normalized
    using its actual active-token count, not the nominal effective batch size.
    Numerical identity with the providers' private optimizer implementations is
    not assumed; token masks and the explicit loss objective are testable here.
    """
    import torch

    settings.validate()
    if not train_rows or not validation_rows or any(not row for row in [*train_rows, *validation_rows]):
        raise PipelineError("training and validation need nonempty rendered conversations")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise PipelineError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, weight_decay=0.0)
    best_loss = patience_loss = math.inf
    best_epoch = stale_epochs = steps = 0
    history = []
    for epoch in range(1, settings.max_epochs + 1):
        ordered = list(train_rows)
        random.Random(settings.seed + epoch - 1).shuffle(ordered)
        examples = [item for row in ordered for item in row]
        model.train()
        train_sum = train_count = 0
        for start in range(0, len(examples), settings.batch_size):
            group = examples[start:start + settings.batch_size]
            denominator = sum(example.active_tokens for example in group)
            optimizer.zero_grad(set_to_none=True)
            for example in group:
                loss = _loss(model, example, device)
                (loss * (example.active_tokens / denominator)).backward()
                train_sum += float(loss.detach()) * example.active_tokens
                train_count += example.active_tokens
            # Abort instead of exporting an adapter corrupted by overflow.
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            print(json.dumps({"smithtune_event": "optimizer_step", "step": steps}), flush=True)
        model.eval()
        val_sum = val_count = 0
        with torch.no_grad():
            for row in validation_rows:
                for example in row:
                    loss = _loss(model, example, device)
                    val_sum += float(loss) * example.active_tokens
                    val_count += example.active_tokens
        validation_loss = val_sum / val_count
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch = validation_loss, epoch
        if validation_loss < patience_loss - settings.early_stopping_min_delta:
            patience_loss, stale_epochs = validation_loss, 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch, "train_loss": train_sum / train_count, "validation_loss": validation_loss,
            "optimizer_steps": steps, "new_best": improved, "train_target_tokens": train_count,
            "validation_target_tokens": val_count,
        }
        history.append(record)
        state = {
            "schema_version": 1, "completed_epoch": epoch, "optimizer_steps": steps,
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "patience_loss": patience_loss, "stale_epochs": stale_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "optimizer": optimizer.state_dict(),
        }
        if on_epoch:
            on_epoch(record, state)
        print(f"epoch={epoch} train_loss={record['train_loss']:.6f} validation_loss={validation_loss:.6f}", flush=True)
        if settings.early_stopping_patience is not None and stale_epochs >= settings.early_stopping_patience:
            break
    return {"best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs": history, "optimizer_steps": steps}


def run(input_dir: Path, model_dir: Path, checkpoint_dir: Path, *, device: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, set_seed

    report_stage("read_bundle")
    bundle, spec, rows, settings, rank = read_bundle(input_dir)
    # The bundle decides the device: the planner chose it from the instance type.
    device = settings.device if device is None else device
    report_stage("cuda_preflight")
    if device != "cpu" and not torch.cuda.is_available():
        raise PipelineError("the training worker requires a CUDA GPU; unit smoke tests run the loop on CPU")
    if device != "cpu" and not torch.cuda.is_bf16_supported():
        raise PipelineError("the initial worker requires a GPU with bfloat16 support")
    print(json.dumps({"smithtune_event": "runtime", **gpu_diagnostics()}), flush=True)
    set_seed(settings.seed)
    report_stage("rendering")
    renderer = load_training_renderer(spec)
    train_rows = render_conversations(rows["train"], spec, renderer)
    validation_rows = render_conversations(rows["validation"], spec, renderer)
    report_stage("loading_model")
    base = AutoModelForImageTextToText.from_pretrained(
        spec.base_model, revision=model_revision(spec), dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        trust_remote_code=False, attn_implementation="sdpa",
    )
    base.config.use_cache = False
    report_stage("configuring_lora")
    model = configure_lora(base, rank)
    report_stage("moving_model")
    model = model.to(device)
    report_stage("gradient_checkpointing")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_epoch(record, state):
        destination = checkpoint_dir / f"epoch-{record['epoch']}"
        destination.mkdir(exist_ok=False)
        model.save_pretrained(destination / "adapter", safe_serialization=True, save_embedding_layers=False)
        torch.save(state, destination / "training-state.pt")
        # The complete marker is written last. Only a marked epoch is coherent.
        _json_dump(destination / "complete.json", {"record": record, "bundle": bundle})
        _json_dump(checkpoint_dir / "latest.json", {"epoch": record["epoch"]})

    report_stage("training")
    if settings.trainer_backend == "trl":
        from smithtune.training.trl_trainer import train
        result = train(model, train_rows, validation_rows, settings, device=device, on_epoch=save_epoch,
                       tokenizer=renderer.tokenizer, output_dir=checkpoint_dir / "trainer")
    else:
        result = train_loop(model, train_rows, validation_rows, settings, device=device, on_epoch=save_epoch)
    report_stage("exporting")
    best = checkpoint_dir / f"epoch-{result['best_epoch']}" / "adapter"
    shutil.copytree(best, model_dir / "adapter")
    renderer.tokenizer.save_pretrained(model_dir / "tokenizer")
    base.config.save_pretrained(model_dir / "base_config")
    result.update({
        "provider": spec.provider, "schema_version": 1, "run_id": bundle["run_id"],
        "base_model": spec.base_model, "base_model_revision": model_revision(spec),
        "artifact_type": "peft_lora_adapter", "adapter_path": "adapter",
        "bundle": bundle, "model": _load_json(input_dir / "prepared" / "manifest.json")["model"],
        "loss_reduction": "mean_over_active_tokens_per_optimizer_group",
        "trainer_backend": settings.trainer_backend,
        "lora_targets": "language attention q_proj,k_proj,v_proj,o_proj; alpha=2*rank",
        "max_sequence_tokens": max(len(item.input_ids) for group in [*train_rows, *validation_rows] for item in group),
        "peak_cuda_memory_bytes": 0 if device == "cpu" else torch.cuda.max_memory_allocated(),
    })
    _json_dump(model_dir / "result.json", result)
    write_artifact_manifest(model_dir)
    return result

"""Client-driven LoRA SFT shared by Tinker-compatible training APIs.

Baseten Loops and Tinker expose the same trainer operations: forward,
forward_backward, optim_step, save_state, and save_weights_for_sampler.
Smithtune renders each assistant target locally, packs the rendered datums into
microbatches, accumulates them to the effective batch size, validates
forward-only, and selects the best checkpoint itself. Each provider supplies
provisioning, the meaning of its loss metrics, an optional guard, and cleanup.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from smithtune.providers.base import ModelSpec, PipelineError
from smithtune.providers.validation import training_schedule


@dataclass(frozen=True)
class ProviderErrors:
    """Name a provider in messages and raise its own error types."""

    label: str
    data: type[PipelineError]
    runtime: type[Exception]


class TrainerOperations(Protocol):
    """One provider's interpretation of the shared trainer operations.

    Loss methods return the mean loss over the microbatch's active target
    tokens, that token count, and the provider's finite metrics.
    """

    def forward_backward(
        self, microbatch: list[Any], *, group_active_tokens: float,
    ) -> tuple[float, float, dict[str, float]]: ...

    def forward(self, microbatch: list[Any]) -> tuple[float, float, dict[str, float]]: ...

    def optim_step(self, learning_rate: float) -> dict[str, float]: ...

    def save_state(self, name: str) -> str: ...

    def save_weights_for_sampler(self, name: str) -> str: ...


class NoGuard:
    """Run without an active-time ceiling."""

    elapsed_seconds = 0.0

    def check(self, operation: str) -> None:
        return None


def pack_microbatches(
    datums: Iterable[Any], token_budget: int, *, error: type[Exception] = PipelineError,
) -> list[list[Any]]:
    """Greedily pack in order and reject trajectories larger than the budget."""
    if token_budget < 1:
        raise error("microbatch token budget must be positive")

    batches: list[list[Any]] = []
    current: list[Any] = []
    current_tokens = 0
    for datum in datums:
        token_count = datum_token_count(datum, error=error)
        if token_count > token_budget:
            raise error("datum exceeds the configured microbatch token budget")
        if current and current_tokens + token_count > token_budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(datum)
        current_tokens += token_count
    if current:
        batches.append(current)
    return batches


def plan_accumulation_groups(
    microbatches: Iterable[Iterable[Any]],
    *,
    effective_batch_size: int,
    error: type[Exception] = PipelineError,
) -> list[list[list[Any]]]:
    """Group ordered microbatches for example-counted gradient accumulation."""
    if effective_batch_size < 1:
        raise error("effective batch size must be positive")

    groups: list[list[list[Any]]] = []
    current: list[list[Any]] = []
    examples = 0
    for batch in microbatches:
        remaining = list(batch)
        if not remaining:
            raise error("microbatches must not be empty")
        while remaining:
            capacity = effective_batch_size - examples
            segment, remaining = remaining[:capacity], remaining[capacity:]
            current.append(segment)
            examples += len(segment)
            if examples == effective_batch_size:
                groups.append(current)
                current = []
                examples = 0
    if current:
        groups.append(current)
    return groups


def to_datum(
    rendered: Any,
    types: Any,
    max_sequence_tokens: int,
    *,
    ignore_index: int | None,
    error: type[Exception] = PipelineError,
) -> Any:
    """Shift one rendered target into a datum with binary next-token weights.

    Providers that need a sentinel for excluded targets pass ``ignore_index``;
    otherwise excluded positions keep their real token with zero weight.
    """
    try:
        token_ids = [int(token) for token in rendered.token_ids]
        token_weights = [float(weight) for weight in rendered.token_weights]
    except (AttributeError, TypeError, ValueError) as exc:
        raise error("renderer returned invalid token IDs or weights") from exc
    if len(token_ids) < 2:
        raise error("rendered trajectory needs at least two tokens")
    if len(token_ids) != len(token_weights):
        raise error("rendered token IDs and weights have different lengths")
    if len(token_ids) > max_sequence_tokens:
        raise error(f"rendered trajectory exceeds the {max_sequence_tokens:,}-token limit")
    if any(not math.isfinite(weight) or weight not in (0.0, 1.0) for weight in token_weights):
        raise error("renderer weights must be finite binary assistant masks")

    inputs = token_ids[:-1]
    weights = token_weights[1:]
    if not any(weights):
        raise error("rendered trajectory has no trainable assistant tokens")
    targets = [
        token if weight or ignore_index is None else ignore_index
        for token, weight in zip(token_ids[1:], weights, strict=True)
    ]
    return types.Datum(
        model_input=types.ModelInput.from_ints(inputs),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=targets, dtype="int64", shape=[len(inputs)]),
            "weights": types.TensorData(data=weights, dtype="float32", shape=[len(inputs)]),
        },
    )


def datum_token_count(datum: Any, *, error: type[Exception] = PipelineError) -> int:
    try:
        count = len(datum.model_input.to_ints())
    except (AttributeError, TypeError) as exc:
        raise error("datum has no readable model input tokens") from exc
    if count < 1:
        raise error("datum has no model input tokens")
    return count


def datum_weights(datum: Any) -> list[float] | None:
    weights = getattr(datum, "loss_fn_inputs", {}).get("weights")
    values = getattr(weights, "data", None)
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def batch_active_tokens(batch: list[Any]) -> float:
    count = 0.0
    for datum in batch:
        weights = datum_weights(datum)
        count += datum_token_count(datum) if weights is None else sum(weights)
    return count


def finite_metrics(result: Any, *, operation: str, error: type[Exception]) -> dict[str, float]:
    try:
        raw_metrics = getattr(result, "metrics", None) or {}
        metrics = {str(name): float(value) for name, value in dict(raw_metrics).items()}
        loss = getattr(result, "loss", None)
        if loss is not None:
            metrics.setdefault("loss", float(loss))
    except (TypeError, ValueError) as exc:
        raise error(f"{operation} returned invalid metrics") from exc
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise error(f"{operation} returned non-finite metrics: {', '.join(sorted(invalid))}")
    return metrics


def checkpoint_path(value: Any, *, operation: str, error: type[Exception]) -> str:
    path = getattr(value, "path", None)
    if not isinstance(path, str) or not path:
        raise error(f"{operation} returned no checkpoint path")
    return path


def checkpoint_name(run_id: str, suffix: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "-", f"{run_id}-{suffix}".lower()).strip("-")
    if len(candidate) <= 54:
        return candidate
    digest = hashlib.sha256(candidate.encode()).hexdigest()[:8]
    return f"{candidate[:45].rstrip('-')}-{digest}"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def audit_identity(data_dir: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    prepared = data_dir / "prepared"
    source = manifest["langsmith"]
    renderer_identity = {
        "renderer": plan["config"]["renderer"],
        "tokenizer_revision": plan["config"]["tokenizer_revision"],
    }
    return {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "model_identity": plan["model"],
        "model_sha256": sha256_json(plan["model"]),
        "source_dataset_identity": {
            "workspace_id": source.get("workspace_id"),
            "dataset_id": source.get("dataset_id"),
            "examples": source.get("examples"),
            "source_examples_sha256": manifest.get("source_examples_sha256"),
        },
        "split_files_sha256": {
            split: sha256_file(prepared / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
        "renderer_sha256": sha256_json(renderer_identity),
        "settings_sha256": sha256_json(
            {"model": plan["model"], "budget": plan["budget"], "config": plan["config"]}
        ),
    }


def validate_canonical_rows(rows: list[dict[str, Any]], *, split: str) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            raise PipelineError(f"{split} row {index} has no canonical messages")
        if not all(isinstance(message, dict) for message in messages):
            raise PipelineError(f"{split} row {index} has invalid canonical messages")
        if tools is not None and not isinstance(tools, list):
            raise PipelineError(f"{split} row {index} has invalid tool declarations")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # JSON strings may contain Unicode line separators such as U+2028, which
    # str.splitlines() would treat as record boundaries; JSONL uses only "\n".
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    if not all(isinstance(row, dict) for row in rows):
        raise PipelineError(f"prepared data rows in {path.name} must be objects")
    return rows


def load_prepared_data(
    data_dir: Path, *, label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = Path(data_dir) / "prepared"
    try:
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        train_rows = read_jsonl(prepared / "train.jsonl")
        validation_rows = read_jsonl(prepared / "validation.jsonl")
        test_rows = read_jsonl(prepared / "test.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read prepared {label} data: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PipelineError(f"prepared {label} manifest is not an object")
    from smithtune.dataset import require_current_preparation
    require_current_preparation(manifest)
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise PipelineError(f"prepared {label} manifest has no valid split counts")
    for partition, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        count = split.get(partition)
        if isinstance(count, bool) or not isinstance(count, int) or count != len(rows):
            raise PipelineError(f"prepared {partition} row count differs from manifest")
        if partition in {"train", "validation"} and not rows:
            raise PipelineError(f"prepared dataset has no {partition} rows")
    return manifest, train_rows, validation_rows


def validate_dataset_counts(manifest: dict[str, Any], *, label: str) -> None:
    source = manifest.get("langsmith")
    split = manifest.get("split")
    if not isinstance(source, dict) or not isinstance(split, dict):
        raise PipelineError(f"prepared {label} manifest has no valid dataset counts")
    counts = [source.get("examples"), *(split.get(name) for name in ("train", "validation", "test"))]
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise PipelineError(f"prepared {label} dataset counts must be non-negative integers")
    if counts[0] < sum(counts[1:]):
        raise PipelineError(f"prepared {label} split counts exceed source examples")


def prepared_max_sequence_tokens(manifest: dict[str, Any], model: ModelSpec, *, label: str) -> int:
    audit = manifest.get("audit")
    maximum = audit.get("max_context_tokens") if isinstance(audit, dict) else None
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 2
        or maximum > model.max_seq_len
    ):
        raise PipelineError(f"prepared {label} data has no valid measured maximum context length")
    return maximum


def assistant_target_count(rows: list[dict[str, Any]]) -> int:
    from smithtune.dataset import _has_message_content

    return sum(
        message.get("role") == "assistant" and _has_message_content(message)
        for row in rows for message in row["messages"]
    )


def estimated_schedule(rows: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    """Estimate optimizer steps from assistant targets without rendering."""
    targets = assistant_target_count(rows)
    return {
        **training_schedule(targets, settings.batch_size, settings.max_epochs, settings.validation_every_steps),
        "training_units": targets,
        "training_unit": "assistant_targets",
        "estimate": True,
    }


@dataclass
class RenderedSplits:
    """Rendered datums that fit the trainer, checked before paid provisioning."""

    train: list[list[Any]]
    validation_batches: list[list[Any]]
    dropped_train_rows: list[dict[str, Any]]
    prepared_train_rows: int

    @property
    def training_units(self) -> int:
        return sum(len(values) for values in self.train)


def render_splits(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    render: Callable[[dict[str, Any]], list[Any]],
    max_seq_len: int,
    microbatch_token_budget: int,
    max_dropped_training_rows: int,
    errors: ProviderErrors,
) -> RenderedSplits:
    """Render once, apply the drop policy, and check packing before provisioning."""
    rendered_train = [render(row) for row in train_rows]
    rendered_validation = [render(row) for row in validation_rows]
    if any(not values for values in rendered_train + rendered_validation):
        raise errors.data("every prepared row must render training data")

    def too_long(values: list[Any]) -> bool:
        return any(datum_token_count(datum, error=errors.data) > max_seq_len for datum in values)

    dropped = [
        {
            "prepared_index": index,
            "model_input_tokens": max(datum_token_count(datum, error=errors.data) for datum in values),
            "reason": f"exceeds trainer limit of {max_seq_len:,} tokens",
        }
        for index, values in enumerate(rendered_train)
        if too_long(values)
    ]
    if len(dropped) > max_dropped_training_rows:
        raise errors.data(
            f"{len(dropped)} training rows exceed the {errors.label} trainer limit; "
            f"configured maximum dropped training rows is {max_dropped_training_rows}"
        )
    if any(too_long(values) for values in rendered_validation):
        raise errors.data(f"validation data exceeds the {errors.label} trainer sequence limit")
    dropped_indices = {item["prepared_index"] for item in dropped}
    retained = [values for index, values in enumerate(rendered_train) if index not in dropped_indices]
    if not retained:
        raise errors.data("no training rows remain within the trainer limit")
    # Check the same packing constraints before paid provisioning.
    pack_microbatches(
        (datum for values in retained for datum in values), microbatch_token_budget, error=errors.data,
    )
    validation_batches = pack_microbatches(
        (datum for values in rendered_validation for datum in values), microbatch_token_budget,
        error=errors.data,
    )
    return RenderedSplits(retained, validation_batches, dropped, len(train_rows))


def rendered_schedule(splits: RenderedSplits, settings: Any) -> dict[str, Any]:
    units = splits.training_units
    return {
        **training_schedule(units, settings.batch_size, settings.max_epochs, settings.validation_every_steps),
        "training_units": units,
        "training_unit": "rendered_datums",
        "estimate": False,
    }


@dataclass
class SFTLoop:
    """Train epochs, validate, and keep the best coherent checkpoint pair.

    State is kept on the instance so callers can record partial progress when
    training stops early, fails, or is interrupted.
    """

    trainer: TrainerOperations
    settings: Any
    run_id: str
    splits: RenderedSplits
    run_dir: Path
    update_state: Callable[..., None]
    errors: ProviderErrors
    guard: Any = field(default_factory=NoGuard)
    # Guard errors raised after a checkpoint save are deferred until that
    # checkpoint is recorded, unless the checkpoint would become the new best.
    deferred: tuple[type[BaseException], ...] = ()
    retry: Callable[[Callable[[], Any]], Any] = field(default=lambda operation: operation())
    state_extras: Callable[[], dict[str, Any]] = field(default=dict)
    epochs: list[dict[str, Any]] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    step_losses: list[dict[str, Any]] = field(default_factory=list)
    completed_step: int = 0
    lowest_loss: float = math.inf
    best_epoch: int | None = None
    best_step: int | None = None
    best_state_uri: str | None = None
    best_sampler_uri: str | None = None
    last_state_uri: str | None = None
    stopped_early: bool = False

    @property
    def epochs_path(self) -> Path:
        return self.run_dir / "epochs.json"

    @property
    def validations_path(self) -> Path:
        return self.run_dir / "validations.json"

    def run(self) -> None:
        settings = self.settings
        patience_loss = math.inf
        epochs_without_patience_improvement = 0
        for epoch in range(1, settings.max_epochs + 1):
            epoch_rows = list(self.splits.train)
            random.Random(settings.seed + epoch - 1).shuffle(epoch_rows)
            train_datums = [datum for values in epoch_rows for datum in values]
            microbatches = pack_microbatches(
                train_datums, settings.microbatch_token_budget, error=self.errors.data,
            )
            groups = plan_accumulation_groups(
                microbatches, effective_batch_size=settings.batch_size, error=self.errors.data,
            )
            epoch_weighted_loss = 0.0
            epoch_active_tokens = 0.0
            optimizer_steps = 0
            for group_index, group in enumerate(groups, start=1):
                group_active_tokens = sum(batch_active_tokens(microbatch) for microbatch in group)
                for microbatch in group:
                    self.guard.check("training step")
                    loss, active_tokens, metrics = self.trainer.forward_backward(
                        microbatch, group_active_tokens=group_active_tokens,
                    )
                    epoch_weighted_loss += loss * active_tokens
                    epoch_active_tokens += active_tokens
                    self.step_losses.append({
                        "epoch": epoch,
                        "operation": "forward_backward",
                        "examples": len(microbatch),
                        "loss": loss,
                        "active_tokens": active_tokens,
                        "metrics": metrics,
                    })
                self.guard.check("optimizer step")
                self.trainer.optim_step(settings.learning_rate)
                optimizer_steps += 1
                self.completed_step += 1
                epoch_end = group_index == len(groups)
                interval_due = (
                    settings.validation_every_steps is not None
                    and self.completed_step % settings.validation_every_steps == 0
                )
                if not (epoch_end or interval_due):
                    continue
                validation, post_checkpoint_error = self.validate(epoch, epoch_end=epoch_end)
                if epoch_end:
                    if epoch_active_tokens <= 0:
                        raise self.errors.runtime("epoch returned no active target tokens")
                    validation_loss = validation["validation_loss"]
                    # Patience counts epochs, independent of interval checks.
                    if validation_loss < patience_loss - settings.early_stopping_min_delta:
                        patience_loss = validation_loss
                        epochs_without_patience_improvement = 0
                    else:
                        epochs_without_patience_improvement += 1
                    self.epochs.append({
                        "epoch": epoch,
                        "step": self.completed_step,
                        "train_loss": epoch_weighted_loss / epoch_active_tokens,
                        "validation_loss": validation_loss,
                        "optimizer_steps": optimizer_steps,
                        "resumable_state_uri": validation["resumable_state_uri"],
                        "sampler_weights_uri": validation["sampler_weights_uri"],
                        "new_best": validation["new_best"],
                    })
                    atomic_json(self.epochs_path, self.epochs)
                    self.update_state(completed_epoch=epoch)
                if post_checkpoint_error is not None:
                    raise post_checkpoint_error
            if (
                settings.early_stopping_patience is not None
                and epochs_without_patience_improvement >= settings.early_stopping_patience
            ):
                self.stopped_early = True
                break

    def validate(self, epoch: int, *, epoch_end: bool) -> tuple[dict[str, Any], BaseException | None]:
        weighted_loss = 0.0
        active_total = 0.0
        for microbatch in self.splits.validation_batches:
            def validation_forward(microbatch=microbatch):
                self.guard.check("validation step")
                return self.trainer.forward(microbatch)

            loss, active_tokens, metrics = self.retry(validation_forward)
            weighted_loss += loss * active_tokens
            active_total += active_tokens
            self.step_losses.append({
                "epoch": epoch,
                "step": self.completed_step,
                "operation": "validation_forward",
                "examples": len(microbatch),
                "loss": loss,
                "active_tokens": active_tokens,
                "metrics": metrics,
            })
        if active_total <= 0:
            raise self.errors.runtime("validation returned no active target tokens")
        validation_loss = weighted_loss / active_total
        improved = validation_loss < self.lowest_loss
        suffix = (
            f"epoch-{epoch}" if self.settings.validation_every_steps is None
            else f"step-{self.completed_step}"
        )
        post_checkpoint_error: BaseException | None = None
        self.guard.check("validation state checkpoint")
        state_uri = self.trainer.save_state(checkpoint_name(self.run_id, f"state-{suffix}"))
        try:
            self.guard.check("validation state checkpoint")
        except self.deferred as exc:
            if improved:
                raise
            post_checkpoint_error = exc
        sampler_uri: str | None = None
        if improved:
            self.guard.check("best checkpoint")
            sampler_uri = self.trainer.save_weights_for_sampler(
                checkpoint_name(self.run_id, f"sampler-{suffix}")
            )
            try:
                self.guard.check("best checkpoint")
            except self.deferred as exc:
                post_checkpoint_error = exc

        # A best checkpoint always pairs resume state and sampler weights
        # from the same completed optimizer update.
        validation = {
            "epoch": epoch,
            "step": self.completed_step,
            "epoch_end": epoch_end,
            "validation_loss": validation_loss,
            "resumable_state_uri": state_uri,
            "sampler_weights_uri": sampler_uri,
            "new_best": improved,
        }
        atomic_json(self.validations_path, [*self.validations, validation])
        self.validations.append(validation)
        self.last_state_uri = state_uri
        if improved:
            self.lowest_loss = validation_loss
            self.best_epoch = epoch
            self.best_step = self.completed_step
            self.best_state_uri = state_uri
            self.best_sampler_uri = sampler_uri
        self.update_state(
            completed_step=self.completed_step,
            last_resumable_state_uri=self.last_state_uri,
            best_resumable_state_uri=self.best_state_uri,
            best_sampler_weights_uri=self.best_sampler_uri,
            best_epoch=self.best_epoch,
            best_step=self.best_step,
            **self.state_extras(),
        )
        return validation, post_checkpoint_error

    def summary(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "validations": self.validations,
            "step_losses": self.step_losses,
            "completed_step": self.completed_step,
            "last_resumable_state_uri": self.last_state_uri,
            "best_resumable_state_uri": self.best_state_uri,
            "best_sampler_weights_uri": self.best_sampler_uri,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "stopped_early": self.stopped_early,
        }

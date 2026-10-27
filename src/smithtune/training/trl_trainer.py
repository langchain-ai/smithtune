"""Default SFTTrainer backend with explicit SmithTune masks and token-normalized loss.

No tokenization, template rewriting, packing or truncation occurs in TRL. The
loss is TRL's chunked next-token NLL: ignored positions never reach lm_head and
scored positions are projected in checkpointed chunks, so full logits are never
materialized. The reference torch loop uses the same patch for parity tests.
"""
from __future__ import annotations

from collections import deque
import math
import random

from smithtune.providers.base import PipelineError


def train(model, train_rows, validation_rows, settings, *, tokenizer, output_dir, device, on_epoch=None):
    import torch
    from datasets import Dataset
    from torch.utils.data import Sampler
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from smithtune.training.hf_sft import _loss

    settings.validate()
    if not train_rows or not validation_rows or any(not row for row in [*train_rows, *validation_rows]):
        raise PipelineError("training and validation need nonempty rendered conversations")
    if any(not example.active_tokens for rows in (train_rows, validation_rows) for row in rows for example in row):
        raise PipelineError("all training/validation datums need active assistant targets")
    if device not in {"cpu", "cuda:0"}:
        raise PipelineError("the initial TRL worker supports a single CPU or CUDA device")

    groups, flattened = [], []
    for row in train_rows:
        groups.append(list(range(len(flattened), len(flattened) + len(row))))
        flattened.extend(row)
    dataset = Dataset.from_list([{"input_ids": item.input_ids, "labels": item.labels} for item in flattened])
    # Active-token totals per optimizer group, in training order, so logged
    # step losses (sum / group tokens) can be turned back into token sums.
    group_tokens: deque[int] = deque()

    class ConversationSampler(Sampler):
        def __init__(self):
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return len(flattened)

        def __iter__(self):
            order = list(groups)
            random.Random(settings.seed + self.epoch).shuffle(order)
            indices = [index for group in order for index in group]
            for start in range(0, len(indices), settings.batch_size):
                group_tokens.append(sum(flattened[index].active_tokens for index in indices[start:start + settings.batch_size]))
            return iter(indices)

    class Trainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return ConversationSampler()

    def collate(items):
        if len(items) != 1:
            raise PipelineError("the initial TRL collator expects one full datum per microbatch")
        ids = torch.tensor([items[0]["input_ids"]], dtype=torch.long)
        return {"input_ids": ids, "labels": torch.tensor([items[0]["labels"]], dtype=torch.long),
                "attention_mask": torch.ones_like(ids)}

    train_sum = train_targets = 0.0
    best_loss = patience_loss = math.inf
    best_epoch = stale_epochs = 0
    history = []

    class EpochCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            nonlocal train_sum, train_targets
            if logs and "loss" in logs and group_tokens:
                # HF logs the step loss rounded to four decimals; records are reporting only.
                tokens = group_tokens.popleft()
                if not math.isfinite(logs["loss"]):
                    raise PipelineError("non-finite TRL training loss")
                train_sum += logs["loss"] * tokens
                train_targets += tokens

        def on_epoch_end(self, args, state, control, **kwargs):
            nonlocal best_loss, patience_loss, best_epoch, stale_epochs, train_sum, train_targets
            epoch = len(history) + 1
            model.eval()
            val_sum = val_count = 0
            with torch.no_grad():
                for row in validation_rows:
                    for example in row:
                        val_sum += float(_loss(model, example, device)) * example.active_tokens
                        val_count += example.active_tokens
            validation_loss = val_sum / val_count
            improved = validation_loss < best_loss
            if improved:
                best_loss, best_epoch = validation_loss, epoch
            if validation_loss < patience_loss - settings.early_stopping_min_delta:
                patience_loss, stale_epochs = validation_loss, 0
            else:
                stale_epochs += 1
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters() if parameter.requires_grad):
                raise PipelineError("non-finite TRL adapter weights")
            record = {"epoch": epoch, "train_loss": train_sum / max(train_targets, 1), "validation_loss": validation_loss,
                      "optimizer_steps": state.global_step, "new_best": improved,
                      "train_target_tokens": int(train_targets), "validation_target_tokens": val_count}
            history.append(record)
            snapshot = {"schema_version": 1, "completed_epoch": epoch, "optimizer_steps": state.global_step,
                        "best_epoch": best_epoch, "best_validation_loss": best_loss,
                        "patience_loss": patience_loss, "stale_epochs": stale_epochs,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "optimizer": kwargs["optimizer"].state_dict()}
            if on_epoch:
                on_epoch(record, snapshot)
            train_sum = train_targets = 0.0
            group_tokens.clear()
            control.should_training_stop = (settings.early_stopping_patience is not None
                                            and stale_epochs >= settings.early_stopping_patience)
            model.train()
            return control

    config = SFTConfig(
        output_dir=str(output_dir), per_device_train_batch_size=1, gradient_accumulation_steps=settings.batch_size,
        num_train_epochs=settings.max_epochs, learning_rate=settings.learning_rate,
        optim="adamw_torch", weight_decay=0.0, lr_scheduler_type="constant", max_grad_norm=1.0,
        use_cpu=device == "cpu", bf16=device != "cpu", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=None, packing=False, padding_free=False,
        loss_type="chunked_nll", assistant_only_loss=False, completion_only_loss=False, remove_unused_columns=False,
        seed=settings.seed, data_seed=settings.seed, report_to="none", disable_tqdm=True,
        save_strategy="no", eval_strategy="no", logging_strategy="steps", logging_steps=1, dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer,
                      data_collator=collate, callbacks=[EpochCallback()])
    # Qwen3.5 declares accepts_loss_kwargs=False; without this, num_items_in_batch
    # is withheld and the loss silently becomes a mean of per-datum means.
    trainer.model_accepts_loss_kwargs = True
    if trainer.accelerator.num_processes != 1:
        raise PipelineError("distributed TRL training has not been validated; use one process")
    trainer.train()
    return {"best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs": history,
            "optimizer_steps": trainer.state.global_step}

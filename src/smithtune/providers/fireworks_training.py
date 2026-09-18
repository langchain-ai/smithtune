"""Keep one serverless session through SFT epochs and replay.

The epoch loop follows the pinned Fireworks SFT recipe. Rendering, streaming
data loading, validation loss, optimizer submission, and checkpoints use its
runtime helpers; smithtune owns early stopping and the session lifetime.
"""

from __future__ import annotations

import functools
import math
import os
import sys
from collections import deque
from contextlib import ExitStack, redirect_stdout
from pathlib import Path

from smithtune.artifacts import _json_dump, _utc_now
from smithtune.providers.fireworks_sampling import create_service
from smithtune.providers.base import PipelineError


def _render_conversation(row):
    """Worker entry point shared by actual training and eager validation."""
    from training.recipes import sft_loop
    from smithtune.bindings import training_targets
    from smithtune.rendering import render_fireworks_target

    state = sft_loop._worker_state
    rendered = [datum for target in training_targets(row)
                for datum in render_fireworks_target(target, state["renderer"])]
    if not rendered or any(not 2 <= len(d.token_ids) <= state["max_seq_len"] or
                           not any(w > 0 for w in d.token_weights) for d in rendered):
        raise PipelineError(f"prepared conversation {row['_source']['example_id']} has an invalid training target; prepare again")
    return [d.datum for d in rendered]


class ServerlessTraining:
    def __init__(self, config, run_dir: Path):
        from training.recipes import sft_loop
        from training.utils import make_render_dataloader
        from training.utils.resource_autosizing import select_render_worker_count
        import torch

        self.cfg = config
        self.run_dir = run_dir
        self._stack = ExitStack()
        self.step = 0
        self.consumed = 0
        self.history = []
        self.training_completed = False
        self.init_args = (
            config.tokenizer_model, config.renderer_name, config.train_on_what,
            config.max_seq_len, config.tokenizer_revision,
            config.thinking_trace_history_mode, config.tokenizer_trust_remote_code,
        )
        sft_loop._init_render_worker(*self.init_args)
        from training.utils import JsonlRenderDataset

        self.dataset = JsonlRenderDataset(config.dataset, _render_conversation,
                                         max_examples=config.max_examples, row_index_key=sft_loop.JSONL_ROW_INDEX_KEY)
        validation = JsonlRenderDataset(config.evaluation_dataset, _render_conversation,
                                       row_index_key=sft_loop.JSONL_ROW_INDEX_KEY)
        self.validation = sft_loop._render_eagerly(validation, len(validation))
        if not self.validation:
            raise PipelineError("prepared validation data has no usable training targets")
        self.generator = torch.Generator()
        self.batch_size = min(config.batch_size, len(self.dataset))
        workers = select_render_worker_count(available_parallel_tasks=len(self.dataset))
        self.loader = make_render_dataloader(
            self.dataset, batch_size=self.batch_size, num_workers=workers,
            shuffle=True, generator=self.generator,
            worker_init_fn=functools.partial(sft_loop._init_render_worker, *self.init_args),
            group_by_length=config.group_by_length,
            length_group_factor=config.length_group_factor,
            sizes=self.dataset.approx_row_sizes() if config.group_by_length else None,
        )
        self.total_steps = math.ceil(len(self.dataset) / self.batch_size) * config.epochs

    def __enter__(self):
        from fireworks.training.sdk import FireworksClient
        from training.utils import ReconnectableClient, RunnerIO
        from training.utils.checkpoints import TrainingCheckpoints
        from training.utils.serverless import ServerlessCheckpointClient
        from training.utils.runner_state import start_running
        from smithtune.providers.fireworks import CLIENT_SOURCE, FIREWORKS_BASE_URL

        try:
            self.runner = self._stack.enter_context(RunnerIO(self.cfg.runner))
            self.runner.set_accelerator_info(None, None, profile=None)
            self.service = create_service()
            self._stack.callback(self.service.close)
            training_client = self.service.create_lora_training_client(
                self.cfg.base_model, rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha,
            )
            self.runner.mark_serverless()
            self.job_id = self.service.training_session_id
            self.client = ReconnectableClient.from_training_client(
                training_client, base_model=self.cfg.base_model,
                lora_rank=self.cfg.lora_rank, job_id=self.job_id, service=self.service,
            )
            control = FireworksClient(
                api_key=os.environ["FIREWORKS_API_KEY"], base_url=FIREWORKS_BASE_URL,
                additional_headers={"X-Fireworks-Client-Source": CLIENT_SOURCE,
                                    "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]},
            )
            self._stack.callback(control.close)
            self.checkpoints = TrainingCheckpoints(
                self.client, ServerlessCheckpointClient(control, control.account_id),
                trainer_id=self.job_id, log_path=self.cfg.log_path,
                lora_rank=self.cfg.lora_rank, serverless=True, current_run_id=training_client.run_id,
            )
            self.checkpoints.resume(init_from_checkpoint=self.cfg.init_from_checkpoint)
            self.receipt = {"session_id": self.job_id, "training_run_id": training_client.run_id,
                            "base_model": self.cfg.base_model, "client_status": "open", "started_at_utc": _utc_now()}
            _json_dump(self.run_dir / "session.json", self.receipt)
            start_running(self.runner, total_steps=self.total_steps)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            # A later replay failure does not turn completed training into a
            # failed training job. Its status is recorded by the evaluator.
            self._stack.__exit__(None, None, None) if self.training_completed else self._stack.__exit__(exc_type, exc, tb)
        except BaseException:
            self.receipt["client_status"] = "cleanup_failed"
            _json_dump(self.run_dir / "session.json", self.receipt)
            raise
        self.receipt.update(client_status="closed", closed_at_utc=_utc_now())
        _json_dump(self.run_dir / "session.json", self.receipt)

    def run_epoch(self, epoch: int, _checkpoint: str | None):
        from tinker import AdamParams
        from training.recipes import sft_loop
        from training.utils import DEFAULT_ADAM
        from training.utils.client import DEFAULT_TIMEOUT_S
        from training.utils.runner_state import write_running_step
        from smithtune.providers.fireworks import _epoch_checkpoints

        self.generator.manual_seed(self.cfg.seed + epoch - 1)
        pending = deque()
        epoch_steps = 0

        def collect():
            step, tokens, weight, forward, optimizer = pending.popleft()
            try:
                output = forward.result(timeout=DEFAULT_TIMEOUT_S)
            finally:
                optimizer.result(timeout=DEFAULT_TIMEOUT_S)
            loss = output.metrics.get("loss:sum")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or weight <= 0:
                raise PipelineError("Fireworks training returned no finite loss")
            write_running_step(
                self.runner, step=step, total_steps=self.total_steps, tokens=tokens,
                metrics={"train/loss": loss / weight, "train/epoch": epoch,
                         "train/lr": self.cfg.learning_rate, "train/total_tokens": tokens},
            )

        try:
            for raw_batch in self.loader:
                self.consumed += len(raw_batch)
                batch = sft_loop._flatten_rendered_batch(raw_batch)
                if not batch:
                    raise PipelineError("prepared training batch has no usable targets")
                self.step += 1
                epoch_steps += 1
                tokens = sum(len(d.model_input.to_ints()) for d in batch)
                adam = AdamParams(learning_rate=self.cfg.learning_rate, **DEFAULT_ADAM)
                forward = self.client.submit_forward_backward(batch, loss_fn="cross_entropy")
                try:
                    optimizer = self.client.submit_optim_step(adam)
                except BaseException:
                    forward.result(timeout=DEFAULT_TIMEOUT_S)
                    raise
                pending.append((self.step, tokens, sft_loop._batch_loss_weight(batch), forward, optimizer))
                if len(pending) >= self.cfg.pipeline_depth:
                    collect()
            while pending:
                collect()
        finally:
            # Resolve submitted operations before releasing the session.
            while pending:
                try:
                    collect()
                except Exception:
                    pass
        if epoch_steps == 0:
            raise PipelineError("prepared training data has no usable batches")
        with redirect_stdout(sys.stderr):
            loss = sft_loop.run_eval(self.validation, self.client, self.cfg.batch_size, self.step, epoch - 1)
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise PipelineError("Fireworks validation returned no finite loss")
        self.runner.append_metrics(self.step, {"eval/loss": loss, "eval/epoch": epoch})
        self.checkpoints.save(f"epoch-{epoch}", resumable=True, promotable=True, data_consumed=self.consumed)
        result = {"job_id": self.job_id, "steps": self.step, "eval_loss": loss,
                  **_epoch_checkpoints(self.job_id)}
        self.history.append({**result, "epoch": epoch})
        _json_dump(self.run_dir / "epochs.json", self.history)
        return result

    def snapshot(self, checkpoint: str) -> str:
        """Restore the selected epoch in this session before publishing weights."""
        self.client.load_state(checkpoint)
        return self.client.save_weights_for_sampler("replay").path

    def complete(self):
        from training.utils.runner_state import write_completed

        write_completed(self.runner, step=self.step, total_steps=self.step)
        self.training_completed = True

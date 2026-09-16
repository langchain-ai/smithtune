"""Fireworks model configuration, SFT, checkpoint promotion, and serving."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smithtune.artifacts import _json_dump, _load_json, _load_jsonl, _run, _utc_now
from smithtune.capabilities import preflight_model
from smithtune.dataset import (
    DEFAULT_TEST_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    _model_from_manifest,
    _prepared_split,
    prepare_dataset,
)
from smithtune.inference_contract import InferenceContract
from smithtune.models import resolve_model_options, resolve_prepared_model
from smithtune.providers.base import (
    CommonSFTSettings,
    ModelOptions,
    ModelSpec,
    PipelineError,
    ReasoningPolicy,
    TrainingOptions,
)
from smithtune.rendering import SFT_TARGET_POLICY, load_training_renderer, resolve_rendering_model


TRAINING_BASE_URL = "https://api.fireworks.ai/training/v1/serverless"
FIREWORKS_BASE_URL = "https://api.fireworks.ai"
INFERENCE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
COOKBOOK_COMMIT = "09bbe1170c804eb5e5d7e7286f0feeef4e26055a"
CLIENT_SOURCE = "fireworks-training-skill/2.0.0"
RESOURCE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
MODEL_SPECS = {
    "qwen3p8-27b": ModelSpec(
        name="qwen3p8-27b",
        base_model="accounts/fireworks/models/qwen3p8-27b",
        tokenizer_model="Qwen/Qwen3.8-27B",
        tokenizer_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        renderer="qwen3_8_preserved",
        max_seq_len=131_072,
        thinking_trace_history_mode="preserved",
        supports_reasoning_content=True,
        requires_tool_declarations=True,
    ),
    "kimi-k3": ModelSpec(
        name="kimi-k3",
        base_model="accounts/fireworks/models/kimi-k3",
        tokenizer_model="moonshotai/Kimi-K3",
        tokenizer_revision="301be1b88c89c0d3a763da6301352cb8fe399e90",
        renderer="kimi_k3",
        supports_reasoning_content=True,
        max_seq_len=196_608,
        trust_remote_code=True,
        requires_tool_declarations=True,
    ),
    "deepseek-v4-flash-0731": ModelSpec(
        name="deepseek-v4-flash-0731",
        base_model="accounts/fireworks/models/deepseek-v4-flash-0731",
        tokenizer_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        tokenizer_revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        renderer="deepseek_v4",
        max_seq_len=262_144,
        supports_reasoning_content=True,
        requires_tool_declarations=True,
    ),
    "muse-glimmer-30b": ModelSpec(
        name="muse-glimmer-30b",
        base_model="accounts/fireworks/models/muse-glimmer-30b",
        tokenizer_model="meta-models/Muse-Glimmer-30B",
        tokenizer_revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        renderer="muse_glimmer",
        max_seq_len=131_072,
        supports_reasoning_content=True,
        requires_tool_declarations=True,
    ),
}

DEFAULT_MODEL = MODEL_SPECS["qwen3p8-27b"]


@dataclass(frozen=True)
class SFTSettings(CommonSFTSettings):
    lora_rank: int | None = None
    lora_alpha: int = 32
    pipeline_depth: int = 4

    def validate(self) -> None:
        super().validate()
        if self.lora_rank is not None and self.lora_rank < 1:
            raise PipelineError("lora_rank must be positive")
        if self.lora_alpha < 1 or self.pipeline_depth < 1:
            raise PipelineError("lora_alpha and pipeline_depth must be positive")


def _require_confirm(value: bool, action: str) -> None:
    if not value:
        raise PipelineError(
            f"{action} changes Fireworks resources or incurs cost; rerun with --confirm"
        )


def _set_skill_session(run_dir: Path | None = None) -> None:
    """Keep one Fireworks attribution ID across train, promote, and deploy."""
    if os.environ.get("FIREWORKS_SESSION_ID"):
        os.environ["FIREWORKS_CLIENT_SOURCE"] = CLIENT_SOURCE
        return
    if run_dir is not None:
        run_manifest = run_dir / "run.md"
        if run_manifest.is_file():
            match = re.search(
                r"^skill_session_id: ([0-9a-f-]{36})$",
                run_manifest.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if match:
                os.environ["FIREWORKS_SESSION_ID"] = match.group(1)
    os.environ.setdefault("FIREWORKS_SESSION_ID", str(uuid.uuid4()))
    os.environ["FIREWORKS_CLIENT_SOURCE"] = CLIENT_SOURCE


def _write_run_md(
    path: Path,
    plan: dict[str, Any],
    phase: str,
    next_action: str,
) -> None:
    session_id = os.environ["FIREWORKS_SESSION_ID"]
    text = f"""# Fireworks training run

status: planned
phase: {phase}
updated_at_utc: {_utc_now()}
skill_session_id: {session_id}
skill_client_source: {CLIENT_SOURCE}

## Intent

task: Train {plan['base_model']} on LangSmith message trajectories.
method: training-api-serverless
success_metric: Held-out cross-entropy loss and perplexity are recorded.

## Resolved plan

approval: operator supplied `--confirm` for this stage

```json
{json.dumps(plan, indent=2, sort_keys=True)}
```

## Progress

next_action: {next_action}
"""
    path.write_text(text, encoding="utf-8")


def _epoch_checkpoints(job_id: str) -> dict[str, str]:
    from fireworks.training.sdk import FireworksClient

    client = FireworksClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=FIREWORKS_BASE_URL,
        additional_headers={
            "X-Fireworks-Client-Source": CLIENT_SOURCE,
            "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"],
        },
    )
    try:
        account_id = client.account_id
        rows = client.list_training_session_checkpoints(
            f"accounts/{account_id}/trainingSessions/{job_id}"
        )
    finally:
        client.close()
    resumable = sorted(
        (
            row
            for row in rows
            if (row.get("checkpointType") or "").endswith(
                ("TRAINING", "TRAINING_LORA")
            )
        ),
        key=lambda row: row.get("createTime", ""),
        reverse=True,
    )
    if not resumable or not isinstance(resumable[0].get("name"), str):
        raise PipelineError(f"training job {job_id} has no resumable checkpoint")
    short_name = resumable[0]["name"].rstrip("/").rsplit("/", 1)[-1]
    match = re.match(r"^(run-[0-9a-f]{32})[:-](.+)$", short_name)
    if not match:
        raise PipelineError(
            f"training job {job_id} returned an invalid serverless checkpoint"
        )
    run_id, checkpoint_name = match.groups()
    promotable = sorted(
        (
            row
            for row in rows
            if row.get("promotable")
            and (row.get("name") or "").rstrip("/").rsplit("/", 1)[-1].startswith(run_id)
        ),
        key=lambda row: row.get("createTime", ""),
        reverse=True,
    )
    if not promotable or not isinstance(promotable[0].get("name"), str):
        raise PipelineError(f"training job {job_id} has no promotable checkpoint")
    return {
        "resume_checkpoint": f"{account_id}/{run_id}/{checkpoint_name}",
        "promotable_checkpoint": promotable[0]["name"],
    }


def run_early_stopping(
    settings: SFTSettings,
    run_epoch: Callable[[int, str | None], dict[str, Any]],
    initial_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Run epochs and select the checkpoint with the best validation loss."""
    settings.validate()
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    patience_loss = math.inf
    stale_epochs = 0
    checkpoint = initial_checkpoint
    for epoch in range(1, settings.max_epochs + 1):
        result = run_epoch(epoch, checkpoint)
        loss = result.get("eval_loss")
        if (
            not isinstance(loss, (int, float))
            or isinstance(loss, bool)
            or not math.isfinite(loss)
        ):
            raise PipelineError(f"epoch {epoch} returned no finite eval_loss")
        if not isinstance(result.get("resume_checkpoint"), str):
            raise PipelineError(f"epoch {epoch} returned no resume checkpoint")
        result = {**result, "epoch": epoch, "eval_loss": float(loss)}
        history.append(result)
        checkpoint = result["resume_checkpoint"]
        if best is None or result["eval_loss"] < best["eval_loss"]:
            best = result
        # Patience tracks significant improvement independently of checkpoint selection.
        if result["eval_loss"] < patience_loss - settings.early_stopping_min_delta:
            patience_loss = result["eval_loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.early_stopping_patience:
                break
    assert best is not None
    return {
        "epochs": history,
        "best": best,
        "stopped_early": len(history) < settings.max_epochs,
    }


def _validate_model(model: ModelSpec) -> None:
    model.validate()
    if model.provider != "fireworks" or not model.base_model.startswith(
        "accounts/fireworks/models/"
    ):
        raise PipelineError("Fireworks base model is invalid")
    if model.training_context_limit != model.max_seq_len:
        raise PipelineError("Fireworks preparation and trainer context limits must match")


class FireworksProvider:
    """Own the Fireworks training and deployment lifecycle."""

    name = "fireworks"

    def model_from_options(self, options: ModelOptions) -> ModelSpec:
        model = resolve_model_options(options, MODEL_SPECS, provider=self.name)
        _validate_model(model)
        return model

    def settings_from_options(self, options: TrainingOptions) -> SFTSettings:
        if any(value is not None for value in (
            options.microbatch_token_budget, options.max_spend_usd, options.hourly_rate_usd,
            options.replicas, options.spend_reserve_fraction, options.max_dropped_training_rows
        )):
            raise PipelineError(
                "--microbatch-token-budget, --max-spend-usd, --hourly-rate-usd, --replicas, "
                "--spend-reserve-fraction, and --max-dropped-training-rows are Baseten-only"
            )
        defaults = SFTSettings()
        settings = SFTSettings(
            max_epochs=options.max_epochs,
            early_stopping_patience=options.early_stopping_patience,
            early_stopping_min_delta=options.early_stopping_min_delta,
            learning_rate=options.learning_rate,
            batch_size=options.batch_size,
            seed=options.seed,
            lora_rank=options.lora_rank,
            lora_alpha=defaults.lora_alpha if options.lora_alpha is None else options.lora_alpha,
            pipeline_depth=defaults.pipeline_depth if options.pipeline_depth is None else options.pipeline_depth,
        )
        settings.validate()
        return settings

    def prepare(
        self,
        workspace_id: str,
        dataset_id: str,
        data_dir: Path,
        *,
        model_options: ModelOptions,
        inference_contract: InferenceContract | None = None,
        source_workspace_id: str | None = None,
        split_from: Path | None = None,
        reasoning_policy: ReasoningPolicy = "omit",
        validation_fraction: float | None = None,
        test_fraction: float | None = None,
        fetch: bool = True,
        check_render: bool = True,
    ) -> dict[str, Any]:
        model = resolve_rendering_model(preflight_model(self.model_from_options(model_options)))
        return prepare_dataset(
            workspace_id,
            dataset_id,
            model,
            data_dir,
            inference_contract=inference_contract,
            source_workspace_id=source_workspace_id,
            split_from=split_from,
            reasoning_policy=reasoning_policy,
            validation_fraction=(
                DEFAULT_VALIDATION_FRACTION if validation_fraction is None else validation_fraction
            ),
            test_fraction=DEFAULT_TEST_FRACTION if test_fraction is None else test_fraction,
            fetch=fetch,
            check_render=check_render,
        )

    def plan(
        self,
        data_dir: Path,
        run_id: str,
        settings: SFTSettings,
        *, replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings.validate()
        manifest = _load_json(data_dir / "prepared" / "manifest.json")
        if not isinstance(manifest, dict):
            raise PipelineError("prepared manifest is not an object")
        split = _prepared_split(manifest)
        for partition in ("train", "validation"):
            if split[partition] < 1:
                raise PipelineError(f"prepared dataset has no {partition} rows")
        source = manifest.get("langsmith")
        if not isinstance(source, dict):
            raise PipelineError("prepared manifest has no LangSmith source summary")
        model = resolve_prepared_model(manifest, MODEL_SPECS, provider="fireworks", allow_legacy=True)
        _validate_model(model)
        lora_rank = settings.lora_rank or model.default_lora_rank
        value = {
            "run_id": run_id,
            "method": "Fireworks Training API serverless LoRA SFT",
            "training_api": TRAINING_BASE_URL,
            "base_model": model.base_model,
            "source_examples_sha256": manifest.get("source_examples_sha256"),
            "dataset": {
                "source_rows": source.get("examples"),
                "train_rows": split["train"],
                "validation_rows": split["validation"],
                "test_rows": split["test"],
            },
            "cookbook": {
                "commit": COOKBOOK_COMMIT,
                "recipe": "smithtune.providers.fireworks_training (uses the pinned SFT runtime helpers)",
            },
            "config": {
                "tokenizer_model": model.tokenizer_model,
                "tokenizer_revision": model.tokenizer_revision,
                "renderer": model.renderer,
                "thinking_trace_history_mode": model.thinking_trace_history_mode,
                "loss_target": "all assistant text and tool calls",
                "max_seq_len": model.max_seq_len,
                "lora_rank": lora_rank,
                "lora_alpha": settings.lora_alpha,
                "learning_rate": settings.learning_rate,
                "max_epochs": settings.max_epochs,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
                "batch_size": settings.batch_size,
                "pipeline_depth": settings.pipeline_depth,
                "seed": settings.seed,
                "group_by_length": True,
                "adam_beta2": 0.95,
                "weight_decay": 0.01,
                "warmup_steps": 0,
            },
            "evaluation": "cross-entropy loss and perplexity on validation after each epoch; test is reserved for replay",
            "checkpoint": "each epoch ends with a resumable and promotable checkpoint; the best epoch is selected",
            "cost": "serverless token charges at the current account rate; resolve in the Fireworks console before confirmation",
            "deployment": "not included in training; a promoted LoRA needs a separately confirmed on-demand deployment",
        }
        if replay is not None:
            from smithtune.evaluation import prepare_replay_evaluation

            if replay["concurrency"] < 1:
                raise PipelineError("evaluation concurrency must be positive")
            with tempfile.TemporaryDirectory(prefix="smithtune-eval-plan-") as temporary:
                preview = prepare_replay_evaluation(
                    data_dir, Path(temporary), replay["max_points_per_trajectory"], replay["max_output_tokens"],
                )
            value["replay"] = {**preview, **replay, "serving_mode": "serverless", "evaluated_models": 2}
        return value

    def train(
        self, data_dir: Path, run_dir: Path, run_id: str, settings: SFTSettings,
        *, confirm: bool, init_from_checkpoint: str | None,
        replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _require_confirm(confirm, "training and replay" if replay is not None else "training")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise PipelineError(f"run directory must be new or empty: {run_dir}")
        plan = self.plan(data_dir, run_id, settings, replay=replay)
        if not os.environ.get("FIREWORKS_API_KEY"):
            raise PipelineError("FIREWORKS_API_KEY is not set")
        model = _model_from_manifest(_load_json(data_dir / "prepared" / "manifest.json"))
        preflight_model(model)
        load_training_renderer(model)
        from training.recipes import sft_loop
        from training.utils import RunnerConfig, WandBConfig
        from smithtune.providers.fireworks_training import ServerlessTraining

        run_dir.mkdir(parents=True, exist_ok=True)
        _set_skill_session()
        os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL
        _json_dump(run_dir / "plan.json", plan)
        if replay is not None:
            from smithtune.evaluation import ensure_judge_calibration, prepare_replay_evaluation, validate_judge_credentials
            from smithtune.inference import _chat_completion

            validate_judge_credentials(replay["judge_model"])
            prepare_replay_evaluation(
                data_dir, run_dir / "replay", replay["max_points_per_trajectory"], replay["max_output_tokens"],
            )
            ensure_judge_calibration(_load_jsonl(run_dir / "replay" / "cases.jsonl"),
                                     run_dir / "replay", replay["judge_model"], _chat_completion)
        cfg = sft_loop.Config(
            log_path=str(run_dir), base_model=model.base_model,
            dataset=str(data_dir / "prepared" / "train.jsonl"),
            evaluation_dataset=str(data_dir / "prepared" / "validation.jsonl"),
            tokenizer_model=model.tokenizer_model, tokenizer_revision=model.tokenizer_revision,
            tokenizer_trust_remote_code=model.trust_remote_code, renderer_name=model.renderer,
            thinking_trace_history_mode=model.thinking_trace_history_mode,
            train_on_what=SFT_TARGET_POLICY, serverless=True,
            lora_rank=settings.lora_rank or model.default_lora_rank, lora_alpha=settings.lora_alpha,
            max_seq_len=model.max_seq_len, learning_rate=settings.learning_rate,
            epochs=settings.max_epochs, batch_size=settings.batch_size,
            pipeline_depth=settings.pipeline_depth, seed=settings.seed, group_by_length=True,
            init_from_checkpoint=init_from_checkpoint, wandb=WandBConfig(project=None),
            runner=RunnerConfig(status_file=str(run_dir / "status.json"),
                                metadata_file=str(run_dir / "metadata.json"),
                                metrics_file=str(run_dir / "metrics.jsonl")),
        )
        _write_run_md(run_dir / "run.md", plan, "job_running", "wait for training and replay")
        result = None
        try:
            with ServerlessTraining(cfg, run_dir) as session:
                result = run_early_stopping(settings, session.run_epoch, init_from_checkpoint)
                _json_dump(run_dir / "epochs.json", result["epochs"])
                _json_dump(run_dir / "result.json", result)
                session.complete()
                if replay is not None:
                    from smithtune.evaluation import run_replay_evaluation
                    from smithtune.providers.fireworks_sampling import FireworksReplaySampler

                    checkpoint = result["best"]["resume_checkpoint"]
                    sampler = FireworksReplaySampler(
                        model, checkpoint, run_dir / "replay", service=session.service,
                        snapshot=session.snapshot(checkpoint),
                        lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                    )
                    _write_run_md(run_dir / "run.md", plan, "evaluating", "wait for replay")
                    result["replay"] = run_replay_evaluation(
                        data_dir, run_dir / "replay", checkpoint,
                        base_model=model.base_model, replay_sampler=sampler, confirm=True, **replay,
                    )
                    _json_dump(run_dir / "result.json", result)
        except BaseException:
            phase = "replay_incomplete" if result is not None else "failed"
            _write_run_md(run_dir / "run.md", plan, phase,
                          "training completed; inspect replay results" if result is not None else "inspect status.json and metrics.jsonl")
            raise
        _write_run_md(run_dir / "run.md", plan, "training_completed", "inspect replay/summary.json" if replay is not None else "run evaluate --run-dir")
        return result

    def promote(self, run_dir: Path, output_model_id: str, *, confirm: bool) -> None:
        _require_confirm(confirm, "checkpoint promotion")
        _validate_resource_id(output_model_id, "output model id")
        if not os.environ.get("FIREWORKS_API_KEY"):
            raise PipelineError("FIREWORKS_API_KEY is not set")
        result = _load_json(run_dir / "result.json")
        best = result.get("best")
        job_id = best.get("job_id") if isinstance(best, dict) else None
        checkpoint = best.get("promotable_checkpoint") if isinstance(best, dict) else None
        if not isinstance(job_id, str) or not isinstance(checkpoint, str):
            raise PipelineError("run result does not contain a best job and checkpoint")
        plan = _load_json(run_dir / "plan.json")
        base_model = plan.get("base_model")
        if not isinstance(base_model, str):
            raise PipelineError("run plan does not contain a base model")
        _set_skill_session(run_dir)
        os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL
        from fireworks.training.sdk import FireworksClient

        client = FireworksClient(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=FIREWORKS_BASE_URL,
            additional_headers={
                "X-Fireworks-Client-Source": CLIENT_SOURCE,
                "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"],
            },
        )
        try:
            client.promote_session_checkpoint(checkpoint, output_model_id, base_model)
        finally:
            client.close()
        _json_dump(
            run_dir / "promotion.json",
            {"promoted_at_utc": _utc_now(), "job_id": job_id, "checkpoint": checkpoint, "output_model_id": output_model_id},
        )

    def deploy(
        self,
        run_dir: Path,
        account_id: str,
        output_model_id: str,
        deployment_id: str,
        deployment_shape: str,
        *,
        confirm: bool,
    ) -> dict[str, Any]:
        _require_confirm(confirm, "deployment and smoke-test inference")
        _validate_resource_id(output_model_id, "output model id")
        _validate_resource_id(deployment_id, "deployment id")
        if not os.environ.get("FIREWORKS_API_KEY"):
            raise PipelineError("FIREWORKS_API_KEY is not set")
        _set_skill_session(run_dir)
        model = f"accounts/{account_id}/models/{output_model_id}"
        deployment = f"accounts/{account_id}/deployments/{deployment_id}"
        _run(
            [
                "firectl",
                "deployment",
                "create",
                model,
                "--deployment-id",
                deployment_id,
                "--deployment-shape",
                deployment_shape,
                "--account-id",
                account_id,
                "--wait",
            ]
        )
        model_route = f"{model}#{deployment}"
        endpoint = {"inference_url": INFERENCE_URL, "model": model_route, "deployment": deployment,
                    "smoke_test": {"status": "pending"}}
        receipt = run_dir / "endpoint.json"
        _json_dump(receipt, endpoint)
        try:
            endpoint["smoke_test"] = _inference_smoke_test(model_route)
        except Exception:
            endpoint["smoke_test"] = {"status": "failed"}
            _json_dump(receipt, endpoint)
            raise
        _json_dump(receipt, endpoint)
        return endpoint

    def undeploy(self, account_id: str, deployment_id: str, *, confirm: bool) -> None:
        _require_confirm(confirm, "deployment deletion")
        _validate_resource_id(deployment_id, "deployment id")
        _run(
            [
                "firectl",
                "deployment",
                "delete",
                f"accounts/{account_id}/deployments/{deployment_id}",
                "--account-id",
                account_id,
                "--ignore-checks",
                "--wait",
            ]
        )


def _validate_resource_id(value: str, label: str) -> None:
    if not RESOURCE_ID.fullmatch(value):
        raise PipelineError(f"{label} must use 1-63 lowercase letters, digits, or hyphens")


def _inference_smoke_test(model_route: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model_route,
            "messages": [{"role": "user", "content": "Reply with the single word ready."}],
            "temperature": 0,
            "max_tokens": 512,
        }
    ).encode()
    request = urllib.request.Request(
        INFERENCE_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
            "Content-Type": "application/json",
            "X-Fireworks-Client-Source": CLIENT_SOURCE,
            "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"inference smoke test failed with HTTP {exc.code}") from exc
    choices = body.get("choices", [])
    if not choices:
        raise PipelineError("inference smoke test returned no choices")
    return {"http_status": 200, "finish_reason": choices[0].get("finish_reason")}

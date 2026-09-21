#!/usr/bin/env python3
"""Prepare LangSmith trajectories and orchestrate provider-specific SFT."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

from smithtune import data_rights, dataset, dataset_workflow, triage
from smithtune.evaluation import replay as replay_evaluation
from smithtune.evaluation import langsmith as reporting
from smithtune.inference_contract import ContractError, load_inference_contract
from smithtune.inference import ANTHROPIC_ENDPOINTS, BasetenEndpoint, anthropic_connection
from smithtune.providers.baseten import (
    MODEL_SPECS as BASETEN_MODEL_SPECS,
    BasetenRuntimeError,
    BasetenSFTSettings,
)
from smithtune.providers.fireworks import (
    MODEL_SPECS as FIREWORKS_MODEL_SPECS,
    FireworksProvider,
    SFTSettings as FireworksSFTSettings,
)
from smithtune.providers.base import CommonSFTSettings, ModelOptions, PipelineError, TrainingOptions
from smithtune.providers import PROVIDERS, baseten_deployment, get_provider
from smithtune.rendering import DEFAULT_REPLAY_MAX_TOKENS
from smithtune import get_version
from smithtune.doctor import diagnose
from smithtune.artifacts import _json_dump, _load_json, output_lock
from smithtune.progress import command_status


def _add_replay_options(command):
    command.add_argument("--judge-model", default=replay_evaluation.DEFAULT_JUDGE_MODEL)
    command.add_argument("--concurrency", type=int, default=replay_evaluation.DEFAULT_EVALUATION_CONCURRENCY)
    command.add_argument("--max-points-per-trajectory", type=int, help="optional replay cap; default scores every assistant action")
    command.add_argument("--max-output-tokens", type=int, default=DEFAULT_REPLAY_MAX_TOKENS)


def _training_replay(args):
    values = {key: getattr(args, key) for key in (
        "judge_model", "concurrency", "max_points_per_trajectory", "max_output_tokens",
    )}
    if not args.evaluate:
        defaults = {"judge_model": replay_evaluation.DEFAULT_JUDGE_MODEL,
                    "concurrency": replay_evaluation.DEFAULT_EVALUATION_CONCURRENCY,
                    "max_points_per_trajectory": None, "max_output_tokens": DEFAULT_REPLAY_MAX_TOKENS}
        if values != defaults:
            raise PipelineError("replay options require --evaluate")
        return {}
    return {"replay": values}


def _parser() -> argparse.ArgumentParser:
    project = Path.cwd()
    parser = argparse.ArgumentParser(
        prog="smithtune", description=__doc__,
        epilog=f"Data Rights and Permitted Use: {data_rights.DOCUMENT_URL}",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {get_version()}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report installed dependencies and configuration without network calls")
    sub.add_parser(
        "acknowledge-data-rights", help="acknowledge reading Data Rights and Permitted Use",
        description=f"Read {data_rights.DOCUMENT_URL}, then acknowledge in an interactive terminal. No network calls are made.",
    )

    models = sub.add_parser("models", help="show supported training models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser(
        "list", help="list smithtune's supported models without network calls",
        description="List smithtune's supported training models as JSON. No credentials or downloads are needed; live provider availability is checked during prepare and train.",
    )
    models_list.add_argument(
        "--provider", choices=tuple(PROVIDERS), help="filter by provider (default: both providers)",
    )

    curate = sub.add_parser("dataset", help="curate trajectories from a tracing project")
    curate_sub = curate.add_subparsers(dest="dataset_command", required=True)
    for name, help_text in {
        "pull": "download trajectories and tool contracts to a saved directory",
        "triage": "preview local council judging; --confirm runs it",
        "push": "preview a dataset upload; --confirm uploads",
        "create": "download, optionally judge, and upload with one saved directory",
        "resume": "show pending stages; --confirm continues saved work",
    }.items():
        command = curate_sub.add_parser(name, help=help_text, description=help_text)
        command.add_argument("directory", nargs="?", type=Path, help="saved directory (generated for a new pull/create)")
        if name in {"pull", "create"}:
            command.add_argument("--workspace-id")
            command.add_argument("--project-id")
            command.add_argument("--start-time", help="inclusive root start time (default: 24 hours before end)")
            command.add_argument("--end-time", help="exclusive root start time (default: now)")
            command.add_argument("--filter", help="LangSmith root-run filter; create skips council when no judging criteria are supplied")
            command.add_argument("--limit", type=int, help="distinct trajectories to select (default: 100, maximum: 2000)")
        if name in {"push", "create"}:
            destination = command.add_mutually_exclusive_group()
            destination.add_argument("--name", help="new dataset name; saved for resume")
            destination.add_argument("--dataset-id", help="existing dataset to extend")
        if name == "create":
            command.add_argument("--no-triage", action="store_true", default=None, help="download and upload without council judging")
        if name in {"triage", "create"}:
            command.add_argument("--judges", help="comma-separated model aliases or provider:model; requests council judging")
            command.add_argument("--rule", action="append", help="criterion requiring council judging; repeat for multiple rules")
            command.add_argument("--rubric", type=Path, help="UTF-8 file with selection criteria; requests council judging and saves the text for resume")
            command.add_argument("--config", type=Path, help=argparse.SUPPRESS)
            command.add_argument("--runner", choices=("api", "deepagent"), help=argparse.SUPPRESS)
            command.add_argument("--max-output-tokens", type=int, help=argparse.SUPPRESS)
            command.add_argument("--attempts", type=int, help=argparse.SUPPRESS)
        if name in {"pull", "triage", "create"}:
            command.add_argument("--concurrency", type=int, help="concurrent tasks (default: 4); downloads cap at 4, judges at 16")
        if name != "pull":
            command.add_argument("--confirm", action="store_true", help="run this workflow's paid judging and uploads; otherwise preview")

    publish_splits = curate_sub.add_parser(
        "publish-splits",
        help="publish and verify existing prepared split memberships",
        description="Publish train, validation, and test memberships from existing prepared artifacts without fetching, converting, rendering, or splitting the dataset again.",
    )
    publish_splits.add_argument(
        "--data-dir", type=Path, required=True,
        help="existing prepared dataset directory containing raw/ and prepared/ artifacts",
    )

    skill = sub.add_parser("skill", help="export the packaged SFT selection skill for any agent")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    skill_export = skill_sub.add_parser("export")
    skill_export.add_argument("--output", type=Path, required=True, help="parent directory for sft-trace-triage/SKILL.md")

    capture_contract = sub.add_parser(
        "capture-contract",
        help="collect all function tools from a sample conversation; reject provider built-ins",
    )
    capture_contract.add_argument("--workspace-id", required=True)
    capture_contract.add_argument("--run-id", required=True, help="LLM run ID used to locate the sample thread; scans every LLM call in that thread")
    capture_contract.add_argument("--output", type=Path, required=True)

    prep = sub.add_parser("prepare", help="fetch, convert, split, and validate all trajectories")
    prep.add_argument("--provider", choices=tuple(PROVIDERS), default="fireworks")
    prep.add_argument("--data-dir", type=Path, default=project / "data", help="dataset directory (default: ./data in the current working directory)")
    prep.add_argument("--workspace-id", required=True)
    prep.add_argument("--dataset-id", required=True)
    prep.add_argument("--source-workspace-id", help="default workspace for automatic source tool capture; example metadata.source_workspace_id takes precedence (default: dataset workspace)")
    prep.add_argument("--inference-contract", type=Path, help="explicit global tool-schema override; default uses recorded per-assistant tool availability")
    prep.add_argument(
        "--reasoning-policy", choices=["omit", "preserve"], default="omit",
        help="omit source reasoning from SFT and replay (default), or explicitly preserve readable reasoning",
    )
    prep.add_argument(
        "--validation-fraction", type=float,
        help=f"validation fraction for either provider (default: {dataset.DEFAULT_VALIDATION_FRACTION})",
    )
    prep.add_argument(
        "--test-fraction", type=float,
        help=f"test fraction for either provider (default: {dataset.DEFAULT_TEST_FRACTION}; use 0 for no test split)",
    )
    prep.add_argument("--split-from", type=Path, help="reuse split assignments from a previous data directory")
    prep.add_argument("--model", required=True, help="provider model ID or supported model alias")
    prep.add_argument("--max-seq-len", type=int, help="lower the selected model's preparation and training context limit")
    prep.add_argument("--no-fetch", action="store_true", help="reuse the raw export and cached tool schemas; split synchronization still contacts LangSmith")
    prep.add_argument("--no-sync-splits", action="store_true", help="prepare locally without publishing LangSmith dataset splits")
    prep.add_argument("--skip-render-check", action="store_true", help=argparse.SUPPRESS)

    plan = sub.add_parser("plan", help="print the resolved training plan without provisioning resources")
    plan.add_argument(
        "--provider", choices=tuple(PROVIDERS), default="fireworks",
        help="training provider (default: %(default)s)",
    )
    plan.add_argument("--data-dir", type=Path, default=project / "data", help="dataset directory (default: ./data in the current working directory)")
    plan.add_argument("--run-id", default="langsmith-sft", help="label for this preview (default: %(default)s); not reserved for training")

    training = sub.add_parser("train", help="run paid serverless SFT after plan approval")
    training.add_argument(
        "--provider", choices=tuple(PROVIDERS), default="fireworks",
        help="training provider (default: %(default)s)",
    )
    training.add_argument("--data-dir", type=Path, default=project / "data", help="dataset directory (default: ./data in the current working directory)")
    training.add_argument("--run-dir", type=Path, help="output directory (default: ./runs/<run-id>); must be new or empty")
    training.add_argument("--run-id", help="run name (default: generated from the UTC timestamp and a random suffix)")
    training.add_argument("--init-from-checkpoint")
    training.add_argument("--confirm", action="store_true")

    common_defaults = CommonSFTSettings()
    fireworks_defaults = FireworksSFTSettings()
    baseten_defaults = BasetenSFTSettings()
    for command in (plan, training):
        command.add_argument("--evaluate", action="store_true", help="compare base and best checkpoint through provider samplers after training")
        _add_replay_options(command)
        shared = command.add_argument_group(
            "Shared training options", "Supported by both Fireworks and Baseten.",
        )
        shared.add_argument(
            "--max-epochs", type=int, default=common_defaults.max_epochs,
            help="maximum training epochs (default: %(default)s)",
        )
        shared.add_argument(
            "--early-stopping-patience", type=int, default=common_defaults.early_stopping_patience,
            help="epochs without sufficient validation improvement before stopping (default: %(default)s)",
        )
        shared.add_argument(
            "--early-stopping-min-delta", type=float, default=common_defaults.early_stopping_min_delta,
            help="minimum validation-loss improvement (default: %(default)s)",
        )
        shared.add_argument(
            "--learning-rate", type=float, default=common_defaults.learning_rate,
            help="optimizer learning rate (default: %(default)s)",
        )
        shared.add_argument(
            "--batch-size", type=int, default=common_defaults.batch_size,
            help="training batch size; Baseten accumulates microbatches to this effective size (default: %(default)s)",
        )
        shared.add_argument(
            "--seed", type=int, default=common_defaults.seed,
            help="training random seed (default: %(default)s)",
        )
        shared.add_argument(
            "--lora-rank", type=int,
            help="LoRA rank for either provider (default: the prepared model profile's default_lora_rank)",
        )
        fireworks = command.add_argument_group(
            "Fireworks-only options", "Require --provider fireworks; rejected by Baseten.",
        )
        fireworks.add_argument(
            "--lora-alpha", type=int,
            help=f"LoRA scaling factor (default: {fireworks_defaults.lora_alpha})",
        )
        fireworks.add_argument(
            "--pipeline-depth", type=int,
            help=f"training pipeline depth (default: {fireworks_defaults.pipeline_depth})",
        )
        baseten = command.add_argument_group(
            "Baseten-only options", "Require --provider baseten; rejected by Fireworks.",
        )
        baseten.add_argument(
            "--microbatch-token-budget", type=int,
            help="token budget per microbatch (default: the prepared model's max_seq_len)",
        )
        baseten.add_argument(
            "--max-spend-usd", type=float,
            help="active-time spend ceiling in USD; requires --hourly-rate-usd (default: disabled)",
        )
        baseten.add_argument(
            "--hourly-rate-usd", type=float,
            help="total hourly rate in USD for the spend guard; requires --max-spend-usd (default: unset)",
        )
        baseten.add_argument(
            "--replicas", type=int,
            help=f"training replicas (default: {baseten_defaults.replicas})",
        )
        baseten.add_argument(
            "--spend-reserve-fraction", type=float,
            help=f"fraction of spend ceiling reserved for cleanup (default: {baseten_defaults.spend_reserve_fraction})",
        )
        baseten.add_argument(
            "--max-dropped-training-rows", type=int,
            help=f"maximum training rows excluded above the trainer context limit (default: {baseten_defaults.max_dropped_training_rows})",
        )

    promotion = sub.add_parser("promote", help="promote the final checkpoint after separate approval")
    promotion.add_argument("--run-dir", type=Path, required=True)
    promotion.add_argument("--output-model-id", required=True)
    promotion.add_argument("--confirm", action="store_true")

    deployment = sub.add_parser("deploy", help="create an on-demand endpoint and test it")
    deployment.add_argument("--run-dir", type=Path, required=True)
    deployment.add_argument("--provider", choices=tuple(PROVIDERS), default="fireworks")
    deployment.add_argument("--account-id", help="Fireworks account ID")
    deployment.add_argument("--output-model-id", help="Fireworks model ID")
    deployment.add_argument("--deployment-id", help="Fireworks deployment ID")
    deployment.add_argument("--deployment-shape", help="Fireworks deployment shape")
    deployment.add_argument("--accelerator", help="required Baseten GPU allocation, for example H200:1")
    deployment.add_argument("--max-seq-len", type=int, help="required Baseten evaluation context cap, verified against the live server; does not configure serving context")
    deployment.add_argument("--deployment-timeout", type=float, help="Baseten readiness timeout in seconds (default: 1800)")
    deployment.add_argument("--confirm", action="store_true")

    eval_plan = sub.add_parser("eval-plan", help="build held-out trajectory replay cases")
    evaluation = sub.add_parser("evaluate", help="compare base and tuned actions with a calibrated judge")
    for command in (eval_plan, evaluation):
        command.add_argument("--provider", choices=tuple(PROVIDERS), default="fireworks", help="candidate serving provider (default: %(default)s)")
        command.add_argument("--run-dir", type=Path, help="training run directory containing the best checkpoint")
        command.add_argument("--data-dir", type=Path, default=project / "data", help="dataset directory (default: ./data in the current working directory)")
        command.add_argument("--output-dir", type=Path, help="replay results directory (default: <run-dir>/replay)")
        command.add_argument(
            "--max-points-per-trajectory",
            type=int,
            default=replay_evaluation.DEFAULT_REPLAY_POINTS,
            help="optional cap; default evaluates every assistant turn",
        )
        command.add_argument("--max-output-tokens", type=int, default=DEFAULT_REPLAY_MAX_TOKENS)
        serving = command.add_argument_group("Baseten serving options", "Fireworks uses --run-dir and automatically compares the base model.")
        serving.add_argument("--model-id", help="existing Baseten model ID")
        serving.add_argument("--max-seq-len", type=int, help="existing endpoint's configured context limit (required unless using --run-dir)")
        serving.add_argument("--serving-mode", choices=("sampler", "existing", "temporary"), help="default: sampler; explicit endpoint IDs use existing serving")
        serving.add_argument("--accelerator", help="temporary serving GPU allocation, for example H200:1; omit to reuse saved settings")
        serving.add_argument("--deployment-id", help="existing deployment ID")
        serving.add_argument("--deployment-timeout", type=float, default=600, help="temporary deployment readiness timeout in seconds")
        serving.add_argument("--tuned-model", help="checkpoint name served by the endpoint")
        if command is evaluation:
            serving.add_argument("--base-model", help="optional base-model route served by the same endpoint")
    evaluation.add_argument("--concurrency", type=int, default=replay_evaluation.DEFAULT_EVALUATION_CONCURRENCY)
    evaluation.add_argument("--judge-model", default=replay_evaluation.DEFAULT_JUDGE_MODEL,
                            help="judge route (default: direct Anthropic); use anthropic-gateway/<model-id> for the LangSmith gateway")
    evaluation.add_argument("--confirm", action="store_true")

    remove = sub.add_parser("undeploy", help="stop serving capacity for a deployment")
    remove.add_argument("--provider", choices=tuple(PROVIDERS), default="fireworks")
    remove.add_argument("--run-dir", type=Path, help="Baseten run directory containing the deployment receipt")
    remove.add_argument("--account-id", help="Fireworks account ID")
    remove.add_argument("--deployment-id", help="Fireworks deployment ID")
    remove.add_argument("--confirm", action="store_true")
    return parser


def _model_options(args: argparse.Namespace) -> ModelOptions:
    return ModelOptions(**{field.name: getattr(args, field.name) for field in fields(ModelOptions)})


def _settings_from_args(args: argparse.Namespace) -> CommonSFTSettings:
    options = TrainingOptions(**{
        field.name: getattr(args, field.name) for field in fields(TrainingOptions)
    })
    return get_provider(args.provider).settings_from_options(options)


def _temporary_baseten_plan(args) -> dict | None:
    if args.run_dir is not None and args.run_dir.resolve() == args.output_dir.resolve():
        raise PipelineError("--run-dir and --output-dir must be different directories")
    if args.serving_mode != "temporary":
        if args.accelerator is not None:
            raise PipelineError("--accelerator requires --provider baseten --serving-mode temporary")
        return None
    if args.provider != "baseten":
        raise PipelineError("--serving-mode temporary requires --provider baseten")
    if args.run_dir is None:
        raise PipelineError("Baseten temporary evaluation requires --run-dir")
    if any(value is not None for value in (
        args.model_id, args.deployment_id, args.tuned_model,
    )):
        raise PipelineError("Baseten temporary evaluation uses the training run; omit --model-id, --deployment-id, and --tuned-model")
    if getattr(args, "base_model", None) is not None:
        raise PipelineError("Baseten temporary evaluation does not support --base-model; the generated endpoint serves the checkpoint route")
    if args.command == "evaluate":
        if not args.confirm:
            raise PipelineError("Baseten temporary deployment and evaluation require --confirm")
        if args.concurrency < 1:
            raise PipelineError("evaluation concurrency must be positive")
        if args.max_output_tokens < 1:
            raise PipelineError("--max-output-tokens must be positive")
        if not os.environ.get("BASETEN_API_KEY", "").strip():
            raise PipelineError("BASETEN_API_KEY is not set")
        judge_provider = args.judge_model.partition("/")[0]
        if judge_provider in ANTHROPIC_ENDPOINTS:
            anthropic_connection(judge_provider)
        elif not os.environ.get("FIREWORKS_API_KEY", "").strip():
            raise PipelineError("FIREWORKS_API_KEY is not set for the judge")
    return baseten_deployment.plan(
        args.run_dir, accelerator=args.accelerator, max_seq_len=args.max_seq_len,
        timeout=args.deployment_timeout,
    )


def _run_evaluation(args, endpoint: BasetenEndpoint | None, tuned_model: str, *,
                    unlocked: bool = False, baseten_lifecycle=None, baseten_cleanup=None) -> dict:
    if args.run_dir is not None:
        baseten_deployment.validate_evaluation_model(args.run_dir, args.data_dir)
    run = replay_evaluation.run_replay_evaluation
    if unlocked:
        run = getattr(run, "__wrapped__", run)
    lifecycle_options = {}
    if baseten_lifecycle is not None:
        lifecycle_options = {"baseten_lifecycle": baseten_lifecycle, "baseten_cleanup": baseten_cleanup}
    return run(
        args.data_dir, args.output_dir, tuned_model, args.judge_model,
        base_model=args.base_model, concurrency=args.concurrency,
        max_points_per_trajectory=args.max_points_per_trajectory,
        max_output_tokens=args.max_output_tokens, confirm=args.confirm,
        training=replay_evaluation.training_metadata(args.run_dir),
        baseten_endpoint=endpoint, **lifecycle_options,
    )


def _run_temporary_evaluation(args, temporary_plan: dict) -> dict:
    # Check prepared data without overwriting a saved evaluation before its identity is validated.
    with tempfile.TemporaryDirectory(prefix="smithtune-eval-preflight-") as scratch:
        replay_plan = replay_evaluation.prepare_replay_evaluation(
            args.data_dir, Path(scratch), args.max_points_per_trajectory,
            args.max_output_tokens, baseten_context_limit=temporary_plan["settings"]["max_seq_len"],
        )
    if replay_plan["training_base_model"] != temporary_plan["checkpoint"]["base_model"]:
        raise PipelineError("prepared data base model differs from the Baseten training checkpoint")
    replay_evaluation.preflight_langsmith(args.data_dir)

    def temporary():
        return baseten_deployment.temporary(
            args.run_dir, accelerator=args.accelerator, max_seq_len=args.max_seq_len,
            timeout=args.deployment_timeout, confirm=args.confirm,
        )

    with output_lock(args.output_dir):
        if (args.run_dir / "endpoint.json").exists():
            endpoint, tuned_model = baseten_deployment.load_endpoint(args.run_dir, require_ready=False)
            endpoint = BasetenEndpoint(endpoint.model_id, endpoint.deployment_id, temporary_plan["settings"]["max_seq_len"])

            @contextmanager
            def lifecycle():
                with temporary() as serving:
                    if serving != (endpoint, tuned_model):
                        raise PipelineError("Baseten endpoint identity changed before evaluation")
                    yield tuned_model

            return _run_evaluation(
                args, endpoint, tuned_model, unlocked=True, baseten_lifecycle=lifecycle(),
                baseten_cleanup=lambda: baseten_deployment.undeploy(args.run_dir, confirm=True),
            )
        if any((args.output_dir / name).exists() for name in ("evaluation-config.json", "results.jsonl")):
            raise PipelineError("evaluation output already exists without a saved Baseten endpoint; use a new --output-dir")
        with ExitStack() as serving_stack:
            endpoint, tuned_model = serving_stack.enter_context(temporary())

            @contextmanager
            def lifecycle():
                try:
                    yield tuned_model
                finally:
                    # Release serving before the shared evaluator publishes results.
                    serving_stack.close()

            return _run_evaluation(args, endpoint, tuned_model, unlocked=True,
                                   baseten_lifecycle=lifecycle())


def _run_fireworks_evaluation(args):
    from smithtune.providers.fireworks_sampling import FireworksReplaySampler, checkpoint_from_run

    if args.serving_mode not in (None, "existing", "sampler") or any(value is not None for value in (
        args.model_id, args.deployment_id, args.max_seq_len, args.tuned_model,
        args.accelerator,
    )) or args.deployment_timeout != 600:
        raise PipelineError("Fireworks evaluation uses the serverless sampler and --run-dir; omit endpoint and deployment options")
    model = best = None
    if args.run_dir is not None:
        if args.run_dir.resolve() == args.output_dir.resolve():
            raise PipelineError("--run-dir and --output-dir must be different directories")
        model, best = checkpoint_from_run(args.data_dir, args.run_dir)
    if args.command == "eval-plan":
        dataset._require_prepared_provider(_load_json(args.data_dir / "prepared" / "manifest.json"), "fireworks")
        with output_lock(args.output_dir):
            value = replay_evaluation.prepare_replay_evaluation(
                args.data_dir, args.output_dir, args.max_points_per_trajectory, args.max_output_tokens,
            )
            value["serving_mode"] = "serverless"
            if best is not None:
                value["checkpoint"] = best["resume_checkpoint"]
            _json_dump(args.output_dir / "plan.json", value)
            return value
    if best is None:
        raise PipelineError("Fireworks evaluate requires --run-dir containing the saved training result")
    if args.base_model not in (None, model.base_model):
        raise PipelineError("serverless replay compares the checkpoint with its own base model")
    if not args.confirm:
        raise PipelineError("model and judge inference require --confirm")
    replay_evaluation.validate_judge_credentials(args.judge_model)
    sampler = FireworksReplaySampler(model, best["resume_checkpoint"], args.output_dir,
                                     lora_rank=best["lora_rank"], lora_alpha=best["lora_alpha"])
    return replay_evaluation.run_replay_evaluation(
        args.data_dir, args.output_dir, best["resume_checkpoint"], args.judge_model,
        base_model=model.base_model, concurrency=args.concurrency,
        max_points_per_trajectory=args.max_points_per_trajectory, max_output_tokens=args.max_output_tokens,
        replay_sampler=sampler, confirm=args.confirm, training=replay_evaluation.training_metadata(args.run_dir),
    )


def _run_baseten_sampler_evaluation(args):
    from smithtune.providers.baseten_sampling import BasetenReplaySampler, checkpoint_from_run

    if any(value is not None for value in (
        args.model_id, args.deployment_id, args.max_seq_len, args.tuned_model,
        args.accelerator,
    )) or args.deployment_timeout != 600:
        raise PipelineError("Baseten sampler replay uses --run-dir; omit endpoint and deployment options")
    dataset._require_prepared_provider(_load_json(args.data_dir / "prepared" / "manifest.json"), "baseten")
    model = checkpoint = None
    if args.run_dir is not None:
        if args.run_dir.resolve() == args.output_dir.resolve():
            raise PipelineError("--run-dir and --output-dir must be different directories")
        model, checkpoint = checkpoint_from_run(args.data_dir, args.run_dir)
    if args.command == "eval-plan":
        with output_lock(args.output_dir):
            value = replay_evaluation.prepare_replay_evaluation(
                args.data_dir, args.output_dir, args.max_points_per_trajectory, args.max_output_tokens,
            )
            value.update(serving_mode="sampler", evaluated_models=2,
                         capacity="dedicated", cleanup="deactivate both sampler deployments on exit")
            if checkpoint is not None:
                value["checkpoint"] = checkpoint
            _json_dump(args.output_dir / "plan.json", value)
            return value
    if checkpoint is None:
        raise PipelineError("Baseten sampler evaluation requires --run-dir containing the saved training result")
    if args.base_model not in (None, model.base_model):
        raise PipelineError("sampler replay compares the checkpoint with its own base model")
    if not args.confirm:
        raise PipelineError("sampler compute and judge inference require --confirm")
    replay_evaluation.validate_judge_credentials(args.judge_model)
    sampler = BasetenReplaySampler(model, checkpoint, args.output_dir)
    return replay_evaluation.run_replay_evaluation(
        args.data_dir, args.output_dir, checkpoint, args.judge_model,
        base_model=model.base_model, concurrency=args.concurrency,
        max_points_per_trajectory=args.max_points_per_trajectory, max_output_tokens=args.max_output_tokens,
        replay_sampler=sampler, confirm=args.confirm, training=replay_evaluation.training_metadata(args.run_dir),
    )


def _eval_baseten_endpoint(args) -> BasetenEndpoint | None:
    if args.provider != "baseten":
        if args.model_id is not None or args.max_seq_len is not None or args.run_dir is not None:
            raise PipelineError("--model-id, --max-seq-len, and --run-dir require --provider baseten")
        return None
    if args.serving_mode not in (None, "existing") or args.deployment_timeout != 600:
        raise PipelineError("Baseten evaluation supports existing endpoints only; omit Fireworks deployment options")
    if args.run_dir is not None:
        if any(value is not None for value in (args.model_id, args.deployment_id, args.max_seq_len, args.tuned_model)):
            raise PipelineError("--run-dir uses the saved Baseten endpoint; omit --model-id, --deployment-id, --max-seq-len, and --tuned-model")
        endpoint, args.tuned_model = baseten_deployment.load_endpoint(args.run_dir)
        return endpoint
    if args.model_id is None or args.deployment_id is None or args.max_seq_len is None:
        raise PipelineError("Baseten evaluation requires --run-dir or --model-id, --deployment-id, and --max-seq-len")
    endpoint = BasetenEndpoint(args.model_id, args.deployment_id, args.max_seq_len)
    endpoint.validate()
    return endpoint


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["dataset"] and len(arguments) > 1 and arguments[1] in (*dataset_workflow.STAGES, "create", "resume"):
        flags = {argument.split("=", 1)[0] for argument in arguments[2:]}
        for old, replacement in {"--run-dir": "dataset create DIR", "--output": "dataset pull DIR",
                                 "--triage-dir": "dataset push DIR", "--output-dir": "dataset triage DIR"}.items():
            if old in flags:
                parser.error(f"{old} was replaced by the directory argument; use {replacement}")
        if arguments[1] == "triage" and flags & {"--workspace-id", "--project-id", "--start-time", "--end-time", "--filter", "--limit", "--seed"}:
            parser.error("triage uses saved local trajectories; run dataset pull DIR with source flags first")
    args = parser.parse_args(arguments)
    activity = ExitStack()
    try:
        if args.command not in {"doctor", "models", "skill"}:
            receipt = data_rights.require_acknowledgment()
            if args.command == "acknowledge-data-rights":
                print(json.dumps(receipt, indent=2, sort_keys=True))
                return
        command = " ".join(filter(None, (args.command, getattr(args, f"{args.command}_command", None))))
        activity.enter_context(command_status(f"Running {command}"))
        if args.command == "doctor":
            value = diagnose()
        elif args.command == "skill":
            value = triage.export_skill(args.output)
        elif args.command == "models":
            profiles = {"baseten": BASETEN_MODEL_SPECS, "fireworks": FIREWORKS_MODEL_SPECS}
            value = {
                "source": "smithtune_support_registry",
                "live_availability_checked": False,
                "models": [
                    {
                        "provider": provider,
                        "alias": alias,
                        "model_id": model.base_model,
                        "training_context_limit": model.training_context_limit,
                    }
                    for provider, specs in sorted(profiles.items())
                    if args.provider is None or args.provider == provider
                    for alias, model in sorted(specs.items())
                ],
            }
        elif args.command == "dataset":
            if args.dataset_command == "publish-splits":
                value = reporting.publish_prepared_splits(args.data_dir)
            else:
                options = {key: getattr(args, key, None) for key in (
                    "workspace_id", "project_id", "start_time", "end_time", "filter", "limit",
                    "name", "dataset_id", "no_triage", "concurrency", "attempts", "max_output_tokens",
                )}
                options.update(judges=args.judges.split(",") if getattr(args, "judges", None) is not None else None,
                               rules=getattr(args, "rule", None), config_path=getattr(args, "config", None),
                               runner_mode=getattr(args, "runner", None), rubric_path=getattr(args, "rubric", None))
                value = dataset_workflow.run(args.dataset_command, args.directory,
                                             confirm=getattr(args, "confirm", False), **options)
        elif args.command == "capture-contract":
            value = dataset.capture_inference_contract(
                args.workspace_id,
                args.run_id,
                args.output,
            )
        elif args.command == "prepare":
            contract = None
            if args.inference_contract is not None:
                try:
                    contract = load_inference_contract(args.inference_contract)
                except ContractError as exc:
                    raise PipelineError(f"invalid inference contract: {exc}") from exc
            provider = get_provider(args.provider)
            value = provider.prepare(
                args.workspace_id,
                args.dataset_id,
                args.data_dir,
                model_options=_model_options(args),
                reasoning_policy=args.reasoning_policy,
                inference_contract=contract,
                source_workspace_id=args.source_workspace_id,
                split_from=args.split_from,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                fetch=not args.no_fetch,
                check_render=not args.skip_render_check,
                sync_splits=not args.no_sync_splits,
            )
        elif args.command == "plan":
            provider = get_provider(args.provider)
            value = provider.plan(args.data_dir, args.run_id, _settings_from_args(args), **_training_replay(args))
        elif args.command == "train":
            provider = get_provider(args.provider)
            run_id = args.run_id
            if run_id is None:
                run_id = f"sft-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:12]}"
            run_dir = args.run_dir
            if run_dir is None:
                if run_id in {"", ".", ".."} or "/" in run_id or "\\" in run_id:
                    raise PipelineError("run ID must be a single directory name when --run-dir is omitted")
                run_dir = Path.cwd() / "runs" / run_id
            print(f"Run ID: {run_id}\nRun directory: {run_dir.resolve()}", file=sys.stderr)
            result = provider.train(
                args.data_dir,
                run_dir,
                run_id,
                _settings_from_args(args),
                confirm=args.confirm,
                init_from_checkpoint=args.init_from_checkpoint,
                **_training_replay(args),
            )
            value = {**result, "run_id": run_id, "run_dir": str(run_dir.resolve())}
        elif args.command == "promote":
            FireworksProvider().promote(args.run_dir, args.output_model_id, confirm=args.confirm)
            value = {"status": "promoted", "output_model_id": args.output_model_id}
        elif args.command == "deploy":
            fireworks_options = (args.account_id, args.output_model_id, args.deployment_id, args.deployment_shape)
            if args.provider == "baseten":
                if any(value is not None for value in fireworks_options):
                    raise PipelineError("Baseten deploy does not accept --account-id, --output-model-id, --deployment-id, or --deployment-shape")
                if args.accelerator is None or args.max_seq_len is None:
                    raise PipelineError("Baseten deploy requires --accelerator and --max-seq-len")
                value = baseten_deployment.deploy(
                    args.run_dir, accelerator=args.accelerator, max_seq_len=args.max_seq_len,
                    timeout=args.deployment_timeout if args.deployment_timeout is not None else 1800,
                    confirm=args.confirm,
                )
            else:
                if any(value is not None for value in (args.accelerator, args.max_seq_len, args.deployment_timeout)):
                    raise PipelineError("--accelerator, --max-seq-len, and --deployment-timeout require --provider baseten")
                if not all(fireworks_options):
                    raise PipelineError("Fireworks deploy requires --account-id, --output-model-id, --deployment-id, and --deployment-shape")
                value = FireworksProvider().deploy(args.run_dir, *fireworks_options, confirm=args.confirm)
        elif args.command in ("eval-plan", "evaluate") and args.provider == "fireworks":
            if args.output_dir is None:
                if args.run_dir is None:
                    raise PipelineError("supply --run-dir or --output-dir")
                args.output_dir = args.run_dir / "replay"
            value = _run_fireworks_evaluation(args)
        elif args.command in ("eval-plan", "evaluate") and (
            args.serving_mode == "sampler" or (
                args.serving_mode is None and args.model_id is None and args.deployment_id is None
            )
        ):
            if args.output_dir is None:
                if args.run_dir is None:
                    raise PipelineError("supply --run-dir or --output-dir")
                args.output_dir = args.run_dir / "replay"
            value = _run_baseten_sampler_evaluation(args)
        elif args.command == "eval-plan":
            if args.output_dir is None:
                if args.run_dir is None:
                    raise PipelineError("supply --run-dir or --output-dir")
                args.output_dir = args.run_dir / "replay"
            temporary_plan = _temporary_baseten_plan(args)
            baseten_endpoint = None if temporary_plan is not None else _eval_baseten_endpoint(args)
            context = {"baseten_context_limit": temporary_plan["settings"]["max_seq_len"]} if temporary_plan is not None else {}
            if args.run_dir is not None:
                baseten_deployment.validate_evaluation_model(args.run_dir, args.data_dir)
            with output_lock(args.output_dir):
                value = replay_evaluation.prepare_replay_evaluation(
                    args.data_dir,
                    args.output_dir,
                    args.max_points_per_trajectory,
                    args.max_output_tokens,
                    baseten_endpoint=baseten_endpoint,
                    **context,
                )
                if temporary_plan is not None:
                    value["deployment"] = temporary_plan
                    _json_dump(args.output_dir / "plan.json", value)
        elif args.command == "evaluate":
            if args.output_dir is None:
                if args.run_dir is None:
                    raise PipelineError("supply --run-dir or --output-dir")
                args.output_dir = args.run_dir / "replay"
            temporary_plan = _temporary_baseten_plan(args)
            if temporary_plan is not None:
                value = _run_temporary_evaluation(args, temporary_plan)
            else:
                baseten_endpoint = _eval_baseten_endpoint(args)
                if not args.tuned_model:
                    raise PipelineError("evaluate requires --tuned-model, or --provider baseten with --run-dir")
                value = _run_evaluation(args, baseten_endpoint, args.tuned_model)
        else:
            if args.provider == "baseten":
                if args.account_id is not None or args.deployment_id is not None:
                    raise PipelineError("Baseten undeploy uses --run-dir; omit --account-id and --deployment-id")
                if args.run_dir is None:
                    raise PipelineError("Baseten undeploy requires --run-dir")
                value = baseten_deployment.undeploy(args.run_dir, confirm=args.confirm)
            else:
                if args.run_dir is not None:
                    raise PipelineError("undeploy --run-dir requires --provider baseten")
                if not args.account_id or not args.deployment_id:
                    raise PipelineError("Fireworks undeploy requires --account-id and --deployment-id")
                FireworksProvider().undeploy(args.account_id, args.deployment_id, confirm=args.confirm)
                value = {"status": "deleted", "deployment_id": args.deployment_id}
    except (PipelineError, BasetenRuntimeError, subprocess.CalledProcessError) as exc:
        activity.close()
        parser.error(str(exc))
    finally:
        activity.close()
    print(json.dumps(value, indent=2, sort_keys=True))
    if args.command == "dataset" and value.get("status") == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

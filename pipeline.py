#!/usr/bin/env python3
"""Prepare LangSmith trajectories and orchestrate provider-specific SFT."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import fields
from pathlib import Path

import dataset
import evaluation as replay_evaluation
from inference_contract import ContractError, load_inference_contract
from providers.baseten import BasetenSFTSettings
from providers.fireworks import FireworksProvider, SFTSettings as FireworksSFTSettings
from providers.base import CommonSFTSettings, ModelOptions, PipelineError, TrainingOptions
from providers import PROVIDERS, get_provider
from rendering import DEFAULT_REPLAY_MAX_TOKENS


def _parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture_contract = sub.add_parser(
        "capture-contract",
        help="capture an inference contract from one approved LangSmith LLM run",
    )
    capture_contract.add_argument("--workspace-id", required=True)
    capture_contract.add_argument("--run-id", required=True)
    capture_contract.add_argument("--output", type=Path, required=True)

    prep = sub.add_parser("prepare", help="fetch, convert, split, and validate all trajectories")
    prep.add_argument("--provider", choices=tuple(PROVIDERS), default="fireworks")
    prep.add_argument("--data-dir", type=Path, default=project / "data")
    prep.add_argument("--workspace-id", required=True)
    prep.add_argument("--dataset-id", required=True)
    prep.add_argument("--inference-contract", type=Path)
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
    prep.add_argument("--model-profile", default=ModelOptions().model_profile, help="provider model profile or custom")
    prep.add_argument("--base-model")
    prep.add_argument("--tokenizer-model")
    prep.add_argument("--tokenizer-revision")
    prep.add_argument("--renderer")
    prep.add_argument("--max-seq-len", type=int)
    prep.add_argument("--trainer-max-seq-len", type=int)
    prep.add_argument("--thinking-trace-history-mode", choices=["interleaved", "preserved"])
    prep.add_argument("--trust-remote-code", action="store_true")
    prep.add_argument("--requires-tool-declarations", action="store_true")
    prep.add_argument("--supports-reasoning-content", action="store_true")
    prep.add_argument("--default-lora-rank", type=int)
    prep.add_argument("--no-fetch", action="store_true", help="use the existing raw export")
    prep.add_argument("--skip-render-check", action="store_true", help=argparse.SUPPRESS)

    plan = sub.add_parser("plan", help="print the resolved training plan without provisioning resources")
    plan.add_argument(
        "--provider", choices=tuple(PROVIDERS), default="fireworks",
        help="training provider (default: %(default)s)",
    )
    plan.add_argument("--data-dir", type=Path, default=project / "data")
    plan.add_argument("--run-id", default="langsmith-sft")

    training = sub.add_parser("train", help="run paid serverless SFT after plan approval")
    training.add_argument(
        "--provider", choices=tuple(PROVIDERS), default="fireworks",
        help="training provider (default: %(default)s)",
    )
    training.add_argument("--data-dir", type=Path, default=project / "data")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument("--init-from-checkpoint")
    training.add_argument("--confirm", action="store_true")

    common_defaults = CommonSFTSettings()
    fireworks_defaults = FireworksSFTSettings()
    baseten_defaults = BasetenSFTSettings()
    for command in (plan, training):
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
    deployment.add_argument("--account-id", required=True)
    deployment.add_argument("--output-model-id", required=True)
    deployment.add_argument("--deployment-id", required=True)
    deployment.add_argument("--deployment-shape", required=True)
    deployment.add_argument("--confirm", action="store_true")

    eval_plan = sub.add_parser("eval-plan", help="build held-out trajectory replay cases")
    evaluation = sub.add_parser("evaluate", help="compare base and tuned actions with a calibrated judge")
    for command in (eval_plan, evaluation):
        command.add_argument("--data-dir", type=Path, default=project / "data")
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument(
            "--max-points-per-trajectory",
            type=int,
            default=replay_evaluation.DEFAULT_REPLAY_POINTS,
            help="optional cap; default evaluates every assistant turn",
        )
        command.add_argument("--max-output-tokens", type=int, default=DEFAULT_REPLAY_MAX_TOKENS)
    evaluation.add_argument("--tuned-model", required=True)
    evaluation.add_argument(
        "--base-model",
        help="optional base-model serving route for before-versus-after comparison",
    )
    evaluation.add_argument("--concurrency", type=int, default=replay_evaluation.DEFAULT_EVALUATION_CONCURRENCY)
    evaluation.add_argument("--judge-model", default=replay_evaluation.DEFAULT_JUDGE_MODEL)
    evaluation.add_argument("--confirm", action="store_true")

    remove = sub.add_parser("undeploy", help="delete the on-demand deployment")
    remove.add_argument("--account-id", required=True)
    remove.add_argument("--deployment-id", required=True)
    remove.add_argument("--confirm", action="store_true")
    return parser


def _model_options(args: argparse.Namespace) -> ModelOptions:
    return ModelOptions(**{field.name: getattr(args, field.name) for field in fields(ModelOptions)})


def _settings_from_args(args: argparse.Namespace) -> CommonSFTSettings:
    options = TrainingOptions(**{
        field.name: getattr(args, field.name) for field in fields(TrainingOptions)
    })
    return get_provider(args.provider).settings_from_options(options)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "capture-contract":
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
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                fetch=not args.no_fetch,
                check_render=not args.skip_render_check,
            )
        elif args.command == "plan":
            provider = get_provider(args.provider)
            value = provider.plan(args.data_dir, args.run_id, _settings_from_args(args))
        elif args.command == "train":
            provider = get_provider(args.provider)
            value = provider.train(
                args.data_dir,
                args.run_dir,
                args.run_id,
                _settings_from_args(args),
                confirm=args.confirm,
                init_from_checkpoint=args.init_from_checkpoint,
            )
        elif args.command == "promote":
            FireworksProvider().promote(args.run_dir, args.output_model_id, confirm=args.confirm)
            value = {"status": "promoted", "output_model_id": args.output_model_id}
        elif args.command == "deploy":
            value = FireworksProvider().deploy(
                args.run_dir,
                args.account_id,
                args.output_model_id,
                args.deployment_id,
                args.deployment_shape,
                confirm=args.confirm,
            )
        elif args.command == "eval-plan":
            value = replay_evaluation.prepare_replay_evaluation(
                args.data_dir,
                args.output_dir,
                args.max_points_per_trajectory,
                args.max_output_tokens,
            )
        elif args.command == "evaluate":
            value = replay_evaluation.run_replay_evaluation(
                args.data_dir,
                args.output_dir,
                args.tuned_model,
                args.judge_model,
                base_model=args.base_model,
                concurrency=args.concurrency,
                max_points_per_trajectory=args.max_points_per_trajectory,
                max_output_tokens=args.max_output_tokens,
                confirm=args.confirm,
            )
        else:
            FireworksProvider().undeploy(args.account_id, args.deployment_id, confirm=args.confirm)
            value = {"status": "deleted", "deployment_id": args.deployment_id}
    except (PipelineError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

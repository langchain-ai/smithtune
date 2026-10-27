"""Experimental customer-account SageMaker jobs; no resources during planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
import re
import shlex
import shutil
from typing import Any

from smithtune.artifacts import _json_dump, _load_json, output_lock
from smithtune.models import resolve_model_options
from smithtune.providers.sagemaker.execution import AWSJobConfig, split_s3_uri
from smithtune.providers.base import CommonSFTSettings, ModelOptions, ModelSpec, PipelineError, TrainingOptions
from smithtune.rendering import resolve_rendering_model
from smithtune.training.contracts import CPU_TRAINING_PROFILES, PREPARED_FILES, WORKER_MODELS, describe_sft_job, load_prepared as load_worker_prepared, sha256_file, WorkerSettings


# Reuse the existing pinned formatter. This is an experimental worker target,
# not a claim that any particular hardware/context combination has been tested.
MODEL_SPECS = {name: replace(spec, provider="sagemaker") for name, spec in WORKER_MODELS.items()}
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}


@dataclass(frozen=True)
class SageMakerSFTSettings(WorkerSettings):
    aws: AWSJobConfig | None = None
    lora_rank: int = 8

    def validate(self) -> None:
        for name in ("max_epochs", "early_stopping_patience", "batch_size", "seed", "lora_rank"):
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
        if self.aws is None:
            raise PipelineError("SageMaker requires --aws-config; see docs/aws-training.md")
        self.aws.validate()
        if (self.device == "cpu") == self.aws.uses_gpu():
            raise PipelineError("worker device must match the instance family: cpu for ml.m*/c*/r*, cuda:0 for GPU types")

    def worker_settings(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(WorkerSettings)} | {"lora_rank": self.lora_rank}


def load_prepared(data_dir: Path):
    return load_worker_prepared(data_dir, expected_provider="sagemaker")


def training_request(config: AWSJobConfig, job_name: str) -> dict:
    config.validate()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", job_name):
        raise PipelineError("SageMaker run ID must be 1–63 alphanumeric/hyphen characters")
    prefix = f"{config.s3_prefix.rstrip('/')}/{job_name}"
    request = {
        "TrainingJobName": job_name,
        "RoleArn": config.role_arn,
        "AlgorithmSpecification": {"TrainingImage": config.training_image, "TrainingInputMode": "File"},
        "InputDataConfig": [{
            "ChannelName": "training", "InputMode": "File",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"{prefix}/input/", "S3DataDistributionType": "FullyReplicated"}},
        }],
        "OutputDataConfig": {"S3OutputPath": f"{prefix}/output"},
        "CheckpointConfig": {"S3Uri": f"{prefix}/checkpoints", "LocalPath": "/opt/ml/checkpoints"},
        "ResourceConfig": {"InstanceType": config.instance_type, "InstanceCount": config.instance_count, "VolumeSizeInGB": config.volume_size_gb},
        "StoppingCondition": {"MaxRuntimeInSeconds": config.max_runtime_seconds},
        "EnableManagedSpotTraining": False,
        "EnableNetworkIsolation": False,
        "Tags": [{"Key": "smithtune:provider", "Value": "sagemaker"}],
    }
    if config.subnets:
        request["VpcConfig"] = {"Subnets": list(config.subnets), "SecurityGroupIds": list(config.security_group_ids)}
    if config.output_kms_key:
        request["OutputDataConfig"]["KmsKeyId"] = config.output_kms_key
    if config.volume_kms_key:
        request["ResourceConfig"]["VolumeKmsKeyId"] = config.volume_kms_key
    if config.instance_preferences:
        del request["ResourceConfig"]["InstanceType"]
        # AWS forbids this field with preferences, including an explicit false.
        del request["EnableManagedSpotTraining"]
        request["ResourceConfig"]["InstancePreferences"] = [
            {"InstanceType": item.instance_type} for item in config.instance_preferences
        ]
    if config.max_pending_seconds is not None:
        request["StoppingCondition"]["MaxPendingTimeInSeconds"] = config.max_pending_seconds
    return request


class SageMakerProvider:
    name = "sagemaker"

    def __init__(self, *, session_factory=None) -> None:
        self._session_factory = session_factory

    def _session(self, config: AWSJobConfig):
        if self._session_factory:
            return self._session_factory(config)
        try:
            import boto3
        except ImportError as exc:
            raise PipelineError("install smithtune's aws extra for SageMaker submission/status") from exc
        # The standard credential chain supports profiles, SSO and workload roles.
        return boto3.Session(profile_name=config.profile, region_name=config.region)

    def model_from_options(self, options: ModelOptions):
        if options.max_seq_len is None:
            raise PipelineError("experimental SageMaker preparation requires an explicit --max-seq-len; start with 4096")
        selected = options.model or "qwen3p5-9b"
        # Profiles share a tokenizer, so match aliases and base models only.
        matches = {alias: spec for alias, spec in MODEL_SPECS.items() if selected in {alias, spec.base_model}}
        if len(matches) != 1:
            raise PipelineError(f"no supported SageMaker model profile for {selected!r}; supported aliases: {', '.join(sorted(MODEL_SPECS))}")
        return resolve_model_options(replace(options, model=selected), matches, provider=self.name)

    def settings_from_options(self, options: TrainingOptions) -> SageMakerSFTSettings:
        rejected = [name for name in (
            "lora_alpha", "pipeline_depth", "training_mode", "training_shape", "provisioning_timeout",
            "microbatch_token_budget", "max_spend_usd", "hourly_rate_usd", "replicas",
            "spend_reserve_fraction", "max_dropped_training_rows",
        ) if getattr(options, name) is not None]
        if rejected:
            raise PipelineError("SageMaker does not accept these provider-specific options: " + ", ".join(rejected))
        if options.validation_every_steps is not None:
            raise PipelineError("--validation-every-steps is not supported by the SageMaker worker; it validates at epoch end")
        config = AWSJobConfig.read(options.aws_config) if options.aws_config else None
        settings = SageMakerSFTSettings(
            **{field.name: getattr(options, field.name) for field in fields(CommonSFTSettings)},
            lora_rank=8 if options.lora_rank is None else options.lora_rank, aws=config,
            trainer_backend=options.aws_trainer or "torch",
            device="cuda:0" if config is None or config.uses_gpu() else "cpu",
        )
        settings.validate()
        return settings

    def prepare(self, workspace_id, dataset_id, data_dir, *, model_options, **kwargs):
        from smithtune.dataset import DEFAULT_TEST_FRACTION, DEFAULT_VALIDATION_FRACTION, prepare_dataset

        if not kwargs.get("check_render", True):
            raise PipelineError("SageMaker preparation requires rendering checks")
        for key, default in (("validation_fraction", DEFAULT_VALIDATION_FRACTION), ("test_fraction", DEFAULT_TEST_FRACTION)):
            if kwargs.get(key) is None:
                kwargs[key] = default
        model = resolve_rendering_model(self.model_from_options(model_options))
        return prepare_dataset(workspace_id, dataset_id, model, data_dir, **kwargs)

    def plan(self, data_dir, run_id, settings, *, replay=None):
        if replay is not None:
            raise PipelineError("SageMaker automatic replay/endpoint deployment is not implemented yet")
        if not isinstance(settings, SageMakerSFTSettings):
            raise PipelineError("SageMaker requires SageMaker SFT settings")
        settings.validate()
        manifest, _rows = load_prepared(data_dir)
        if settings.device == "cpu" and manifest["model"]["name"] not in CPU_TRAINING_PROFILES:
            raise PipelineError("CPU instance types are allowed only for the qwen3p5-0p8b-test pipeline profile; choose GPU instance types")
        request = training_request(settings.aws, run_id)
        rates = settings.aws.instance_rates()
        known_rates = [rate for rate in rates.values() if rate is not None]
        rate = max(known_rates) if known_rates else None
        duration = settings.aws.instance_count * settings.aws.max_runtime_seconds / 3600
        return {
            "provider": self.name, "run_id": run_id, "experimental": True,
            # Submission returns once AWS accepts the job; no local process must stay alive.
            "execution": {"requires_active_client": False},
            "model": manifest["model"], "dataset": manifest["split"],
            "training": describe_sft_job(
                ModelSpec(**manifest["model"]), settings, lora_rank=settings.lora_rank,
                prepared_manifest_uri=f"{settings.aws.s3_prefix.rstrip('/')}/{run_id}/input/prepared/manifest.json",
            ),
            "worker_settings": settings.worker_settings(), "create_training_job": request,
            "input_sha256": {name: sha256_file(Path(data_dir) / "prepared" / name) for name in PREPARED_FILES},
            "cost": {
                "hourly_rate_usd": rate,
                "runtime_compute_estimate_usd": None if rate is None else rate * duration,
                "instance_options": [{"instance_type": name, "hourly_rate_usd": value,
                                      "runtime_compute_estimate_usd": None if value is None else value * duration}
                                     for name, value in rates.items()],
                "note": "User-supplied rates; the headline estimate uses the highest listed rate. This is a runtime estimate, not a billing cap. Provisioning, storage, logs and serving are separate.",
            },
            "warnings": [
                *(["CPU training exercises the pipeline with the 0.8B test profile only; its timings and memory say nothing about GPU runs."]
                  if settings.device == "cpu" else []),
                "Single GPU/process worker; additional GPUs in an instance are not used. Only a short Qwen3.5-9B LoRA smoke on L40S has passed; validate the required context and hardware.",
                "Model context is a formatting ceiling, not a memory/throughput guarantee. No truncation is allowed.",
                "No automatic endpoint, replay, Spot recovery or cross-job resume in this prototype.",
                "The worker downloads public model weights from Hugging Face; VPC jobs need outbound access or a compatible cache.",
            ],
        }

    def train(self, data_dir, run_dir, run_id, settings, *, confirm, init_from_checkpoint=None, replay=None):
        if not confirm:
            raise PipelineError("SageMaker training creates paid resources; review plan and rerun with --confirm")
        if init_from_checkpoint is not None:
            raise PipelineError("SageMaker cross-job resume is not implemented; checkpoints are preserved in S3")
        plan = self.plan(data_dir, run_id, settings, replay=replay)
        run_dir = Path(run_dir)
        with output_lock(run_dir):
            if any(path.name != ".smithtune.lock" for path in run_dir.iterdir()):
                raise PipelineError(f"run directory must be new or empty: {run_dir}")
            _json_dump(run_dir / "plan.json", plan)
            bundle = run_dir / "input"
            bundle.mkdir()
            prepared = bundle / "prepared"
            prepared.mkdir()
            for name in PREPARED_FILES:
                shutil.copyfile(Path(data_dir) / "prepared" / name, prepared / name)
                if sha256_file(prepared / name) != plan["input_sha256"][name]:
                    raise PipelineError("prepared data changed during staging; create a new run")
            _json_dump(bundle / "bundle.json", {
                "schema_version": 1, "run_id": run_id, "files": plan["input_sha256"],
                "settings": plan["worker_settings"], "training": plan["training"],
            })
            _json_dump(run_dir / "request.json", plan["create_training_job"])
            receipt = {
                "provider": self.name, "job_name": run_id, "status": "staging",
                "aws": asdict(settings.aws), "request_sha256": sha256_file(run_dir / "request.json"),
                "stop_command": f"smithtune aws stop --run-dir {shlex.quote(str(run_dir.resolve()))} --confirm",
            }
            _json_dump(run_dir / "job.json", receipt)
            try:
                session = self._session(settings.aws)
                account = session.client("sts").get_caller_identity()["Account"]
                if account != settings.aws.role_arn.split(":")[4]:
                    raise PipelineError("AWS caller account must match the execution role account")
                s3 = session.client("s3")
                client = session.client("sagemaker")
                bucket, prefix = split_s3_uri(settings.aws.s3_prefix)
                try:
                    client.describe_training_job(TrainingJobName=run_id)
                except Exception as exc:
                    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                    if code not in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"}:
                        raise
                else:
                    raise PipelineError("job name already exists; inspect it and choose a new run ID")
                if s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{run_id}/", MaxKeys=1).get("KeyCount", 0):
                    raise PipelineError("S3 run prefix already contains objects; choose a new run ID")
                for path in sorted(bundle.rglob("*")):
                    if path.is_file():
                        extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": settings.aws.output_kms_key} if settings.aws.output_kms_key else {}
                        s3.upload_file(str(path), bucket, f"{prefix}/{run_id}/input/{path.relative_to(bundle).as_posix()}", ExtraArgs=extra)
                # Persist before the mutating call: a timeout is ambiguous. Never
                # automatically retry with another job name or discard the receipt.
                receipt["status"] = "submission_pending"
                _json_dump(run_dir / "job.json", receipt)
                response = client.create_training_job(**plan["create_training_job"])
                receipt.update(status="Submitted", training_job_arn=response["TrainingJobArn"])
                _json_dump(run_dir / "job.json", receipt)
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(
                    f"AWS submission did not complete locally ({type(exc).__name__}); "
                    f"receipt preserved at {run_dir / 'job.json'}. "
                    "Use smithtune aws status before retrying; a submitted job may still be running."
                ) from None
        return receipt

    def status(self, run_dir: Path, *, stop: bool = False, confirm: bool = False) -> dict:
        if stop and not confirm:
            raise PipelineError("stopping a training job requires --confirm")
        with output_lock(run_dir):
            receipt = _load_json(run_dir / "job.json")
            if receipt.get("provider") != self.name:
                raise PipelineError("run is not a SageMaker job")
            request_path = run_dir / "request.json"
            if sha256_file(request_path) != receipt["request_sha256"]:
                raise PipelineError("saved SageMaker request changed")
            config = AWSJobConfig(**receipt["aws"])
            config.validate()
            request = _load_json(request_path)
            if training_request(config, receipt["job_name"]) != request:
                raise PipelineError("saved SageMaker identity differs from its request")
            try:
                client = self._session(config).client("sagemaker")
                result = client.describe_training_job(TrainingJobName=receipt["job_name"])
                remote_inputs = result.get("InputDataConfig", [])
                if (result.get("RoleArn") != config.role_arn
                        or result.get("AlgorithmSpecification", {}).get("TrainingImage") != config.training_image
                        or len(remote_inputs) != 1
                        or remote_inputs[0].get("ChannelName") != "training"
                        or remote_inputs[0].get("DataSource") != request["InputDataConfig"][0]["DataSource"]
                        or result.get("OutputDataConfig", {}).get("S3OutputPath") != request["OutputDataConfig"]["S3OutputPath"]):
                    raise PipelineError("remote job does not match the saved request")
                resources = result.get("ResourceConfig", {})
                for key, expected in request["ResourceConfig"].items():
                    actual = resources.get(key)
                    if key == "InstancePreferences":
                        # Describe may add optional metadata to each preference.
                        matches = (isinstance(actual, list) and len(actual) == len(expected)
                                   and all(isinstance(item, dict) and all(item.get(k) == v for k, v in wanted.items())
                                           for item, wanted in zip(actual, expected, strict=True)))
                    else:
                        matches = actual == expected
                    if not matches:
                        raise PipelineError("remote job resources do not match the saved request")
                selected_type = resources.get("SelectedInstanceType") if config.instance_preferences else resources.get("InstanceType")
                selected_count = resources.get("SelectedInstanceCount") if config.instance_preferences else resources.get("InstanceCount")
                if ((selected_type is not None and selected_type not in config.instance_rates())
                        or (selected_count is not None and selected_count != config.instance_count)):
                    raise PipelineError("remote job selected an unapproved instance allocation")
                state = result["TrainingJobStatus"]
                if stop and state not in TERMINAL_STATUSES and state != "Stopping":
                    client.stop_training_job(TrainingJobName=receipt["job_name"])
                    state = "Stopping"
                receipt.update(status=state, training_job_arn=result["TrainingJobArn"])
                receipt["model_artifacts"] = result.get("ModelArtifacts", {}).get("S3ModelArtifacts")
                receipt["secondary_status"] = result.get("SecondaryStatus")
                receipt["phase"] = {"Pending": "waiting_for_capacity", "Downloading": "starting", "Training": "training",
                                    "Uploading": "uploading"}.get(result.get("SecondaryStatus"), state.lower())
                for source, key in (("CreationTime", "created_at"), ("TrainingStartTime", "training_started_at"), ("TrainingEndTime", "training_ended_at")):
                    receipt[key] = result[source].isoformat() if result.get(source) is not None else None
                receipt["selected_instance_type"] = selected_type
                receipt["selected_instance_count"] = selected_count
                receipt["selected_hourly_rate_usd"] = config.instance_rates().get(selected_type)
                receipt["billable_seconds"] = result.get("BillableTimeInSeconds")
                # Remote failure text may contain data or secret-bearing URLs;
                # point to CloudWatch instead of echoing arbitrary container logs.
                receipt["logs"] = {"group": "/aws/sagemaker/TrainingJobs", "stream_prefix": receipt["job_name"]}
                _json_dump(run_dir / "job.json", receipt)
                return receipt
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(f"AWS job lookup/stop failed ({type(exc).__name__}); saved receipt is unchanged") from None

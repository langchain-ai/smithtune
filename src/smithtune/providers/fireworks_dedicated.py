"""Dedicated LoRA capacity, pinned shape selection, and owned-resource cleanup."""

from __future__ import annotations

import math
import os
import re
import signal
import sys
import threading
import uuid
from datetime import datetime, UTC
from contextlib import ExitStack
from pathlib import Path

from smithtune.artifacts import _json_dump, _utc_now
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, FIREWORKS_BASE_URL, _set_skill_session
from smithtune.providers.fireworks_sampling import FireworksReplaySampler
from smithtune.providers.fireworks_training import ServerlessTraining

_SHAPE = re.compile(r"accounts/[a-zA-Z0-9_-]+/trainingShapes/[a-zA-Z0-9_-]+(?:/versions/[a-zA-Z0-9_-]+)?")
_DEPLOYMENT_SHAPE = re.compile(r"accounts/[a-zA-Z0-9_-]+/deploymentShapes/[a-zA-Z0-9_-]+/versions/[a-zA-Z0-9_-]+")


def validate_shape_name(name: str, *, versioned: bool = False) -> None:
    if not isinstance(name, str) or not _SHAPE.fullmatch(name) or (versioned and "/versions/" not in name):
        raise PipelineError("training shape must be accounts/<account>/trainingShapes/<shape>[/versions/<version>]")


def _client_options():
    if not os.environ.get("FIREWORKS_API_KEY", "").strip():
        raise PipelineError("FIREWORKS_API_KEY is not set; dedicated shape discovery requires read-only account access")
    _set_skill_session()
    return dict(api_key=os.environ["FIREWORKS_API_KEY"], base_url=FIREWORKS_BASE_URL,
                additional_headers={"X-Fireworks-Client-Source": CLIENT_SOURCE,
                                    "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]})


def validate_saved_shape(value, context: int, *, replay: bool = True):
    if not isinstance(value, dict):
        raise PipelineError("saved dedicated run has no training shape; plan again")
    validate_shape_name(value.get("training_shape"), versioned=True)
    if value.get("trainer_mode") != "LORA_TRAINER":
        raise PipelineError("dedicated SFT requires a LoRA-capable training shape")
    limit = value.get("max_supported_context_length")
    if type(limit) is not int or limit < context:
        raise PipelineError("training shape context is smaller than the prepared model context")
    timeout = value.get("provisioning_timeout")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 1 <= timeout <= 7200:
        raise PipelineError("saved dedicated provisioning timeout must be between 1 and 7200 seconds")
    if replay and not _DEPLOYMENT_SHAPE.fullmatch(value.get("deployment_shape") or ""):
        raise PipelineError("dedicated replay requires a linked versioned deployment shape")


def resolve_shape(model, lora_rank: int, override: str | None, *, timeout: float, replay: bool):
    """Read the account catalog; never provision resources during planning."""
    from fireworks.training.sdk.trainer import TrainerJobManager
    from training.utils.training_shapes import auto_select_training_shape

    if override is not None:
        validate_shape_name(override)
    manager = TrainerJobManager(**_client_options())
    try:
        shape = override or auto_select_training_shape(
            manager, base_model=model.base_model, lora_rank=lora_rank, max_seq_len=model.max_seq_len,
        )
        profile = manager.resolve_training_profile(shape)
    except (ValueError, RuntimeError) as exc:
        raise PipelineError(f"cannot resolve dedicated LoRA training capacity: {exc}") from exc
    finally:
        manager.close()
    value = {
        "selection": "explicit" if override else "automatic",
        "replay": replay,
        "training_shape": profile.training_shape_version,
        "trainer_mode": profile.trainer_mode,
        "max_supported_context_length": profile.max_supported_context_length,
        "node_count": profile.node_count,
        "accelerator_type": profile.accelerator_type,
        "accelerator_count": profile.accelerator_count,
        "deployment_shape": profile.deployment_shape_version or None,
        "provisioning_timeout": timeout,
    }
    validate_saved_shape(value, model.max_seq_len, replay=replay)
    if (type(value["node_count"]) is not int or value["node_count"] < 1 or
            type(value["accelerator_count"]) is not int or value["accelerator_count"] < 1 or
            not value["accelerator_type"]):
        raise PipelineError("training shape has no usable hardware details")
    return value


class TerminationRequested(BaseException):
    """SIGTERM requests orderly cleanup of dedicated resources."""


class DedicatedResources:
    """Persist owned IDs before creation, including SDK partial-provision failures.

    SDK delete helpers swallow errors, and managed close cannot clean up before
    its readiness handle exists. Use the control plane directly and record each
    outcome so cleanup failure cannot look like released capacity.
    """

    def __init__(self, receipt_path: Path):
        self.path = receipt_path
        self.receipt = {"training_mode": "dedicated", "resources": [], "status": "provisioning",
                        "started_at_utc": _utc_now()}
        self._stack = ExitStack()
        self._clients = ExitStack()

    def __enter__(self):
        from fireworks.training.sdk import FireworksClient

        self.control = FireworksClient(**_client_options())
        self._stack.callback(self.control.close)
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)
            self._stack.callback(signal.signal, signal.SIGTERM, previous)

            def terminate(_number, _frame):
                raise TerminationRequested("SIGTERM requested Fireworks dedicated cleanup")

            signal.signal(signal.SIGTERM, terminate)
        self._save()
        return self

    def _save(self):
        _json_dump(self.path, self.receipt)

    def reserve(self, collection: str) -> str:
        resource_id = f"smithtune-{uuid.uuid4().hex}"
        name = f"accounts/{self.control.account_id}/{collection}/{resource_id}"
        # Refuse to claim existing resources, even if an ID were to collide.
        response = self.control._get(f"/v1/{name}", timeout=30)
        if response.status_code != 404:
            response.raise_for_status()
            raise PipelineError(f"refusing to take ownership of existing resource {name}")
        self.receipt["resources"].append({"name": name, "status": "reserved"})
        self._save()
        return resource_id

    def service(self, *, base_model, tokenizer_model, rank, context, learning_rate, dedicated, replay):
        from training.utils.config import DeployConfig, TrainerConfig
        from training.utils.service import build_service_client

        validate_saved_shape(dedicated, context, replay=replay)
        job = self.reserve("rlorTrainerJobs")
        deployment = self.reserve("deployments") if replay else None
        timeout = dedicated["provisioning_timeout"]
        service = build_service_client(
            **_client_options(), base_model=base_model, tokenizer_model=tokenizer_model,
            max_lora_rank=rank, max_context_length=context, learning_rate=learning_rate,
            trainer=TrainerConfig(job_id=job, training_shape_id=dedicated["training_shape"],
                                  timeout_s=timeout, pending_timeout_s=timeout),
            deployment=(DeployConfig(deployment_id=deployment, deployment_shape=dedicated["deployment_shape"],
                                     deployment_timeout_s=timeout) if replay else None),
            # This owner deletes through a checked control-plane call after SDK close.
            cleanup_trainer_on_close=False, cleanup_deployment_on_close=None,
        )
        self._clients.callback(service.close)
        self.receipt.update(trainer_job_id=job, tuned_deployment_id=deployment,
                            training_shape=dedicated["training_shape"])
        self._save()
        return service

    def ready(self):
        self.receipt["status"] = "running"
        self._save()

    def __exit__(self, exc_type, exc, tb):
        errors = []
        # Close samplers/training clients before deleting their remote resources.
        # Keep the control connection open through all deletion attempts.
        resources = list(reversed(self.receipt["resources"]))
        try:
            try:
                self._clients.close()
            except Exception:
                errors.append("SDK client close")
            for resource in resources:
                name = resource["name"]
                suffix = "?ignoreChecks=true&hard=true" if "/deployments/" in name else ""
                try:
                    response = self.control._delete(f"/v1/{name}{suffix}", timeout=30)
                    if response.status_code != 404:
                        response.raise_for_status()
                    resource["status"] = "delete_requested"
                except Exception:
                    resource["status"] = "cleanup_failed"
                    errors.append(name)
                self._save()
        finally:
            try:
                self._stack.close()
            except Exception:
                errors.append("SDK client close")
            self.receipt.update(status="cleanup_failed" if errors else "closed", closed_at_utc=_utc_now())
            self._save()
        if errors:
            failure = PipelineError(f"dedicated cleanup failed for {', '.join(errors)}; inspect {self.path} and delete remaining resources in Fireworks")
            if exc is not None:
                raise failure from exc
            raise failure


def checkpoint_parts(checkpoint: str):
    if not isinstance(checkpoint, str):
        raise PipelineError("dedicated replay checkpoint must be <trainer-job>:<checkpoint>")
    job, separator, name = checkpoint.partition(":")
    if not separator or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", job):
        raise PipelineError("dedicated replay checkpoint must be <trainer-job>:<checkpoint>")
    if any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", part) for part in name.split("/")):
        raise PipelineError("dedicated replay checkpoint has an invalid logical name")
    return job, name


def resolve_checkpoint(client, checkpoint: str):
    """Translate a cookbook cross-job reference into the SDK storage path."""
    job, name = checkpoint_parts(checkpoint)
    return client.resolve_checkpoint_path(name, source_job_id=job)


def _checkpoint_time(row):
    try:
        parsed = datetime.fromisoformat((row.get("createTime") or "").replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()
    except (ValueError, TypeError):
        return float("-inf")


class DedicatedTraining(ServerlessTraining):
    """Reuse the streaming epoch loop with a dedicated LoRA trainer."""

    def __init__(self, config, run_dir: Path, *, dedicated, reasoning_effort=None):
        super().__init__(config, run_dir, reasoning_effort=reasoning_effort)
        self.dedicated = dedicated

    def __enter__(self):
        from training.utils import ReconnectableClient, RunnerIO
        from training.utils.checkpoints import TrainingCheckpoints
        from training.utils.runner_state import start_running

        try:
            self.runner = self._stack.enter_context(RunnerIO(self.cfg.runner))
            self.resources = self._stack.enter_context(DedicatedResources(self.run_dir / "session.json"))
            self.service = self.resources.service(
                base_model=self.cfg.base_model, tokenizer_model=self.cfg.tokenizer_model,
                rank=self.cfg.lora_rank, context=self.cfg.max_seq_len, learning_rate=self.cfg.learning_rate,
                dedicated=self.dedicated, replay=self.dedicated.get("replay", False),
            )
            raw_client = self.service.create_training_client(
                self.cfg.base_model, lora_rank=self.cfg.lora_rank, lora_alpha=self.cfg.lora_alpha,
            )
            self.job_id = self.service.trainer_job_id
            self.client = ReconnectableClient.from_training_client(
                raw_client, base_model=self.cfg.base_model, lora_rank=self.cfg.lora_rank,
                job_id=self.job_id, service=self.service,
            )
            self.runner.set_accelerator_info(self.service.accelerator_type, self.service.accelerator_count,
                                             profile=self.service.training_profile)
            self.checkpoints = TrainingCheckpoints(
                self.client, self.resources.control, trainer_id=self.job_id,
                log_path=self.cfg.log_path, lora_rank=self.cfg.lora_rank,
            )
            self.checkpoints.resume(init_from_checkpoint=self.cfg.init_from_checkpoint)
            self.resources.ready()
            start_running(self.runner, total_steps=self.total_steps)
            return self
        except BaseException:
            self._stack.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.training_completed:
            return self._stack.__exit__(None, None, None)
        return self._stack.__exit__(exc_type, exc, tb)

    def snapshot(self, checkpoint: str) -> str:
        job, name = checkpoint_parts(checkpoint)
        path = self.client.resolve_checkpoint_path(name) if job == self.job_id else resolve_checkpoint(self.client, checkpoint)
        self.client.load_state(path)
        return self.client.save_weights_for_sampler("replay").path

    def epoch_checkpoints(self):
        rows = [row for row in self.resources.control.list_checkpoints(self.job_id)
                if isinstance(row.get("name"), str) and row["name"]]
        training = sorted((row for row in rows if (row.get("checkpointType") or "").endswith(("TRAINING", "TRAINING_LORA"))),
                          key=_checkpoint_time, reverse=True)
        promotable = sorted((row for row in rows if row.get("promotable")),
                            key=_checkpoint_time, reverse=True)
        if not training or not promotable:
            raise PipelineError(f"dedicated trainer {self.job_id} has no resumable and promotable checkpoint")
        return {"resume_checkpoint": f"{self.job_id}:{training[0]['name'].rsplit('/', 1)[-1]}",
                "promotable_checkpoint": promotable[0]["name"]}


class DedicatedReplaySampler(FireworksReplaySampler):
    """Compare tuned weights with a separate immutable base deployment."""

    def __init__(self, model, checkpoint: str, output_dir: Path, *, dedicated, **kwargs):
        checkpoint_parts(checkpoint)
        super().__init__(model, checkpoint, output_dir, **kwargs)
        validate_saved_shape(dedicated, self.model.max_seq_len)
        self.dedicated = dedicated
        self.config.update(serving_mode="dedicated", training_shape=dedicated["training_shape"],
                           deployment_shape=dedicated["deployment_shape"])

    def __enter__(self):
        from fireworks.training.sdk import DeploymentConfig

        try:
            resources = self._stack.enter_context(DedicatedResources(self.output_dir / "resources.json"))
            if self.service is None:
                self.service = resources.service(
                    base_model=self.model.base_model, tokenizer_model=self.model.tokenizer_model,
                    rank=self.lora_rank, context=self.model.max_seq_len, learning_rate=1e-4,
                    dedicated=self.dedicated, replay=True,
                )
                client = self.service.create_training_client(
                    self.model.base_model, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                )
                client.load_state(resolve_checkpoint(client, self.checkpoint)).result()
                self.snapshot = client.save_weights_for_sampler(f"replay-{uuid.uuid4().hex[:8]}").result().path
            tuned = self.service.create_sampling_client(model_path=self.snapshot, tokenizer=self.renderer.tokenizer)
            self._stack.callback(tuned.close)
            self._samplers[self.checkpoint] = tuned
            base_id = resources.reserve("deployments")
            base_backend = self.service.create_inference_deployment_sampler(
                DeploymentConfig(deployment_id=base_id, base_model=self.model.base_model,
                                 deployment_shape=self.dedicated["deployment_shape"],
                                 min_replica_count=1, max_replica_count=1, enable_hot_load=False),
                timeout_s=self.dedicated["provisioning_timeout"], cleanup_on_close=None,
                tokenizer=self.renderer.tokenizer,
            )
            base = self.service.create_sampling_client(deployment_sampler=base_backend, tokenizer=self.renderer.tokenizer)
            self._stack.callback(base.close)
            self._samplers[self.model.base_model] = base
            resources.ready()
            self.receipt = {**self.config, "trainer_job_id": self.service.trainer_job_id,
                            "tuned_deployment_id": self.service.deployment_id, "base_deployment_id": base_id,
                            "snapshot": self.snapshot, "started_at_utc": _utc_now(), "status": "running"}
            _json_dump(self.output_dir / "sampler.json", self.receipt)
            return self.checkpoint
        except BaseException:
            self._stack.__exit__(*sys.exc_info())
            raise

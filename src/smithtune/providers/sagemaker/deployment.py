"""Customer-account endpoint lifecycle with durable ownership and explicit billing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import time
import uuid

from smithtune.artifacts import _json_dump, _load_json, output_lock
from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker import AWSInstancePreference, AWSJobConfig, SageMakerProvider, split_s3_uri
from smithtune.providers.sagemaker.artifacts import _verify_origin, package_artifact
from smithtune.training.contracts import sha256_file, verify_artifact


@dataclass(frozen=True)
class EndpointSettings:
    serving_image: str
    instance_type: str | None
    hourly_rate_usd: float | None
    max_context_tokens: int
    max_new_tokens: int = 256
    startup_timeout_seconds: int = 600
    inference_ami_version: str | None = None
    provision_timeout_seconds: int | None = None
    instance_preferences: tuple[AWSInstancePreference, ...] = ()

    def __post_init__(self):
        if isinstance(self.instance_preferences, (list, tuple)):
            try:
                preferences = tuple(AWSInstancePreference(**item) if isinstance(item, dict) else item
                                    for item in self.instance_preferences)
            except TypeError as exc:
                raise PipelineError("each endpoint preference needs instance_type and hourly_rate_usd") from exc
            object.__setattr__(self, "instance_preferences", preferences)

    def instance_rates(self):
        if self.instance_preferences:
            return {item.instance_type: item.hourly_rate_usd for item in self.instance_preferences}
        return {self.instance_type: self.hourly_rate_usd}

    @classmethod
    def read(cls, path):
        value = _load_json(path)
        if not isinstance(value, dict):
            raise PipelineError("endpoint config must be a JSON object")
        try:
            return cls(**({"instance_type": None, "hourly_rate_usd": None} | value))
        except TypeError as exc:
            raise PipelineError("invalid endpoint config; see docs/aws-training.md") from exc

    def validate(self, job):
        if not isinstance(self.instance_preferences, tuple) or len(self.instance_preferences) > 5:
            raise PipelineError("endpoint instance_preferences must be an ordered list of 1–5 GPU options")
        if self.instance_preferences:
            if self.instance_type is not None or self.hourly_rate_usd is not None:
                raise PipelineError("use endpoint instance_type/hourly_rate_usd or instance_preferences, not both")
            if any(not isinstance(item, AWSInstancePreference) for item in self.instance_preferences):
                raise PipelineError("each endpoint preference needs instance_type and hourly_rate_usd")
            if self.provision_timeout_seconds is None:
                raise PipelineError("endpoint instance_preferences requires provision_timeout_seconds")
        # Reuse account/region/image/GPU validation without conflating hosting
        # selection with the training allocation or its fallback preferences.
        choices = self.instance_preferences or (AWSInstancePreference(self.instance_type, self.hourly_rate_usd),)
        for item in choices:
            replace(job, training_image=self.serving_image, instance_type=item.instance_type,
                    instance_preferences=(), hourly_rate_usd=item.hourly_rate_usd,
                    max_pending_seconds=None).validate()
            if (isinstance(item.hourly_rate_usd, bool) or not isinstance(item.hourly_rate_usd, (int, float))
                    or not math.isfinite(item.hourly_rate_usd) or item.hourly_rate_usd <= 0):
                raise PipelineError("endpoint hourly_rate_usd must be positive")
        if len(self.instance_rates()) != len(choices):
            raise PipelineError("endpoint instance_preferences must not repeat an instance type")
        for value, low, high in ((self.max_context_tokens, 2, 262144), (self.max_new_tokens, 1, 1024),
                                 (self.startup_timeout_seconds, 60, 3600)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise PipelineError("invalid endpoint context/output/startup limit")
        if self.inference_ami_version is not None and (not isinstance(self.inference_ami_version, str) or self.inference_ami_version not in {
            "al2-ami-sagemaker-inference-gpu-2", "al2-ami-sagemaker-inference-gpu-2-1",
            "al2-ami-sagemaker-inference-gpu-3-1", "al2023-ami-sagemaker-inference-gpu-4-1",
        }):
            raise PipelineError("inference_ami_version must be a documented SageMaker GPU inference AMI")
        if self.provision_timeout_seconds is not None and (
            isinstance(self.provision_timeout_seconds, bool) or not isinstance(self.provision_timeout_seconds, int)
            or not 300 <= self.provision_timeout_seconds <= 3600
        ):
            raise PipelineError("provision_timeout_seconds must be an integer in [300, 3600]")


def plan(run_dir: Path, settings: EndpointSettings, name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", name):
        raise PipelineError("invalid SageMaker endpoint name")
    receipt = _load_json(run_dir / "job.json")
    if receipt.get("provider") != "sagemaker" or receipt.get("status") != "Completed":
        raise PipelineError("deploy requires a completed SageMaker run")
    job = AWSJobConfig(**receipt["aws"])
    job.validate()
    settings.validate(job)
    result, spec = verify_artifact(run_dir / "artifact")
    _verify_origin(result, run_dir, receipt)
    if settings.max_context_tokens > spec.training_context_limit or settings.max_new_tokens >= settings.max_context_tokens:
        raise PipelineError("serving limits exceed the model profile")
    artifact_uri = f"{job.s3_prefix.rstrip('/')}/deployments/{name}/{sha256_file(run_dir / 'artifact/artifact.json')}/model.tar.gz"
    model = {
        "ModelName": name, "ExecutionRoleArn": job.role_arn,
        "PrimaryContainer": {"Image": settings.serving_image, "ModelDataUrl": artifact_uri,
                             "Environment": {"SMITHTUNE_MAX_CONTEXT_TOKENS": str(settings.max_context_tokens),
                                             "SMITHTUNE_MAX_NEW_TOKENS": str(settings.max_new_tokens)}},
        "EnableNetworkIsolation": False,
    }
    if job.subnets:
        model["VpcConfig"] = {"Subnets": list(job.subnets), "SecurityGroupIds": list(job.security_group_ids)}
    endpoint_config = {
        "EndpointConfigName": name,
        "ProductionVariants": [{"VariantName": "AllTraffic", "ModelName": name,
                                "InitialInstanceCount": 1, "InstanceType": settings.instance_type,
                                "InitialVariantWeight": 1.0, "ModelDataDownloadTimeoutInSeconds": settings.startup_timeout_seconds,
                                "ContainerStartupHealthCheckTimeoutInSeconds": settings.startup_timeout_seconds}],
    }
    variant = endpoint_config["ProductionVariants"][0]
    settings_payload = asdict(settings)
    settings_payload.pop("instance_preferences")
    cost = {"hourly_compute_usd": max(settings.instance_rates().values()), "continuous_billing": True,
            "note": "One dedicated instance, billed until deletion; storage and other services additional. No automatic AWS TTL."}
    if settings.instance_preferences:
        del variant["InstanceType"]
        variant["InstancePools"] = [{"InstanceType": item.instance_type, "Priority": priority}
                                    for priority, item in enumerate(settings.instance_preferences, start=1)]
        settings_payload["instance_preferences"] = [asdict(item) for item in settings.instance_preferences]
        cost["instance_hourly_rates_usd"] = settings.instance_rates()
        cost["note"] += " Hourly estimate uses the highest approved pool rate, not the sum."
    if settings.inference_ami_version is not None:
        variant["InferenceAmiVersion"] = settings.inference_ami_version
    if settings.provision_timeout_seconds is not None:
        variant["VariantInstanceProvisionTimeoutInSeconds"] = settings.provision_timeout_seconds
    if job.volume_kms_key:
        endpoint_config["KmsKeyId"] = job.volume_kms_key
    return {
        "schema_version": 1, "provider": "sagemaker", "source_run_id": result["run_id"],
        "region": job.region, "endpoint_name": name, "settings": settings_payload,
        "artifact_sha256": sha256_file(run_dir / "artifact/artifact.json"),
        "requests": {"model": model, "config": endpoint_config,
                     "endpoint": {"EndpointName": name, "EndpointConfigName": name}},
        "cost": cost,
        "cleanup_command": f"smithtune aws undeploy --run-dir {shlex.quote(str(run_dir.resolve()))} --confirm",
        "limitations": ["Experimental greedy single-GPU serving; throughput and long context are unvalidated.",
                        "Model weights download from the pinned Hugging Face revision; outbound access/cache required.",
                        "InService confirms readiness, not model quality. Use invoke for a functional check."],
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _missing(exc):
    error = getattr(exc, "response", {}).get("Error", {})
    return error.get("Code") in {"ResourceNotFound", "ResourceNotFoundException"} or (
        error.get("Code") == "ValidationException" and any(
            marker in error.get("Message", "").lower() for marker in ("could not find", "does not exist", "not found")))


def _describe(client, kind, name):
    operation, field = {"model": ("describe_model", "ModelName"), "config": ("describe_endpoint_config", "EndpointConfigName"),
                        "endpoint": ("describe_endpoint", "EndpointName")}[kind]
    try:
        return getattr(client, operation)(**{field: name})
    except Exception as exc:
        if _missing(exc):
            return None
        raise


def _owns(client, kind, remote, receipt):
    arn_field = {"model": "ModelArn", "config": "EndpointConfigArn", "endpoint": "EndpointArn"}[kind]
    tags = {item["Key"]: item["Value"] for item in client.list_tags(ResourceArn=remote[arn_field])["Tags"]}
    if tags.get("smithtune:deployment") != receipt["owner"]:
        raise PipelineError("refusing to use/delete an endpoint resource not owned by this receipt")
    request = receipt["plan"]["requests"][kind]
    keys = {"model": ("ExecutionRoleArn", "PrimaryContainer", "VpcConfig", "EnableNetworkIsolation"),
            "config": ("ProductionVariants", "KmsKeyId"), "endpoint": ("EndpointConfigName",)}[kind]
    for key in keys:
        expected, actual = request.get(key), remote.get(key)
        if key == "PrimaryContainer":
            matches = isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items())
        elif key == "ProductionVariants":
            matches = (isinstance(actual, list) and len(actual) == len(expected)
                       and all(all(item.get(k) == v for k, v in wanted.items()) for item, wanted in zip(actual, expected, strict=True)))
        else:
            matches = actual == expected
        if not matches:
            raise PipelineError("remote endpoint resource differs from its recorded request")


def _session(run_dir, session_factory):
    job = AWSJobConfig(**_load_json(run_dir / "job.json")["aws"])
    job.validate()
    session = session_factory(job) if session_factory else SageMakerProvider()._session(job)
    if session.client("sts").get_caller_identity()["Account"] != job.role_arn.split(":")[4]:
        raise PipelineError("AWS caller account must match the saved training account")
    return session, job


def _load_receipt(run_dir):
    receipt = _load_json(run_dir / "endpoint/receipt.json")
    if receipt.get("schema_version") != 1 or _digest(receipt["plan"]) != receipt.get("plan_sha256"):
        raise PipelineError("invalid or changed deployment receipt")
    return receipt


def _archive_deleted(run_dir, receipt, name, session_factory):
    """Keep prior deployment evidence when explicitly starting a new endpoint."""
    if receipt["status"] != "Deleted" or name == receipt["plan"]["endpoint_name"]:
        raise PipelineError("use a new endpoint name after the previous deployment is fully Deleted")
    session, _job = _session(run_dir, session_factory)
    client = session.client("sagemaker")
    if any(_describe(client, kind, receipt["plan"]["endpoint_name"]) for kind in ("endpoint", "config", "model")):
        raise PipelineError("previous endpoint resources still exist; complete cleanup first")
    folder = run_dir / "endpoint"
    archive = folder / "history" / receipt["owner"]
    archive.mkdir(parents=True, exist_ok=True)
    # Copy receipts first so an interruption cannot destroy the cleanup evidence.
    for filename in ("receipt.json", "smoke-result.json", "smoke-generations.json"):
        if (folder / filename).exists():
            shutil.copy2(folder / filename, archive / filename)
    for filename in ("model.tar.gz", "smoke-result.json", "smoke-generations.json", "receipt.json"):
        (folder / filename).unlink(missing_ok=True)


def _confirmed(receipt, kind):
    confirmed = receipt.setdefault("confirmed", [])
    if kind not in confirmed:
        confirmed.append(kind)


def _cleanup_uncertain(receipt):
    # Even an acknowledged create can be temporarily invisible to Describe.
    # Require an owned resource to have reached the deletion path before its
    # later absence can prove cleanup. External deletion needs manual review.
    return bool(set(receipt["attempted"]) - set(receipt.get("delete_requested", [])) - set(receipt.get("rejected", {})))


def deploy(run_dir: Path, settings: EndpointSettings, name: str, *, confirm=False, session_factory=None):
    if not confirm:
        raise PipelineError("endpoint deployment creates continuously billed resources; review deploy-plan and use --confirm")
    proposal = plan(run_dir, settings, name)
    folder = run_dir / "endpoint"
    with output_lock(folder):
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            receipt = _load_receipt(run_dir)
            if receipt["status"] == "Deleted":
                _archive_deleted(run_dir, receipt, name, session_factory)
        if receipt_path.exists():
            receipt = _load_receipt(run_dir)
            if receipt["plan"] != proposal or receipt["status"] in {"Deleting", "Deleted"} or receipt.get("cleanup_requested"):
                raise PipelineError("existing deployment receipt cannot be replaced; finish its lifecycle first")
        else:
            receipt = {"schema_version": 1, "owner": uuid.uuid4().hex, "plan": proposal,
                       "plan_sha256": _digest(proposal), "status": "Planned", "attempted": []}
            _json_dump(receipt_path, receipt)
        try:
            session, job = _session(run_dir, session_factory)
            client = session.client("sagemaker")
            registry, image_path = settings.serving_image.split("/", 1)
            repository, digest = image_path.split("@", 1)
            images = session.client("ecr").describe_images(
                registryId=registry.split(".")[0], repositoryName=repository, imageIds=[{"imageDigest": digest}],
            )["imageDetails"]
            if (len(images) != 1 or images[0].get("imageDigest") != digest
                    or images[0].get("imageManifestMediaType") != "application/vnd.docker.distribution.manifest.v2+json"):
                raise PipelineError("SageMaker hosting requires a Docker V2 schema 2 image; rebuild with oci-mediatypes=false (see AWS guide)")
            # Check all names before any create. Matching names alone never confer ownership.
            existing = {kind: _describe(client, kind, name) for kind in ("model", "config", "endpoint")}
            for kind, remote in existing.items():
                if remote:
                    _owns(client, kind, remote, receipt)
                    _confirmed(receipt, kind)
                elif kind in receipt["attempted"]:
                    raise PipelineError("a prior create response was uncertain; inspect endpoint-status before retrying")
            if not existing["model"]:
                archive = folder / "model.tar.gz"
                package_artifact(run_dir / "artifact", archive)
                # A unique ownership prefix prevents replacing a previous deployment's package.
                uri = proposal["requests"]["model"]["PrimaryContainer"]["ModelDataUrl"]
                bucket, key = split_s3_uri(uri)
                s3 = session.client("s3")
                # The plan's hash-addressed artifact key may already hold the same content.
                # Never overwrite it; subsequent deployment uses the saved version.
                artifact_present = False
                try:
                    head = s3.head_object(Bucket=bucket, Key=key)
                except Exception as exc:
                    error = getattr(exc, "response", {}).get("Error", {}).get("Code")
                    if error not in {"404", "NoSuchKey", "NotFound"}:
                        raise
                else:
                    if head.get("Metadata", {}).get("smithtune-manifest") != proposal["artifact_sha256"]:
                        raise PipelineError("serving artifact key is occupied by different content")
                    artifact_present = True
                if not artifact_present:
                    extra = {"Metadata": {"smithtune-manifest": proposal["artifact_sha256"]}}
                    if job.output_kms_key:
                        extra.update(ServerSideEncryption="aws:kms", SSEKMSKeyId=job.output_kms_key)
                    s3.upload_file(str(archive), bucket, key, ExtraArgs=extra)
            for kind, operation in (("model", "create_model"), ("config", "create_endpoint_config"), ("endpoint", "create_endpoint")):
                if existing[kind]:
                    continue
                receipt["attempted"].append(kind)
                receipt["status"] = f"Creating_{kind}"
                _json_dump(receipt_path, receipt)
                request = proposal["requests"][kind] | {"Tags": [{"Key": "smithtune:deployment", "Value": receipt["owner"]}]}
                try:
                    getattr(client, operation)(**request)
                except Exception as exc:
                    response = getattr(exc, "response", {})
                    error_code = response.get("Error", {}).get("Code")
                    metadata = response.get("ResponseMetadata", {})
                    # A direct, non-retried service rejection proves this create
                    # did not succeed. A timeout/retried request remains ambiguous.
                    if (error_code in {"ValidationException", "AccessDeniedException", "ResourceLimitExceeded"}
                            and metadata.get("RetryAttempts") == 0):
                        receipt.setdefault("rejected", {})[kind] = {
                            "error_code": error_code, "request_id": metadata.get("RequestId"),
                        }
                        _json_dump(receipt_path, receipt)
                    raise
                _confirmed(receipt, kind)
                _json_dump(receipt_path, receipt)
            receipt["status"] = "Creating"
            _json_dump(receipt_path, receipt)
            return receipt
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"endpoint submission incomplete ({type(exc).__name__}); receipt retained at {receipt_path}; use endpoint-status or undeploy") from None


def status(run_dir: Path, *, session_factory=None):
    with output_lock(run_dir / "endpoint"):
        receipt = _load_receipt(run_dir)
        try:
            session, _job = _session(run_dir, session_factory)
            client = session.client("sagemaker")
            remote = _describe(client, "endpoint", receipt["plan"]["endpoint_name"])
            if remote:
                _owns(client, "endpoint", remote, receipt)
                receipt["status"] = remote["EndpointStatus"]
                if remote.get("FailureReason"):
                    receipt["failure_reason"] = remote["FailureReason"]
                pools = [pool for variant in remote.get("ProductionVariants", []) for pool in variant.get("InstancePools", [])]
                if pools:
                    receipt["provisioned_instances"] = {pool["InstanceType"]: pool.get("CurrentInstanceCount", 0) for pool in pools}
            else:
                receipt["status"] = "EndpointAbsent"
            receipt["resources_present"] = []
            for kind in ("model", "config", "endpoint"):
                item = _describe(client, kind, receipt["plan"]["endpoint_name"])
                if item:
                    _owns(client, kind, item, receipt)
                    _confirmed(receipt, kind)
                    receipt["resources_present"].append(kind)
            if receipt.get("cleanup_requested"):
                unseen = (set(receipt["attempted"]) - set(receipt["resources_present"])
                          - set(receipt.get("delete_requested", [])) - set(receipt.get("rejected", {})))
                if unseen:
                    receipt["status"] = "CleanupUncertain"
                elif not receipt["resources_present"]:
                    receipt["status"] = "Deleted"
            _json_dump(run_dir / "endpoint/receipt.json", receipt)
            return receipt
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"endpoint status failed ({type(exc).__name__}); receipt preserved") from None


def undeploy(run_dir: Path, *, confirm=False, session_factory=None):
    if not confirm:
        raise PipelineError("endpoint deletion requires --confirm")
    with output_lock(run_dir / "endpoint"):
        receipt = _load_receipt(run_dir)
        try:
            session, _job = _session(run_dir, session_factory)
            client, name = session.client("sagemaker"), receipt["plan"]["endpoint_name"]
            receipt["cleanup_requested"] = True
            _json_dump(run_dir / "endpoint/receipt.json", receipt)
            for kind, operation, field in (("endpoint", "delete_endpoint", "EndpointName"),
                                            ("config", "delete_endpoint_config", "EndpointConfigName"),
                                            ("model", "delete_model", "ModelName")):
                remote = _describe(client, kind, name)
                if not remote:
                    if (kind in receipt["attempted"] and kind not in receipt.get("delete_requested", [])
                            and kind not in receipt.get("rejected", {})):
                        receipt["status"] = "CleanupUncertain"
                        _json_dump(run_dir / "endpoint/receipt.json", receipt)
                        return receipt  # A creation may still be in flight; keep its dependencies.
                    continue
                _owns(client, kind, remote, receipt)
                _confirmed(receipt, kind)
                deleted = receipt.setdefault("delete_requested", [])
                if kind not in deleted:
                    deleted.append(kind)
                _json_dump(run_dir / "endpoint/receipt.json", receipt)
                if kind != "endpoint" or remote["EndpointStatus"] != "Deleting":
                    try:
                        getattr(client, operation)(**{field: name})
                    except Exception as exc:
                        error = getattr(exc, "response", {}).get("Error", {})
                        if (kind == "endpoint" and error.get("Code") == "ValidationException"
                                and error.get("Message", "").startswith("Cannot update in-progress endpoint")):
                            # Hosting cannot cancel an endpoint while AWS is creating it.
                            # Keep its dependencies and retry deletion within the caller's
                            # cleanup deadline. This rejection is not an accepted delete.
                            deleted.remove(kind)
                            receipt["status"] = "CleanupPending"
                            receipt["cleanup_pending_reason"] = "aws_endpoint_operation_in_progress"
                            _json_dump(run_dir / "endpoint/receipt.json", receipt)
                            return receipt
                        raise
                receipt.pop("cleanup_pending_reason", None)
                if kind == "endpoint":
                    receipt["status"] = "Deleting"
                    _json_dump(run_dir / "endpoint/receipt.json", receipt)
                    return receipt  # AWS deletion is asynchronous; preserve dependencies until absent.
            receipt["status"] = "CleanupUncertain" if _cleanup_uncertain(receipt) else "Deleted"
            _json_dump(run_dir / "endpoint/receipt.json", receipt)
            return receipt
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"endpoint cleanup incomplete ({type(exc).__name__}); repeat the saved cleanup command") from None


def invoke(run_dir: Path, payload: dict, *, session_factory=None):
    receipt = status(run_dir, session_factory=session_factory)
    if receipt["status"] != "InService":
        raise PipelineError("endpoint is not InService")
    session, _job = _session(run_dir, session_factory)
    body = json.dumps(payload, allow_nan=False).encode()
    if len(body) > 6 * 1024 * 1024:
        raise PipelineError("endpoint input exceeds 6 MiB")
    try:
        response = session.client("sagemaker-runtime").invoke_endpoint(
            EndpointName=receipt["plan"]["endpoint_name"], ContentType="application/json", Accept="application/json", Body=body)
        value = json.loads(response["Body"].read())
        if value.get("source_run_id") != receipt["plan"]["source_run_id"]:
            raise PipelineError("endpoint response belongs to a different training run")
        return value
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"endpoint invocation failed ({type(exc).__name__}); inspect customer CloudWatch logs") from None


def wait(run_dir, *, timeout=600, deleting=False, session_factory=None, sleep=time.sleep, clock=time.monotonic):
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise PipelineError("endpoint wait timeout must be in (0, 3600]")
    deadline = clock() + timeout
    while True:
        value = undeploy(run_dir, confirm=True, session_factory=session_factory) if deleting else status(run_dir, session_factory=session_factory)
        if value["status"] in {"Deleted", "InService", "Failed", "OutOfService"}:
            return value
        remaining = deadline - clock()
        if remaining <= 0:
            return value | {"wait_timed_out": True}
        sleep(min(10, remaining))


def smoke(run_dir, settings, name, payload, *, confirm=False, timeout=600, session_factory=None):
    """Deploy one fresh temporary endpoint, invoke it, and always request cleanup.

    Readiness timeout is not an AWS billing limit. Persist the cleanup outcome
    even on failure; callers must finish cleanup if AWS deletion remains pending.
    """
    if not confirm:
        raise PipelineError("temporary endpoint smoke creates paid resources; use --confirm")
    if (run_dir / "endpoint/receipt.json").exists() and _load_receipt(run_dir)["status"] != "Deleted":
        raise PipelineError("temporary smoke requires a fresh endpoint receipt; it cannot borrow an existing endpoint")
    # Validate the complete request before creating any receipt or resources.
    plan(run_dir, settings, name)
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise PipelineError("smoke timeout must be in (0, 3600]")
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list) or not payload["messages"]:
        raise PipelineError("smoke input must contain nonempty messages")
    results, primary, cleanup = [], None, None
    try:
        deploy(run_dir, settings, name, confirm=True, session_factory=session_factory)
        ready = wait(run_dir, timeout=timeout, session_factory=session_factory)
        if ready["status"] != "InService":
            raise PipelineError("temporary endpoint did not become ready before its deadline")
        for variant in ("base", "tuned"):
            results.append(invoke(run_dir, payload | {"variant": variant}, session_factory=session_factory))
            _json_dump(run_dir / "endpoint/smoke-generations.json", results)
    except Exception as exc:
        primary = exc
    finally:
        if ((run_dir / "endpoint/receipt.json").exists()
                and _load_receipt(run_dir)["plan"]["endpoint_name"] == name):
            try:
                cleanup = wait(run_dir, timeout=timeout, deleting=True, session_factory=session_factory)
            except Exception as exc:
                cleanup = {"status": "CleanupFailed", "error_type": type(exc).__name__}
            _json_dump(run_dir / "endpoint/smoke-result.json", {
                "generation_count": len(results), "error_type": type(primary).__name__ if primary else None,
                "all_formats_valid": len(results) == 2 and all(value.get("message", {}).get("sampling", {}).get("format_valid") is True for value in results),
                "cleanup_status": cleanup["status"], "cleanup_complete": cleanup["status"] == "Deleted",
            })
    if cleanup is not None and cleanup["status"] != "Deleted":
        raise PipelineError("temporary endpoint cleanup is incomplete; run the cleanup command in endpoint/receipt.json") from None
    if primary is not None:
        raise PipelineError(f"temporary endpoint smoke failed ({type(primary).__name__}); owned resources cleaned up") from None
    return {"generations": results, "cleanup_complete": True, "status": "Completed",
            "all_formats_valid": all(value.get("message", {}).get("sampling", {}).get("format_valid") is True for value in results)}

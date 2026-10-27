"""Validated SageMaker compute, image and storage configuration; no allocation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from smithtune.artifacts import _load_json
from smithtune.providers.base import PipelineError


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (parsed.scheme != "s3" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or parsed.query or parsed.fragment or not parsed.path.strip("/")
            or any(part in {".", "..", ""} for part in parsed.path.strip("/").split("/"))):
        raise PipelineError("s3_prefix must be s3://bucket/nonempty-prefix without query or traversal")
    return parsed.netloc, parsed.path.strip("/")


@dataclass(frozen=True)
class AWSInstancePreference:
    instance_type: str
    hourly_rate_usd: float


def _validate_instance_type(value) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"ml\.(?:g|p)[a-z0-9]+\.[a-z0-9]+", value):
        raise PipelineError("choose an explicit GPU SageMaker instance_type")


def _validate_hourly_rate(value) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise PipelineError("hourly_rate_usd must be finite and positive")


@dataclass(frozen=True)
class AWSJobConfig:
    region: str
    role_arn: str
    training_image: str
    s3_prefix: str
    max_runtime_seconds: int
    instance_type: str | None = None
    instance_count: int = 1
    volume_size_gb: int = 100
    profile: str | None = None
    hourly_rate_usd: float | None = None
    subnets: tuple[str, ...] = ()
    security_group_ids: tuple[str, ...] = ()
    output_kms_key: str | None = None
    volume_kms_key: str | None = None
    instance_preferences: tuple[AWSInstancePreference, ...] = ()
    max_pending_seconds: int | None = None

    def __post_init__(self) -> None:
        for name in ("subnets", "security_group_ids"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        if isinstance(self.instance_preferences, (list, tuple)):
            try:
                preferences = tuple(AWSInstancePreference(**item) if isinstance(item, dict) else item
                                    for item in self.instance_preferences)
            except TypeError as exc:
                raise PipelineError("each instance preference needs instance_type and hourly_rate_usd") from exc
            object.__setattr__(self, "instance_preferences", preferences)

    def instance_rates(self) -> dict[str, float | None]:
        if self.instance_preferences:
            return {item.instance_type: item.hourly_rate_usd for item in self.instance_preferences}
        return {self.instance_type: self.hourly_rate_usd}

    @classmethod
    def read(cls, path: Path) -> AWSJobConfig:
        value = _load_json(path)
        if not isinstance(value, dict):
            raise PipelineError("AWS config must be a JSON object")
        try:
            config = cls(**value)
        except TypeError as exc:
            raise PipelineError("AWS config has missing or unknown fields; see docs/aws-training.md") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.region, str) or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", self.region):
            raise PipelineError("AWS config requires an explicit region")
        if not isinstance(self.role_arn, str) or not re.fullmatch(r"arn:aws(?:-us-gov|-cn)?:iam::\d{12}:role/[\w+=,.@/-]+", self.role_arn):
            raise PipelineError("AWS config requires an execution role ARN")
        image = re.fullmatch(
            r"\d{12}\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/[a-z0-9._/-]+@sha256:[a-f0-9]{64}",
            self.training_image if isinstance(self.training_image, str) else "",
        )
        if image is None or image.group(1) != self.region:
            raise PipelineError("training_image must be an ECR image pinned by digest in the job region")
        if not isinstance(self.s3_prefix, str):
            raise PipelineError("s3_prefix must be an S3 URI")
        split_s3_uri(self.s3_prefix)
        if not isinstance(self.instance_preferences, tuple) or len(self.instance_preferences) > 5:
            raise PipelineError("instance_preferences must be an ordered list of 1–5 GPU options")
        if self.instance_preferences:
            if self.instance_type is not None or self.hourly_rate_usd is not None:
                raise PipelineError("use either instance_type/hourly_rate_usd or instance_preferences, not both")
            for item in self.instance_preferences:
                if not isinstance(item, AWSInstancePreference):
                    raise PipelineError("each instance preference needs instance_type and hourly_rate_usd")
                _validate_instance_type(item.instance_type)
                _validate_hourly_rate(item.hourly_rate_usd)
            if len(self.instance_rates()) != len(self.instance_preferences):
                raise PipelineError("instance_preferences must not repeat an instance type")
            if self.max_pending_seconds is None:
                raise PipelineError("instance_preferences requires max_pending_seconds (AWS minimum: 1800)")
        else:
            _validate_instance_type(self.instance_type)
        for name, maximum in (("instance_count", 1), ("volume_size_gb", 16384), ("max_runtime_seconds", 86400)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise PipelineError(f"{name} must be an integer in [1, {maximum}]; this worker is single-instance")
        if self.profile is not None and (not isinstance(self.profile, str) or not self.profile.strip()):
            raise PipelineError("profile must be a nonempty AWS profile name")
        if self.hourly_rate_usd is not None:
            _validate_hourly_rate(self.hourly_rate_usd)
        if self.max_pending_seconds is not None and (
            isinstance(self.max_pending_seconds, bool) or not isinstance(self.max_pending_seconds, int)
            or not 1800 <= self.max_pending_seconds <= 86400
        ):
            raise PipelineError("max_pending_seconds must be an integer in [1800, 86400]")
        for values, prefix in ((self.subnets, "subnet-"), (self.security_group_ids, "sg-")):
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not re.fullmatch(prefix + r"[a-f0-9]+", value) for value in values
            ):
                raise PipelineError("invalid VPC subnet/security group IDs")
        if bool(self.subnets) != bool(self.security_group_ids):
            raise PipelineError("subnets and security_group_ids must be supplied together")
        for name in ("output_kms_key", "volume_kms_key"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise PipelineError(f"{name} must be a nonempty key reference")



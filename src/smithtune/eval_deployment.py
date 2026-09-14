"""Temporary, owned Fireworks deployments for replay evaluation."""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, FIREWORKS_BASE_URL, _set_skill_session, _validate_resource_id


class ControlError(PipelineError):
    def __init__(self, status: int | None):
        self.status = status
        super().__init__(f"Fireworks control request failed: HTTP {status}" if status else "Fireworks control request failed; outcome may be unknown")


def control_request(method: str, path: str, body: dict | None = None) -> dict:
    """Use only the official control API; never expose response bodies or keys."""
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise PipelineError("FIREWORKS_API_KEY is not set")
    _set_skill_session()
    request = urllib.request.Request(
        FIREWORKS_BASE_URL + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                 "Content-Type": "application/json", "X-Fireworks-Client-Source": CLIENT_SOURCE,
                 "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            result = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raise ControlError(exc.code) from None
    except (OSError, ValueError):
        raise ControlError(None) from None
    if not isinstance(result, dict):
        raise ControlError(None)
    return result


@dataclass(frozen=True)
class EvalDeployment:
    model: str
    account_id: str
    deployment_id: str
    deployment_shape: str
    timeout: float = 600

    def validate(self) -> None:
        for value, label in ((self.account_id, "account id"), (self.deployment_id, "deployment id")):
            _validate_resource_id(value, label)
        match = re.fullmatch(r"accounts/([a-z0-9-]+)/models/([a-z0-9-]+)", self.model)
        if not match or match[1] != self.account_id or self.account_id == "fireworks":
            raise PipelineError("preemptible evaluation requires a promoted model in the selected account")
        _validate_resource_id(match[2], "model id")
        if not re.fullmatch(r"accounts/[a-z0-9-]+/deploymentShapes/[a-zA-Z0-9_.-]+(?:/versions/[a-zA-Z0-9_.-]+)?", self.deployment_shape):
            raise PipelineError("a full Fireworks deployment shape resource is required")
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 3600:
            raise PipelineError("deployment timeout must be between 1 and 3600 seconds")

    @property
    def resource(self) -> str:
        return f"accounts/{self.account_id}/deployments/{self.deployment_id}"

    @property
    def route(self) -> str:
        return f"{self.model}#{self.resource}"

    def plan(self) -> dict:
        self.validate()
        return {**asdict(self), "serving_mode": "preemptible", "preemptible": True,
                "min_replica_count": 1, "max_replica_count": 1, "cleanup": "delete owned deployment after evaluation",
                "cost": "borrowed capacity; model and judge inference charges use current account rates",
                "interruption": "capacity may disappear; rerun with the same output directory to resume"}


class TemporaryDeployment:
    """Reconcile intent, verify ownership, and clean up even if creation fails."""

    def __init__(self, config: EvalDeployment, output_dir: Path, *, request=None, sleeper=time.sleep, clock=time.monotonic):
        config.validate()
        self.config = config
        self.request = request or control_request
        self.sleep, self.clock = sleeper, clock
        self.path = output_dir / "deployments" / f"{config.deployment_id}.json"
        self.receipt = None
        self.owned = False
        self.creation_pending = False

    def _get(self) -> dict | None:
        try:
            result = self.request("GET", "/v1/" + self.config.resource)
        except ControlError as exc:
            if exc.status == 404:
                return None
            raise
        return None if result.get("state") == "DELETED" else result

    def _matches(self, resource: dict) -> bool:
        return bool(self.receipt and resource.get("description") == self.receipt["owner"]
                    and resource.get("name") == self.config.resource
                    and resource.get("baseModel") == self.config.model)

    def _save(self, state: str, **extra) -> None:
        self.receipt.update(state=state, **extra)
        _json_dump(self.path, self.receipt)

    def _load_receipt(self) -> None:
        if self.path.exists():
            self.receipt = _load_json(self.path)
            if not isinstance(self.receipt, dict) or self.receipt.get("config") != asdict(self.config) or not isinstance(self.receipt.get("owner"), str):
                raise PipelineError("temporary deployment receipt has different settings; use another deployment ID")

    def cleanup_existing(self) -> None:
        """Retry cleanup after the last case was saved, without creating capacity."""
        self._load_receipt()
        if self.receipt is not None:
            self.owned = True
            self._cleanup()

    def __enter__(self) -> str:
        existing = self._get()
        self._load_receipt()
        if existing and not self._matches(existing):
            raise PipelineError("deployment already exists and is not owned by this evaluation; use another ID")
        if self.receipt is None:
            self.receipt = {"schema_version": 1, "config": asdict(self.config),
                            "owner": "smithtune-eval:" + str(uuid4()), "deployment": self.config.resource,
                            "cleanup_command": f"smithtune undeploy --account-id {self.config.account_id} --deployment-id {self.config.deployment_id} --confirm"}
        self._save("creating" if existing is None else "waiting")
        # From this point any lost create response is reconciled through the
        # saved ownership marker. An unrelated existing resource is never deleted.
        self.owned = True
        try:
            if existing is None:
                self.creation_pending = True
                try:
                    self.request("POST", f"/v1/accounts/{self.config.account_id}/deployments?deploymentId={self.config.deployment_id}", {
                        "baseModel": self.config.model, "deploymentShape": self.config.deployment_shape,
                        "minReplicaCount": 1, "maxReplicaCount": 1, "preemptible": True,
                        "description": self.receipt["owner"],
                    })
                except ControlError:
                    resource = self._get()
                    if resource is None or not self._matches(resource):
                        raise
                self.creation_pending = False
            deadline = self.clock() + self.config.timeout
            while True:
                resource = self._get()
                if resource is None:
                    raise PipelineError("temporary deployment disappeared; evaluation can be resumed")
                if not self._matches(resource):
                    raise PipelineError("temporary deployment ownership changed")
                if resource.get("preemptible") is not True:
                    raise PipelineError("Fireworks did not confirm preemptible mode; refusing standard capacity")
                shape = resource.get("deploymentShape", "")
                expected = self.config.deployment_shape
                if not isinstance(shape, str) or not (shape == expected or ("/versions/" not in expected and shape.startswith(expected + "/versions/"))):
                    raise PipelineError("Fireworks returned a different deployment shape")
                if any(type(resource.get(key)) is not int or resource[key] != 1 for key in ("minReplicaCount", "maxReplicaCount")):
                    raise PipelineError("Fireworks returned different replica limits")
                if resource.get("state") == "READY":
                    self._save("ready", route=self.config.route)
                    return self.config.route
                if resource.get("state") in {"FAILED", "DELETING"}:
                    raise PipelineError("temporary deployment is unavailable; evaluation can be resumed")
                if self.clock() >= deadline:
                    raise PipelineError("temporary deployment readiness timed out")
                self.sleep(min(2, max(0, deadline - self.clock())))
        except BaseException:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if not self.owned:
            return
        try:
            resource = self._get()
            if resource is None and self.creation_pending:
                raise PipelineError("deployment creation outcome is still unknown")
            if resource is not None:
                if not self._matches(resource):
                    raise PipelineError("refusing to delete a deployment with different ownership")
                self._save("deleting")
                try:
                    self.request("DELETE", "/v1/" + self.config.resource + "?ignoreChecks=true")
                except ControlError as exc:
                    if exc.status != 404:
                        raise
                deadline = self.clock() + min(self.config.timeout, 120)
                while self._get() is not None:
                    if self.clock() >= deadline:
                        raise PipelineError("deployment deletion was not confirmed before timeout")
                    self.sleep(2)
            self._save("deleted")
        except (PipelineError, OSError):
            self._save("cleanup_required")
            raise PipelineError(f"temporary deployment cleanup needs attention; inspect {self.path}; {self.receipt['cleanup_command']}") from None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._cleanup()

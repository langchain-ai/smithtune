"""Fixed-checkpoint replay through owned Baseten Loops GPU samplers."""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from smithtune.artifacts import _json_dump, _load_json, _utc_now
from smithtune.capabilities import open_without_redirects
from smithtune.dataset import _model_from_manifest, _require_prepared_provider
from smithtune.providers.base import PipelineError
from smithtune.rendering import load_training_renderer


API_ROOT = "https://api.baseten.co"
_CHECKPOINT = re.compile(r"bt://loops:([A-Za-z0-9_-]+)/sampler_weights/([A-Za-z0-9_.-]+)")


def _resource_id(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise PipelineError("Baseten returned an invalid resource ID; inspect sampler.json")
    return value


def checkpoint_from_run(data_dir: Path, run_dir: Path) -> tuple[Any, str]:
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    _require_prepared_provider(manifest, "baseten")
    model = _model_from_manifest(manifest)
    plan = _load_json(run_dir / "plan.json")
    result = _load_json(run_dir / "result.json")
    if not isinstance(plan, dict) or not isinstance(result, dict) or any(
        item.get("provider") != "baseten" for item in (plan, result)
    ):
        raise PipelineError("Baseten evaluation requires a completed Baseten training run")
    checkpoint = result.get("best_sampler_weights_uri")
    match = _CHECKPOINT.fullmatch(checkpoint) if isinstance(checkpoint, str) else None
    if not match or match[1] != result.get("baseten_run_id"):
        raise PipelineError("Baseten evaluation requires the saved best sampler checkpoint in --run-dir")
    if plan.get("base_model") != model.base_model:
        raise PipelineError("prepared data base model differs from the Baseten training run")
    return model, checkpoint


def _request(method: str, path: str, body: dict | None = None) -> dict:
    """Control-plane writes are attempted once; ambiguous outcomes need reconciliation."""
    key = os.environ.get("BASETEN_API_KEY", "").strip()
    if not key:
        raise PipelineError("BASETEN_API_KEY is not set")
    request = urllib.request.Request(
        API_ROOT + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with open_without_redirects(request, timeout=300 if method == "POST" else 60) as response:
            payload = response.read()
            result = json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"Baseten sampler control request failed: HTTP {exc.code}; inspect sampler.json") from None
    except (OSError, ValueError):
        raise PipelineError("Baseten sampler control request failed; write outcome may be unknown; inspect sampler.json") from None
    if not isinstance(result, dict):
        raise PipelineError("Baseten sampler control response is not an object; inspect sampler.json")
    return result


class _SamplerService:
    def __init__(self):
        if not os.environ.get("BASETEN_API_KEY", "").strip():
            raise PipelineError("BASETEN_API_KEY is not set")
        if any(name.startswith("LOOPS_REUSE_") for name in os.environ):
            raise PipelineError("Baseten replay rejects Loops resource-reuse overrides")
        if os.environ.get("LOOPS_BASE_URL", API_ROOT).rstrip("/") != API_ROOT:
            raise PipelineError("Baseten replay requires the standard Baseten API URL")
        self.root = "/v1/loops"
        team_name = os.environ.get("LOOPS_TEAM")
        if team_name:
            teams = _request("GET", "/v1/teams").get("teams", [])
            matches = [team for team in teams if isinstance(team, dict) and team.get("name") == team_name]
            if len(matches) != 1:
                raise PipelineError("LOOPS_TEAM must identify exactly one Baseten team")
            self.root = f"/v1/teams/{_resource_id(matches[0].get('id'))}/loops"

    def validate_checkpoint(self, checkpoint: str, base_model: str) -> None:
        match = _CHECKPOINT.fullmatch(checkpoint)
        assert match is not None
        response = _request("GET", "/v1/loops/checkpoints?" + urllib.parse.urlencode({"checkpoint_path": checkpoint}))
        values = response.get("checkpoints")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise PipelineError("Baseten must return exactly one checkpoint for the saved sampler URI")
        value = values[0]
        if (value.get("run_id") != match[1] or value.get("checkpoint_id") != match[2]
                or value.get("target") != "sampler" or value.get("base_model") != base_model
                or value.get("sync_status") not in (None, "COMPLETE")):
            raise PipelineError("Baseten sampler checkpoint metadata differs from the saved run or is still syncing")

    def create_session(self) -> str:
        return _resource_id(_request("POST", self.root + "/sessions").get("session", {}).get("id"))

    def create_sampler(self, body: dict) -> dict:
        return _request("POST", self.root + "/samplers", body).get("sampler", {})

    def connect(self, resource: dict, model, session_id: str, ready_timeout: float):
        from baseten.loops.sampling_client import BasetenDeployment, SamplingClient

        url = urllib.parse.urlsplit(resource.get("base_url", ""))
        if (url.scheme != "https" or not url.hostname or not url.hostname.endswith(".api.baseten.co")
                or url.username or url.password or url.port not in (None, 443) or url.query or url.fragment):
            raise PipelineError("Baseten returned an invalid sampler URL; inspect sampler.json")
        return SamplingClient(
            api_key=os.environ["BASETEN_API_KEY"], base_model=model.base_model,
            deployment=BasetenDeployment(
                base_url=resource["base_url"], model_id=resource["model_id"],
                deployment_id=resource["deployment_id"], api_base_url=API_ROOT,
            ),
            sampler_id=resource["id"], session_id=session_id,
            checkpoint_path=resource.get("checkpoint"), ready_timeout=ready_timeout,
        )

    def deactivate(self, resource: dict) -> None:
        path = f"/v1/models/{_resource_id(resource['model_id'])}/deployments/{_resource_id(resource['deployment_id'])}"
        _request("POST", path + "/deactivate")
        deadline = time.monotonic() + 120
        while True:
            if _request("GET", path).get("status") == "INACTIVE":
                return
            if time.monotonic() >= deadline:
                raise PipelineError("Baseten sampler deactivation has not reached INACTIVE")
            time.sleep(2)

    def is_inactive(self, resource: dict) -> bool:
        model_id = _resource_id(resource.get("model_id"))
        deployment_id = _resource_id(resource.get("deployment_id"))
        deployment = _request("GET", f"/v1/models/{model_id}/deployments/{deployment_id}")
        return (deployment.get("id") == deployment_id and deployment.get("model_id") == model_id
                and deployment.get("status") == "INACTIVE")


class BasetenReplaySampler:
    """Own two standalone samplers; never restore or publish trainer weights."""

    def __init__(self, model, checkpoint: str, output_dir: Path, *, ready_timeout: float = 3600, service=None):
        if model.provider != "baseten" or not isinstance(checkpoint, str) or not _CHECKPOINT.fullmatch(checkpoint):
            raise PipelineError("Baseten replay requires a saved sampler checkpoint and its Baseten model")
        if not math.isfinite(ready_timeout) or ready_timeout <= 0:
            raise PipelineError("sampler readiness timeout must be positive and finite")
        self.model = model
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.ready_timeout = ready_timeout
        self.service = service
        self._check_previous_receipt()
        self.renderer = load_training_renderer(model)
        self._samplers: dict[str, Any] = {}
        from smithtune.providers.baseten_sampling_formats import PARSING_VERSION

        self.config = {"parsing_version": PARSING_VERSION, "provider": "baseten", "serving_mode": "sampler", "checkpoint": checkpoint,
                       "base_model": model.base_model, "max_seq_len": model.max_seq_len,
                       "tokenizer_revision": model.tokenizer_revision, "template_sha256": model.template_sha256,
                       "rendering_version": model.rendering_version, "renderer": model.renderer}
        self.receipt: dict[str, Any] = {}

    def _save(self):
        _json_dump(self.output_dir / "sampler.json", self.receipt)

    def _check_previous_receipt(self):
        path = self.output_dir / "sampler.json"
        if not path.exists():
            return
        previous = _load_json(path)
        if isinstance(previous, dict) and previous.get("status") == "closed":
            return
        resources = previous.get("resources") if isinstance(previous, dict) else None
        if not isinstance(resources, list) or not resources or any(not isinstance(resource, dict) for resource in resources):
            raise PipelineError(f"previous Baseten sampler resources need reconciliation; inspect {path} before retrying")
        for resource in resources:
            for field in ("model_id", "deployment_id"):
                if not isinstance(resource.get(field), str) or re.fullmatch(r"[A-Za-z0-9_-]+", resource[field]) is None:
                    raise PipelineError(f"previous Baseten sampler creation needs reconciliation; inspect {path} and its session ID before retrying")
        service = self.service or _SamplerService()
        if not all(service.is_inactive(resource) for resource in resources):
            raise PipelineError(f"previous Baseten sampler resources need reconciliation; inspect {path} and its cleanup commands before retrying")

    def __enter__(self):
        # Recheck under the evaluator's output lock before replacing the receipt.
        self._check_previous_receipt()
        self.service = self.service or _SamplerService()
        self.service.validate_checkpoint(self.checkpoint, self.model.base_model)
        self.receipt = {**self.config, "started_at_utc": _utc_now(), "status": "starting", "resources": []}
        self._save()
        try:
            self.receipt["session_id"] = self.service.create_session()
            self._save()
            for identity, options in (
                (self.checkpoint, {"model_path": self.checkpoint}),
                (self.model.base_model, {"base_model": self.model.base_model}),
            ):
                resource = {"identity": identity, "status": "create_outcome_unknown"}
                self.receipt["resources"].append(resource)
                self._save()
                created = self.service.create_sampler({"session_id": self.receipt["session_id"],
                                                       "max_seq_length": self.model.max_seq_len, **options})
                # Save returned identities before validating unrelated response fields,
                # constructing SDK clients, or waiting for paid GPUs to become ready.
                if isinstance(created, dict):
                    for field in ("id", "model_id", "deployment_id", "base_url"):
                        if field in created:
                            resource[field] = created[field]
                self._save()
                for field in ("model_id", "deployment_id"):
                    _resource_id(resource.get(field))
                resource["cleanup_command"] = (
                    'curl --fail-with-body -X POST -H "Authorization: Api-Key $BASETEN_API_KEY" '
                    f"{API_ROOT}/v1/models/{resource['model_id']}/deployments/{resource['deployment_id']}/deactivate"
                )
                self._save()
                _resource_id(resource.get("id"))
                resource.update(status="starting", owned=True, checkpoint=options.get("model_path"))
                self._save()
                sampler = self.service.connect(resource, self.model, self.receipt["session_id"], self.ready_timeout)
                self._samplers[identity] = sampler
                sampler.ensure_ready(self.ready_timeout)
                resource["status"] = "active"
                self._save()
            self.receipt["status"] = "running"
            self._save()
            return self.checkpoint
        except BaseException:
            self._cleanup()
            raise

    def _cleanup(self):
        failures = []
        for resource in reversed(self.receipt.get("resources", [])):
            sampler = self._samplers.pop(resource["identity"], None)
            if sampler is not None:
                try:
                    sampler.close()
                except BaseException as exc:
                    failures.append(type(exc).__name__)
            try:
                # Valid IDs remain usable for cleanup even if creation returned a
                # malformed sampler URL or ID and full receipt validation failed.
                _resource_id(resource.get("model_id"))
                _resource_id(resource.get("deployment_id"))
                self.service.deactivate(resource)
                resource["status"] = "inactive"
            except BaseException as exc:
                resource["status"] = "cleanup_required"
                failures.append(type(exc).__name__)
        self.receipt.update(status="cleanup_required" if failures else "closed", closed_at_utc=_utc_now())
        self._save()
        if failures:
            raise PipelineError(
                f"Baseten sampler cleanup needs attention; GPU charges may continue. Inspect {self.output_dir / 'sampler.json'} "
                "and deactivate its recorded deployments; reconcile unknown creates using the saved session ID."
            )

    def __exit__(self, exc_type, exc, tb):
        self._cleanup()

    def generate(self, model, messages, max_tokens, json_mode=False, request_contract=None):
        from baseten.loops import ModelInput, SamplingParams
        from smithtune.providers.baseten_sampling_formats import parse_completion, stop_sequences

        if model not in self._samplers or json_mode:
            raise PipelineError("Baseten replay can sample only its checkpoint and matching base model")
        if type(max_tokens) is not int or max_tokens < 1:
            raise PipelineError("max output tokens must be positive")
        tools = list(request_contract.tools) if request_contract is not None else []
        tokens = self.renderer.prompt_tokens(messages, tools=tools)
        if len(tokens) + max_tokens > self.model.max_seq_len:
            raise PipelineError("replay prompt and output budget exceed model context limit")
        response = self._samplers[model].sample(
            prompt=ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0, stop=stop_sequences(self.renderer, self.model)),
        )
        if len(response.sequences or []) != 1:
            raise PipelineError("Baseten sampler returned no unique completion")
        sequence = response.sequences[0]
        output = list(sequence.tokens or [])
        candidate = parse_completion(self.renderer, self.model, output, sequence.stop_reason, tools=tools)
        candidate.setdefault("sampling", {}).update(
            session_id=self.receipt["session_id"], checkpoint=self.checkpoint if model == self.checkpoint else None,
            prompt_tokens=len(tokens), output_tokens=len(output),
        )
        return candidate

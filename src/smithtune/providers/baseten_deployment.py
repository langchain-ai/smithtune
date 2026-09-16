"""Owned Baseten checkpoint deployments, with recoverable local receipts."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from smithtune.artifacts import _json_dump, _load_json, exclusive_output, output_lock
from smithtune.providers.baseten_truss import prepare_deployment
from smithtune.capabilities import open_without_redirects
from smithtune.inference import BasetenEndpoint, _baseten_chat_completion
from smithtune.inference_contract import ContractError, json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import _require_confirm


class ControlError(PipelineError):
    def __init__(self, status: int | None):
        self.status = status
        super().__init__(f"Baseten control request failed: HTTP {status}" if status else
                         "Baseten control request failed; write outcome may be unknown")


def _request(method: str, path: str, *, endpoint: BasetenEndpoint | None = None) -> dict:
    key = os.environ.get("BASETEN_API_KEY", "").strip()
    if not key:
        raise PipelineError("BASETEN_API_KEY is not set")
    origin = "https://api.baseten.co" if endpoint is None else endpoint.url.removesuffix("/chat/completions")
    request = urllib.request.Request(origin + path, method=method,
                                    headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with open_without_redirects(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ControlError(exc.code) from None
    except (OSError, ValueError):
        raise ControlError(None) from None
    if not isinstance(result, dict):
        raise ControlError(None)
    return result


def _training_identity(run_dir: Path) -> dict:
    plan = _load_json(run_dir / "plan.json")
    result = _load_json(run_dir / "result.json")
    if not isinstance(plan, dict) or not isinstance(result, dict) or any(
        item.get("provider") != "baseten" for item in (plan, result)
    ):
        raise PipelineError("deploy requires a Baseten training run")
    uri = result.get("best_sampler_weights_uri")
    match = re.fullmatch(r"bt://loops:([A-Za-z0-9_-]+)/sampler_weights/([A-Za-z0-9_.-]+)", uri if isinstance(uri, str) else "")
    base = plan.get("base_model")
    if not match or match[1] != result.get("baseten_run_id") or not isinstance(base, str) or not base:
        raise PipelineError("training run has no valid best sampler checkpoint; resumable training state cannot be deployed")
    return {"checkpoint_path": uri, "run_id": match[1], "checkpoint_name": match[2], "base_model": base}


def _checkpoint(identity: dict) -> dict:
    response = _request("GET", "/v1/loops/checkpoints?" + urllib.parse.urlencode({"checkpoint_path": identity["checkpoint_path"]}))
    checkpoints = response.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1 or not isinstance(checkpoints[0], dict):
        raise PipelineError("Baseten must return exactly one checkpoint for the saved sampler URI")
    checkpoint = checkpoints[0]
    if (checkpoint.get("run_id") != identity["run_id"]
            or checkpoint.get("checkpoint_id") != identity["checkpoint_name"]
            or checkpoint.get("base_model") != identity["base_model"]
            or checkpoint.get("target") != "sampler"
            or not isinstance(checkpoint.get("id"), str)
            or re.fullmatch(r"[A-Za-z0-9_-]+", checkpoint["id"]) is None):
        raise PipelineError("Baseten checkpoint metadata does not match the saved training run")
    if checkpoint.get("sync_status") not in (None, "COMPLETE"):
        raise PipelineError("Baseten sampler checkpoint is still syncing; retry deployment later")
    return {**identity, "checkpoint_id": checkpoint["id"]}


def _settings(accelerator: str, max_seq_len: int, timeout: float) -> dict:
    if not isinstance(accelerator, str) or re.fullmatch(r"[A-Z][A-Z0-9_]*:[1-9][0-9]*", accelerator) is None:
        raise PipelineError("--accelerator must specify GPU type and count, for example H200:1")
    if type(max_seq_len) is not int or max_seq_len < 1024:
        raise PipelineError("--max-seq-len must be at least 1024 for deployment smoke tests")
    if not math.isfinite(timeout) or not 1 <= timeout <= 7200:
        raise PipelineError("deployment timeout must be between 1 and 7200 seconds")
    return {"accelerator": accelerator, "max_seq_len": max_seq_len}


def _receipt(run_dir: Path) -> dict:
    value = _load_json(run_dir / "endpoint.json")
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("provider") != "baseten" or value.get("owned") is not True
            or not isinstance(value.get("checkpoint"), dict) or not isinstance(value.get("settings"), dict)):
        raise PipelineError("invalid owned Baseten endpoint receipt")
    return value


def _matching_settings(receipt: dict, settings: dict) -> bool:
    if receipt["settings"] == settings:
        return True
    # Before the first successful smoke test, let a user correct an excessive
    # replay cap without recreating already allocated serving resources.
    return ("advertised_max_seq_len" not in receipt and
            receipt["settings"].get("accelerator") == settings["accelerator"])


def _endpoint(receipt: dict) -> BasetenEndpoint:
    data = receipt.get("endpoint")
    if not isinstance(data, dict):
        raise PipelineError("deployment creation outcome is unknown; inspect endpoint.json and reconcile the saved model name in Baseten before retrying")
    endpoint = BasetenEndpoint(data.get("model_id"), data.get("deployment_id"), data.get("max_seq_len"))
    endpoint.validate()
    return endpoint


def load_endpoint(run_dir: Path, *, require_ready: bool = True) -> tuple[BasetenEndpoint, str]:
    receipt = _receipt(run_dir)
    if require_ready and receipt.get("state") != "ready":
        raise PipelineError("saved Baseten deployment is not ready; rerun deploy or inspect endpoint.json")
    endpoint = _endpoint(receipt)
    name = receipt["checkpoint"].get("checkpoint_name")
    if not isinstance(name, str) or not name:
        raise PipelineError("saved Baseten endpoint has no checkpoint route")
    return endpoint, name


def _save(run_dir: Path, receipt: dict, state: str, **extra) -> None:
    receipt.update(state=state, **extra)
    _json_dump(run_dir / "endpoint.json", receipt)


def _resource(endpoint: BasetenEndpoint) -> str:
    endpoint.validate()
    return f"/v1/models/{endpoint.model_id}/deployments/{endpoint.deployment_id}"


def _status(endpoint: BasetenEndpoint) -> str:
    result = _request("GET", _resource(endpoint))
    if result.get("id") != endpoint.deployment_id or result.get("model_id") != endpoint.model_id:
        raise PipelineError("Baseten returned a different deployment identity")
    status = result.get("status")
    if not isinstance(status, str):
        raise PipelineError("Baseten returned no deployment status")
    return status


def _wait(endpoint: BasetenEndpoint, timeout: float, *, inactive: bool = False) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status = _status(endpoint)
        except ControlError as exc:
            if inactive and exc.status == 404:
                return
            raise
        if status in ({"INACTIVE"} if inactive else {"ACTIVE", "SCALED_TO_ZERO"}):
            return
        if not inactive and status in {"UNHEALTHY", "DEPLOY_FAILED", "BUILD_FAILED", "BUILD_STOPPED", "FAILED", "DEACTIVATING"}:
            raise PipelineError(f"Baseten deployment is {status}; inspect endpoint.json for cleanup")
        if time.monotonic() >= deadline:
            raise PipelineError("Baseten deployment readiness timed out" if not inactive else "Baseten deactivation timed out")
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def _smoke(endpoint: BasetenEndpoint, checkpoint_name: str) -> int:
    catalog = _request("GET", "/models", endpoint=endpoint)
    data = catalog.get("data")
    if not isinstance(data, list):
        raise PipelineError("Baseten returned no serving model catalog")
    matches = [item for item in data if isinstance(item, dict) and item.get("id") == checkpoint_name]
    if len(matches) != 1:
        raise PipelineError("Baseten does not advertise the saved checkpoint name on /v1/models")
    context = matches[0].get("max_model_len")
    if type(context) is not int or context < endpoint.max_seq_len:
        raise PipelineError("Baseten does not advertise enough serving context for --max-seq-len; lower the cap or configure serving through Baseten")
    text = _baseten_chat_completion(checkpoint_name, [{"role": "user", "content": "Reply with the word hello."}], 256, endpoint=endpoint)
    if not isinstance(text.get("content"), str) or not text["content"].strip() or text.get("tool_calls"):
        raise PipelineError("Baseten text smoke test returned no visible answer")
    tools = [{"type": "function", "function": {
        "name": "lookup_number", "description": "Look up the requested number. Use this tool when asked to look up a number.",
        "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"], "additionalProperties": False},
    }}]
    contract = parse_inference_contract({"schema_version": 1, "format": "main_model_inference_contract",
                                         "tools": tools, "tools_sha256": json_sha256(tools), "provenance": {},
                                         "inference_settings": {"tool_choice": "auto"}})
    messages = [{"role": "user", "content": "Use lookup_number to look up number 7. Do not answer without calling it."}]
    call = _baseten_chat_completion(checkpoint_name, messages, 512, request_contract=contract, endpoint=endpoint)
    calls = call.get("tool_calls")
    try:
        contract.validate_messages([call])
        valid = (isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0].get("id"), str)
                 and bool(calls[0]["id"]) and calls[0].get("type") == "function"
                 and calls[0]["function"]["name"] == "lookup_number"
                 and json.loads(calls[0]["function"]["arguments"]) == {"number": 7})
    except (ContractError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise PipelineError("Baseten tool smoke test failed; the generated serving template/model did not return a valid tool call")
    messages.extend([{**call, "role": "assistant"}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "The value is seven."}])
    reply = _baseten_chat_completion(checkpoint_name, messages, 256, request_contract=contract, endpoint=endpoint)
    if not isinstance(reply.get("content"), str) or not reply["content"].strip() or reply.get("tool_calls"):
        raise PipelineError("Baseten tool-result smoke test returned no visible answer")
    return context


@exclusive_output("run_dir")
def deploy(run_dir: Path, *, accelerator: str, max_seq_len: int,
           timeout: float = 1800, confirm: bool) -> dict:
    """Deploy or resume one owned endpoint; failed creates are never blindly retried."""
    _require_confirm(confirm, "Baseten dedicated deployment and smoke-test inference")
    settings = _settings(accelerator, max_seq_len, timeout)
    identity = _training_identity(run_dir)
    path = run_dir / "endpoint.json"
    if path.exists():
        receipt = _receipt(run_dir)
        if not _matching_settings(receipt, settings) or any(receipt["checkpoint"].get(k) != v for k, v in identity.items()):
            raise PipelineError("existing deployment uses different checkpoint or settings; keep its receipt for cleanup")
        endpoint = _endpoint(receipt)
        if receipt["settings"] != settings:
            endpoint = BasetenEndpoint(endpoint.model_id, endpoint.deployment_id, max_seq_len)
            receipt.update(settings=settings, endpoint=endpoint.to_dict())
            _save(run_dir, receipt, "waiting")
    else:
        checkpoint = _checkpoint(identity)
        model_name = "smithtune-" + uuid4().hex
        create = prepare_deployment(checkpoint_id=checkpoint["checkpoint_id"], model_name=model_name,
                                    accelerator=accelerator)
        receipt = {"schema_version": 1, "provider": "baseten", "owned": True,
                   "model_name": model_name, "checkpoint": checkpoint, "settings": settings,
                   "cleanup_command": f"smithtune undeploy --provider baseten --run-dir {shlex.quote(str(run_dir.resolve()))} --confirm"}
        _save(run_dir, receipt, "creating")
        try:
            created = create()
            version = created.get("model_version") if isinstance(created, dict) else None
            if not isinstance(version, dict):
                raise PipelineError("Baseten create returned no deployment IDs; creation outcome is unknown")
            endpoint = BasetenEndpoint(version.get("model_id"), version.get("id"), max_seq_len)
            endpoint.validate()
            _save(run_dir, receipt, "created", endpoint=endpoint.to_dict())
            # IDs are durable before interpreting the server-generated configuration.
            config = created.get("truss_config")
            if not isinstance(config, str):
                raise PipelineError("Baseten returned no generated serving configuration")
            try:
                config = json.loads(config)
            except ValueError:
                raise PipelineError("Baseten returned an invalid serving configuration; IDs remain in endpoint.json") from None
            if not isinstance(config, dict):
                raise PipelineError("Baseten returned an invalid serving configuration")
            _json_dump(run_dir / "baseten-serving-config.json", config)
        except BaseException:
            _save(run_dir, receipt, "needs_attention" if receipt.get("endpoint") else "creation_unknown")
            raise
    try:
        status = _status(endpoint)
        if status == "INACTIVE":
            _save(run_dir, receipt, "activating")
            _request("POST", _resource(endpoint) + "/activate")
        _save(run_dir, receipt, "waiting")
        _wait(endpoint, timeout)
        _save(run_dir, receipt, "smoke_testing")
        advertised_context = _smoke(endpoint, receipt["checkpoint"]["checkpoint_name"])
        _save(run_dir, receipt, "ready", advertised_max_seq_len=advertised_context)
    except BaseException:
        _save(run_dir, receipt, "needs_attention")
        raise
    return receipt


@exclusive_output("run_dir")
def undeploy(run_dir: Path, *, confirm: bool) -> dict:
    """Stop only the recorded serving deployment, retaining checkpoint and config."""
    _require_confirm(confirm, "Baseten deployment deactivation")
    receipt = _receipt(run_dir)
    endpoint = _endpoint(receipt)
    try:
        try:
            status = _status(endpoint)
        except ControlError as exc:
            if exc.status != 404:
                raise
            status = "INACTIVE"
        if status != "INACTIVE":
            _save(run_dir, receipt, "deactivating")
            if status != "DEACTIVATING":
                _request("POST", _resource(endpoint) + "/deactivate")
            _wait(endpoint, 120, inactive=True)
        _save(run_dir, receipt, "inactive")
    except BaseException:
        _save(run_dir, receipt, "cleanup_required")
        raise
    return receipt


def plan(run_dir: Path, *, accelerator: str | None = None, max_seq_len: int | None = None,
         timeout: float = 1800) -> dict:
    """Resolve dedicated serving settings locally, without allocating compute."""
    identity = _training_identity(run_dir)
    saved = _receipt(run_dir) if (run_dir / "endpoint.json").exists() else None
    defaults = saved["settings"] if saved else {}
    settings = _settings(
        accelerator if accelerator is not None else defaults.get("accelerator"),
        max_seq_len if max_seq_len is not None else defaults.get("max_seq_len"),
        timeout,
    )
    if saved and (not _matching_settings(saved, settings) or any(saved["checkpoint"].get(k) != v for k, v in identity.items())):
        raise PipelineError("existing deployment uses different checkpoint or settings; keep its receipt for cleanup")
    return {"provider": "baseten", "serving_mode": "temporary", "checkpoint": identity,
            "settings": settings, "timeout": timeout,
            "cost": "dedicated GPU deployment plus model and judge inference",
            "cleanup": "deactivate the owned deployment after evaluation; retain checkpoint and resource IDs"}


@contextmanager
def temporary(run_dir: Path, *, accelerator: str | None = None, max_seq_len: int | None = None,
              timeout: float = 1800, confirm: bool):
    """Keep the same resource IDs on resume and release owned compute on exit."""
    _require_confirm(confirm, "temporary Baseten dedicated deployment and evaluation")
    with output_lock(run_dir):
        config = plan(run_dir, accelerator=accelerator, max_seq_len=max_seq_len,
                      timeout=timeout)
        try:
            deploy.__wrapped__(run_dir, **config["settings"], timeout=timeout, confirm=True)
            yield load_endpoint(run_dir)
        finally:
            if (run_dir / "endpoint.json").exists():
                receipt = _receipt(run_dir)
                if receipt.get("endpoint"):
                    try:
                        undeploy.__wrapped__(run_dir, confirm=True)
                    except BaseException as exc:
                        raise PipelineError(
                            f"Baseten cleanup needs attention; inspect {run_dir / 'endpoint.json'}; "
                            f"{receipt.get('cleanup_command', '')}"
                        ) from exc


def validate_evaluation_model(run_dir: Path, data_dir: Path) -> None:
    """Use the checkpoint's tokenizer for replay context accounting."""
    from smithtune.dataset import _require_prepared_provider

    identity = _training_identity(run_dir)
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    model = _require_prepared_provider(manifest, "baseten")
    if model.base_model != identity["base_model"]:
        raise PipelineError("prepared data uses a different base model from the Baseten training checkpoint")

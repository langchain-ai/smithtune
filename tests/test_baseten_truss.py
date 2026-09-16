from types import SimpleNamespace
import logging
import sys
import threading

import pytest

from smithtune.providers import baseten_truss
from smithtune.providers.base import PipelineError


@pytest.fixture
def deployment_adapter(monkeypatch):
    calls = []
    result = {"model_version": {"id": "deployment-1", "model_id": "model-1"}, "truss_config": "invalid JSON"}
    instance = SimpleNamespace(id="instance-1", gpu_type="H200", gpu_count=1, node_count=1)

    def create(request):
        calls.append(("create", request))
        return result

    api = SimpleNamespace(
        create_model_version_from_inference_template=create,
        get_instance_types=lambda: [instance],
    )

    def remote(url, *, api_key):
        assert url == "https://app.baseten.co"
        assert api_key == "test-key"
        return SimpleNamespace(api=api)

    def build(config, provider, *, dry_run):
        calls.append(("prepare", config))
        assert provider.api is api
        assert dry_run is False
        return {"instance_type_id": "instance-1"}

    modules = {
        "truss.base.truss_config": SimpleNamespace(Accelerator=lambda value: value, AcceleratorSpec=SimpleNamespace),
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints": SimpleNamespace(_build_inference_template_request=build),
        "truss.cli.train.types": SimpleNamespace(DeployCheckpointsConfigComplete=SimpleNamespace),
        "truss.remote.baseten.remote": SimpleNamespace(BasetenRemote=remote),
        "truss_train.definitions": SimpleNamespace(
            CheckpointList=SimpleNamespace, Compute=SimpleNamespace,
            DeployCheckpointsRuntime=SimpleNamespace, SecretReference=SimpleNamespace,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    monkeypatch.setattr(baseten_truss, "version", lambda name: "0.18.30")
    return SimpleNamespace(calls=calls, result=result, instance=instance, api=api)


def prepare():
    return baseten_truss.prepare_deployment(
        checkpoint_id="checkpoint-1", model_name="smithtune-example",
        accelerator="H200:1", hf_token_secret="hf_access_token",
    )


def test_prepare_does_not_create_and_returns_unparsed_resource_ids(deployment_adapter):
    create = prepare()
    assert [name for name, _ in deployment_adapter.calls] == ["prepare"]
    config = deployment_adapter.calls[0][1]
    assert config.checkpoint_details.loops_checkpoint_ids == ["checkpoint-1"]
    assert config.runtime.environment_variables["HF_TOKEN"].name == "hf_access_token"
    assert create() is deployment_adapter.result
    assert [name for name, _ in deployment_adapter.calls] == ["prepare", "create"]
    with pytest.raises(PipelineError, match="already been attempted"):
        create()


def test_create_failure_is_unknown_sanitized_and_not_retried(deployment_adapter):
    calls = []

    def fail(request):
        calls.append(request)
        raise RuntimeError("sensitive provider response")

    deployment_adapter.api.create_model_version_from_inference_template = fail
    create = prepare()
    with pytest.raises(PipelineError, match="outcome is unknown") as error:
        create()
    assert "sensitive" not in str(error.value)
    assert error.value.__suppress_context__
    with pytest.raises(PipelineError, match="already been attempted"):
        create()
    assert len(calls) == 1


def test_larger_gpu_fallback_is_rejected_before_create(deployment_adapter):
    deployment_adapter.instance.gpu_count = 2
    with pytest.raises(PipelineError, match="exact GPU allocation"):
        prepare()
    assert [name for name, _ in deployment_adapter.calls] == ["prepare"]


def test_unsupported_truss_pin_fails_before_provider_read(deployment_adapter, monkeypatch):
    monkeypatch.setattr(baseten_truss, "version", lambda name: "0.18.31")
    with pytest.raises(PipelineError, match="truss==0.18.30"):
        prepare()
    assert deployment_adapter.calls == []


def test_invalid_accelerator_count_fails_before_provider_read(deployment_adapter):
    with pytest.raises(PipelineError, match="optional positive count"):
        baseten_truss.prepare_deployment(
            checkpoint_id="checkpoint-1", model_name="smithtune-example",
            accelerator="H200:1:8", hf_token_secret="hf_access_token",
        )
    assert deployment_adapter.calls == []


@pytest.mark.parametrize("phase", ["prepare", "create"])
def test_sensitive_provider_logs_are_scoped_to_calling_thread(deployment_adapter, caplog, phase):
    logger = logging.getLogger("truss.remote.baseten.api")
    existing_filters = logger.filters[:]

    def fail(*args):
        logger.error("sensitive provider response")
        logging.getLogger("unrelated").error("unrelated current thread")
        thread = threading.Thread(target=lambda: logger.error("other provider thread"))
        thread.start()
        thread.join()
        raise RuntimeError("sensitive provider exception")

    if phase == "prepare":
        deployment_adapter.api.get_instance_types = fail
        operation = prepare
    else:
        deployment_adapter.api.create_model_version_from_inference_template = fail
        operation = prepare()
    with pytest.raises(PipelineError) as error:
        operation()
    assert "sensitive" not in str(error.value)
    assert "sensitive" not in caplog.text
    assert "unrelated current thread" in caplog.text
    assert "other provider thread" in caplog.text
    assert logger.filters == existing_filters
    logger.error("provider logging restored")
    assert "provider logging restored" in caplog.text


def test_released_truss_builds_checkpoint_request_without_live_api(monkeypatch):
    pytest.importorskip("truss", reason="Requires the optional baseten-deploy extra")
    from truss.remote.baseten.api import BasetenApi

    calls = []
    instance = SimpleNamespace(id="instance-1", gpu_type="H200", gpu_count=1, node_count=1)
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    monkeypatch.setattr(BasetenApi, "get_instance_types", lambda self: [instance])

    def create(self, request):
        calls.append(request)
        return {"model_version": {"id": "deployment-1", "model_id": "model-1"}, "truss_config": "invalid JSON"}

    monkeypatch.setattr(BasetenApi, "create_model_version_from_inference_template", create)
    operation = prepare()
    assert calls == []
    assert operation()["model_version"]["id"] == "deployment-1"
    assert calls == [{
        "metadata": {"oracle_name": "smithtune-example"},
        "weights_sources": [{
            "weight_source_type": "B10_LOOPS_CHECKPOINTING",
            "b10_loops_checkpoint_weights_source": {"checkpoint": {"loops_checkpoint_id": "checkpoint-1"}},
        }],
        "inference_stack": {
            "stack_type": "VLLM",
            "environment_variables": [{"name": "HF_TOKEN", "value": "hf_access_token", "is_secret_reference": True}],
        },
        "instance_type_id": "instance-1",
        "dry_run": False,
    }]

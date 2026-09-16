import copy
import io
import json
import sys
import urllib.error
from types import SimpleNamespace

import pytest

from smithtune.providers import baseten_sampling as sampling
from smithtune.artifacts import _json_dump
from smithtune.providers.base import PipelineError
from smithtune.providers.baseten import DEFAULT_MODEL
from test_pipeline import write_manifest


CHECKPOINT = "bt://loops:run-1/sampler_weights/best-2"


@pytest.fixture
def renderer(monkeypatch):
    rendered = []

    def prompt(messages, tools):
        rendered.append(copy.deepcopy((messages, tools)))
        return [10, 11, 12]

    renderer = SimpleNamespace(tokenizer=SimpleNamespace(), prompt_tokens=prompt)
    monkeypatch.setattr(sampling, "load_training_renderer", lambda _: renderer)
    monkeypatch.setitem(sys.modules, "smithtune.providers.baseten_sampling_formats", SimpleNamespace(
        PARSING_VERSION="test-v1", stop_sequences=lambda *_: ["<end>"],
        parse_completion=lambda *args, **kwargs: {"role": "assistant", "content": "answer",
                                                 "sampling": {"format_valid": True}},
    ))
    return rendered


class FakeService:
    def __init__(self, directory, *, fail_ready=None, fail_create=None, fail_cleanup=None, missing_field=None):
        self.directory = directory
        self.fail_ready = fail_ready
        self.fail_create = fail_create
        self.fail_cleanup = fail_cleanup
        self.missing_field = missing_field
        self.events = []
        self.requests = []
        self.created = 0

    def validate_checkpoint(self, checkpoint, model):
        self.events.append(("validate", checkpoint, model))

    def create_session(self):
        self.events.append("session")
        return "session-1"

    def create_sampler(self, body):
        self.created += 1
        self.events.append(("create", body))
        if self.fail_create == self.created:
            raise TimeoutError("unknown create outcome")
        resource = {"id": f"sampler-{self.created}", "model_id": f"model-{self.created}",
                    "deployment_id": f"deployment-{self.created}", "base_url": "https://model-id.api.baseten.co/sync"}
        if self.missing_field:
            resource.pop(self.missing_field)
        return resource

    def connect(self, resource, model, session_id, ready_timeout):
        service = self
        index = self.created
        # The paid resource is recoverable even if SDK construction fails.
        saved = json.loads((self.directory / "sampler.json").read_text())
        assert saved["resources"][-1]["deployment_id"] == resource["deployment_id"]

        class Client:
            def ensure_ready(self, timeout):
                service.events.append(("ready", index))
                if service.fail_ready == index:
                    raise KeyboardInterrupt("readiness interrupted")

            def close(self):
                service.events.append(("close", index))

            def sample(self, **kwargs):
                service.requests.append(kwargs)
                return SimpleNamespace(sequences=[SimpleNamespace(tokens=[21, 22], stop_reason="stop")])

        return Client()

    def deactivate(self, resource):
        self.events.append(("deactivate", resource["deployment_id"]))
        if resource["deployment_id"] == self.fail_cleanup:
            raise OSError("cleanup failed")

    def is_inactive(self, resource):
        return False


def test_fixed_checkpoint_and_matching_base_use_official_tokens_and_explicit_cleanup(tmp_path, renderer):
    service = FakeService(tmp_path)
    adapter = sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)
    messages = [{"role": "user", "content": "café"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with adapter as identity:
        assert identity == CHECKPOINT
        result = adapter.generate(identity, messages, 32, request_contract=SimpleNamespace(tools=tools))
        adapter.generate(DEFAULT_MODEL.base_model, messages, 32)
        with pytest.raises(PipelineError, match="matching base model"):
            adapter.generate("wrong-model", messages, 32)
        with pytest.raises(PipelineError, match="context limit"):
            adapter.generate(identity, messages, DEFAULT_MODEL.max_seq_len)
    creates = [event[1] for event in service.events if isinstance(event, tuple) and event[0] == "create"]
    assert creates == [
        {"session_id": "session-1", "max_seq_length": DEFAULT_MODEL.max_seq_len, "model_path": CHECKPOINT},
        {"session_id": "session-1", "max_seq_length": DEFAULT_MODEL.max_seq_len, "base_model": DEFAULT_MODEL.base_model},
    ]
    assert renderer[:2] == [(messages, tools), (messages, [])]
    request = service.requests[0]
    assert type(request["prompt"]).__module__.startswith("baseten.loops")
    assert request["prompt"].to_ints() == [10, 11, 12]
    assert request["num_samples"] == 1
    assert request["sampling_params"].temperature == 0
    assert request["sampling_params"].max_tokens == 32
    assert request["sampling_params"].stop == ["<end>"]
    assert result["sampling"]["format_valid"] is True
    assert result["sampling"]["checkpoint"] == CHECKPOINT
    assert result["sampling"]["prompt_tokens"] == 3
    assert result["sampling"]["output_tokens"] == 2
    assert service.events[-4:] == [("close", 2), ("deactivate", "deployment-2"), ("close", 1), ("deactivate", "deployment-1")]
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["status"] == "closed"
    assert all(resource["status"] == "inactive" for resource in receipt["resources"])
    assert "deactivate" in receipt["resources"][0]["cleanup_command"]
    assert adapter.config["serving_mode"] == "sampler"


@pytest.mark.parametrize("failing", [1, 2])
def test_readiness_interrupt_deactivates_all_created_resources(tmp_path, renderer, failing):
    service = FakeService(tmp_path, fail_ready=failing)
    with pytest.raises(KeyboardInterrupt), sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service):
        pass
    assert [("deactivate", f"deployment-{index}") for index in reversed(range(1, failing + 1))] == [
        event for event in service.events if isinstance(event, tuple) and event[0] == "deactivate"
    ]
    assert json.loads((tmp_path / "sampler.json").read_text())["status"] == "closed"


def test_body_failure_deactivates_every_resource_even_when_one_cleanup_fails(tmp_path, renderer):
    service = FakeService(tmp_path, fail_cleanup="deployment-2")
    with pytest.raises(PipelineError, match="GPU charges may continue"), sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service):
        raise RuntimeError("judge failed")
    assert ("deactivate", "deployment-1") in service.events
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["status"] == "cleanup_required"
    assert [resource["status"] for resource in receipt["resources"]] == ["inactive", "cleanup_required"]
    with pytest.raises(PipelineError, match="reconciliation"):
        sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)


def test_unknown_create_is_never_retried_and_blocks_new_allocation(tmp_path, renderer):
    service = FakeService(tmp_path, fail_create=2)
    with pytest.raises(PipelineError, match="reconcile unknown creates"), sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service):
        pass
    assert service.created == 2
    assert ("deactivate", "deployment-1") in service.events
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["session_id"] == "session-1"
    assert receipt["resources"][1]["status"] == "cleanup_required"
    with pytest.raises(PipelineError, match="reconciliation"):
        sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)
    assert service.created == 2


def test_incomplete_create_response_still_deactivates_returned_deployment(tmp_path, renderer):
    service = FakeService(tmp_path, missing_field="id")
    with pytest.raises(PipelineError, match="invalid resource ID"), sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service):
        pass
    assert ("deactivate", "deployment-1") in service.events
    receipt = json.loads((tmp_path / "sampler.json").read_text())
    assert receipt["status"] == "closed"
    assert "cleanup_command" in receipt["resources"][0]


def test_saved_run_accepts_only_best_sampler_weights_for_the_matching_base(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data, model=DEFAULT_MODEL)
    _json_dump(run / "plan.json", {"provider": "baseten", "base_model": DEFAULT_MODEL.base_model})
    result = {"provider": "baseten", "baseten_run_id": "run-1", "best_sampler_weights_uri": CHECKPOINT,
              "last_resumable_state_uri": "bt://loops:run-1/checkpoints/last"}
    _json_dump(run / "result.json", result)
    model, checkpoint = sampling.checkpoint_from_run(data, run)
    assert model.base_model == DEFAULT_MODEL.base_model
    assert checkpoint == CHECKPOINT
    _json_dump(run / "result.json", {**result, "best_sampler_weights_uri": result["last_resumable_state_uri"]})
    with pytest.raises(PipelineError, match="best sampler checkpoint"):
        sampling.checkpoint_from_run(data, run)
    _json_dump(run / "result.json", {**result, "baseten_run_id": "other"})
    with pytest.raises(PipelineError, match="best sampler checkpoint"):
        sampling.checkpoint_from_run(data, run)
    _json_dump(run / "result.json", result)
    _json_dump(run / "plan.json", {"provider": "baseten", "base_model": "different"})
    with pytest.raises(PipelineError, match="base model differs"):
        sampling.checkpoint_from_run(data, run)


def test_control_transport_single_attempt_redacts_error_body_and_uses_api_key(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "private-key")
    requests = []

    def fail(request, timeout):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, 500, "private-key", {}, io.BytesIO(b"private-key"))

    monkeypatch.setattr(sampling, "open_without_redirects", fail)
    with pytest.raises(PipelineError, match="HTTP 500") as error:
        sampling._request("POST", "/v1/loops/samplers", {"session_id": "session-1"})
    assert "private-key" not in str(error.value)
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") == "Api-Key private-key"
    assert json.loads(requests[0].data) == {"session_id": "session-1"}


def test_checkpoint_remote_identity_is_checked_before_creation(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    monkeypatch.setattr(sampling, "_request", lambda *_: {"checkpoints": [{
        "run_id": "run-1", "checkpoint_id": "best-2", "target": "sampler", "base_model": "wrong",
    }]})
    with pytest.raises(PipelineError, match="metadata differs"):
        sampling._SamplerService().validate_checkpoint(CHECKPOINT, DEFAULT_MODEL.base_model)


def test_gpu_deactivation_waits_for_inactive(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    events = []

    def request(method, path):
        events.append((method, path))
        return {"status": "INACTIVE"}

    monkeypatch.setattr(sampling, "_request", request)
    sampling._SamplerService().deactivate({"model_id": "m1", "deployment_id": "d1"})
    assert events == [("POST", "/v1/models/m1/deployments/d1/deactivate"), ("GET", "/v1/models/m1/deployments/d1")]


def test_sdk_client_is_pinned_to_saved_weights_with_no_live_trainer(monkeypatch):
    from baseten.loops import sampling_client

    monkeypatch.setenv("BASETEN_API_KEY", "test")
    calls = []
    monkeypatch.setattr(sampling_client, "SamplingClient", lambda **kwargs: calls.append(kwargs) or object())
    service = sampling._SamplerService()
    resource = {"id": "sampler1", "model_id": "model1", "deployment_id": "deployment1",
                "base_url": "https://model-model1.api.baseten.co/deployment/deployment1/sync", "checkpoint": CHECKPOINT}
    service.connect(resource, DEFAULT_MODEL, "session1", 12)
    assert calls[0]["checkpoint_path"] == CHECKPOINT
    assert calls[0]["base_model"] == DEFAULT_MODEL.base_model
    assert calls[0]["session_id"] == "session1"
    assert calls[0]["sampler_id"] == "sampler1"
    assert calls[0]["ready_timeout"] == 12
    assert "run_id" not in calls[0]
    assert "min_policy_version" not in calls[0]
    assert calls[0]["deployment"].api_base_url == sampling.API_ROOT
    with pytest.raises(PipelineError, match="invalid sampler URL"):
        service.connect({**resource, "base_url": "https://untrusted.example/sync"}, DEFAULT_MODEL, "session1", 12)
    assert len(calls) == 1


def test_persistence_failure_after_create_still_deactivates_gpu(tmp_path, renderer, monkeypatch):
    service = FakeService(tmp_path)
    adapter = sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)
    original = adapter._save

    def save():
        if adapter.receipt.get("resources") and adapter.receipt["resources"][0].get("model_id"):
            raise OSError("disk full")
        original()

    monkeypatch.setattr(adapter, "_save", save)
    with pytest.raises(OSError, match="disk full"), adapter:
        pass
    assert ("deactivate", "deployment-1") in service.events


def test_manually_deactivated_receipt_can_resume_without_constructor_writes(tmp_path, renderer):
    service = FakeService(tmp_path)
    previous = {"status": "cleanup_required", "session_id": "old-session", "resources": [
        {"identity": CHECKPOINT, "model_id": "old-model", "deployment_id": "old-deployment", "status": "cleanup_required"},
    ]}
    _json_dump(tmp_path / "sampler.json", previous)
    before = (tmp_path / "sampler.json").read_bytes()
    checked = []

    def is_inactive(resource):
        checked.append(resource["deployment_id"])
        return True

    service.is_inactive = is_inactive
    adapter = sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)
    assert checked == ["old-deployment"]
    assert service.created == 0
    assert (tmp_path / "sampler.json").read_bytes() == before
    with adapter:
        assert checked == ["old-deployment", "old-deployment"]
    assert json.loads((tmp_path / "sampler.json").read_text())["status"] == "closed"


def test_manually_cleaned_receipt_is_rechecked_before_creating(tmp_path, renderer):
    service = FakeService(tmp_path)
    _json_dump(tmp_path / "sampler.json", {"status": "running", "resources": [
        {"model_id": "old-model", "deployment_id": "old-deployment"},
    ]})
    before = (tmp_path / "sampler.json").read_bytes()
    inactive = iter([True, False])
    service.is_inactive = lambda _: next(inactive)
    adapter = sampling.BasetenReplaySampler(DEFAULT_MODEL, CHECKPOINT, tmp_path, service=service)
    with pytest.raises(PipelineError, match="reconciliation"), adapter:
        pass
    assert service.created == 0
    assert (tmp_path / "sampler.json").read_bytes() == before


@pytest.mark.parametrize("response,expected", [
    ({"id": "d1", "model_id": "m1", "status": "INACTIVE"}, True),
    ({"id": "d1", "model_id": "m1", "status": "DEACTIVATING"}, False),
    ({"id": "different", "model_id": "m1", "status": "INACTIVE"}, False),
])
def test_manual_cleanup_verification_reads_exact_deployment(monkeypatch, response, expected):
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    calls = []
    monkeypatch.setattr(sampling, "_request", lambda *args: calls.append(args) or response)
    assert sampling._SamplerService().is_inactive({"model_id": "m1", "deployment_id": "d1"}) is expected
    assert calls == [("GET", "/v1/models/m1/deployments/d1")]

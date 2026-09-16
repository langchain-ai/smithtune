"""Owned deployment recovery and serving checks without paid provider calls."""

import copy
import io
import json
import urllib.error
import urllib.parse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smithtune.providers import baseten_deployment as deployment
from smithtune.inference import BasetenEndpoint
from smithtune.providers.base import PipelineError


ENDPOINT = BasetenEndpoint("model123", "deploy123", 8192)
RESOURCE = "/v1/models/model123/deployments/deploy123"
IDENTITY = {"checkpoint_path": "bt://loops:run-123/sampler_weights/checkpoint-7",
            "run_id": "run-123", "checkpoint_name": "checkpoint-7", "base_model": "Qwen/Qwen3-8B"}
CHECKPOINT = {"id": "checkpoint123", "run_id": "run-123", "checkpoint_id": "checkpoint-7",
              "base_model": "Qwen/Qwen3-8B", "target": "sampler", "sync_status": "COMPLETE"}


def read_receipt(run):
    return json.loads((run / "endpoint.json").read_text())


def deploy(run, **kwargs):
    return deployment.deploy(run, **{"accelerator": "H200:1", "max_seq_len": 8192,
                                     "confirm": True, **kwargs})


@pytest.fixture
def run(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"provider": "baseten", "base_model": IDENTITY["base_model"]}))
    (tmp_path / "result.json").write_text(json.dumps({"provider": "baseten", "baseten_run_id": "run-123",
                                                  "best_sampler_weights_uri": IDENTITY["checkpoint_path"]}))
    return tmp_path


@pytest.fixture
def backend(monkeypatch):
    def request(method, path, **kwargs):
        if path.startswith("/v1/loops/checkpoints?"):
            assert method == "GET"
            assert urllib.parse.parse_qs(urllib.parse.urlsplit(path).query) == {
                "checkpoint_path": [IDENTITY["checkpoint_path"]],
            }
            return {"checkpoints": [copy.deepcopy(CHECKPOINT)]}
        assert path == RESOURCE
        assert method == "GET"
        return {"id": "deploy123", "model_id": "model123", "status": "ACTIVE"}

    api = Mock(side_effect=request)
    create = Mock(return_value={"model_version": {"model_id": "model123", "id": "deploy123"},
                                "truss_config": '{"model_name": "generated-model"}'})
    prepare = Mock(return_value=create)
    smoke = Mock(return_value=16384)
    monkeypatch.setattr(deployment, "_request", api)
    monkeypatch.setattr(deployment, "prepare_deployment", prepare)
    monkeypatch.setattr(deployment, "_smoke", smoke)
    monkeypatch.setattr(deployment.time, "sleep", lambda *_: pytest.fail("unexpected polling"))
    return SimpleNamespace(api=api, create=create, prepare=prepare, smoke=smoke)


def test_create_records_intent_before_provider_call_and_finishes_ready(run, backend):
    raw_result = backend.create.return_value

    def create():
        receipt = read_receipt(run)
        assert receipt["state"] == "creating"
        assert receipt["checkpoint"] == {**IDENTITY, "checkpoint_id": "checkpoint123"}
        assert receipt["model_name"] == backend.prepare.call_args.kwargs["model_name"]
        assert "endpoint" not in receipt
        return raw_result

    backend.create.side_effect = create
    receipt = deploy(run)
    assert receipt == read_receipt(run)
    assert receipt["state"] == "ready"
    assert receipt["endpoint"] == ENDPOINT.to_dict()
    assert receipt["advertised_max_seq_len"] == 16384
    assert "undeploy --provider baseten" in receipt["cleanup_command"]
    assert backend.prepare.call_args.kwargs == {
        "checkpoint_id": "checkpoint123", "model_name": receipt["model_name"],
        "accelerator": "H200:1",
    }
    backend.create.assert_called_once_with()
    backend.smoke.assert_called_once_with(ENDPOINT, "checkpoint-7")
    assert json.loads((run / "baseten-serving-config.json").read_text()) == {"model_name": "generated-model"}


def test_confirmation_is_required_before_reading_or_provisioning(run, backend):
    (run / "plan.json").unlink()
    with pytest.raises(PipelineError, match="confirm"):
        deploy(run, confirm=False)
    backend.api.assert_not_called()
    backend.prepare.assert_not_called()


@pytest.mark.parametrize("file,change", [
    ("plan.json", {"provider": "fireworks"}),
    ("plan.json", {"base_model": ""}),
    ("result.json", {"provider": "fireworks"}),
    ("result.json", {"baseten_run_id": "other-run"}),
    ("result.json", {"best_sampler_weights_uri": "bt://loops:run-123/training_state/checkpoint-7"}),
    ("result.json", {"best_sampler_weights_uri": None}),
])
def test_invalid_training_identity_cannot_provision(run, backend, file, change):
    path = run / file
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(PipelineError):
        deploy(run)
    backend.api.assert_not_called()
    backend.prepare.assert_not_called()
    assert not (run / "endpoint.json").exists()


@pytest.mark.parametrize("change", [
    {"run_id": "other"}, {"checkpoint_id": "other"}, {"base_model": "other"},
    {"target": "training"}, {"id": "../other"}, {"sync_status": "IN_PROGRESS"},
])
def test_checkpoint_mismatch_or_syncing_never_creates(run, backend, change):
    backend.api.side_effect = None
    backend.api.return_value = {"checkpoints": [{**CHECKPOINT, **change}]}
    with pytest.raises(PipelineError, match="match|syncing"):
        deploy(run)
    backend.prepare.assert_not_called()
    assert not (run / "endpoint.json").exists()


@pytest.mark.parametrize("checkpoints", [[], [CHECKPOINT, CHECKPOINT], [None]])
def test_checkpoint_lookup_must_resolve_exactly_one_record(run, backend, checkpoints):
    backend.api.side_effect = None
    backend.api.return_value = {"checkpoints": checkpoints}
    with pytest.raises(PipelineError, match="exactly one"):
        deploy(run)
    backend.prepare.assert_not_called()


@pytest.mark.parametrize("config", [None, "not-json", "[]"])
def test_created_ids_survive_invalid_generated_config(run, backend, config):
    backend.create.return_value["truss_config"] = config
    with pytest.raises(PipelineError, match="configuration"):
        deploy(run)
    receipt = read_receipt(run)
    assert receipt["state"] == "needs_attention"
    assert receipt["endpoint"] == ENDPOINT.to_dict()
    backend.smoke.assert_not_called()
    deploy(run)
    backend.create.assert_called_once()


def test_smoke_failure_keeps_ids_and_resumes_without_create(run, backend):
    def smoke(*_):
        assert read_receipt(run)["endpoint"] == ENDPOINT.to_dict()
        raise PipelineError("smoke failed")

    backend.smoke.side_effect = smoke
    with pytest.raises(PipelineError, match="smoke failed"):
        deploy(run)
    assert read_receipt(run)["state"] == "needs_attention"
    backend.smoke.side_effect = None
    assert deploy(run)["state"] == "ready"
    backend.create.assert_called_once()


@pytest.mark.parametrize("failure", ["transport", "interrupt", "missing_ids"])
def test_unknown_create_is_durable_and_never_retried(run, backend, failure):
    if failure == "missing_ids":
        backend.create.return_value = {}
    else:
        backend.create.side_effect = PipelineError("unknown create") if failure == "transport" else KeyboardInterrupt()
    with pytest.raises((PipelineError, KeyboardInterrupt)):
        deploy(run)
    before = (run / "endpoint.json").read_bytes()
    assert read_receipt(run)["state"] == "creation_unknown"
    with pytest.raises(PipelineError, match="outcome is unknown"):
        deploy(run)
    assert (run / "endpoint.json").read_bytes() == before
    backend.create.assert_called_once()
    backend.prepare.assert_called_once()


@pytest.mark.parametrize("change", [{"accelerator": "H200:2"}, {"max_seq_len": 4096}])
def test_changed_settings_preserve_existing_receipt(run, backend, change):
    deploy(run)
    before = (run / "endpoint.json").read_bytes()
    backend.api.reset_mock()
    with pytest.raises(PipelineError, match="different checkpoint or settings"):
        deploy(run, **change)
    assert (run / "endpoint.json").read_bytes() == before
    backend.api.assert_not_called()
    backend.create.assert_called_once()


def test_changed_training_identity_preserves_existing_receipt(run, backend):
    deploy(run)
    before = (run / "endpoint.json").read_bytes()
    path = run / "result.json"
    result = json.loads(path.read_text())
    result["best_sampler_weights_uri"] = "bt://loops:run-123/sampler_weights/checkpoint-8"
    path.write_text(json.dumps(result))
    backend.api.reset_mock()
    with pytest.raises(PipelineError, match="different checkpoint or settings"):
        deploy(run)
    assert (run / "endpoint.json").read_bytes() == before
    backend.api.assert_not_called()
    backend.create.assert_called_once()


def test_ready_receipt_rechecks_serving_and_reactivates_inactive_endpoint(run, backend):
    deploy(run)
    assert deployment.load_endpoint(run) == (ENDPOINT, "checkpoint-7")
    deploy(run)
    backend.api.reset_mock()
    backend.api.side_effect = [
        {"id": "deploy123", "model_id": "model123", "status": "INACTIVE"}, {},
        {"id": "deploy123", "model_id": "model123", "status": "ACTIVE"},
    ]
    assert deploy(run)["state"] == "ready"
    assert [(call.args[0], call.args[1]) for call in backend.api.call_args_list] == [
        ("GET", RESOURCE), ("POST", RESOURCE + "/activate"), ("GET", RESOURCE),
    ]
    backend.create.assert_called_once()
    assert backend.smoke.call_count == 3


@pytest.mark.parametrize("state", ["creating", "creation_unknown", "needs_attention", "inactive", "cleanup_required"])
def test_only_ready_receipts_can_be_used_for_replay(run, backend, state):
    deploy(run)
    receipt = read_receipt(run)
    receipt["state"] = state
    (run / "endpoint.json").write_text(json.dumps(receipt))
    with pytest.raises(PipelineError, match="not ready"):
        deployment.load_endpoint(run)


@pytest.mark.parametrize("change", [{"owned": False}, {"provider": "fireworks"}, {"schema_version": 2},
                                     {"endpoint": {"model_id": "../model", "deployment_id": "deploy123", "max_seq_len": 8192}}])
def test_cleanup_rejects_unowned_or_invalid_endpoint_before_network(run, backend, change):
    deploy(run)
    receipt = {**read_receipt(run), **change}
    (run / "endpoint.json").write_text(json.dumps(receipt))
    before = (run / "endpoint.json").read_bytes()
    backend.api.reset_mock()
    with pytest.raises(PipelineError):
        deployment.undeploy(run, confirm=True)
    assert (run / "endpoint.json").read_bytes() == before
    backend.api.assert_not_called()


def test_cleanup_failure_retries_only_recorded_deployment_and_preserves_training(run, backend):
    deploy(run)
    saved = {name: (run / name).read_bytes() for name in ("plan.json", "result.json", "baseten-serving-config.json")}
    backend.api.reset_mock()
    active = {"id": "deploy123", "model_id": "model123", "status": "ACTIVE"}
    backend.api.side_effect = [active, deployment.ControlError(503)]
    with pytest.raises(deployment.ControlError):
        deployment.undeploy(run, confirm=True)
    assert read_receipt(run)["state"] == "cleanup_required"
    assert read_receipt(run)["endpoint"] == ENDPOINT.to_dict()
    backend.api.side_effect = [active, {}, {**active, "status": "INACTIVE"}]
    assert deployment.undeploy(run, confirm=True)["state"] == "inactive"
    assert [(call.args[0], call.args[1]) for call in backend.api.call_args_list] == [
        ("GET", RESOURCE), ("POST", RESOURCE + "/deactivate"),
        ("GET", RESOURCE), ("POST", RESOURCE + "/deactivate"), ("GET", RESOURCE),
    ]
    assert {name: (run / name).read_bytes() for name in saved} == saved


@pytest.mark.parametrize("status", ["INACTIVE", "DEACTIVATING", "missing"])
def test_cleanup_is_idempotent_without_reposting_deactivation(run, backend, status):
    deploy(run)
    backend.api.reset_mock()
    backend.api.side_effect = [deployment.ControlError(404)] if status == "missing" else [
        {"id": "deploy123", "model_id": "model123", "status": status},
        {"id": "deploy123", "model_id": "model123", "status": "INACTIVE"},
    ]
    assert deployment.undeploy(run, confirm=True)["state"] == "inactive"
    assert all(call.args == ("GET", RESOURCE) for call in backend.api.call_args_list)


def test_readiness_timeout_is_bounded_and_preserves_ids(run, backend, monkeypatch):
    backend.api.side_effect = [
        {"checkpoints": [CHECKPOINT]},
        {"id": "deploy123", "model_id": "model123", "status": "BUILDING"},
        {"id": "deploy123", "model_id": "model123", "status": "BUILDING"},
    ]
    monkeypatch.setattr(deployment.time, "monotonic", Mock(side_effect=[0, 10]))
    with pytest.raises(PipelineError, match="timed out"):
        deploy(run, timeout=1)
    assert read_receipt(run)["state"] == "needs_attention"
    assert read_receipt(run)["endpoint"] == ENDPOINT.to_dict()
    backend.smoke.assert_not_called()


@pytest.mark.parametrize("endpoint,path,url", [
    (None, RESOURCE, "https://api.baseten.co" + RESOURCE),
    (ENDPOINT, "/models", ENDPOINT.url.removesuffix("/chat/completions") + "/models"),
])
def test_control_transport_uses_redirect_safe_opener_and_exact_origin(monkeypatch, endpoint, path, url):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    opener = Mock(return_value=io.BytesIO(b'{"ok": true}'))
    monkeypatch.setattr(deployment, "open_without_redirects", opener)
    assert deployment._request("GET", path, endpoint=endpoint) == {"ok": True}
    request = opener.call_args.args[0]
    assert request.full_url == url
    assert request.get_header("Authorization") == "Bearer test-key"
    assert opener.call_args.kwargs == {"timeout": 60}


@pytest.mark.parametrize("failure,status", [("redirect", 302), ("http", 403), ("transport", None),
                                            ("invalid_json", None), ("non_object", None)])
def test_control_transport_sanitizes_errors_and_rejects_redirects(monkeypatch, failure, status):
    monkeypatch.setenv("BASETEN_API_KEY", "private-key")

    def open_request(request, **kwargs):
        if status:
            raise urllib.error.HTTPError(request.full_url, status, "private-key", {}, io.BytesIO(b"private-body"))
        if failure == "transport":
            raise urllib.error.URLError("private-key private-body")
        return io.BytesIO(b"private-body" if failure == "invalid_json" else b"[]")

    monkeypatch.setattr(deployment, "open_without_redirects", open_request)
    with pytest.raises(deployment.ControlError) as error:
        deployment._request("GET", RESOURCE)
    assert error.value.status == status
    assert "private-key" not in str(error.value)
    assert "private-body" not in str(error.value)
    assert error.value.__context__ is None or error.value.__suppress_context__


def test_control_transport_requires_credential_before_network(monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    opener = Mock()
    monkeypatch.setattr(deployment, "open_without_redirects", opener)
    with pytest.raises(PipelineError, match="BASETEN_API_KEY"):
        deployment._request("GET", RESOURCE)
    opener.assert_not_called()


def tool_candidate():
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "lookup_number", "arguments": '{"number": 7}'}},
    ]}


@pytest.fixture
def smoke_backend(monkeypatch):
    api = Mock(return_value={"data": [{"id": "checkpoint-7", "max_model_len": 16384}]})
    chat = Mock(side_effect=[{"role": "assistant", "content": "hello"}, tool_candidate(),
                             {"role": "assistant", "content": "seven"}])
    monkeypatch.setattr(deployment, "_request", api)
    monkeypatch.setattr(deployment, "_baseten_chat_completion", chat)
    return SimpleNamespace(api=api, chat=chat)


def test_smoke_exercises_text_tool_call_and_tool_result_context(smoke_backend):
    assert deployment._smoke(ENDPOINT, "checkpoint-7") == 16384
    smoke_backend.api.assert_called_once_with("GET", "/models", endpoint=ENDPOINT)
    text, tool, result = smoke_backend.chat.call_args_list
    assert text.args[0] == tool.args[0] == result.args[0] == "checkpoint-7"
    assert [call.args[2] for call in (text, tool, result)] == [256, 512, 256]
    assert all(call.kwargs["endpoint"] == ENDPOINT for call in (text, tool, result))
    assert "request_contract" not in text.kwargs
    assert tool.kwargs["request_contract"] == result.kwargs["request_contract"]
    assert result.args[1][-2:] == [tool_candidate(),
                                  {"role": "tool", "tool_call_id": "call-1", "content": "The value is seven."}]
    assert tool.kwargs["request_contract"].tools[0]["function"]["name"] == "lookup_number"


@pytest.mark.parametrize("checkpoint_context", [{}, {"max_model_len": None}])
def test_smoke_uses_named_parent_context_for_lora(smoke_backend, checkpoint_context):
    smoke_backend.api.return_value = {"data": [
        {"id": "baseten-model", "max_model_len": 262144},
        {"id": "checkpoint-7", "parent": "baseten-model", **checkpoint_context},
    ]}
    assert deployment._smoke(ENDPOINT, "checkpoint-7") == 262144
    assert smoke_backend.chat.call_count == 3
    assert all(call.args[0] == "checkpoint-7" for call in smoke_backend.chat.call_args_list)


@pytest.mark.parametrize("context,parents", [
    (None, []),
    (None, [{"id": "unrelated", "max_model_len": 262144}]),
    (None, [{"id": "baseten-model", "max_model_len": 262144}] * 2),
    (None, [{"id": "baseten-model", "max_model_len": None}]),
    (None, [{"id": "baseten-model", "max_model_len": True}]),
    (None, [{"id": "baseten-model", "max_model_len": 4096}]),
    (4096, [{"id": "baseten-model", "max_model_len": 262144}]),
    (True, [{"id": "baseten-model", "max_model_len": 262144}]),
])
def test_smoke_parent_fallback_requires_unique_valid_context(smoke_backend, context, parents):
    smoke_backend.api.return_value = {"data": [
        {"id": "checkpoint-7", "parent": "baseten-model", "max_model_len": context}, *parents,
    ]}
    with pytest.raises(PipelineError, match="context"):
        deployment._smoke(ENDPOINT, "checkpoint-7")
    smoke_backend.chat.assert_not_called()


@pytest.mark.parametrize("catalog", [
    {}, {"data": []}, {"data": [{"id": "other", "max_model_len": 16384}]},
    {"data": [{"id": "checkpoint-7", "max_model_len": 4096}]},
    {"data": [{"id": "checkpoint-7", "max_model_len": True}]},
])
def test_smoke_requires_exact_advertised_route_and_context(smoke_backend, catalog):
    smoke_backend.api.return_value = catalog
    with pytest.raises(PipelineError, match="catalog|checkpoint|context"):
        deployment._smoke(ENDPOINT, "checkpoint-7")
    smoke_backend.chat.assert_not_called()


@pytest.mark.parametrize("stage,candidate,match", [
    (0, {"content": " "}, "text smoke"),
    (0, tool_candidate(), "text smoke"),
    (1, {"role": "assistant", "content": "seven"}, "tool smoke"),
    (1, {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                            "function": {"name": "lookup_number", "arguments": '{"number": "7"}'}}]}, "tool smoke"),
    (1, {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                            "function": {"name": "lookup_number", "arguments": '{"number": 8}'}}]}, "tool smoke"),
    (2, {"content": ""}, "tool-result smoke"),
    (2, tool_candidate(), "tool-result smoke"),
])
def test_smoke_rejects_invalid_candidate_at_each_stage(smoke_backend, stage, candidate, match):
    responses = [{"content": "hello"}, tool_candidate(), {"content": "seven"}]
    responses[stage] = candidate
    smoke_backend.chat.side_effect = responses
    with pytest.raises(PipelineError, match=match):
        deployment._smoke(ENDPOINT, "checkpoint-7")
    assert smoke_backend.chat.call_count == stage + 1


def serving_status(status="ACTIVE"):
    return {"id": "deploy123", "model_id": "model123", "status": status}


def test_temporary_creates_cleans_up_and_resumes_same_endpoint(run, backend):
    backend.api.side_effect = [
        {"checkpoints": [CHECKPOINT]}, serving_status(), serving_status("SCALED_TO_ZERO"),
        serving_status(), {}, serving_status("INACTIVE"),
    ]
    with deployment.temporary(run, accelerator="H200:1", max_seq_len=8192, confirm=True) as value:
        assert value == (ENDPOINT, "checkpoint-7")
        assert read_receipt(run)["state"] == "ready"
    assert read_receipt(run)["state"] == "inactive"
    backend.api.reset_mock()
    backend.api.side_effect = [
        serving_status("INACTIVE"), {}, serving_status(),
        serving_status(), {}, serving_status("INACTIVE"),
    ]
    with deployment.temporary(run, confirm=True) as resumed:
        assert resumed == value
    assert read_receipt(run)["state"] == "inactive"
    backend.create.assert_called_once()
    assert [(call.args[0], call.args[1]) for call in backend.api.call_args_list] == [
        ("GET", RESOURCE), ("POST", RESOURCE + "/activate"), ("GET", RESOURCE),
        ("GET", RESOURCE), ("POST", RESOURCE + "/deactivate"), ("GET", RESOURCE),
    ]


@pytest.mark.parametrize("stage", ["smoke", "evaluation"])
def test_temporary_cleans_up_after_failure(run, backend, stage):
    backend.api.side_effect = [
        {"checkpoints": [CHECKPOINT]}, serving_status(), serving_status(),
        serving_status(), {}, serving_status("INACTIVE"),
    ]
    if stage == "smoke":
        backend.smoke.side_effect = PipelineError("smoke failed")
    with (
        pytest.raises(PipelineError, match=f"{stage} failed"),
        deployment.temporary(run, accelerator="H200:1", max_seq_len=8192, confirm=True),
    ):
        assert stage == "evaluation"
        raise PipelineError("evaluation failed")
    assert read_receipt(run)["state"] == "inactive"
    assert backend.api.call_args_list[-2].args == ("POST", RESOURCE + "/deactivate")
    assert read_receipt(run)["endpoint"] == ENDPOINT.to_dict()


def test_temporary_unknown_create_cannot_guess_cleanup_target(run, backend):
    backend.create.side_effect = PipelineError("unknown create")
    with (
        pytest.raises(PipelineError, match="unknown create"),
        deployment.temporary(run, accelerator="H200:1", max_seq_len=8192, confirm=True),
    ):
        pytest.fail("unknown deployment cannot serve evaluation")
    assert read_receipt(run)["state"] == "creation_unknown"
    assert backend.api.call_count == 1
    assert backend.api.call_args.args[0] == "GET"
    with pytest.raises(PipelineError, match="outcome is unknown"), deployment.temporary(run, confirm=True):
        pytest.fail("unknown deployment cannot resume")
    backend.create.assert_called_once()
    assert backend.api.call_count == 1


def test_temporary_cleanup_failure_keeps_recoverable_receipt(run, backend):
    backend.api.side_effect = [
        {"checkpoints": [CHECKPOINT]}, serving_status(), serving_status(),
        serving_status(), deployment.ControlError(503),
    ]
    with (
        pytest.raises(PipelineError, match="cleanup needs attention") as error,
        deployment.temporary(run, accelerator="H200:1", max_seq_len=8192, confirm=True),
    ):
        pass
    assert read_receipt(run)["state"] == "cleanup_required"
    assert read_receipt(run)["endpoint"] == ENDPOINT.to_dict()
    assert "undeploy --provider baseten" in str(error.value)


def test_temporary_plan_is_offline_and_reuses_saved_settings(run, backend):
    explicit = deployment.plan(run, accelerator="H200:1", max_seq_len=8192)
    assert explicit["checkpoint"] == IDENTITY
    assert explicit["settings"] == {"accelerator": "H200:1", "max_seq_len": 8192}
    assert explicit["serving_mode"] == "temporary"
    backend.api.assert_not_called()
    backend.prepare.assert_not_called()
    assert not (run / "endpoint.json").exists()
    deploy(run)
    backend.api.reset_mock()
    before = (run / "endpoint.json").read_bytes()
    assert deployment.plan(run) == explicit
    with pytest.raises(PipelineError, match="different checkpoint or settings"):
        deployment.plan(run, max_seq_len=4096)
    backend.api.assert_not_called()
    assert (run / "endpoint.json").read_bytes() == before


def test_failed_smoke_allows_correcting_context_cap_without_recreating(run, backend):
    backend.smoke.side_effect = PipelineError("serving context too small")
    with pytest.raises(PipelineError, match="context"):
        deploy(run)
    backend.smoke.side_effect = None
    backend.smoke.return_value = 4096
    receipt = deploy(run, max_seq_len=4096)
    assert receipt["endpoint"]["max_seq_len"] == 4096
    assert receipt["settings"]["max_seq_len"] == 4096
    assert receipt["state"] == "ready"
    backend.create.assert_called_once()

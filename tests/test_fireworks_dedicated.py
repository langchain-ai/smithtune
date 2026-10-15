"""Dedicated capacity contracts without provisioning paid resources."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from smithtune import cli
from smithtune.providers import fireworks, fireworks_dedicated as dedicated, fireworks_sampling
from smithtune.providers.base import PipelineError, TrainingOptions
from test_fireworks_sampling import Future, fake_renderer
from test_pipeline import write_manifest

SHAPE = "accounts/fireworks/trainingShapes/lora-qwen"
VERSION = SHAPE + "/versions/test-v1"
SERVING = "accounts/fireworks/deploymentShapes/lora-qwen/versions/test-v2"
MODEL = fireworks.DEFAULT_MODEL


def shape_config(**overrides):
    return {"training_shape": VERSION, "deployment_shape": SERVING,
            "trainer_mode": "LORA_TRAINER", "max_supported_context_length": MODEL.max_seq_len,
            "node_count": 1, "accelerator_count": 8, "accelerator_type": "NVIDIA_H200_141GB",
            "provisioning_timeout": 1800, "replay": False, **overrides}


@pytest.fixture
def control(monkeypatch):
    import fireworks.training.sdk as sdk

    instance = Mock(account_id="test-account")
    instance._get.return_value = httpx.Response(404, request=httpx.Request("GET", "https://example.test"))
    instance._delete.return_value = httpx.Response(200, request=httpx.Request("DELETE", "https://example.test"))
    monkeypatch.setattr(sdk, "FireworksClient", Mock(return_value=instance))
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    return instance


@pytest.mark.parametrize("override", [None, SHAPE, VERSION])
def test_shape_resolution_selects_validated_lora_and_pins_hardware(monkeypatch, override):
    import fireworks.training.sdk.trainer as trainer
    from training.utils import training_shapes

    config = shape_config()
    manager = Mock()
    manager.resolve_training_profile.return_value = SimpleNamespace(
        training_shape_version=VERSION, deployment_shape_version=SERVING,
        **{key: config[key] for key in ("trainer_mode", "max_supported_context_length", "node_count", "accelerator_type", "accelerator_count")},
    )
    select = Mock(return_value=SHAPE)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    monkeypatch.setattr(trainer, "TrainerJobManager", Mock(return_value=manager))
    monkeypatch.setattr(training_shapes, "auto_select_training_shape", select)
    result = dedicated.resolve_shape(MODEL, 4, override, timeout=600, replay=True)
    assert result == {**config, "selection": "explicit" if override else "automatic",
                      "provisioning_timeout": 600, "replay": True}
    if override:
        select.assert_not_called()
    else:
        select.assert_called_once_with(manager, base_model=MODEL.base_model, lora_rank=4, max_seq_len=MODEL.max_seq_len)
    manager.resolve_training_profile.assert_called_once_with(override or SHAPE)
    manager.close.assert_called_once()


@pytest.mark.parametrize("changes,match", [
    ({"trainer_mode": "POLICY_TRAINER"}, "LoRA-capable"),
    ({"max_supported_context_length": 1024}, "context"),
    ({"training_shape": SHAPE}, "training shape"),
    ({"deployment_shape": None}, "deployment shape"),
    ({"provisioning_timeout": float("nan")}, "timeout"),
])
def test_invalid_saved_capacity_fails_before_provisioning(changes, match):
    with pytest.raises(PipelineError, match=match):
        dedicated.validate_saved_shape(shape_config(**changes), MODEL.max_seq_len)


@pytest.mark.parametrize("options", [
    TrainingOptions(training_shape=SHAPE),
    TrainingOptions(provisioning_timeout=1800),
    TrainingOptions(training_mode="full"),
    TrainingOptions(training_mode="dedicated", lora_rank=0),
])
def test_training_options_reject_inapplicable_or_non_lora_modes(options):
    with pytest.raises(PipelineError):
        fireworks.FireworksProvider().settings_from_options(options)


def test_plan_defaults_offline_and_dedicated_shows_capacity(tmp_path, monkeypatch):
    write_manifest(tmp_path)
    resolver = Mock(return_value=shape_config())
    monkeypatch.setattr(dedicated, "resolve_shape", resolver)
    provider = fireworks.FireworksProvider()
    default = provider.plan(tmp_path, "test", fireworks.SFTSettings())
    assert default["training_api"] == fireworks.TRAINING_BASE_URL
    resolver.assert_not_called()
    plan = provider.plan(tmp_path, "test", fireworks.SFTSettings(training_mode="dedicated"))
    assert plan["training_mode"] == "dedicated"
    assert plan["dedicated"]["training_shape"] == VERSION
    assert plan["dedicated"]["accelerator_count"] == 8
    assert "time charges" in plan["cost"]
    resolver.assert_called_once_with(MODEL, MODEL.default_lora_rank, None, timeout=1800, replay=False)


@pytest.mark.parametrize("fail", [False, True])
def test_owned_capacity_deleted_even_after_partial_provisioning(tmp_path, control, fail):
    path = tmp_path / "session.json"
    with pytest.raises(RuntimeError, match="readiness failed") if fail else nullcontext(), dedicated.DedicatedResources(path) as owner:
        trainer = owner.reserve("rlorTrainerJobs")
        serving = owner.reserve("deployments")
        saved = json.loads(path.read_text())
        assert saved["resources"][0]["name"].endswith(trainer)
        if fail:
            raise RuntimeError("readiness failed")
        owner.ready()
    calls = [call.args[0] for call in control._delete.call_args_list]
    assert calls == [f"/v1/accounts/test-account/deployments/{serving}?ignoreChecks=true&hard=true",
                     f"/v1/accounts/test-account/rlorTrainerJobs/{trainer}"]
    saved = json.loads(path.read_text())
    assert saved["status"] == "closed"
    assert all(item["status"] == "delete_requested" for item in saved["resources"])


def test_cleanup_failure_persists_ids_and_still_cleans_other_resources(tmp_path, control):
    request = httpx.Request("DELETE", "https://example.test")
    control._delete.side_effect = [httpx.Response(503, request=request), httpx.Response(200, request=request)]
    path = tmp_path / "session.json"
    with pytest.raises(PipelineError, match="dedicated cleanup failed"), dedicated.DedicatedResources(path) as owner:
        owner.reserve("rlorTrainerJobs")
        owner.reserve("deployments")
    saved = json.loads(path.read_text())
    assert saved["status"] == "cleanup_failed"
    assert [row["status"] for row in saved["resources"]] == ["delete_requested", "cleanup_failed"]
    assert control._delete.call_count == 2


def test_never_claim_existing_capacity(tmp_path, control):
    control._get.return_value = httpx.Response(200, request=httpx.Request("GET", "https://example.test"))
    with pytest.raises(PipelineError, match="existing resource"), dedicated.DedicatedResources(tmp_path / "session.json") as owner:
        owner.reserve("rlorTrainerJobs")
    control._delete.assert_not_called()


def test_real_cookbook_builder_gets_exact_version_and_bounded_waits(tmp_path, monkeypatch, control):
    from fireworks.training.sdk import FiretitanServiceClient

    service = Mock()
    build = Mock(return_value=service)
    monkeypatch.setattr(FiretitanServiceClient, "from_firetitan_config", build)
    with dedicated.DedicatedResources(tmp_path / "session.json") as owner:
        result = owner.service(base_model=MODEL.base_model, tokenizer_model=MODEL.tokenizer_model,
                               rank=4, context=MODEL.max_seq_len, learning_rate=1e-4,
                               dedicated=shape_config(), replay=False)
        assert result is service
    kwargs = build.call_args.kwargs
    assert kwargs["training_shape_id"] == VERSION
    assert kwargs["max_lora_rank"] == 4
    assert kwargs["create_deployment"] is False
    assert kwargs["trainer_timeout_s"] == kwargs["trainer_pending_timeout_s"] == 1800
    service.close.assert_called_once()


@pytest.mark.parametrize("live", [False, True])
def test_replay_restores_selected_checkpoint_and_uses_separate_base_route(tmp_path, monkeypatch, control, live):
    fake_renderer(monkeypatch)
    events = []
    tuned = SimpleNamespace(close=lambda: events.append("tuned-close"))
    base = SimpleNamespace(close=lambda: events.append("base-close"))
    service = Mock(trainer_job_id="trainer", deployment_id="tuned", training_session_id=None)
    service.create_training_client.return_value = SimpleNamespace(
        resolve_checkpoint_path=lambda name, source_job_id: f"resolved/{source_job_id}/{name}",
        load_state=lambda checkpoint: events.append(("restore", checkpoint)) or Future(None),
        save_weights_for_sampler=lambda _: Future(SimpleNamespace(path="best-snapshot")),
    )
    service.create_sampling_client.side_effect = [tuned, base]
    monkeypatch.setattr(dedicated.DedicatedResources, "service", lambda *_a, **_kw: service)
    kwargs = {"service": service, "snapshot": "best-snapshot"} if live else {}
    sampler = dedicated.DedicatedReplaySampler(MODEL, "trainer:epoch-2", tmp_path, dedicated=shape_config(), **kwargs)
    with sampler:
        assert sampler._samplers["trainer:epoch-2"] is tuned
        assert sampler._samplers[MODEL.base_model] is base
    if not live:
        assert ("restore", "resolved/trainer/epoch-2") in events
    else:
        service.create_training_client.assert_not_called()
    options = service.create_inference_deployment_sampler.call_args
    deployment = options.args[0]
    assert deployment.enable_hot_load is False
    assert deployment.deployment_shape == SERVING
    assert deployment.base_model == MODEL.base_model
    calls = service.create_sampling_client.call_args_list
    assert calls[0].kwargs["model_path"] == "best-snapshot"
    assert calls[1].kwargs["deployment_sampler"] is service.create_inference_deployment_sampler.return_value
    assert "base_model" not in calls[1].kwargs  # SDK ignores this selector on managed deployments.
    assert events[-2:] == ["base-close", "tuned-close"]
    assert json.loads((tmp_path / "resources.json").read_text())["status"] == "closed"


def test_dedicated_checkpoint_selection_and_saved_replay_identity(tmp_path):
    session = object.__new__(dedicated.DedicatedTraining)
    session.job_id = "trainer"
    session.resources = SimpleNamespace(control=Mock())
    session.resources.control.list_checkpoints.return_value = [
        {"name": "accounts/a/rlorTrainerJobs/trainer/checkpoints/epoch-1", "checkpointType": "TRAINING_LORA", "createTime": "2026-10-15T01:02:13Z"},
        {"name": "accounts/a/rlorTrainerJobs/trainer/checkpoints/epoch-2", "checkpointType": "TRAINING_LORA", "createTime": "2026-10-15T01:02:13.123Z"},
        {"name": "accounts/a/rlorTrainerJobs/trainer/checkpoints/epoch-2-abcdefab", "promotable": True, "createTime": "2026-10-15T01:02:13.123Z"},
    ]
    best = {**session.epoch_checkpoints("epoch-2", state_path="epoch-2"), "job_id": "trainer"}
    assert best["resume_checkpoint"] == "trainer:epoch-2"
    write_manifest(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    plan = fireworks.FireworksProvider().plan(tmp_path, "run", fireworks.SFTSettings())
    plan.update(training_mode="dedicated", dedicated=shape_config())
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "result.json").write_text(json.dumps({"best": best}))
    model, restored = fireworks_sampling.checkpoint_from_run(tmp_path, run)
    assert restored["training_mode"] == "dedicated"
    assert restored["dedicated"]["training_shape"] == VERSION
    assert model == MODEL


def test_dedicated_promotion_uses_job_checkpoint_api(tmp_path, control):
    best = {"job_id": "trainer", "promotable_checkpoint": "accounts/test-account/rlorTrainerJobs/trainer/checkpoints/epoch-2"}
    (tmp_path / "result.json").write_text(json.dumps({"best": best}))
    (tmp_path / "plan.json").write_text(json.dumps({"base_model": MODEL.base_model, "training_mode": "dedicated"}))
    fireworks.FireworksProvider().promote(tmp_path, "tuned-model", confirm=True)
    control.promote_checkpoint.assert_called_once_with(name=best["promotable_checkpoint"], output_model_id="tuned-model", base_model=MODEL.base_model)
    control.promote_session_checkpoint.assert_not_called()


def test_dedicated_train_dispatch_keeps_best_epoch_and_pinned_shape(tmp_path, monkeypatch):
    data = tmp_path / "data"
    run = tmp_path / "run"
    write_manifest(data)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    monkeypatch.setattr(dedicated, "resolve_shape", lambda *_a, **_kw: shape_config())
    monkeypatch.setattr(fireworks, "load_training_renderer", lambda _model: None)
    monkeypatch.setattr(fireworks, "preflight_model", Mock(side_effect=AssertionError("serverless catalog must not gate dedicated")))

    class Session:
        def __init__(self, cfg, run_dir, *, dedicated, reasoning_effort):
            assert cfg.serverless is False
            assert dedicated["training_shape"] == VERSION

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run_epoch(self, epoch, _checkpoint):
            return {"job_id": "trainer", "eval_loss": [0.5, 0.8][epoch - 1], "resume_checkpoint": f"trainer:epoch-{epoch}"}

        def complete(self):
            pass

    monkeypatch.setattr(dedicated, "DedicatedTraining", Session)
    result = fireworks.FireworksProvider().train(data, run, "test", fireworks.SFTSettings(training_mode="dedicated", max_epochs=2),
                                                confirm=True, init_from_checkpoint=None)
    assert result["best"]["resume_checkpoint"] == "trainer:epoch-1"
    assert json.loads((run / "plan.json").read_text())["dedicated"]["training_shape"] == VERSION


def test_cli_dedicated_flags_available_for_plan_and_train():
    for command in ("plan", "train"):
        args = cli._parser().parse_args([command, "--data-dir", "data", "--run-id", "test", "--training-mode", "dedicated", "--training-shape", VERSION])
        assert args.training_mode == "dedicated"
        assert args.training_shape == VERSION


def test_integrated_snapshot_loads_own_job_logical_path():
    session = object.__new__(dedicated.DedicatedTraining)
    session.job_id = "trainer"
    session.client = Mock()
    session.client.resolve_checkpoint_path.return_value = "epoch-2"
    session.client.save_weights_for_sampler.return_value = SimpleNamespace(path="selected-snapshot")
    assert session.snapshot("trainer:epoch-2") == "selected-snapshot"
    session.client.resolve_checkpoint_path.assert_called_once_with("epoch-2")
    session.client.load_state.assert_called_once_with("epoch-2")


def test_saved_checkpoint_translation_matches_installed_sdk():
    from fireworks.training.sdk.client import FiretitanTrainingClient

    # resolve_checkpoint_path constructs the SDK's opaque cross-job reference
    # without a network request. The trainer later resolves it server-side.
    client = object.__new__(FiretitanTrainingClient)
    resolved = dedicated.resolve_checkpoint(client, "trainer:epoch-2")
    assert resolved == client.resolve_checkpoint_path("epoch-2", source_job_id="trainer")
    assert resolved != "trainer:epoch-2"


@pytest.mark.parametrize("provision_failure", [False, True])
@pytest.mark.parametrize("qualified_state", [False, True])
def test_dedicated_training_lifecycle_uses_lora_and_cleans_readiness_failure(tmp_path, monkeypatch, control, provision_failure, qualified_state):
    from contextlib import ExitStack
    from training import utils
    from training.utils import checkpoints, runner_state, service as service_module

    events = []
    service = Mock(trainer_job_id="trainer", accelerator_type="NVIDIA_H200_141GB", accelerator_count=8,
                   max_context_length=MODEL.max_seq_len, training_profile=None)
    service.close.side_effect = lambda: events.append("client-close")
    control._delete.side_effect = lambda *_a, **_kw: events.append("delete") or httpx.Response(
        200, request=httpx.Request("DELETE", "https://example.test"),
    )
    if provision_failure:
        service.create_training_client.side_effect = RuntimeError("not ready")
    monkeypatch.setattr(service_module, "build_service_client", lambda **_kw: service)
    monkeypatch.setattr(utils, "RunnerIO", Mock(return_value=nullcontext(Mock())))
    reconnect = Mock()
    monkeypatch.setattr(utils.ReconnectableClient, "from_training_client", Mock(return_value=reconnect))
    checkpoint_adapter = Mock()
    monkeypatch.setattr(checkpoints, "TrainingCheckpoints", Mock(return_value=checkpoint_adapter))
    monkeypatch.setattr(runner_state, "start_running", Mock())
    initial = "source-trainer:epoch-3"
    if qualified_state:
        source = object.__new__(dedicated.DedicatedTraining)
        source.job_id = "source-trainer"
        source.resources = SimpleNamespace(control=Mock())
        source.resources.control.list_checkpoints.return_value = [
            {"name": "accounts/a/rlorTrainerJobs/source-trainer/checkpoints/s-2-abcdefab", "promotable": True},
        ]
        initial = source.epoch_checkpoints("s-2", state_path=f"account/run-{'a' * 32}/s-2")["resume_checkpoint"]
    session = object.__new__(dedicated.DedicatedTraining)
    session.cfg = SimpleNamespace(base_model=MODEL.base_model, tokenizer_model=MODEL.tokenizer_model,
                                  lora_rank=4, lora_alpha=16, max_seq_len=MODEL.max_seq_len,
                                  learning_rate=1e-4, runner=None, log_path=str(tmp_path),
                                  init_from_checkpoint=initial)
    session.run_dir = tmp_path
    session._stack = ExitStack()
    session.dedicated = shape_config()
    session.total_steps = 2
    session.training_completed = False
    with pytest.raises(RuntimeError, match="not ready") if provision_failure else nullcontext(), session:
        if qualified_state:
            checkpoint_adapter.resume.assert_not_called()
            reconnect.resolve_checkpoint_path.assert_called_once_with(f"account/run-{'a' * 32}/s-2")
            reconnect.load_state_with_optimizer.assert_called_once_with(reconnect.resolve_checkpoint_path.return_value)
        else:
            checkpoint_adapter.resume.assert_called_once_with(init_from_checkpoint=initial)
    service.create_training_client.assert_called_once_with(MODEL.base_model, lora_rank=4, lora_alpha=16)
    assert events == ["client-close", "delete"]
    assert json.loads((tmp_path / "session.json").read_text())["status"] == "closed"


@pytest.mark.parametrize("command", ["eval-plan", "evaluate"])
def test_cli_replay_uses_saved_dedicated_shape(tmp_path, monkeypatch, command):
    from smithtune.evaluation import replay

    fake_renderer(monkeypatch)
    write_manifest(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    plan = fireworks.FireworksProvider().plan(tmp_path, "run", fireworks.SFTSettings())
    plan.update(training_mode="dedicated", dedicated=shape_config())
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "result.json").write_text(json.dumps({"best": {"resume_checkpoint": "trainer:epoch-2"}}))
    monkeypatch.setattr(replay, "validate_judge_credentials", lambda *_: None)
    monkeypatch.setattr(replay, "prepare_replay_evaluation", lambda *_: {})
    monkeypatch.setattr(replay, "training_metadata", lambda *_: None)
    monkeypatch.setattr(replay, "run_replay_evaluation", lambda *_a, **kwargs: kwargs["replay_sampler"].config)
    args = cli._parser().parse_args([command, "--provider", "fireworks", "--data-dir", str(tmp_path),
                                    "--run-dir", str(run), "--output-dir", str(tmp_path / "evaluation"),
                                    *(["--confirm"] if command == "evaluate" else [])])
    value = cli._run_fireworks_evaluation(args)
    assert value["serving_mode"] == "dedicated"
    assert (value["training_shape"] if command == "evaluate" else value["dedicated"]["training_shape"]) == VERSION


def test_replay_failure_deletes_partially_created_base_deployment(tmp_path, monkeypatch, control):
    fake_renderer(monkeypatch)
    tuned = Mock()
    service = Mock(trainer_job_id="trainer", deployment_id="tuned")
    service.create_sampling_client.return_value = tuned
    service.create_inference_deployment_sampler.side_effect = RuntimeError("base readiness failed")
    sampler = dedicated.DedicatedReplaySampler(MODEL, "trainer:epoch-2", tmp_path,
                                               dedicated=shape_config(), service=service, snapshot="best")
    with pytest.raises(RuntimeError, match="base readiness failed"), sampler:
        pass
    tuned.close.assert_called_once()
    service.close.assert_not_called()  # The integrated trainer remains owned by its outer context.
    control._delete.assert_called_once()
    assert "/deployments/" in control._delete.call_args.args[0]
    assert json.loads((tmp_path / "resources.json").read_text())["status"] == "closed"


@pytest.mark.parametrize("checkpoint", ["trainer:epoch//2", "trainer:epoch-2/..", "account/run-abc/epoch-2", "trainer:"])
def test_invalid_dedicated_resume_rejected_before_capacity(tmp_path, monkeypatch, checkpoint):
    resolver = Mock(side_effect=AssertionError("must validate resume before shape discovery"))
    monkeypatch.setattr(dedicated, "resolve_shape", resolver)
    with pytest.raises(PipelineError):
        fireworks.FireworksProvider().train(tmp_path, tmp_path / "run", "test",
                                           fireworks.SFTSettings(training_mode="dedicated"),
                                           confirm=True, init_from_checkpoint=checkpoint)
    resolver.assert_not_called()

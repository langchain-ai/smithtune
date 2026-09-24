"""Deployment composes checkpoint promotion without repeating successful writes."""
import json

import pytest

from firectl_fakes import FakeFirectl, shape
from smithtune.providers import fireworks
from smithtune.providers.base import PipelineError


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    checkpoint = "accounts/account-id/trainingSessions/job-2/checkpoints/best"
    (tmp_path / "result.json").write_text(json.dumps({"best": {
        "job_id": "job-2", "promotable_checkpoint": checkpoint,
    }}))
    (tmp_path / "plan.json").write_text(json.dumps({"base_model": "accounts/fireworks/models/base"}))
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    events = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def promote_session_checkpoint(self, *args):
            events.append(("promote", args))

        def close(self):
            events.append(("close",))

    import fireworks.training.sdk as sdk
    monkeypatch.setattr(sdk, "FireworksClient", Client)
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events))
    monkeypatch.setattr(fireworks, "firectl_version", lambda: (1, 8, 9))
    monkeypatch.setattr(fireworks.time, "sleep", lambda _: None)
    monkeypatch.setattr(fireworks, "_inference_smoke_test", lambda route: {"status": "passed"})
    return tmp_path, events


def deploy(directory, **kwargs):
    return fireworks.FireworksProvider().deploy(
        directory, "account-id", "model-id", "endpoint-id", "shape", confirm=True, **kwargs,
    )


def test_deploy_promotes_best_checkpoint_before_creating_endpoint(deployment):
    directory, events = deployment
    endpoint = deploy(directory)
    assert [event[0] for event in events] == ["promote", "close", "deploy"]
    assert events[0][1] == (
        "accounts/account-id/trainingSessions/job-2/checkpoints/best",
        "model-id", "accounts/fireworks/models/base",
    )
    assert events[2][1][3] == "accounts/account-id/models/model-id"
    receipt = json.loads((directory / "promotion.json").read_text())
    assert receipt["checkpoint"] == events[0][1][0]
    assert receipt["base_model"] == events[0][1][2]
    assert json.loads((directory / "endpoint.json").read_text()) == endpoint


@pytest.mark.parametrize("legacy", [False, True])
def test_deploy_reuses_standalone_promotion(deployment, legacy):
    directory, events = deployment
    fireworks.FireworksProvider().promote(directory, "model-id", confirm=True)
    if legacy:
        receipt = json.loads((directory / "promotion.json").read_text())
        receipt.pop("base_model")
        (directory / "promotion.json").write_text(json.dumps(receipt))
    saved = (directory / "promotion.json").read_bytes()
    events.clear()
    deploy(directory)
    assert [event[0] for event in events] == ["deploy"]
    assert (directory / "promotion.json").read_bytes() == saved


def test_failed_deployment_creation_reuses_saved_promotion(deployment, monkeypatch):
    directory, events = deployment

    firectl = FakeFirectl()

    def create(command, **kwargs):
        if command[1:3] == ["deployment", "create"]:
            assert (directory / "promotion.json").exists()
            events.append(("deploy", command))
            if sum(event[0] == "deploy" for event in events) == 1:
                raise PipelineError("deployment creation failed")
        return firectl(command, **kwargs)

    monkeypatch.setattr(fireworks, "_run", create)
    with pytest.raises(PipelineError, match="deployment creation failed"):
        deploy(directory)
    assert not (directory / "endpoint.json").exists()
    deploy(directory)
    assert [event[0] for event in events] == ["promote", "close", "deploy", "deploy"]


def test_promotion_failure_stops_deployment(deployment, monkeypatch):
    directory, events = deployment
    import fireworks.training.sdk as sdk

    def fail(*args):
        raise PipelineError("promotion failed")

    monkeypatch.setattr(sdk.FireworksClient, "promote_session_checkpoint", fail)
    with pytest.raises(PipelineError, match="promotion failed"):
        deploy(directory)
    assert events == [("close",)]
    assert not (directory / "promotion.json").exists()
    assert not (directory / "endpoint.json").exists()


@pytest.mark.parametrize("field", ["checkpoint", "job_id", "base_model"])
def test_conflicting_saved_promotion_stops_before_external_calls(deployment, field):
    directory, events = deployment
    fireworks.FireworksProvider().promote(directory, "model-id", confirm=True)
    receipt = json.loads((directory / "promotion.json").read_text())
    receipt[field] = "another-value"
    (directory / "promotion.json").write_text(json.dumps(receipt))
    events.clear()
    with pytest.raises(PipelineError, match="saved promotion does not match"):
        deploy(directory)
    assert events == []


def test_wrong_account_stops_before_promotion(deployment):
    directory, events = deployment
    with pytest.raises(PipelineError, match="account owning"):
        fireworks.FireworksProvider().deploy(
            directory, "another-account", "model-id", "endpoint-id", "shape", confirm=True,
        )
    assert events == []


@pytest.mark.parametrize("legacy", [False, True])
def test_standalone_promotion_is_reusable_and_can_register_another_name(deployment, legacy):
    directory, events = deployment
    provider = fireworks.FireworksProvider()
    provider.promote(directory, "model-id", confirm=True)
    provider.promote(directory, "model-id", confirm=True)
    assert [event[0] for event in events] == ["promote", "close"]
    if legacy:
        receipt = json.loads((directory / "promotion.json").read_text())
        receipt.pop("base_model")
        (directory / "promotion.json").write_text(json.dumps(receipt))
    provider.promote(directory, "another-model", confirm=True)
    assert [event[0] for event in events] == ["promote", "close", "promote", "close"]
    receipt = json.loads((directory / "promotion.json").read_text())
    assert receipt["output_model_id"] == "another-model"
    assert receipt["previous_promotions"][0]["output_model_id"] == "model-id"
    events.clear()
    deploy(directory)
    assert [event[0] for event in events] == ["deploy"]


def deploy_matched(directory, **kwargs):
    return fireworks.FireworksProvider().deploy(directory, "account-id", "model-id", "endpoint-id", confirm=True, **kwargs)


def test_deploy_matches_a_validated_shape_after_promotion(deployment, monkeypatch):
    directory, events = deployment
    firectl = FakeFirectl(events=events, shapes=[
        shape("Stale shape", "NVIDIA_B200_180GB", latest=False),
        shape("Qwen 1x H100", "NVIDIA_H100_80GB"),
        shape("Qwen 1x B200", "NVIDIA_B200_180GB"),
    ])
    monkeypatch.setattr(fireworks, "_run", firectl)
    endpoint = deploy_matched(directory)
    match = next(c for c in firectl.commands if c[1:3] == ["deployment-shape-version", "match"])
    assert match[-4:] == ["--model", "accounts/account-id/models/model-id", "-o", "json"]
    # The model is promoted before matching, and the first validated shape is used.
    assert [e[0] for e in events] == ["promote", "close", "deploy"]
    create = events[-1][1]
    assert create[create.index("--deployment-shape") + 1] == "accounts/fireworks/deploymentShapes/qwen-1x-h100"
    assert endpoint["deployment_shape"]["source"] == "matched"
    assert endpoint["deployment_shape"]["alternatives"] == ["Qwen 1x B200"]


def test_deploy_waits_for_a_ready_replica_not_just_ready_state(deployment, monkeypatch):
    directory, events = deployment
    firectl = FakeFirectl(events=events, ready_after=2)
    monkeypatch.setattr(fireworks, "_run", firectl)
    smoke = []
    monkeypatch.setattr(fireworks, "_inference_smoke_test", lambda route: smoke.append(firectl.gets) or {"status": "passed"})
    deploy(directory)
    assert smoke == [3]  # smoke test only after a replica is ready


def test_deploy_times_out_waiting_for_capacity_with_cleanup_command(deployment, monkeypatch):
    directory, events = deployment
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events, ready_after=10**9))
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(fireworks.time, "monotonic", lambda: next(clock))
    with pytest.raises(PipelineError, match="no ready replica.*smithtune undeploy --provider fireworks"):
        deploy(directory, timeout=300)
    assert json.loads((directory / "endpoint.json").read_text())["smoke_test"] == {"status": "pending"}


def test_old_firectl_fails_before_promotion_when_matching_is_needed(deployment, monkeypatch):
    directory, events = deployment
    monkeypatch.setattr(fireworks, "firectl_version", lambda: (1, 8, 0))
    with pytest.raises(PipelineError, match="firectl 1.8.5 or newer .*found 1.8.0"):
        deploy_matched(directory)
    assert events == [] and not (directory / "promotion.json").exists()
    deploy(directory)  # an explicit shape still works on old firectl


def test_deploy_preview_shows_the_shape_without_creating_resources(deployment, monkeypatch):
    directory, events = deployment
    firectl = FakeFirectl(events=events)
    monkeypatch.setattr(fireworks, "_run", firectl)
    preview = fireworks.FireworksProvider().deploy(directory, "account-id", "model-id", "endpoint-id", confirm=False)
    assert preview["status"] == "preview" and preview["promotion"] == "pending"
    assert preview["deployment_shape"]["source"] == "matched_after_promotion"
    deploy_matched(directory)
    events.clear()
    firectl.commands.clear()
    preview = fireworks.FireworksProvider().deploy(directory, "account-id", "model-id", "endpoint-id", confirm=False)
    assert preview["promotion"] == "saved" and preview["deployment_shape"]["display_name"] == "Base 1x H100"
    assert events == [] and not any(c[1:3] == ["deployment", "create"] for c in firectl.commands)


def test_agent_block_hands_the_exact_deploy_command_to_the_user(deployment, monkeypatch):
    directory, events = deployment
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events, block_agents=True))
    with pytest.raises(PipelineError) as failure:
        deploy_matched(directory)
    message = str(failure.value)
    assert "cannot create the deployment here" in message and "outside the agent" in message
    # The handoff pins the shape already matched, so it also works on firectl < 1.8.5.
    assert ("smithtune deploy --provider fireworks --run-dir " + str(directory) + " --account-id account-id "
            "--output-model-id model-id --deployment-id endpoint-id --deployment-shape "
            "accounts/fireworks/deploymentShapes/base-1x-h100 --confirm") in message
    assert (directory / "promotion.json").exists() and not (directory / "endpoint.json").exists()


def test_handoff_command_succeeds_outside_the_agent(deployment, monkeypatch):
    directory, events = deployment
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events, block_agents=True))
    with pytest.raises(PipelineError):
        deploy_matched(directory)
    monkeypatch.setattr(fireworks, "firectl_version", lambda: (1, 8, 0))
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events))
    endpoint = fireworks.FireworksProvider().deploy(
        directory, "account-id", "model-id", "endpoint-id", "accounts/fireworks/deploymentShapes/base-1x-h100", confirm=True)
    assert endpoint["deployment_shape"] == {"name": "accounts/fireworks/deploymentShapes/base-1x-h100", "source": "explicit"}
    assert [e[0] for e in events].count("promote") == 1  # the saved promotion is reused


def test_agent_block_hands_undeploy_to_the_user(monkeypatch):
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(block_agents=True))
    with pytest.raises(PipelineError, match="smithtune undeploy --provider fireworks --account-id account-id "
                                            "--deployment-id endpoint-id --confirm"):
        fireworks.FireworksProvider().undeploy("account-id", "endpoint-id", confirm=True)


def test_other_firectl_failures_are_summarized_without_the_raw_command(deployment, monkeypatch):
    directory, events = deployment
    monkeypatch.setattr(fireworks, "_run", FakeFirectl(events=events, fail="Error: quota exceeded for NVIDIA_H100_80GB"))
    with pytest.raises(PipelineError, match="could not create the deployment: Error: quota exceeded") as failure:
        deploy(directory)
    assert "['firectl'" not in str(failure.value)

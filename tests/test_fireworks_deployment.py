"""Deployment composes checkpoint promotion without repeating successful writes."""
import json

import pytest

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
    monkeypatch.setattr(fireworks, "_run", lambda command: events.append(("deploy", command)))
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

    def create(command):
        assert (directory / "promotion.json").exists()
        events.append(("deploy", command))
        if sum(event[0] == "deploy" for event in events) == 1:
            raise PipelineError("deployment creation failed")

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


def test_standalone_promotion_is_reusable_and_can_register_another_name(deployment):
    directory, events = deployment
    provider = fireworks.FireworksProvider()
    provider.promote(directory, "model-id", confirm=True)
    provider.promote(directory, "model-id", confirm=True)
    assert [event[0] for event in events] == ["promote", "close"]
    provider.promote(directory, "another-model", confirm=True)
    assert [event[0] for event in events] == ["promote", "close", "promote", "close"]
    assert json.loads((directory / "promotion.json").read_text())["output_model_id"] == "another-model"

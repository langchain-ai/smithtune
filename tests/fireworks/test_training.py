"""Offline optimizer boundaries and exact checkpoint selection for Fireworks."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smithtune.providers.fireworks import provider as fireworks
from smithtune.providers.fireworks import dedicated as fireworks_dedicated
from smithtune.providers.fireworks import training as fireworks_training
from smithtune.providers.base import PipelineError


RUN = "run-0123456789abcdef0123456789abcdef"


def checkpoint_rows(name, *, run_id=None, timestamp="2026-10-15T01:00:00Z"):
    prefix = f"{run_id}-" if run_id else ""
    base = f"accounts/account/checkpoints/{prefix}{name}"
    return [
        {"name": base, "checkpointType": "TRAINING_LORA", "createTime": timestamp},
        {"name": f"{base}-abcdef01", "promotable": True, "createTime": timestamp},
    ]


def fake_session(tmp_path, monkeypatch, *, dedicated=False, interval=2, steps_per_epoch=3,
                 losses=(0.5,), failure=None, fail_step=4):
    """Use the real epoch loop and resolver with futures at the SDK boundary."""
    from training.recipes import sft_loop
    from training.utils import runner_state

    session_class = fireworks_dedicated.DedicatedTraining if dedicated else fireworks_training.ServerlessTraining
    session = object.__new__(session_class)
    session.cfg = SimpleNamespace(seed=42, batch_size=1, pipeline_depth=4, learning_rate=1e-4)
    session.run_dir = tmp_path
    session.generator = Mock()
    session.runner = Mock()
    session.step = session.consumed = 0
    session.total_steps = steps_per_epoch * 10
    session.history = []
    session.validations = []
    session.validation_every_steps = interval
    session.job_id = "trainer"
    session.training_run_id = RUN
    datum = SimpleNamespace(model_input=SimpleNamespace(to_ints=lambda: [1, 2, 3]),
                            loss_fn_inputs={"weights": SimpleNamespace(data=[0, 1, 1])})
    session.loader = [[datum] for _ in range(steps_per_epoch)]
    session.validation = [datum]
    rows = []
    events = []
    completed = set()
    losses = iter(losses)
    submitted = 0

    class Future:
        def __init__(self, step, kind):
            self.step, self.kind = step, kind

        def result(self, **_kwargs):
            events.append((self.kind, self.step))
            if self.kind == "optimizer":
                if failure == "optimizer" and self.step == fail_step:
                    raise RuntimeError("optimizer failed")
                completed.add(self.step)
            return SimpleNamespace(metrics={"loss:sum": 2.0})

    def submit_forward(_batch, **_kwargs):
        nonlocal submitted
        # The next update cannot enter the pipeline until its predecessor's
        # validation checkpoint and loss are durably associated.
        if submitted and (submitted % steps_per_epoch == 0 or (interval is not None and submitted % interval == 0)):
            saved = json.loads((tmp_path / "validations.json").read_text())
            assert saved[-1]["step"] == submitted
        submitted += 1
        events.append(("submit", submitted))
        return Future(submitted, "forward")

    session.client = SimpleNamespace(submit_forward_backward=submit_forward,
                                     submit_optim_step=lambda _adam: Future(submitted, "optimizer"))
    from fireworks.training.sdk.client import FiretitanTrainingClient
    validator = object.__new__(FiretitanTrainingClient)
    session.client.resolve_checkpoint_path = validator.resolve_checkpoint_path
    session.client.save_state = lambda name: SimpleNamespace(path=name if dedicated else f"account/{RUN}/{name}")
    session.client.save_weights_for_sampler = lambda name, **_kwargs: SimpleNamespace(path=f"{name}-abcdef01")
    session.checkpoint_client = fireworks_training._CheckpointSaveClient(session.client)

    def evaluate(_validation, _client, _batch_size, step, _epoch):
        assert completed == set(range(1, step + 1))
        events.append(("eval", step))
        if failure == "eval" and step == fail_step:
            raise RuntimeError("validation failed")
        return next(losses)

    def save(name, **kwargs):
        assert completed == set(range(1, session.step + 1))
        assert events[-1] == ("eval", session.step)
        assert kwargs == {"resumable": True, "promotable": True, "data_consumed": session.step}
        events.append(("save", session.step))
        if failure == "save" and session.step == fail_step:
            raise RuntimeError("save failed")
        session.checkpoint_client.save_state(name)
        session.checkpoint_client.save_weights_for_sampler(name)
        if failure == "identity" and session.step == fail_step:
            return
        created = checkpoint_rows(name, run_id=None if dedicated else RUN)
        assert max(len(row["name"].rsplit("/", 1)[-1]) for row in created) <= 63
        rows.extend(created)

    session.checkpoints = SimpleNamespace(save=save)
    session.resources = SimpleNamespace(control=SimpleNamespace(list_checkpoints=lambda _job_id: list(rows)))
    monkeypatch.setattr(fireworks, "_session_checkpoint_rows", lambda _job_id: ("account", list(rows)))
    monkeypatch.setattr(sft_loop, "run_eval", evaluate)
    monkeypatch.setattr(runner_state, "write_running_step", lambda *_args, **_kwargs: None)
    return session, events


@pytest.mark.parametrize("dedicated", [False, True])
def test_global_optimizer_intervals_and_epoch_end_are_deduplicated(tmp_path, monkeypatch, dedicated):
    session, events = fake_session(tmp_path, monkeypatch, dedicated=dedicated, losses=[0.5] * 6)
    result = fireworks.run_early_stopping(fireworks.SFTSettings(max_epochs=3), session.run_epoch)

    validations = json.loads((tmp_path / "validations.json").read_text())
    assert [(row["epoch"], row["step"]) for row in validations] == [(1, 2), (1, 3), (2, 4), (2, 6), (3, 8), (3, 9)]
    assert [step for kind, step in events if kind == "eval"] == [2, 3, 4, 6, 8, 9]
    assert [step for kind, step in events if kind == "save"] == [2, 3, 4, 6, 8, 9]
    assert [row["step"] for row in result["epochs"]] == [3, 6, 9]
    assert result["validations"] == validations
    for row in validations:
        assert row["step"] == row["steps"]
        assert row["resume_checkpoint"].endswith(f"s-{row['step']}")
        assert row["promotable_checkpoint"].endswith(f"s-{row['step']}-abcdef01")
    assert [row["step"] for row in json.loads((tmp_path / "epochs.json").read_text())] == [3, 6, 9]


@pytest.mark.parametrize("interval", [None, 20])
def test_default_or_long_interval_validates_final_update_of_each_epoch(tmp_path, monkeypatch, interval):
    session, events = fake_session(tmp_path, monkeypatch, interval=interval, losses=[0.8, 0.5])
    result = fireworks.run_early_stopping(fireworks.SFTSettings(max_epochs=2), session.run_epoch)
    assert [step for kind, step in events if kind == "eval"] == [3, 6]
    assert result["best"]["step"] == 6
    expected_name = "epoch-2" if interval is None else "s-6"
    assert result["best"]["resume_checkpoint"].endswith(expected_name)
    assert (tmp_path / "validations.json").exists()


def test_pipeline_drains_before_eval_and_save_and_waits_for_checkpoint(tmp_path, monkeypatch):
    session, events = fake_session(tmp_path, monkeypatch, steps_per_epoch=5, losses=[0.9, 0.8, 0.7])
    session.run_epoch(1, None)
    assert events == [
        ("submit", 1), ("submit", 2), ("forward", 1), ("optimizer", 1), ("forward", 2), ("optimizer", 2),
        ("eval", 2), ("save", 2),
        ("submit", 3), ("submit", 4), ("forward", 3), ("optimizer", 3), ("forward", 4), ("optimizer", 4),
        ("eval", 4), ("save", 4),
        ("submit", 5), ("forward", 5), ("optimizer", 5), ("eval", 5), ("save", 5),
    ]


@pytest.mark.parametrize("dedicated", [False, True])
@pytest.mark.parametrize("interval", [None, 2])
def test_default_completes_requested_epochs_despite_worse_validation(tmp_path, monkeypatch, dedicated, interval):
    losses = [0.1, 0.8, 0.5, 0.9, 0.7, 1.0] if interval is not None else [0.1, 0.9, 1.0]
    session, _ = fake_session(tmp_path, monkeypatch, dedicated=dedicated, interval=interval, losses=losses)
    result = fireworks.run_early_stopping(
        fireworks.SFTSettings(max_epochs=3, early_stopping_min_delta=100.0), session.run_epoch,
    )
    assert result["stopped_early"] is False
    assert [row["step"] for row in result["epochs"]] == [3, 6, 9]
    assert result["best"]["epoch"] == 1
    assert result["best"]["step"] == (2 if interval is not None else 3)
    assert result["validations"][-1]["step"] == 9


@pytest.mark.parametrize("dedicated", [False, True])
def test_mid_epoch_best_survives_endpoint_and_patience_remains_in_epochs(tmp_path, monkeypatch, dedicated):
    session, _ = fake_session(tmp_path, monkeypatch, dedicated=dedicated, losses=[0.1, 0.8, 0.05, 0.9])
    result = fireworks.run_early_stopping(
        fireworks.SFTSettings(max_epochs=5, early_stopping_patience=1), session.run_epoch,
    )
    assert result["stopped_early"] is True
    assert len(result["epochs"]) == 2
    assert result["best"]["epoch"] == 2
    assert result["best"]["step"] == 4
    assert result["best"]["eval_loss"] == 0.05
    assert result["best"]["resume_checkpoint"].endswith("s-4")
    assert result["best"]["promotable_checkpoint"].endswith("s-4-abcdef01")


@pytest.mark.parametrize("failure,error", [("eval", "validation failed"), ("save", "save failed"), ("identity", "matching promotable")])
def test_later_failure_preserves_completed_validations_and_submits_no_next_update(tmp_path, monkeypatch, failure, error):
    session, events = fake_session(tmp_path, monkeypatch, steps_per_epoch=5, losses=[0.5, 0.4], failure=failure)
    with pytest.raises((RuntimeError, PipelineError), match=error):
        session.run_epoch(1, None)
    saved = json.loads((tmp_path / "validations.json").read_text())
    assert [row["step"] for row in saved] == [2]
    assert saved[0]["resume_checkpoint"].endswith("s-2")
    assert max(step for kind, step in events if kind == "submit") == 4
    assert not (tmp_path / "epochs.json").exists()


def test_optimizer_failure_never_validates_incomplete_state(tmp_path, monkeypatch):
    session, events = fake_session(tmp_path, monkeypatch, failure="optimizer", fail_step=2)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        session.run_epoch(1, None)
    assert not any(kind in ("eval", "save") for kind, _ in events)
    assert not (tmp_path / "validations.json").exists()


@pytest.mark.parametrize("dedicated", [False, True])
def test_exact_saved_identity_beats_timestamps_and_other_runs(tmp_path, monkeypatch, dedicated):
    run_id = None if dedicated else RUN
    old = checkpoint_rows("s-2", run_id=run_id, timestamp="2026-10-15T09:00:00Z")
    current = checkpoint_rows("s-4", run_id=run_id, timestamp="2026-10-15T01:00:00Z")
    unrelated = checkpoint_rows("s-5", run_id=run_id, timestamp="2026-10-15T10:00:00Z")
    rows = [*old, *unrelated, *current]
    if not dedicated:
        rows.extend(checkpoint_rows("s-4", run_id=f"run-{'f' * 32}", timestamp="2026-10-15T12:00:00Z"))
    if dedicated:
        session = object.__new__(fireworks_dedicated.DedicatedTraining)
        session.job_id = "trainer"
        session.resources = SimpleNamespace(control=SimpleNamespace(list_checkpoints=lambda _job: rows))
        result = session.epoch_checkpoints("s-4", state_path="s-4")
        assert result["resume_checkpoint"] == "trainer:s-4"
    else:
        monkeypatch.setattr(fireworks, "_session_checkpoint_rows", lambda _job: ("account", rows))
        result = fireworks._epoch_checkpoints("session", "s-4", run_id=RUN, state_path=f"account/{RUN}/s-4")
        assert result["resume_checkpoint"] == f"account/{RUN}/s-4"
    assert result["promotable_checkpoint"] == current[1]["name"]


def test_plan_and_options_use_trajectory_optimizer_schedule(tmp_path):
    from smithtune.providers.base import TrainingOptions
    from tests.test_pipeline import write_manifest

    write_manifest(tmp_path)
    provider = fireworks.FireworksProvider()
    settings = provider.settings_from_options(TrainingOptions(batch_size=3, max_epochs=3, validation_every_steps=2))
    assert settings.validation_every_steps == 2
    plan = provider.plan(tmp_path, "test", settings)
    assert plan["training_schedule"] == {
        "steps_per_epoch": 3,  # Eight trajectory rows, three per optimizer batch.
        "max_optimizer_steps": 9,
        "expected_validation_passes": 6,
        "step_unit": "optimizer_updates",
        "early_stopping_patience_unit": "epochs",
        "training_units": 8,
        "training_unit": "trajectories",
        "estimate": False,
    }
    assert plan["config"]["validation_every_steps"] == 2


@pytest.mark.parametrize("dedicated", [False, True])
def test_train_replay_and_promotion_use_the_selected_mid_epoch_identity(tmp_path, monkeypatch, dedicated):
    import fireworks.training.sdk as sdk
    from smithtune.evaluation import replay
    from smithtune.providers.fireworks import sampling as fireworks_sampling
    from tests.test_pipeline import write_manifest

    data, run = tmp_path / "data", tmp_path / "run"
    write_manifest(data)
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-value")
    monkeypatch.setattr(fireworks, "preflight_model", lambda _model: None)
    monkeypatch.setattr(fireworks, "load_training_renderer", lambda _model: None)
    monkeypatch.setattr(fireworks, "_load_jsonl", lambda _path: [])
    monkeypatch.setattr(fireworks_dedicated, "resolve_shape", lambda *_args, **_kwargs: {"training_shape": "pinned"})
    for name in ("validate_judge_credentials", "preflight_langsmith", "ensure_judge_calibration"):
        monkeypatch.setattr(replay, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(replay, "prepare_replay_evaluation", lambda *_args, **_kwargs: {})
    session, _ = fake_session(run, monkeypatch, dedicated=dedicated, losses=[0.1, 0.8, 0.4, 0.9])
    session.service = object()
    session.complete = Mock()
    session.snapshot = Mock(return_value="selected-snapshot")

    @contextmanager
    def session_factory(_config, run_dir, **kwargs):
        assert run_dir == run
        assert kwargs["validation_every_steps"] == 2
        if dedicated:
            assert kwargs["dedicated"] == {"training_shape": "pinned"}
        yield session

    if dedicated:
        monkeypatch.setattr(fireworks_dedicated, "DedicatedTraining", session_factory)
        sampler = Mock()
        monkeypatch.setattr(fireworks_dedicated, "DedicatedReplaySampler", sampler)
    else:
        monkeypatch.setattr(fireworks_training, "ServerlessTraining", session_factory)
        sampler = Mock()
        monkeypatch.setattr(fireworks_sampling, "FireworksReplaySampler", sampler)
    evaluate = Mock(return_value={"completed": True})
    monkeypatch.setattr(replay, "run_replay_evaluation", evaluate)
    provider = fireworks.FireworksProvider()
    result = provider.train(
        data, run, "test", fireworks.SFTSettings(max_epochs=2, validation_every_steps=2,
                                                training_mode="dedicated" if dedicated else "serverless"),
        confirm=True, init_from_checkpoint=None,
        replay={"judge_model": "judge", "concurrency": 1, "max_points_per_trajectory": None, "max_output_tokens": 32},
    )
    best = result["best"]
    assert (best["epoch"], best["step"]) == (1, 2)
    assert best["resume_checkpoint"].endswith("s-2")
    session.snapshot.assert_called_once_with(best["resume_checkpoint"])
    assert sampler.call_args.args[1] == best["resume_checkpoint"]
    assert sampler.call_args.kwargs["snapshot"] == "selected-snapshot"
    assert evaluate.call_args.args[2] == best["resume_checkpoint"]
    assert evaluate.call_args.kwargs["training"]["checkpoint_epoch"] == 1
    assert evaluate.call_args.kwargs["training"]["checkpoint_step"] == 2

    control = Mock()
    monkeypatch.setattr(sdk, "FireworksClient", Mock(return_value=control))
    provider.promote(run, "mid-epoch-best", confirm=True)
    if dedicated:
        control.promote_checkpoint.assert_called_once_with(name=best["promotable_checkpoint"],
                                                          output_model_id="mid-epoch-best", base_model=fireworks.DEFAULT_MODEL.base_model)
    else:
        control.promote_session_checkpoint.assert_called_once_with(best["promotable_checkpoint"], "mid-epoch-best", fireworks.DEFAULT_MODEL.base_model)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), True])
def test_invalid_validation_loss_never_publishes_checkpoint(tmp_path, monkeypatch, loss):
    session, events = fake_session(tmp_path, monkeypatch, losses=[loss])
    with pytest.raises(PipelineError, match="finite loss"):
        session.run_epoch(1, None)
    assert not any(kind == "save" for kind, _ in events)
    assert not (tmp_path / "validations.json").exists()


@pytest.mark.parametrize("dedicated", [False, True])
def test_cookbook_reused_metadata_alias_keeps_exact_save_responses_and_earlier_best(tmp_path, monkeypatch, dedicated):
    from training.utils.checkpoints import TrainingCheckpoints
    from smithtune.artifacts import _utc_now

    session, events = fake_session(tmp_path, monkeypatch, dedicated=dedicated, losses=[0.1, 0.8, 0.4, 0.9])
    rows, states, snapshots = [], {}, {}
    run_id = None if dedicated else RUN

    def save_state(name):
        # The service repeatedly consolidates control-plane metadata to one
        # alias while its RPC returns the exact requested state identity.
        alias = checkpoint_rows("step-0", run_id=run_id, timestamp=_utc_now().replace("+00:00", "Z"))[0]
        rows[:] = [row for row in rows if row["name"] != alias["name"]]
        rows.append(alias)
        path = name if dedicated else f"account/{RUN}/{name}"
        states[path] = session.step
        return SimpleNamespace(path=path)

    def save_sampler(name, **_kwargs):
        row = checkpoint_rows(name, run_id=run_id, timestamp=_utc_now().replace("+00:00", "Z"))[1]
        rows.append(row)
        path = f"{name}-abcdef01" if dedicated else f"account/{RUN}/{name}-abcdef01"
        snapshots[path] = session.step
        return SimpleNamespace(path=path)

    session.client.save_state = save_state
    session.client.save_weights_for_sampler = save_sampler
    session.resources.control.list_checkpoints = lambda _job: list(rows)
    monkeypatch.setattr(fireworks, "_session_checkpoint_rows", lambda _job: ("account", list(rows)))
    session.checkpoints = TrainingCheckpoints(
        session.checkpoint_client, session.resources.control, trainer_id=session.job_id, log_path=str(tmp_path),
        serverless=not dedicated, current_run_id=run_id,
        save_appear_timeout_s=0.1, save_stabilize_s=0, save_poll_s=0,
    )
    result = fireworks.run_early_stopping(fireworks.SFTSettings(max_epochs=2), session.run_epoch)
    assert [step for kind, step in events if kind == "eval"] == [2, 3, 4, 6]
    assert len({row["resume_checkpoint"] for row in result["validations"]}) == 4
    assert len([row for row in rows if row.get("checkpointType") == "TRAINING_LORA"]) == 1
    best = result["best"]
    assert best["step"] == 2
    state_path = best["resume_checkpoint"].split(":", 1)[1] if dedicated else best["resume_checkpoint"]
    assert states[state_path] == 2
    assert snapshots[best["sampler_snapshot"]] == 2
    session.client.load_state = Mock(side_effect=AssertionError("best replay must use its saved sampler"))
    assert session.snapshot(best["resume_checkpoint"]) == best["sampler_snapshot"]
    session.client.load_state.assert_not_called()
    if dedicated:
        assert fireworks_dedicated.resolve_checkpoint(session.client, best["resume_checkpoint"]) == "cross_job://trainer/s-2"
    else:
        assert session.client.resolve_checkpoint_path(best["resume_checkpoint"]) == f"account/{RUN}/s-2"
    saved = json.loads((tmp_path / "validations.json").read_text())
    assert saved == result["validations"]


@pytest.mark.parametrize("missing", ["state", "sampler"])
def test_incomplete_save_response_cannot_reuse_previous_validation_identity(tmp_path, monkeypatch, missing):
    session, events = fake_session(tmp_path, monkeypatch, steps_per_epoch=5, losses=[0.1, 0.8])
    complete_save = session.checkpoints.save

    def save(name, **kwargs):
        if session.step != 4:
            return complete_save(name, **kwargs)
        if missing != "state":
            session.checkpoint_client.save_state(name)
        if missing != "sampler":
            session.checkpoint_client.save_weights_for_sampler(name)

    session.checkpoints.save = save
    with pytest.raises(PipelineError, match="both training and sampler"):
        session.run_epoch(1, None)
    saved = json.loads((tmp_path / "validations.json").read_text())
    assert [row["step"] for row in saved] == [2]
    assert saved[0]["sampler_snapshot"] == "s-2-abcdef01"
    assert max(step for kind, step in events if kind == "submit") == 4


@pytest.mark.parametrize("path", [None, "tinker://unsupported/weights", "step//4"])
def test_invalid_state_response_fails_before_next_optimizer_update(tmp_path, monkeypatch, path):
    session, events = fake_session(tmp_path, monkeypatch)
    session.client.save_state = lambda _name: SimpleNamespace(path=path)
    with pytest.raises(PipelineError, match="checkpoint path"):
        session.run_epoch(1, None)
    assert max(step for kind, step in events if kind == "submit") == 2
    assert not (tmp_path / "validations.json").exists()


def test_reused_state_response_is_reported_instead_of_claiming_immutable_continuation(tmp_path, monkeypatch):
    session, _ = fake_session(tmp_path, monkeypatch, steps_per_epoch=5, losses=[0.1, 0.8])
    session.client.save_state = lambda _name: SimpleNamespace(path=f"account/{RUN}/step-0")
    with pytest.raises(PipelineError, match="reused a save_state response path"):
        session.run_epoch(1, None)
    saved = json.loads((tmp_path / "validations.json").read_text())
    assert [row["step"] for row in saved] == [2]
    assert session.snapshot(saved[0]["resume_checkpoint"]) == "s-2-abcdef01"


def test_dedicated_returned_fully_qualified_path_stays_loadable():
    from fireworks.training.sdk.client import FiretitanTrainingClient

    client = object.__new__(FiretitanTrainingClient)
    checkpoint = f"trainer:account/{RUN}/s-2"
    assert fireworks_dedicated.resolve_checkpoint(client, checkpoint) == f"account/{RUN}/s-2"

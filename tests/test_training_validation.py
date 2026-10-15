"""Training schedule and checkpoint provenance contracts."""

import json

import pytest

from smithtune import cli
from smithtune.evaluation.replay import training_metadata
from smithtune.providers.base import CommonSFTSettings, PipelineError
from smithtune.providers.validation import training_schedule


@pytest.mark.parametrize("units,batch,epochs,interval,steps,passes", [
    (320, 32, 3, None, 10, 3),
    (320, 32, 3, 4, 10, 9),
    (96, 32, 3, 4, 3, 5),
    (96, 32, 3, 3, 3, 3),
    (96, 32, 3, 100, 3, 3),
    (1, 32, 3, 1, 1, 3),
    (33, 32, 1, 1, 2, 2),
    (0, 32, 3, 1, 0, 0),
])
def test_schedule_counts_global_intervals_and_epoch_ends_once(
    units, batch, epochs, interval, steps, passes,
):
    plan = training_schedule(units, batch, epochs, interval)
    assert plan["steps_per_epoch"] == steps
    assert plan["max_optimizer_steps"] == steps * epochs
    assert plan["expected_validation_passes"] == passes
    assert plan["step_unit"] == "optimizer_updates"
    assert plan["early_stopping_patience_unit"] == "epochs"


@pytest.mark.parametrize("field", ["validation_every_steps", "early_stopping_patience"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_optional_training_intervals_reject_non_positive_integers(field, value):
    with pytest.raises(PipelineError, match=f"{field}.*positive integer"):
        CommonSFTSettings(**{field: value}).validate()


@pytest.mark.parametrize("provider", ["fireworks", "baseten"])
@pytest.mark.parametrize("command", ["plan", "train"])
def test_validation_option_resolves_for_training_and_planning(provider, command):
    parser = cli._parser()
    args = parser.parse_args([command, "--provider", provider, "--validation-every-steps", "13"])
    assert cli._settings_from_args(args).validation_every_steps == 13
    defaults = parser.parse_args([command, "--provider", provider])
    assert cli._settings_from_args(defaults).validation_every_steps is None
    assert cli._settings_from_args(defaults).early_stopping_patience is None
    opted_in = parser.parse_args([command, "--provider", provider, "--early-stopping-patience", "2"])
    assert cli._settings_from_args(opted_in).early_stopping_patience == 2
    with pytest.raises(PipelineError, match="validation_every_steps"):
        cli._settings_from_args(parser.parse_args([
            command, "--provider", provider, "--validation-every-steps", "0",
        ]))


@pytest.mark.parametrize("result,expected", [
    ({"best": {"epoch": 2, "step": 7}}, {"checkpoint_epoch": 2, "checkpoint_step": 7}),
    ({"best_epoch": 2, "best_step": 7}, {"checkpoint_epoch": 2, "checkpoint_step": 7}),
    ({"best": {"epoch": 2, "steps": 7}}, {"checkpoint_epoch": 2}),
    ({"best_epoch": 2}, {"checkpoint_epoch": 2}),
    ({}, {}),
])
def test_replay_records_selected_checkpoint_step_when_available(tmp_path, result, expected):
    (tmp_path / "plan.json").write_text(json.dumps({"run_id": "training-run"}))
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert training_metadata(tmp_path) == {"parent_training_run_id": "training-run", **expected}

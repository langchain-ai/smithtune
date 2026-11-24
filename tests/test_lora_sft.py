from smithtune.artifacts import _jsonl_dump
from smithtune.providers import lora_sft


def test_prepared_rows_keep_unicode_line_separators_inside_strings(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "first second third\x85end"}]}, {"messages": []}]
    path = tmp_path / "train.jsonl"
    _jsonl_dump(path, rows)

    assert lora_sft.read_jsonl(path) == rows


class _Trainer:
    def __init__(self, losses):
        self.losses = iter(losses)
        self.samplers = []

    def forward_backward(self, microbatch, *, group_active_tokens):
        return 1.0, 1.0, {}

    def forward(self, microbatch):
        return next(self.losses), 1.0, {}

    def optim_step(self, learning_rate):
        return {}

    def save_state(self, name):
        return f"state://{name}"

    def save_weights_for_sampler(self, name):
        self.samplers.append(name)
        return f"sampler://{name}"


class _Stop(Exception):
    pass


class _Guard:
    """Stop only at the named operation."""

    def __init__(self, stop_at=None):
        self.stop_at = stop_at

    def check(self, operation):
        if operation == self.stop_at:
            raise _Stop(operation)


def _loop(tmp_path, losses, **options):
    from types import SimpleNamespace

    from smithtune.providers.base import PipelineError

    datum = SimpleNamespace(model_input=SimpleNamespace(to_ints=lambda: [1, 2]), loss_fn_inputs={})
    settings = SimpleNamespace(max_epochs=len(losses), seed=0, microbatch_token_budget=8, batch_size=1,
                               learning_rate=1e-4, validation_every_steps=None, early_stopping_patience=None,
                               early_stopping_min_delta=0.0)
    splits = lora_sft.RenderedSplits(train=[[datum]], validation_batches=[[datum]], dropped_train_rows=[],
                                     prepared_train_rows=1)
    errors = lora_sft.ProviderErrors("Test", PipelineError, RuntimeError)
    trainer = _Trainer(losses)
    loop = lora_sft.SFTLoop(trainer=trainer, settings=settings, run_id="run", splits=splits, run_dir=tmp_path,
                            update_state=lambda **_: None, errors=errors, **options)
    return loop, trainer


def test_sampler_weights_are_saved_only_for_new_bests_by_default(tmp_path):
    loop, _ = _loop(tmp_path, [1.0, 1.5, 0.5])
    loop.run()
    assert [row["sampler_weights_uri"] for row in loop.validations] == [
        "sampler://run-sampler-epoch-1", None, "sampler://run-sampler-epoch-3"]
    assert loop.best_sampler_uri == "sampler://run-sampler-epoch-3"


def test_opting_in_saves_sampler_weights_at_every_validation_without_changing_the_best(tmp_path):
    loop, _ = _loop(tmp_path, [1.0, 1.5], save_every_sampler=True)
    loop.run()
    assert [row["sampler_weights_uri"] for row in loop.validations] == [
        "sampler://run-sampler-epoch-1", "sampler://run-sampler-epoch-2"]
    assert [row["new_best"] for row in loop.validations] == [True, False]
    assert loop.best_epoch == 1 and loop.best_sampler_uri == "sampler://run-sampler-epoch-1"


def test_a_guard_stop_before_an_optional_sampler_save_still_records_the_validation(tmp_path):
    import pytest

    loop, trainer = _loop(tmp_path, [1.0, 1.5], save_every_sampler=True,
                          guard=_Guard("validation sampler checkpoint"), deferred=(_Stop,))
    with pytest.raises(_Stop):
        loop.run()
    assert [row["sampler_weights_uri"] for row in loop.validations] == ["sampler://run-sampler-epoch-1", None]
    assert trainer.samplers == ["run-sampler-epoch-1"]

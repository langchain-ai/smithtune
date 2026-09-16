"""Unit tests opt into LangSmith I/O explicitly and supply a fake client."""

from functools import wraps

import pytest

from smithtune import dataset, evaluation
from smithtune.providers import fireworks


@pytest.fixture(autouse=True)
def local_langsmith_defaults(monkeypatch, request):
    if request.node.get_closest_marker("sdk_integration"):
        return
    prepare = dataset.prepare_dataset
    replay = evaluation.run_replay_evaluation

    @wraps(prepare)
    def local_prepare(*args, **kwargs):
        kwargs["sync_splits"] = False
        return prepare(*args, **kwargs)

    @wraps(replay)
    def local_replay(*args, **kwargs):
        kwargs["publish"] = False
        return replay(*args, **kwargs)

    def local_unlocked(*args, **kwargs):
        kwargs["publish"] = False
        return replay.__wrapped__(*args, **kwargs)

    local_replay.__wrapped__ = local_unlocked
    monkeypatch.setattr(dataset, "prepare_dataset", local_prepare)
    monkeypatch.setattr(fireworks, "prepare_dataset", local_prepare)
    monkeypatch.setattr(evaluation, "run_replay_evaluation", local_replay)
    monkeypatch.setattr(evaluation, "preflight_langsmith", lambda *_args, **_kwargs: None)

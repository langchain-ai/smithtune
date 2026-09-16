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

    @wraps(prepare)
    def local_prepare(*args, **kwargs):
        kwargs["sync_splits"] = False
        return prepare(*args, **kwargs)

    monkeypatch.setattr(dataset, "prepare_dataset", local_prepare)
    monkeypatch.setattr(fireworks, "prepare_dataset", local_prepare)
    monkeypatch.setattr(evaluation, "preflight_langsmith", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluation.reporting, "bind_evaluation_snapshot", lambda *_args: None)
    monkeypatch.setattr(evaluation.reporting, "publish_evaluation", lambda *_args: {"experiments": {}})

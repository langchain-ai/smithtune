"""Unit tests opt into LangSmith I/O explicitly and supply a fake client."""

from functools import wraps

import pytest

from smithtune import curation, data_rights, dataset, triage
from smithtune.artifacts import _json_dump
from smithtune.evaluation import replay as evaluation
from smithtune.providers import fireworks


@pytest.fixture(autouse=True)
def local_data_rights_acknowledgment(monkeypatch, tmp_path_factory):
    """Existing workflow tests act as an acknowledged user, never using real state."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config")))
    _json_dump(data_rights._receipt_path(), {
        "document_version": data_rights.DOCUMENT_VERSION,
        "document_url": data_rights.DOCUMENT_URL,
        "read_at": "2026-09-20T00:00:00+00:00",
    })


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
    monkeypatch.setattr(evaluation.reporting, "publish_evaluation", lambda *_args: {"comparison_url": "https://smith.langchain.com/test-comparison"})

    class LocalPublisher:
        def __init__(self, *args):
            pass

        def submit(self, results):
            pass

        def close(self):
            return {"comparison_url": "https://smith.langchain.com/test-comparison"}

    monkeypatch.setattr(evaluation.reporting, "BackgroundPublisher", LocalPublisher)


@pytest.fixture(autouse=True)
def immediate_trajectory_retries(monkeypatch):
    """Fake trajectories are indexed at once; tests asserting delays patch this themselves."""
    monkeypatch.setattr(curation, "_sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def default_selection_rule(monkeypatch, request):
    """Council review requires criteria; tests that do not test criteria share one rule."""
    if request.node.get_closest_marker("no_default_rule"):
        return
    load = triage.load_config

    def with_rule(path):
        config = load(path)
        return config if path is not None or config["rules"] else {**config, "rules": ["Keep trajectories that complete the request."]}

    monkeypatch.setattr(triage, "load_config", with_rule)

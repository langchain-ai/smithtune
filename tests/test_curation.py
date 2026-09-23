from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace
from uuid import UUID

import pytest

from smithtune import artifacts
from smithtune import curation
from smithtune.providers.base import PipelineError
from trajectory_fixtures import items


def uid(number):
    return str(UUID(int=number))


def root(number, thread=None):
    return {"id": uid(number), "trace_id": uid(number), "thread_id": thread,
            "start_time": "2026-09-02T12:00:00Z",
            "feedback_stats": {"correctness": {"avg": 0.95, "n": 1}}}


def thread(identity):
    return {"key": "thread_id", "id": identity}


def trace(number):
    return {"key": "trace_id", "id": uid(number)}


COMMON_METADATA = {"trajectory_format": "messages", "conversation_scope": "root",
                   "source_project_id": uid(101), "source_workspace_id": uid(100)}


class API:
    def __init__(self, pages=None):
        self.pages = pages or [[]]
        self.calls = []
        self.contract_calls = []
        self.examples = []
        self.messages = [
            {"role": "system", "content": "Be precise."},
            {"role": "human", "content": "What is 17 × 6?"},
            {"role": "ai", "content": "112."},
            {"role": "human", "content": "Please check."},
            {"role": "ai", "content": "102."},
        ]
        self.failure = None
        self.datasets = {}

    def __call__(self, command, *, capture=False, input=None):
        assert capture
        assert command[:2] == ["langsmith", "api"]
        assert command[command.index("--workspace") + 1] == uid(100)
        path = command[2]
        body = json.loads(input) if input is not None else json.loads(command[command.index("--body") + 1]) if "--body" in command else None
        if path.startswith("/api/v1/sessions/"):
            return SimpleNamespace(stdout=json.dumps({"id": uid(101), "start_time": "2026-09-01T00:00:00+00:00"}))
        self.calls.append((path, copy.deepcopy(body)))
        if self.failure:
            self.failure(path, body)
        if path == "/api/v2/runs/query":
            if body.get("filter", "").startswith("eq(thread_id,"):
                thread_id = json.loads(body["filter"][len("eq(thread_id,"):-1])
                result = {"items": [r for page in self.pages for r in page if r["thread_id"] == thread_id], "next_cursor": None}
            else:
                page = int(body.get("cursor", 0))
                result = {"items": self.pages[page], "next_cursor": str(page + 1) if page + 1 < len(self.pages) else None}
        elif path == "/api/v1/datasets":
            result = copy.deepcopy(body)
            self.datasets[body["id"]] = result
        elif path == "/v1/trajectory":
            assert body["format"] == "ui" and body["include"] == {"system_messages": True}
            assert sum(key in body for key in curation.TRAJECTORY_KEYS) == 1
            result = {"items": items(self.messages, trace_id=body.get("trace_id", uid(1))), "next_cursor": None, "prev_cursor": None}
        elif path == "/api/v1/examples":
            result = {"id": body["id"], "dataset_id": body["dataset_id"], "inputs": body["inputs"],
                      "outputs": body["outputs"], "metadata": body["metadata"]}
            self.examples.append(result)
            result = {"id": result["id"]}
        elif path.endswith("/versions?limit=1"):
            result = [{"as_of": "2026-09-15T00:00:00+00:00"}] if self.examples else []
        elif path.startswith("/api/v1/datasets/"):
            result = self.datasets.get(path.rsplit("/", 1)[1])
            if result is None:
                raise PipelineError("HTTP 404")
        elif path.startswith("/api/v1/examples?"):
            from urllib.parse import parse_qs, urlsplit
            query = parse_qs(urlsplit(path).query)
            offset = int(query["offset"][0])
            result = self.examples[offset:offset + 100]
        elif path.startswith("/api/v1/examples/"):
            result = next((ex for ex in self.examples if ex["id"] == path.rsplit("/", 1)[1]), None)
            if result is None:
                raise PipelineError("HTTP 404")
        else:
            raise AssertionError(path)
        return SimpleNamespace(stdout=json.dumps(result))

    def trajectory_calls(self):
        return [{key: body[key] for key in curation.TRAJECTORY_KEYS if key in body}
                for path, body in self.calls if path == "/v1/trajectory"]


def test_source_time_window_defaults_to_the_day_ending_at_the_resolved_end(monkeypatch):
    monkeypatch.setattr(curation, "_utc_now", lambda: "2026-09-08T12:34:56+00:00")
    assert curation.resolve_time_window(None, None) == (
        "2026-09-07T12:34:56+00:00", "2026-09-08T12:34:56+00:00",
    )
    assert curation.resolve_time_window(None, "2026-09-02T05:00:00-07:00") == (
        "2026-09-01T12:00:00+00:00", "2026-09-02T12:00:00+00:00",
    )
    assert curation.resolve_time_window("2026-09-08T00:00:00Z", None) == (
        "2026-09-08T00:00:00+00:00", "2026-09-08T12:34:56+00:00",
    )


def test_api_uses_stdin_and_run_passes_it_to_subprocess(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"ok": true}')
    monkeypatch.setattr(artifacts.subprocess, "run", run)
    body = {"project_id": uid(101), "thread_id": "private-thread", "format": "messages"}
    assert curation._api(uid(100), "POST", "/v1/trajectory", body) == {"ok": True}
    command, kwargs = calls[0]
    assert command[-2:] == ["--input", "-"]
    assert "private" not in " ".join(command)
    assert json.loads(kwargs["input"]) == body
    assert kwargs["check"] and kwargs["capture_output"]


def test_trajectory_fetch_gives_up_after_three_attempts(tmp_path, monkeypatch):
    api = API([[root(1, "a")]])
    monkeypatch.setattr(curation, "_sleep", lambda seconds: None)
    def always_fail(path, body):
        if path == "/v1/trajectory":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 503")
    api.failure = always_fail
    with pytest.raises(PipelineError, match="HTTP 503 after 3 attempt"):
        curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=api)
    assert sum(path == "/v1/trajectory" for path, _ in api.calls) == 3
    assert not any(path == "/api/v1/examples" for path, _ in api.calls)


@pytest.mark.parametrize("recover", [True, False])
def test_later_page_retries_only_that_page_and_never_writes_partial_example(tmp_path, monkeypatch, recover):
    api = API([[root(1, "a")]])
    sleeps, cursors = [], []
    monkeypatch.setattr(curation, "_sleep", sleeps.append)

    def runner(command, **kwargs):
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        body = json.loads(kwargs["input"])
        cursor = body.get("cursor")
        cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(stdout=json.dumps({"items": items(api.messages[:2]), "next_cursor": "second"}))
        if not recover or cursors.count("second") < 3:
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 503")
        return SimpleNamespace(stdout=json.dumps({"items": items(api.messages[2:]), "next_cursor": None, "prev_cursor": "first"}))

    if recover:
        result = curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=runner)
        assert result["messages"] == api.messages
    else:
        with pytest.raises(PipelineError, match="HTTP 503 after 3 attempt"):
            curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=runner)
        assert api.examples == []
    assert cursors == [None, "second", "second", "second"]
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("oversized_cursor", [None, "second"])
@pytest.mark.parametrize("stream", ["output", "stderr"])
@pytest.mark.parametrize("recover", [True, False])
def test_oversized_page_narrows_same_cursor_without_partial_import(
    tmp_path, monkeypatch, oversized_cursor, stream, recover,
):
    api = API([[root(1, "a")]])
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a value.",
            "parameters": {"type": "object", "properties": {}}}}
    entries = items(api.messages, trace_id=uid(1), run_id=uid(1001))
    entries[4]["message"]["available_tools"] = [tool]
    entries[4]["metadata"] = {"run_id": uid(1002), "trace_id": uid(2)}
    calls, sleeps = [], []
    monkeypatch.setattr(curation, "_sleep", sleeps.append)
    error = json.dumps({"status": 400, "detail":
        "Narrow the requested trajectory page and retry: "
        "trajectory view exceeded response data size limit: private source content"}) + "\nHTTP 400"

    def runner(command, **kwargs):
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        body = json.loads(kwargs["input"])
        calls.append(body)
        assert body["include"] == {"system_messages": True}
        assert body["format"] == "ui"
        assert body["thread_id"] == "a" and body["project_id"] == uid(101)
        assert api.examples == []
        cursor = body.get("cursor")
        if cursor == oversized_cursor and (not recover or body.get("page_size") != 1):
            raise subprocess.CalledProcessError(1, command, **{stream: error})
        page = {"items": entries[:3], "next_cursor": "second"} if cursor is None else {
            "items": entries[3:], "next_cursor": None}
        return SimpleNamespace(stdout=json.dumps(page))

    if recover:
        result = curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=runner)
        assert result["messages"] == api.messages
        assert result["source"] == {
            "schema_version": 1, "assistant_runs": [
                {"message_index": 2, "run_id": uid(1001), "trace_id": uid(1), "tools": []},
                {"message_index": 4, "run_id": uid(1002), "trace_id": uid(2), "tools": [tool]},
            ]}
        assert calls[-1]["page_size"] == 1
    else:
        result = curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=runner)
        assert result["messages"] == [] and result["source"] is None
        assert "trajectory fetch limit" in result["training_error"]
        assert "private source content" not in result["training_error"]
    failed = [body for body in calls if body.get("cursor") == oversized_cursor]
    assert len(failed) == 2
    assert failed[1] == {**failed[0], "page_size": 1}
    assert sleeps == []


@pytest.mark.parametrize("error", [
    "HTTP 400: unrelated private response",
    "HTTP 503: Narrow the requested trajectory page; trajectory view exceeded response data size limit",
])
def test_unrelated_errors_do_not_narrow_trajectory_pages(monkeypatch, error):
    calls = []
    monkeypatch.setattr(curation, "_sleep", lambda _: None)

    def runner(command, **kwargs):
        calls.append(json.loads(kwargs["input"]))
        raise subprocess.CalledProcessError(1, command, output=error)

    with pytest.raises(PipelineError, match=r"HTTP [45]\d\d after 3 attempt") as caught:
        curation._fetch_trajectory(uid(100), uid(101), {"key": "thread_id", "id": "a"}, runner=runner)
    assert "private" not in str(caught.value)
    assert len(calls) == 3 and all("page_size" not in body for body in calls)


@pytest.mark.parametrize("cursor", ["", 123, [], {}, "second"])
def test_invalid_or_repeated_later_cursor_never_writes_partial_example(tmp_path, cursor):
    api = API([[root(1, "a")]])
    calls = []

    def runner(command, **kwargs):
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        calls.append(json.loads(kwargs["input"]).get("cursor"))
        return SimpleNamespace(stdout=json.dumps({"items": items(api.messages), "next_cursor": "second" if len(calls) == 1 else cursor}))

    with pytest.raises(PipelineError, match="invalid or repeated continuation cursor"):
        curation._fetch_trajectory(uid(100), uid(101), thread("a"), runner=runner)
    assert calls == [None, "second"]
    assert api.examples == []

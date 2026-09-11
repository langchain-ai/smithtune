from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
from urllib.parse import parse_qs, urlparse

import pytest

import artifacts
import curation
import dataset
import pipeline
from providers.base import PipelineError
from providers.fireworks import DEFAULT_MODEL


def uid(number):
    return str(UUID(int=number))


def root(number, thread=None):
    return {"id": uid(number), "trace_id": uid(number), "thread_id": thread,
            "start_time": "2026-09-02T12:00:00Z",
            "feedback_stats": {"correctness": {"avg": 0.95, "n": 1}}}


class API:
    def __init__(self, pages=None):
        self.pages = pages or [[]]
        self.calls = []
        self.examples = []
        self.messages = [
            {"role": "system", "content": "Be precise."},
            {"role": "human", "content": "What is 17 × 6?"},
            {"role": "ai", "content": "112."},
            {"role": "human", "content": "Please check."},
            {"role": "ai", "content": "102."},
        ]
        self.failure = None
        self.next_trajectory_cursor = None

    def __call__(self, command, *, capture=False, input=None):
        assert capture
        assert command[:2] == ["langsmith", "api"]
        assert command[command.index("--workspace") + 1] == uid(100)
        path = command[2]
        body = json.loads(input) if input is not None else None
        self.calls.append((path, copy.deepcopy(body)))
        if self.failure:
            self.failure(path, body)
        if path == "/api/v2/runs/query":
            page = int(body.get("cursor", 0))
            result = {"items": self.pages[page], "next_cursor": str(page + 1) if page + 1 < len(self.pages) else None}
        elif path == "/api/v1/datasets":
            result = {"id": uid(200)}
        elif path == "/v1/trajectory":
            result = {"messages": self.messages, "next_cursor": self.next_trajectory_cursor, "prev_cursor": None}
        elif path == "/api/v1/examples":
            result = {"id": uid(300 + len(self.examples)), **copy.deepcopy(body)}
            self.examples.append(result)
        elif path.endswith("/thread-imports"):
            result = {"id": uid(300 + len(self.examples)),
                      "inputs": {"messages": copy.deepcopy(self.messages)}, "outputs": None,
                      "source_thread_id": body["thread_ids"][0], "metadata": body["metadata"]}
            self.examples.append(result)
            result = {"count": 1, "example_ids": [result["id"]]}
        else:
            raise AssertionError(path)
        return SimpleNamespace(stdout=json.dumps(result))


def select(tmp_path, api, **overrides):
    return curation.select_dataset(**{
        "workspace_id": uid(100), "project_id": uid(101),
        "start_time": "2026-09-01T00:00:00Z", "end_time": "2026-09-08T00:00:00Z",
        "scope": "thread", "output": tmp_path / "selection.json", "runner": api,
        **overrides,
    })


def test_selection_pages_deduplicates_then_samples(tmp_path):
    roots = [root(1, "a"), root(2, "a"), root(3, "b"), root(4)]
    api = API([roots[:2], roots[2:]])
    result = select(tmp_path, api, limit=2, filter='has(tags, "reviewed")')
    saved = json.loads(Path(result["selection"]).read_text())
    assert saved["selected_ids"] == ["a", "b"]
    assert (result["matching_roots"], result["distinct_threads"], result["excluded_unthreaded_roots"]) == (4, 2, 1)
    assert result["selected_examples"] == result["eligible_examples"] == 2
    assert result["preview"][0]["feedback_stats"] == roots[0]["feedback_stats"]
    assert result["preview"][-1]["selected"] is False
    assert api.calls[0][1]["is_root"] is True
    assert api.calls[0][1]["page_size"] == 100
    assert api.calls[0][1]["filter"] == 'and(has(tags, "reviewed"), lt(start_time, "2026-09-08T00:00:00+00:00"))'
    assert api.calls[1][1]["cursor"] == "1"
    assert "messages" not in json.dumps(saved)


@pytest.mark.parametrize("scope", ["trace", "thread"])
def test_seeded_sample_does_not_depend_on_pages(tmp_path, scope):
    roots = [root(i, f"thread-{i // 2}") for i in range(1, 21)]
    first = select(tmp_path / "first", API([roots[:8], roots[8:]]), scope=scope, limit=4, seed=17)
    second = select(tmp_path / "second", API([list(reversed(roots[8:])), list(reversed(roots[:8]))]), scope=scope, limit=4, seed=17)
    assert json.loads(Path(first["selection"]).read_text())["selected_ids"] == json.loads(Path(second["selection"]).read_text())["selected_ids"]


def test_no_limit_keeps_all_traces_and_deduplicates_pages(tmp_path):
    result = select(tmp_path, API([[root(1, "a")], [root(1, "a"), root(2, "a"), root(3)]]), scope="trace")
    assert result["selected_examples"] == 3
    assert result["excluded_unthreaded_roots"] == 0


@pytest.mark.parametrize("overrides", [{"limit": 0}, {"limit": -1}, {"scope": "all"},
    {"start_time": "2026-09-01"}, {"end_time": "2026-08-01T00:00:00Z"}, {"project_id": "bad"}])
def test_invalid_selection_arguments_do_not_query(tmp_path, overrides):
    api = API()
    with pytest.raises(PipelineError):
        select(tmp_path, api, **overrides)
    assert api.calls == []


def test_empty_selection_and_overwrite_fail_before_writes(tmp_path):
    api = API()
    select(tmp_path, api)
    with pytest.raises(PipelineError, match="already exists"):
        select(tmp_path, api)
    with pytest.raises(PipelineError, match="empty selections"):
        curation.create_dataset(selection=tmp_path / "selection.json", name="new", runner=api)
    assert len(api.calls) == 1
    assert not (tmp_path / "selection.import.json").exists()


@pytest.mark.parametrize("page", [{"items": None}, {"items": [root(1)], "next_cursor": "repeat"}])
def test_malformed_or_repeated_pages_are_rejected(tmp_path, page):
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps(page))
    with pytest.raises(PipelineError, match="page|cursor"):
        select(tmp_path, runner)
    assert not (tmp_path / "selection.json").exists()


@pytest.mark.parametrize("scope", ["trace", "thread"])
@pytest.mark.parametrize("system_prompt", [None, "System one", "Different system"])
def test_create_preserves_messages_and_uses_saved_ids(tmp_path, scope, system_prompt):
    api = API([[root(1, "a")]])
    api.messages = [m for m in api.messages if m["role"] != "system"]
    if system_prompt:
        api.messages.insert(0, {"role": "system", "content": system_prompt})
    select(tmp_path, api, scope=scope)
    api.calls.clear()
    result = curation.create_dataset(selection=tmp_path / "selection.json", name="new", runner=api)
    assert result["example_count"] == 1
    assert api.examples[0]["inputs"]["messages"] == api.messages
    assert api.examples[0]["outputs"] is None
    assert api.examples[0]["metadata"]["trajectory_format"] == "messages"
    assert not any(path.endswith("runs/query") for path, _ in api.calls)
    if scope == "thread":
        assert len(api.calls) == 2
        assert api.calls[-1][1]["thread_ids"] == ["a"]
        assert api.examples[0]["source_thread_id"] == "a"
    else:
        assert api.calls[1][1]["include"] == {"system_messages": True}
        assert api.examples[0]["metadata"]["source_trace_id"] == uid(1)
        assert api.examples[0]["metadata"]["source_thread_id"] == "a"
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete"
    assert receipt["pending_write"] is None
    assert "messages" not in receipt
    assert "What is 17" not in json.dumps(receipt)
    calls = len(api.calls)
    with pytest.raises(PipelineError, match="new writable path"):
        curation.create_dataset(selection=tmp_path / "selection.json", name="again", runner=api)
    assert len(api.calls) == calls


def test_partial_failure_has_receipt_and_never_retries(tmp_path):
    api = API([[root(1, "a"), root(2, "b")]])
    select(tmp_path, api)
    def fail(path, body):
        if path.endswith("thread-imports") and body["thread_ids"] == ["b"]:
            raise subprocess.CalledProcessError(1, "langsmith", stderr="request timed out; private message")
    api.failure = fail
    with pytest.raises(PipelineError, match="confirmed=1") as error:
        curation.create_dataset(selection=tmp_path / "selection.json", name="new", runner=api)
    assert "private message" not in str(error.value)
    assert "outcome may be unknown" in str(error.value)
    assert sum(path.endswith("thread-imports") for path, _ in api.calls) == 2
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["current_source_id"] == "b"
    assert receipt["dataset_id"] == uid(200)
    assert receipt["example_ids"] == [uid(300)]


@pytest.mark.parametrize("cursor,empty", [("unexpected", False), (None, True)])
def test_bad_trajectory_never_creates_example(tmp_path, cursor, empty):
    api = API([[root(1)]])
    api.next_trajectory_cursor = cursor
    if empty:
        api.messages = []
    select(tmp_path, api, scope="trace")
    with pytest.raises(PipelineError, match="no messages|continuation"):
        curation.create_dataset(selection=tmp_path / "selection.json", name="new", runner=api)
    assert not api.examples
    assert json.loads((tmp_path / "selection.import.json").read_text())["pending_write"] is None


def test_existing_dataset_name_and_unknown_dataset_write(tmp_path):
    api = API([[root(1)]])
    select(tmp_path, api, scope="trace")
    def fail(path, body):
        raise subprocess.CalledProcessError(1, "langsmith", stderr="Error: HTTP 409")
    api.failure = fail
    with pytest.raises(PipelineError, match="dataset name already exists"):
        curation.create_dataset(selection=tmp_path / "selection.json", name="existing", runner=api)
    assert len(api.calls) == 2
    assert json.loads((tmp_path / "selection.import.json").read_text())["dataset_name"] == "existing"


def test_saved_selection_rejects_unknown_ids(tmp_path):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    path = tmp_path / "selection.json"
    saved = json.loads(path.read_text())
    saved["selected_ids"] = ["not-a-match"]
    path.write_text(json.dumps(saved))
    with pytest.raises(PipelineError, match="belong"):
        curation.create_dataset(selection=path, name="new", runner=api)
    assert len(api.calls) == 1


def test_api_uses_stdin_and_run_passes_it_to_subprocess(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='{"ok": true}')
    monkeypatch.setattr(artifacts.subprocess, "run", run)
    assert curation._api(uid(100), "POST", "/api/v1/examples", {"messages": ["private"]}) == {"ok": True}
    command, kwargs = calls[0]
    assert command[-2:] == ["--input", "-"]
    assert "private" not in " ".join(command)
    assert json.loads(kwargs["input"]) == {"messages": ["private"]}
    assert kwargs["check"] and kwargs["capture_output"]


def test_parser_defaults_and_dispatch(tmp_path, monkeypatch, capsys):
    args = ["dataset", "select", "--workspace-id", uid(100), "--project-id", uid(101),
            "--start-time", "2026-09-01T00:00:00Z", "--end-time", "2026-09-08T00:00:00Z",
            "--scope", "trace", "--output", str(tmp_path / "selection.json")]
    parsed = pipeline._parser().parse_args(args)
    assert parsed.seed == 42 and parsed.limit is None
    calls = []
    monkeypatch.setattr(curation, "select_dataset", lambda **kwargs: calls.append(kwargs) or {"selected_examples": 3})
    monkeypatch.setattr(sys, "argv", ["pipeline.py", *args])
    pipeline.main()
    assert calls[0]["scope"] == "trace"
    assert json.loads(capsys.readouterr().out)["selected_examples"] == 3
    monkeypatch.setattr(curation, "create_dataset", lambda **kwargs: calls.append(kwargs) or {"dataset_id": uid(200)})
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "dataset", "create", "--selection", "data/selection.json", "--name", "new"])
    pipeline.main()
    assert calls[-1] == {"selection": Path("data/selection.json"), "name": "new"}
    assert json.loads(capsys.readouterr().out)["dataset_id"] == uid(200)
    with pytest.raises(SystemExit):
        pipeline._parser().parse_args(args[:args.index("--scope")] + args[args.index("--output"):])


@pytest.mark.parametrize("scope", ["trace", "thread"])
def test_select_create_download_prepare(tmp_path, scope):
    api = API([[root(i, f"thread-{i}" if scope == "thread" else None) for i in range(1, 11)]])
    def source_messages(path, body):
        if path == "/v1/trajectory" or path.endswith("thread-imports"):
            identity = body.get("trace_id") or body["thread_ids"][0]
            api.messages = [{"role": "human", "content": f"Question {identity}"},
                            {"role": "ai", "content": f"Answer {identity}"}]
    api.failure = source_messages
    select(tmp_path, api, scope=scope)
    imported = curation.create_dataset(selection=tmp_path / "selection.json", name="new", runner=api)
    def download(command, *, capture=False):
        if command[1:3] == ["dataset", "get"]:
            return SimpleNamespace(stdout=json.dumps({"id": uid(200), "name": "new", "example_count": len(api.examples)}))
        if command[1:3] == ["dataset", "export"]:
            Path(command[4]).write_text(json.dumps([{"inputs": ex["inputs"], "outputs": ex["outputs"]} for ex in api.examples]))
            return SimpleNamespace(stdout="")
        assert command[1] == "api" and command[2].startswith("/api/v1/examples?")
        query = parse_qs(urlparse(command[2]).query)
        assert query["dataset"] == [uid(200)]
        offset, limit = int(query["offset"][0]), int(query["limit"][0])
        return SimpleNamespace(stdout=json.dumps(api.examples[offset:offset + limit]))
    data_dir = tmp_path / "data"
    dataset.download_dataset(uid(100), imported["dataset_id"], data_dir / "raw", runner=download)
    manifest = dataset.prepare_dataset(uid(100), imported["dataset_id"], DEFAULT_MODEL, data_dir,
                                       fetch=False, check_render=False)
    assert [manifest["split"][key] for key in ("train", "validation", "test")] == [8, 1, 1]
    rows = [json.loads(line) for line in (data_dir / "prepared" / "train.jsonl").read_text().splitlines()]
    # Provider JSONL strips provenance; the preserved raw examples keep it.
    assert all(row["messages"][0]["role"] == "user" for row in rows)
    source_rows = dataset.prepare_sft_rows(api.examples)
    assert all(row["_source"][f"source_{scope}_id"] for row in source_rows)

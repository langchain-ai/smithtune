from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
from urllib.parse import parse_qs, urlparse

import pytest

from smithtune import artifacts
from smithtune import curation
from smithtune import dataset
from smithtune.dataset_artifacts import load_conversation
from smithtune import cli as pipeline
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL


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
        if path.startswith("/api/v1/sessions/") or (path == "/api/v2/runs/query" and body.get("selects") != ["ID", "TRACE_ID", "THREAD_ID", "START_TIME"]):
            self.contract_calls.append((path, copy.deepcopy(body)))
            if path.startswith("/api/v1/sessions/"):
                result = {"id": uid(101), "start_time": "2026-09-01T00:00:00+00:00"}
            elif body.get("is_root"):
                result = {"items": [{"id": uid(1), "project_id": uid(101)}]}
            else:
                trace_id = body.get("trace_id") or uid(1)
                result = {"items": [{"id": uid(1001), "trace_id": trace_id, "project_id": uid(101),
                                    "run_type": "llm", "extra": {"invocation_params": {"tools": []}}}]}
            return SimpleNamespace(stdout=json.dumps(result))
        self.calls.append((path, copy.deepcopy(body)))
        if self.failure:
            self.failure(path, body)
        if path == "/api/v2/runs/query":
            page = int(body.get("cursor", 0))
            result = {"items": self.pages[page], "next_cursor": str(page + 1) if page + 1 < len(self.pages) else None}
        elif path == "/api/v1/datasets":
            result = copy.deepcopy(body)
            self.datasets[body["id"]] = result
        elif path == "/v1/trajectory":
            assert body["format"] == "messages" and body["include"] == {"system_messages": True}
            assert sum(key in body for key in curation.TRAJECTORY_KEYS) == 1
            result = {"messages": copy.deepcopy(self.messages), "next_cursor": None, "prev_cursor": None}
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


def select(tmp_path, api, **overrides):
    return curation._select_dataset(**{
        "workspace_id": uid(100), "project_id": uid(101),
        "start_time": "2026-09-01T00:00:00Z", "end_time": "2026-09-08T00:00:00Z",
        "output": tmp_path / "selection.json", "runner": api, "limit": 100,
        **overrides,
    })


def create(tmp_path, api, **overrides):
    return curation.create_dataset(**{
        "workspace_id": uid(100), "project_id": uid(101), "name": "new",
        "start_time": "2026-09-01T00:00:00Z", "end_time": "2026-09-08T00:00:00Z",
        "output": tmp_path / "selection.json", "runner": api, "limit": 100, "concurrency": 1, **overrides,
    })


def saved_selection(tmp_path):
    return json.loads((tmp_path / "selection.json").read_text())


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


def test_selection_keys_roots_by_thread_then_trace(tmp_path):
    roots = [root(1, "a"), root(2, "a"), root(3, "b"), root(4)]
    api = API([roots[:2], roots[2:]])
    result = select(tmp_path, api, filter='has(tags, "reviewed")')
    saved = saved_selection(tmp_path)
    assert saved["schema_version"] == 2
    assert saved["selected"] == [thread("a"), thread("b"), trace(4)]
    assert saved["query"] == {"start_time": "2026-09-01T00:00:00+00:00", "end_time": "2026-09-08T00:00:00+00:00",
                              "filter": 'has(tags, "reviewed")', "limit": 100}
    assert "seed" not in json.dumps(saved) and "feedback_stats" not in json.dumps(saved)
    assert (result["matching_roots"], result["distinct_conversations"], result["selected_examples"]) == (4, 3, 3)
    assert result["trace_keyed_examples"] == 1
    assert [item["selected"] for item in result["preview"]] == [True, True, True, True]
    assert "feedback_stats" not in result["preview"][0]
    assert api.calls[0][1]["is_root"] is True
    assert api.calls[0][1]["page_size"] == 100
    assert api.calls[0][1]["selects"] == ["ID", "TRACE_ID", "THREAD_ID", "START_TIME"]
    assert api.calls[0][1]["filter"] == 'and(has(tags, "reviewed"), lt(start_time, "2026-09-08T00:00:00+00:00"))'
    assert api.calls[1][1]["cursor"] == "1"
    assert "messages" not in json.dumps(saved)


def test_selection_stops_paging_once_limit_is_reached(tmp_path):
    roots = [root(i, f"thread-{(i + 1) // 2}") for i in range(1, 21)]
    api = API([roots[:8], roots[8:12], roots[12:]])
    result = select(tmp_path, api, limit=5)
    # Page one yields four threads; page two reaches five; page three is never requested.
    assert result["pages_fetched"] == 2
    assert [path for path, _ in api.calls] == ["/api/v2/runs/query"] * 2
    assert saved_selection(tmp_path)["selected"] == [thread(f"thread-{i}") for i in range(1, 6)]
    assert result["matching_roots"] == 12 and result["distinct_conversations"] == 6
    assert result["selected_examples"] == 5
    assert [item["selected"] for item in result["preview"]] == [True] * 10 + [False] * 2


def test_selection_is_deterministic_in_query_order(tmp_path):
    roots = [root(i, f"thread-{i // 2}") for i in range(1, 21)]
    first = select(tmp_path / "first", API([roots[:8], roots[8:]]), limit=4)
    second = select(tmp_path / "second", API([roots[:8], roots[8:]]), limit=4)
    assert saved_selection(tmp_path / "first")["selected"] == saved_selection(tmp_path / "second")["selected"]
    assert first["selected_examples"] == second["selected_examples"] == 4


def test_limit_larger_than_matches_keeps_all_conversations_and_deduplicates_pages(tmp_path):
    result = select(tmp_path, API([[root(1, "a")], [root(1, "a"), root(2, "a"), root(3), root(4, "b")]]), limit=curation.MAX_LIMIT)
    assert result["matching_roots"] == 4
    assert result["selected_examples"] == result["distinct_conversations"] == 3
    assert result["trace_keyed_examples"] == 1
    assert saved_selection(tmp_path)["selected"] == [thread("a"), trace(3), thread("b")]


@pytest.mark.parametrize("overrides", [{"limit": 0}, {"limit": -1}, {"limit": None}, {"limit": "5"}, {"limit": curation.MAX_LIMIT + 1},
    {"start_time": "2026-09-01"}, {"end_time": "2026-08-01T00:00:00Z"}, {"project_id": "bad"}])
def test_invalid_selection_arguments_do_not_query(tmp_path, overrides):
    api = API()
    with pytest.raises(PipelineError):
        select(tmp_path, api, **overrides)
    assert api.calls == []


def test_empty_selection_and_overwrite_fail_before_writes(tmp_path):
    api = API([[]])
    result = select(tmp_path, api)
    assert result["selected_examples"] == result["distinct_conversations"] == 0
    with pytest.raises(PipelineError, match="already exists"):
        select(tmp_path, api)
    with pytest.raises(PipelineError, match="empty selections"):
        curation._import_selection(selection=tmp_path / "selection.json", name="new", runner=api)
    assert len(api.calls) == 1
    assert not (tmp_path / "selection.import.json").exists()


@pytest.mark.parametrize("page", [{"items": None}, {"items": [root(1)], "next_cursor": "repeat"}])
def test_malformed_or_repeated_pages_are_rejected(tmp_path, page):
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps(page))
    with pytest.raises(PipelineError, match="page|cursor"):
        select(tmp_path, runner, limit=5)
    assert not (tmp_path / "selection.json").exists()


def test_import_fetches_each_trajectory_and_creates_one_example(tmp_path):
    api = API([[root(1, "a"), root(2, "a"), root(3)]])
    select(tmp_path, api)
    api.calls.clear()
    api.pages = [[root(4, "b")]]
    result = curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=api)
    assert result["example_count"] == 2
    assert api.calls == [
        ("/api/v1/datasets", {"id": result["dataset_id"], "name": "new", "data_type": "kv"}),
        ("/v1/trajectory", {"project_id": uid(101), "thread_id": "a",
                            "format": "messages", "include": {"system_messages": True}}),
        ("/api/v1/examples", {"id": api.examples[0]["id"], "dataset_id": result["dataset_id"], "inputs": {"messages": api.messages}, "outputs": None,
                              "metadata": {**COMMON_METADATA, "source_scope": "thread", "source_scope_id": "a"}}),
        ("/v1/trajectory", {"project_id": uid(101), "trace_id": uid(3),
                            "format": "messages", "include": {"system_messages": True}}),
        ("/api/v1/examples", {"id": api.examples[1]["id"], "dataset_id": result["dataset_id"], "inputs": {"messages": api.messages}, "outputs": None,
                              "metadata": {**COMMON_METADATA, "source_scope": "trace", "source_scope_id": uid(3)}}),
    ]
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete"
    assert receipt["pending_write"] is None
    assert "messages" not in receipt
    assert "What is 17" not in json.dumps(receipt)
    calls = len(api.calls)
    with pytest.raises(PipelineError, match="destination or selection changed"):
        curation._import_selection(selection=tmp_path / "selection.json", name="again", runner=api)
    assert len(api.calls) == calls


def test_partial_failure_records_saved_input_and_pending_write(tmp_path):
    api = API([[root(1, "a"), root(2, "b")]])
    select(tmp_path, api)
    def fail(path, body):
        if path == "/api/v1/examples" and body["metadata"]["source_scope_id"] == "b":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="request timed out; private message")
    api.failure = fail
    with pytest.raises(PipelineError, match="request failed.*import incomplete") as error:
        curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=api)
    assert "private message" not in str(error.value)
    assert api.trajectory_calls() == [{"thread_id": "a"}, {"thread_id": "b"}]
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete"
    pending = receipt["pending_write"]
    saved = load_conversation(Path(pending["conversation"]))["example"]
    assert pending["source"]["scope_id"] == "b"
    assert saved["inputs"]["messages"] == api.messages
    assert saved["metadata"]["source_scope_id"] == "b"
    assert len(list((tmp_path / "conversations").glob("*.json"))) == 2
    assert receipt["dataset_id"] == next(iter(api.datasets))
    assert receipt["confirmed_example_ids"] == [api.examples[0]["id"]]


@pytest.mark.parametrize("response", [
    {"messages": [], "next_cursor": None}, {"messages": None}, [],
    {"messages": [{"role": "human", "content": "hi"}], "next_cursor": "more"},
])
def test_incomplete_trajectory_fails_before_the_example_write(tmp_path, response):
    api = API([[root(1, "a"), root(2)]])
    select(tmp_path, api)
    def runner(command, **kwargs):
        result = api(command, **kwargs)
        return SimpleNamespace(stdout=json.dumps(response)) if command[2] == "/v1/trajectory" else result
    with pytest.raises(PipelineError, match="thread_id a returned") as error:
        curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=runner)
    assert "outcome may be unknown" not in str(error.value)
    assert not any(path == "/api/v1/examples" for path, _ in api.calls)
    assert not (tmp_path / "conversations").exists()
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["pending_write"] is None
    assert receipt["confirmed_example_ids"] == []


@pytest.mark.parametrize("response", [{}, {"id": "invalid"}, None])
def test_unconfirmed_example_write_records_pending_write(tmp_path, response):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    def runner(command, **kwargs):
        result = api(command, **kwargs)
        return SimpleNamespace(stdout=json.dumps(response)) if command[2] == "/api/v1/examples" else result
    with pytest.raises(PipelineError, match="did not confirm its saved ID"):
        curation._import_selection(selection=tmp_path / "selection.json", name="new", runner=runner)
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete"
    pending = receipt["pending_write"]
    assert load_conversation(Path(pending["conversation"]))["example"]["inputs"]["messages"] == api.messages
    assert receipt["confirmed_example_ids"] == [] and pending["source"]["scope_id"] == "a"


def test_saved_v1_thread_selection_must_be_regenerated(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({
        "schema_version": 1, "created_at_utc": "2026-09-01T00:00:00Z",
        "workspace_id": uid(100), "project_id": uid(101), "scope": "thread",
        "query": {"start_time": "2026-09-01T00:00:00Z", "end_time": "2026-09-08T00:00:00Z",
                  "filter": None, "limit": None, "seed": 42},
        "matches": [root(1, "existing-thread")], "selected_ids": ["existing-thread"],
    }))
    api = API()
    with pytest.raises(PipelineError, match="schema_version 2.*regenerate"):
        curation._import_selection(selection=path, name="new", runner=api)
    assert api.calls == []
    assert not (tmp_path / "selection.import.json").exists()


@pytest.mark.parametrize("selected", [
    [{"key": "trace_id", "id": uid(1)}],          # thread root reinterpreted as a trace
    [{"key": "thread_id", "id": uid(2)}],         # trace root reinterpreted as a thread
    [{"key": "thread_id", "id": "not-a-match"}],
    [{"key": "thread_id", "id": "a"}, {"key": "thread_id", "id": "a"}],
    [{"key": "run_id", "id": uid(1)}], ["a"], [],
])
def test_saved_selection_rejects_unknown_or_reinterpreted_ids(tmp_path, selected):
    api = API([[root(1, "a"), root(2)]])
    select(tmp_path, api)
    path = tmp_path / "selection.json"
    saved = saved_selection(tmp_path)
    saved["selected"] = selected
    path.write_text(json.dumps(saved))
    with pytest.raises(PipelineError):
        curation._import_selection(selection=path, name="new", runner=api)
    assert len(api.calls) == 1
    assert not (tmp_path / "selection.import.json").exists()


def test_existing_dataset_name_and_unknown_dataset_write(tmp_path):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    def fail(path, body):
        raise subprocess.CalledProcessError(1, "langsmith", stderr="Error: HTTP 409")
    api.failure = fail
    with pytest.raises(PipelineError, match="dataset name already exists"):
        curation._import_selection(selection=tmp_path / "selection.json", name="existing", runner=api)
    assert len(api.calls) == 2
    assert json.loads((tmp_path / "selection.import.json").read_text())["dataset_name"] == "existing"


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


def test_parser_defaults_and_dispatch(tmp_path, monkeypatch, capsys):
    from smithtune import dataset_workflow
    from test_triage import API as TriageAPI

    args = ["dataset", "create", "--workspace-id", uid(100), "--project-id", uid(101), "--name", "new"]
    parsed = pipeline._parser().parse_args(args)
    assert parsed.limit is None and parsed.directory is None and not hasattr(parsed, "seed")
    assert parsed.concurrency is None and pipeline._parser().parse_args([*args, "--concurrency", "2"]).concurrency == 2
    api = TriageAPI()
    original = dataset_workflow.run
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curation, "_utc_now", lambda: "2026-09-03T00:00:00+00:00")
    monkeypatch.setattr(dataset_workflow, "run", lambda *a, **kw: original(*a, **kw, runner=api))
    pipeline.main([*args, "--filter", 'has(tags, "reviewed")', "--limit", "1", "--confirm"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "Dataset directory" in captured.err
    assert result["dataset_id"] == next(iter(api.datasets)) and result["example_count"] == 1
    assert result["downloaded"] == 1
    saved = json.loads((Path(result["run_dir"]) / "snapshot.json").read_text())
    assert saved["source"]["limit"] == 1 and saved["source"]["filter"] == 'has(tags, "reviewed")'
    assert saved["source"]["start_time"] == "2026-09-02T00:00:00+00:00"
    assert saved["source"]["end_time"] == "2026-09-03T00:00:00+00:00"
    assert Path(result["receipt"]).exists()
    for obsolete in (["dataset", "select"], [*args, "--selection", "saved.json"], [*args, "--scope", "trace"],
                     [*args, "--seed", "17"]):
        with pytest.raises(SystemExit) as error:
            pipeline._parser().parse_args(obsolete)
        assert error.value.code == 2


def test_create_download_prepare(tmp_path):
    # Eight threads of two traces each plus four standalone traces.
    api = API([[root(i, f"thread-{(i + 1) // 2}") for i in range(1, 17)] + [root(i) for i in range(17, 21)]])
    def source_messages(path, body):
        if path == "/v1/trajectory":
            identity = body.get("thread_id") or body["trace_id"]
            api.messages = [{"role": "human", "content": f"Question {identity}"},
                            {"role": "ai", "content": f"Answer {identity}"}]
    api.failure = source_messages
    imported = create(tmp_path, api)
    assert imported["example_count"] == 12 and imported["trace_keyed_examples"] == 4
    def download(command, *, capture=False):
        if command[1:3] == ["dataset", "get"]:
            return SimpleNamespace(stdout=json.dumps({"id": imported["dataset_id"], "name": "new", "example_count": len(api.examples)}))
        if command[1:3] == ["dataset", "export"]:
            Path(command[4]).write_text(json.dumps([{"inputs": ex["inputs"], "outputs": ex["outputs"]} for ex in api.examples]))
            return SimpleNamespace(stdout="")
        assert command[1] == "api" and command[2].startswith("/api/v1/examples?")
        query = parse_qs(urlparse(command[2]).query)
        assert query["dataset"] == [imported["dataset_id"]]
        offset, limit = int(query["offset"][0]), int(query["limit"][0])
        return SimpleNamespace(stdout=json.dumps(api.examples[offset:offset + limit]))
    data_dir = tmp_path / "data"
    dataset.download_dataset(uid(100), imported["dataset_id"], data_dir / "raw", runner=download)
    from test_example_tools import write_empty_tool_snapshot
    write_empty_tool_snapshot(data_dir, api.examples, imported["dataset_id"], uid(100))
    manifest = dataset.prepare_dataset(uid(100), imported["dataset_id"], DEFAULT_MODEL, data_dir,
                                       fetch=False, check_render=False)
    split = [manifest["split"][key] for key in ("train", "validation", "test")]
    assert sum(split) == 12
    rows = [json.loads(line) for line in (data_dir / "prepared" / "train.jsonl").read_text().splitlines()]
    # Provider JSONL strips provenance; the preserved raw examples keep it.
    assert all(row["messages"][0]["role"] == "user" for row in rows)
    source_rows = dataset.prepare_sft_rows(api.examples)
    assert sum(row["_source"]["source_scope"] == "thread" for row in source_rows) == 8
    assert sum(row["_source"]["source_scope"] == "trace" for row in source_rows) == 4


def test_create_freezes_paginated_selection_before_import(tmp_path):
    api = API([[root(1, "a"), root(2, "a")], [root(3, "b"), root(4)]])
    def source_changes(path, body):
        if path == "/api/v1/datasets":
            assert saved_selection(tmp_path)["selected"] == [thread("a"), thread("b"), trace(4)]
            api.pages = [[root(5, "new-thread")]]
    api.failure = source_changes
    result = create(tmp_path, api)
    assert result["example_count"] == 3 and result["trace_keyed_examples"] == 1
    assert api.trajectory_calls() == [{"thread_id": "a"}, {"thread_id": "b"}, {"trace_id": uid(4)}]
    assert sum(path == "/api/v2/runs/query" for path, _ in api.calls) == 2
    assert all(ex["inputs"]["messages"] == api.messages for ex in api.examples)


def test_create_with_no_matches_does_not_create_empty_dataset(tmp_path):
    api = API([[]])
    with pytest.raises(PipelineError, match="no conversations matched.*no dataset was created"):
        create(tmp_path, api)
    assert [path for path, _ in api.calls] == ["/api/v2/runs/query"]
    assert (tmp_path / "selection.json").exists()
    assert not (tmp_path / "selection.import.json").exists()


def test_create_default_paths_are_unique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = create(tmp_path, API([[root(1, "a")]]), output=None)
    second = create(tmp_path, API([[root(1, "a")]]), name="another", output=None)
    assert first["selection"] != second["selection"]
    assert first["receipt"] != second["receipt"]
    for result in (first, second):
        run_dir = Path(result["run_dir"])
        assert run_dir.parent == Path("data/datasets")
        assert Path(result["selection"]) == run_dir / "selection.json"
        conversation, = (run_dir / "conversations").glob("*.json")
        assert load_conversation(conversation)["example"]["metadata"]["source_scope_id"] == "a"
        assert json.loads(Path(result["receipt"]).read_text())["status"] == "complete"


def test_create_explicit_run_directory_and_conflicting_output(tmp_path):
    api = API([[root(1, "a")]])

    def check_saved_before_upload(path, body):
        if path == "/api/v1/examples":
            saved, = (tmp_path / "conversations").glob("*.json")
            assert load_conversation(saved)["example"] == {key: value for key, value in body.items() if key not in {"dataset_id", "id"}}

    api.failure = check_saved_before_upload
    result = create(tmp_path, api, output=None, run_dir=tmp_path)
    assert result["run_dir"] == str(tmp_path)
    before = len(api.calls)
    with pytest.raises(PipelineError, match="use --run-dir or --output"):
        create(tmp_path, api, run_dir=tmp_path)
    assert len(api.calls) == before


def test_conversation_save_failure_prevents_upload(tmp_path, monkeypatch):
    from smithtune import dataset_artifacts

    api = API([[root(1, "a")]])

    def disk_full(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(dataset_artifacts, "_json_dump", disk_full)
    with pytest.raises(PipelineError, match="cannot save conversation"):
        create(tmp_path, api)
    assert not any(path == "/api/v1/examples" for path, _ in api.calls)
    assert not list((tmp_path / "conversations").glob("*.json"))


def test_create_validates_name_before_querying(tmp_path):
    api = API([[root(1, "a")]])
    with pytest.raises(PipelineError, match="name must be a nonempty string"):
        create(tmp_path, api, name=" ")
    assert api.calls == []


def test_create_failure_reports_partial_dataset_and_saved_receipt(tmp_path, monkeypatch):
    api = API([[root(1, "a"), root(2)]])
    def fail(path, body):
        if path == "/v1/trajectory" and body.get("trace_id") == uid(2):
            raise subprocess.CalledProcessError(1, "langsmith", stderr="timeout")
    api.failure = fail
    monkeypatch.setattr(curation, "_sleep", lambda seconds: None)
    with pytest.raises(PipelineError, match="after 3 attempt.*receipt="):
        create(tmp_path, api)
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["dataset_id"] == next(iter(api.datasets))
    assert receipt["confirmed_example_ids"] == [api.examples[0]["id"]]
    assert api.trajectory_calls() == [{"thread_id": "a"}] + [{"trace_id": uid(2)}] * 3


def test_transient_trajectory_failures_are_retried_but_example_writes_are_not(tmp_path, monkeypatch):
    api = API([[root(1, "a"), root(2, "b")]])
    sleeps = []
    monkeypatch.setattr(curation, "_sleep", sleeps.append)
    attempts = {"a": 0, "b": 0}
    def flaky(path, body):
        if path == "/v1/trajectory":
            attempts[body["thread_id"]] += 1
            if body["thread_id"] == "a" and attempts["a"] < 3:
                raise subprocess.CalledProcessError(1, "langsmith", stderr="connection reset")
        if path == "/api/v1/examples" and body["metadata"]["source_scope_id"] == "b":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="connection reset")
    api.failure = flaky
    with pytest.raises(PipelineError, match="request failed.*import incomplete"):
        create(tmp_path, api)
    assert attempts == {"a": 3, "b": 1}
    assert sleeps == [1.0, 2.0]
    assert sum(path == "/api/v1/examples" for path, _ in api.calls) == 2


def test_trajectory_fetch_gives_up_after_three_attempts(tmp_path, monkeypatch):
    api = API([[root(1, "a")]])
    monkeypatch.setattr(curation, "_sleep", lambda seconds: None)
    def always_fail(path, body):
        if path == "/v1/trajectory":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 503")
    api.failure = always_fail
    with pytest.raises(PipelineError, match="HTTP 503 after 3 attempt"):
        create(tmp_path, api)
    assert sum(path == "/v1/trajectory" for path, _ in api.calls) == 3
    assert not any(path == "/api/v1/examples" for path, _ in api.calls)


def test_concurrent_import_bounds_in_flight_work_and_drains_after_failure(tmp_path, monkeypatch):
    import threading
    import time
    monkeypatch.setattr(curation, "_sleep", lambda seconds: None)
    api = API([[root(i, f"thread-{i}") for i in range(1, 13)]])
    gate, active, peak = threading.Lock(), [0], [0]
    def fail_first_and_slow_others(path, body):
        if path != "/v1/trajectory":
            return
        if body.get("thread_id") == "thread-1":
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 500")
        with gate:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with gate:
            active[0] -= 1
    api.failure = fail_first_and_slow_others
    with pytest.raises(PipelineError, match="HTTP 500.*after 3 attempt") as error:
        create(tmp_path, api, concurrency=4)
    assert "outcome may be unknown" not in str(error.value)
    assert 1 < peak[0] <= 3
    # The first batch of four starts together; after the failure nothing new is
    # submitted, and the three slow fetches already in flight drain to completion.
    started = [next(iter(call.values())) for call in api.trajectory_calls()]
    assert sorted(set(started)) == [f"thread-{i}" for i in range(1, 5)]
    assert started.count("thread-1") == 3 and len(started) == 6
    receipt = json.loads((tmp_path / "selection.import.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["pending_write"] is None
    assert sorted(receipt["confirmed_example_ids"]) == sorted(ex["id"] for ex in api.examples)
    assert len(receipt["confirmed_example_ids"]) == 0


def test_concurrent_import_completes_every_selected_conversation(tmp_path):
    api = API([[root(i, f"thread-{i}") for i in range(1, 10)] + [root(i) for i in range(10, 13)]])
    result = create(tmp_path, api, concurrency=3)
    assert result["example_count"] == 12 and len(api.examples) == 12
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["status"] == "complete" and receipt["pending_write"] is None
    assert sorted(receipt["confirmed_example_ids"]) == sorted(ex["id"] for ex in api.examples)
    assert sorted(json.dumps(call, sort_keys=True) for call in api.trajectory_calls()) == sorted(
        json.dumps({"thread_id": f"thread-{i}"}) for i in range(1, 10)) + sorted(
        json.dumps({"trace_id": uid(i)}) for i in range(10, 13))


def test_import_collects_all_pages_before_writing_one_example(tmp_path):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    pages = {
        None: {"messages": api.messages[:2], "next_cursor": "second", "prev_cursor": None},
        "second": {"messages": [], "next_cursor": "third", "prev_cursor": "first"},
        "third": {"messages": api.messages[2:], "next_cursor": None, "prev_cursor": "second"},
    }
    cursors = []

    def runner(command, **kwargs):
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory":
            assert api.examples == []
            body = json.loads(kwargs["input"])
            cursor = body.get("cursor")
            cursors.append(cursor)
            return SimpleNamespace(stdout=json.dumps(pages[cursor]))
        return response

    result = curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=runner)
    assert cursors == [None, "second", "third"]
    assert result["example_count"] == 1
    assert api.examples[0]["inputs"]["messages"] == api.messages


@pytest.mark.parametrize("recover", [True, False])
def test_later_page_retries_only_that_page_and_never_writes_partial_example(tmp_path, monkeypatch, recover):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    sleeps, cursors = [], []
    monkeypatch.setattr(curation, "_sleep", sleeps.append)

    def runner(command, **kwargs):
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        body = json.loads(kwargs["input"])
        cursor = body.get("cursor")
        cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(stdout=json.dumps({"messages": api.messages[:2], "next_cursor": "second"}))
        if not recover or cursors.count("second") < 3:
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 503")
        return SimpleNamespace(stdout=json.dumps({"messages": api.messages[2:], "next_cursor": None, "prev_cursor": "first"}))

    if recover:
        curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=runner)
        assert api.examples[0]["inputs"]["messages"] == api.messages
    else:
        with pytest.raises(PipelineError, match="HTTP 503 after 3 attempt"):
            curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=runner)
        assert api.examples == []
    assert cursors == [None, "second", "second", "second"]
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("cursor", ["", 123, [], {}, "second"])
def test_invalid_or_repeated_later_cursor_never_writes_partial_example(tmp_path, cursor):
    api = API([[root(1, "a")]])
    select(tmp_path, api)
    calls = []

    def runner(command, **kwargs):
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        calls.append(json.loads(kwargs["input"]).get("cursor"))
        return SimpleNamespace(stdout=json.dumps({"messages": api.messages, "next_cursor": "second" if len(calls) == 1 else cursor}))

    with pytest.raises(PipelineError, match="invalid or repeated continuation cursor"):
        curation._import_selection(selection=tmp_path / "selection.json", name="new", concurrency=1, runner=runner)
    assert calls == [None, "second"]
    assert api.examples == []

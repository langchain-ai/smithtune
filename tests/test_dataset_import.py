import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest

from smithtune import cli, dataset, dataset_import, dataset_workflow, triage
from smithtune.dataset_artifacts import load_conversation
from smithtune.providers.base import PipelineError
from binding_fixtures import bound_example


def uid(n):
    return str(UUID(int=n))


def example(n, turns=1, **metadata):
    return bound_example({"id": uid(n), "dataset_id": uid(200), "outputs": None,
            "inputs": {"messages": [{"role": role, "content": f"{i}-{role}", "id": f"{i}-{role}"}
                                     for i in range(turns) for role in ("human", "ai")]},
            "metadata": {"trajectory_format": "messages", "conversation_scope": "root",
                          "source_workspace_id": uid(100), "source_project_id": uid(101),
                         "source_scope": "thread", "source_scope_id": f"thread-{n}", **metadata}})


class API:
    def __init__(self, existing=()):
        from test_curation import API as SourceAPI

        self.source = SourceAPI()
        self.existing = list(existing)
        self.calls = []
        self.writes = []
        self.version = "2026-09-15T00:00:00+00:00"
        self.before_write = None

    def __call__(self, command, *, capture=False, input=None):
        assert capture
        assert command[command.index("--workspace") + 1] == uid(100)
        method = command[command.index("--method") + 1]
        path = command[2]
        if path.startswith("/api/v1/sessions/") or path == "/api/v2/runs/query":
            return self.source(command, capture=capture, input=input)
        body = json.loads(input) if input else None
        self.calls.append((method, path, body))
        if method != "GET":
            self.writes.append((method, path, body))
            if self.before_write:
                self.before_write(method, path, body)
            if method == "POST":
                self.existing.append(copy.deepcopy(body))
                value = {"id": body["id"]}
            else:
                old = next(ex for ex in self.existing if ex["id"] == path.rsplit("/", 1)[1])
                old.update(copy.deepcopy(body))
                value = {"message": "Example updated"}
        elif path == f"/api/v1/datasets/{uid(200)}":
            value = {"id": uid(200), "data_type": "kv"}
        elif path.endswith("/versions?limit=1"):
            value = [{"as_of": self.version}] if self.existing else []
        elif path.startswith("/api/v1/examples/"):
            value = next((ex for ex in self.existing if ex["id"] == path.rsplit("/", 1)[1]), None)
            if value is None:
                raise PipelineError("HTTP 404")
        else:
            assert urlsplit(path).path == "/api/v1/examples"
            query = parse_qs(urlsplit(path).query)
            assert query["dataset"] == [uid(200)]
            assert query["as_of"] == [self.version]
            assert query["limit"] == ["100"]
            offset = int(query["offset"][0])
            value = self.existing[offset:offset + 100]
        return SimpleNamespace(stdout=json.dumps(value))


def update(tmp_path, api, incoming, **kwargs):
    return dataset_import.update_dataset(uid(100), uid(200), incoming,
        {dataset._source_key(item, uid(100), None) for item in incoming},
        tmp_path, tmp_path / "receipt.json", runner=api, **kwargs)


def receipt(tmp_path):
    return json.loads((tmp_path / "receipt.json").read_text())


def test_mixed_create_skip_update_preserves_ids_and_metadata(tmp_path):
    api = API([example(1), example(2, note="keep me")])

    def saved_before_write(method, path, body):
        pending = receipt(tmp_path)["pending_write"]
        assert load_conversation(Path(pending["conversation"]))["inputs"] == body["inputs"]
        assert pending["action"] == ("created" if method == "POST" else "updated")

    api.before_write = saved_before_write
    result = update(tmp_path, api, [example(1), example(2, 2), example(3)])
    assert (result["created"], result["updated"], result["skipped"], result["example_count"]) == (1, 1, 1, 3)
    patch, post = api.writes
    assert patch[:2] == ("PATCH", f"/api/v1/examples/{uid(2)}")
    assert patch[2]["metadata"]["note"] == "keep me"
    assert post[2]["dataset_id"] == uid(200)
    assert receipt(tmp_path)["status"] == "complete"
    assert receipt(tmp_path)["pending_write"] is None
    actions = [json.loads(line) for line in (tmp_path / "receipt.actions.jsonl").read_text().splitlines()]
    assert [row["action"] for row in actions] == ["skipped", "updated", "created"]
    assert [row["example_id"] for row in actions] == [uid(1), uid(2), post[2]["id"]]


def test_destination_pagination_is_pinned_and_only_matches_saved(tmp_path):
    api = API([example(n) for n in range(1, 102)])
    update(tmp_path, api, [example(101, 2)])
    queries = [parse_qs(urlsplit(path).query) for method, path, _ in api.calls if path.startswith("/api/v1/examples?")]
    assert [q["offset"] for q in queries] == [["0"], ["100"]]
    assert len(list((tmp_path / "destination/conversations").glob("*.json"))) == 1
    assert api.writes[0][1] == f"/api/v1/examples/{uid(101)}"


def test_duplicate_source_on_later_page_prevents_all_writes(tmp_path):
    api = API([example(n) for n in range(1, 101)] + [example(101, source_scope_id="thread-1")])
    with pytest.raises(PipelineError, match="same source conversation"):
        update(tmp_path, api, [example(102)])
    assert not api.writes


@pytest.mark.parametrize("field,value", [("source_workspace_id", uid(999)), ("source_project_id", uid(999)), ("source_scope", "trace")])
def test_matching_includes_whole_source_namespace(tmp_path, field, value):
    # Scope IDs are UUIDs so both thread and trace identities are valid.
    api = API([example(1, source_scope_id=uid(9), **{field: value})])
    assert update(tmp_path, api, [example(1, source_scope_id=uid(9))])["created"] == 1


@pytest.mark.parametrize("change", ["shorter", "text", "message_id", "other_input", "outputs"])
def test_conflicting_snapshot_stops_without_write(tmp_path, change):
    old, incoming = example(1, 2), example(1, 3)
    if change == "shorter":
        incoming = example(1)
    elif change == "outputs":
        old["outputs"] = {"answer": "must retain"}
    elif change == "other_input":
        incoming["inputs"]["context"] = "new context"
    else:
        incoming["inputs"]["messages"][0]["id" if change == "message_id" else "content"] = "changed"
    api = API([old])
    with pytest.raises(PipelineError, match="exact prefix|not a message trajectory"):
        update(tmp_path, api, [incoming])
    assert not api.writes
    assert receipt(tmp_path)["status"] == "incomplete"
    assert len(list((tmp_path / "conversations").glob("*.json"))) == 1


def test_ordinary_import_skips_unchanged_triaged_but_cannot_extend(tmp_path):
    api = API([example(1, smithtune_triage={"identity_sha256": "old"})])
    assert update(tmp_path / "same", api, [example(1)])["skipped"] == 1
    with pytest.raises(PipelineError, match="rerun triage"):
        update(tmp_path / "extended", api, [example(1, 2)])
    assert not api.writes


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_failed_write_resumes_pending_and_retains_prior_success(tmp_path, method):
    api = API([example(1)] if method == "PATCH" else [])

    def fail_second(*args):
        if len(api.writes) == 2:
            raise subprocess.CalledProcessError(1, "langsmith", stderr="HTTP 504")

    api.before_write = fail_second
    with pytest.raises(PipelineError, match="HTTP 504.*incomplete.*receipt.json"):
        update(tmp_path, api, [example(2), example(1, 2)])
    saved = receipt(tmp_path)
    assert len(api.writes) == 2 and api.writes[-1][0] == method
    assert saved["created"] == 1 and saved["status"] == "incomplete"
    assert saved["pending_write"]["source"]["scope_id"] == "thread-1"
    assert load_conversation(Path(saved["pending_write"]["conversation"])) == example(1, 2)
    assert len((tmp_path / "receipt.actions.jsonl").read_text().splitlines()) == 1
    api.before_write = None
    result = update(tmp_path, api, [example(2), example(1, 2)])
    assert result["example_count"] == 2
    assert len(api.writes) == 3
    assert api.writes[-1] == api.writes[-2]


def test_missing_incoming_source_does_not_mark_complete(tmp_path):
    api = API()
    with pytest.raises(PipelineError, match="before every selected conversation"):
        dataset_import.update_dataset(uid(100), uid(200), iter([]),
            {dataset._source_key(example(1), uid(100), None)}, tmp_path, tmp_path / "receipt.json", runner=api)
    assert receipt(tmp_path)["status"] == "incomplete"


@pytest.mark.parametrize("targets", [["--name", "new", "--dataset-id", uid(200)]])
def test_cli_rejects_conflicting_destinations(targets):
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args(["dataset", "push", *targets])
    assert exc.value.code == 2


def test_cli_existing_dataset_ordinary_path(tmp_path, monkeypatch, capsys):
    from smithtune.triage_source import load_snapshot
    from test_triage import API as SourceAPI, source

    dataset_workflow.run("pull", tmp_path, runner=SourceAPI(), **{key: value for key, value in source().items() if key != "seed"})
    old = copy.deepcopy(load_snapshot(tmp_path)["units"][0]["example"])
    old.update(id=uid(1), dataset_id=uid(200))
    old["inputs"]["messages"] = old["inputs"]["messages"][:3]
    old["metadata"]["smithtune_source"]["assistant_runs"] = old["metadata"]["smithtune_source"]["assistant_runs"][:1]
    destination = API([old])
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *a, **kw: original(*a, **kw, runner=destination))
    cli.main(["dataset", "push", str(tmp_path), "--dataset-id", uid(200), "--confirm"])
    result = json.loads(capsys.readouterr().out)
    assert result["updated"] == 1
    assert destination.writes[0][0] == "PATCH"
    assert (tmp_path / "snapshot.json").exists()


def test_fresh_triage_updates_messages_and_contract_together(tmp_path, monkeypatch, capsys):
    from test_triage import run

    run(tmp_path)
    incoming, = triage.selected_examples(tmp_path)
    old = copy.deepcopy(incoming)
    old.update(id=uid(1), dataset_id=uid(200))
    old["inputs"]["messages"] = old["inputs"]["messages"][:3]
    old["metadata"]["smithtune_source"]["assistant_runs"] = old["metadata"]["smithtune_source"]["assistant_runs"][:1]
    old["metadata"].update(note="retain", smithtune_triage={"identity_sha256": "old", "contract": {"old": True}})
    api = API([old])
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(*args, **kwargs, runner=api))
    cli.main(["dataset", "push", str(tmp_path), "--dataset-id", uid(200), "--confirm"])
    assert json.loads(capsys.readouterr().out)["updated"] == 1
    method, path, body = api.writes[0]
    assert (method, path) == ("PATCH", f"/api/v1/examples/{uid(1)}")
    assert body["metadata"]["smithtune_triage"] == incoming["metadata"]["smithtune_triage"]
    assert body["metadata"]["note"] == "retain"
    assert body["inputs"] == incoming["inputs"]
    from smithtune.bindings import validate_bound_messages
    assert len(validate_bound_messages({"id": uid(1), **body})) == 2


def test_failed_triage_never_reaches_destination(tmp_path):
    from test_triage import API as SourceAPI, source

    triage.run_triage(source(), tmp_path, runner=SourceAPI(), judge_call=lambda *_: {"keep": 1}, confirm=True, attempts=1)
    api = API()
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.create_triaged_dataset(tmp_path, dataset_id=uid(200), confirm=True, runner=api)
    assert not api.calls


@pytest.mark.parametrize("changed", [1, 1.0])
def test_nested_argument_type_change_is_not_an_exact_prefix(tmp_path, changed):
    old = example(1)
    old["inputs"]["messages"][-1]["tool_calls"] = [{"name": "search", "args": {"enabled": True}}]
    incoming = copy.deepcopy(old)
    incoming["inputs"]["messages"][-1]["tool_calls"][0]["args"]["enabled"] = changed
    incoming["inputs"]["messages"].extend(example(1, 2)["inputs"]["messages"][2:])
    api = API([old])
    with pytest.raises(PipelineError, match="exact prefix"):
        update(tmp_path, api, [incoming])
    assert not api.writes


@pytest.mark.parametrize("malformed", ["foreign_dataset", "repeated_id", "missing_source"])
def test_malformed_destination_fails_before_writes(tmp_path, malformed):
    old = example(1)
    if malformed == "foreign_dataset":
        old["dataset_id"] = uid(999)
    elif malformed == "missing_source":
        old["metadata"].pop("source_scope_id")
    api = API([old, old] if malformed == "repeated_id" else [old])
    with pytest.raises(PipelineError, match="another dataset|repeated an example|source_scope_id"):
        update(tmp_path, api, [example(2)])
    assert not api.writes


def test_duplicate_incoming_cannot_create_second_example(tmp_path):
    api = API()
    with pytest.raises(PipelineError, match="unique, selected source"):
        update(tmp_path, api, [example(1), example(1)])
    assert len(api.writes) == 1


def test_fresh_triage_can_label_unchanged_ordinary_example(tmp_path):
    from test_triage import run

    run(tmp_path)
    incoming, = triage.selected_examples(tmp_path)
    old = copy.deepcopy(incoming)
    old.update(id=uid(1), dataset_id=uid(200))
    old["metadata"].pop("smithtune_triage")
    api = API([old])
    result = triage.create_triaged_dataset(tmp_path, dataset_id=uid(200), confirm=True, runner=api)
    assert result["updated"] == 1
    assert api.writes[0][2]["metadata"]["smithtune_triage"] == incoming["metadata"]["smithtune_triage"]

"""Interrupted curation must reuse evidence and reconcile uncertain remote writes."""

import json

import pytest

from smithtune import checkpoint, cli, curation, dataset, dataset_import, dataset_workflow, triage, triage_source
from smithtune.dataset_artifacts import load_conversation
from smithtune.providers.base import PipelineError
from test_dataset_workflow import run as workflow
from test_dataset_import import API, example, receipt, uid, update
from test_triage import API as TriageAPI, judge_call, source


def test_completed_stages_resume_without_source_or_destination_calls(tmp_path):
    api = TriageAPI()
    workflow(tmp_path, api, "pull")
    workflow(tmp_path, api, "push", name="saved", confirm=True)
    api.calls.clear()
    api.failure = lambda *_: pytest.fail("completed workflow made a remote call")
    assert workflow(tmp_path, api, "resume", confirm=True)["status"] == "complete"
    assert api.calls == []
    units = [load_conversation(path) for path in (tmp_path / "conversations").glob("*.json")]
    assert len(units) == 1 and all(b["tools"] == [] for unit in units for b in unit["example"]["metadata"]["smithtune_source"]["assistant_runs"])


@pytest.mark.parametrize("operation", ["dataset", "example"])
def test_lost_upload_response_is_adopted_by_saved_id(tmp_path, operation):
    api = TriageAPI()
    workflow(tmp_path, api, "pull")
    failed = False

    def lost_response(command, **kwargs):
        nonlocal failed
        response = api(command, **kwargs)
        if not failed and command[2] == f"/api/v1/{operation}s":
            failed = True
            raise PipelineError("connection lost after server saved write")
        return response

    with pytest.raises(PipelineError, match="import incomplete"):
        workflow(tmp_path, lost_response, "push", name="saved", confirm=True)
    result = workflow(tmp_path, api, "resume", confirm=True)
    assert result["created"] == 1
    assert len(api.datasets) == len(api.imported) == 1
    assert sum(command[2] == f"/api/v1/{operation}s" for command, _ in api.calls) == 1


def test_dataset_absent_after_failed_creation_retries_same_id(tmp_path):
    api = TriageAPI()
    workflow(tmp_path, api, "pull")
    api.failure = lambda command: (_ for _ in ()).throw(PipelineError("HTTP 504")) if command[2] == "/api/v1/datasets" else None
    with pytest.raises(PipelineError):
        workflow(tmp_path, api, "push", name="saved", confirm=True)
    saved = json.loads((tmp_path / "dataset-import.json").read_text())
    pending_id = saved["pending_write"]["id"]
    api.failure = None
    assert workflow(tmp_path, api, "resume", confirm=True)["dataset_id"] == pending_id
    posts = [json.loads(body)["id"] for command, body in api.calls if command[2] == "/api/v1/datasets"]
    assert posts == [pending_id, pending_id]


@pytest.mark.parametrize("operation", ["POST", "PATCH"])
def test_lost_example_response_reconciles_without_repeating_write(tmp_path, operation):
    api = API([example(1)] if operation == "PATCH" else [])

    def lost_response(command, **kwargs):
        result = api(command, **kwargs)
        if command[command.index("--method") + 1] == operation and command[2].startswith("/api/v1/examples"):
            raise PipelineError("HTTP 504")
        return result

    incoming = [example(1, 2)]
    with pytest.raises(PipelineError):
        update(tmp_path, lost_response, incoming)
    result = update(tmp_path, api, incoming)
    assert result["updated" if operation == "PATCH" else "created"] == 1
    assert len(api.writes) == 1 and receipt(tmp_path)["pending_write"] is None


@pytest.mark.parametrize("change", ["dataset", "source", "messages", "metadata"])
def test_uncertain_write_cannot_adopt_changed_remote_content(tmp_path, change):
    api = API()

    def lost_response(command, **kwargs):
        result = api(command, **kwargs)
        if command[command.index("--method") + 1] == "POST" and command[2] == "/api/v1/examples":
            raise PipelineError("HTTP 504")
        return result

    with pytest.raises(PipelineError):
        update(tmp_path, lost_response, [example(1)])
    remote = api.existing[0]
    if change == "dataset":
        remote["dataset_id"] = uid(999)
    elif change == "source":
        remote["metadata"]["source_scope_id"] = "another-thread"
    elif change == "messages":
        remote["inputs"]["messages"][0]["content"] = "changed"
    else:
        remote["metadata"]["external-note"] = "edited after write"
    with pytest.raises(PipelineError, match="another dataset|different destination or source|conflicts with changed"):
        update(tmp_path, api, [example(1)])
    assert len(api.writes) == 1 and receipt(tmp_path)["pending_write"] is not None


def test_uncertain_patch_does_not_overwrite_external_edit(tmp_path):
    api = API([example(1)])
    api.before_write = lambda *_: (_ for _ in ()).throw(PipelineError("HTTP 504"))
    with pytest.raises(PipelineError):
        update(tmp_path, api, [example(1, 2)])
    api.before_write = None
    api.existing[0]["metadata"]["external-note"] = "preserve"
    with pytest.raises(PipelineError, match="conflicts with changed"):
        update(tmp_path, api, [example(1, 2)])
    assert len(api.writes) == 1


@pytest.mark.parametrize("change", ["destination", "input", "selection"])
def test_resume_rejects_changed_import_request(tmp_path, change):
    api = API()
    incoming = [example(1)]
    update(tmp_path, api, incoming)
    if change == "destination":
        destination = uid(999)
    else:
        destination = uid(200)
        incoming = [example(1, 2)] if change == "input" else [example(2)]
    count = len(api.calls)
    with pytest.raises(PipelineError, match="changed"):
        dataset_import.update_dataset(uid(100), destination, incoming,
            {dataset._source_key(item, uid(100), None) for item in incoming},
            tmp_path, tmp_path / "receipt.json", runner=api)
    assert len(api.calls) == count


def test_triage_checkpoint_reuses_complete_units_and_votes(tmp_path):
    api = TriageAPI()
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"})

    def fail(command, **kwargs):
        if command[2] == "/v1/trajectory" and json.loads(kwargs["input"]).get("trace_id") == uid(3):
            raise KeyboardInterrupt()

    def interrupted(command, **kwargs):
        fail(command, **kwargs)
        return api(command, **kwargs)
    with pytest.raises(KeyboardInterrupt):
        triage_source.snapshot(source(), tmp_path, runner=interrupted)
    assert len(checkpoint.load(tmp_path)["downloads"]) == 1
    api.failure = None
    api.calls.clear()
    triage.run_triage(source(), tmp_path, runner=api, judge_call=judge_call, confirm=True)
    paths = [command[2] for command, _ in api.calls]
    assert all("/traces/" + uid(n) not in path for path in paths for n in (1, 2))
    assert paths.count("/v1/trajectory") == 1
    assert not (tmp_path / "download").exists()
    manifest = json.loads((tmp_path / "snapshot.json").read_text())
    assert "units" not in manifest and "traces" not in manifest
    assert len(list((tmp_path / "conversations").glob("*.json"))) == 2
    result = triage.run_triage(source(), tmp_path,
        runner=lambda *_a, **_kw: pytest.fail("refetched source"),
        judge_call=lambda *_: pytest.fail("repeated vote"), confirm=True)
    assert result["kept"] == 2


def test_pull_repeat_source_flags_uses_saved_default_window(tmp_path, monkeypatch, capsys):
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *a, **kw: original(*a, **kw, runner=TriageAPI(), judge_call=judge_call))
    monkeypatch.setattr(curation, "_utc_now", lambda: "2026-09-03T00:00:00+00:00")
    args = ["dataset", "pull", str(tmp_path), "--workspace-id", uid(100), "--project-id", uid(101)]
    cli.main(args)
    capsys.readouterr()
    before = (tmp_path / "snapshot.json").read_bytes()
    monkeypatch.setattr(curation, "_utc_now", lambda: "2026-09-04T00:00:00+00:00")
    cli.main(args)
    assert (tmp_path / "snapshot.json").read_bytes() == before


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json", "conversations/nested/file.json"])
def test_checkpoint_rejects_unsafe_paths(tmp_path, path):
    with pytest.raises(PipelineError, match="file reference"):
        checkpoint.read_file(tmp_path, path)


def test_checkpoint_rejects_changed_download_before_upload(tmp_path):
    api = TriageAPI()
    workflow(tmp_path, api, "pull")
    path, = (tmp_path / "conversations").glob("*.json")
    unit = json.loads(path.read_text())
    unit["example"]["inputs"]["messages"][0]["content"] = "tampered"
    path.write_text(json.dumps(unit))
    api.calls.clear()
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        workflow(tmp_path, api, "push", name="saved", confirm=True)
    assert not api.calls


def test_saved_destination_id_cannot_redirect_an_existing_import(tmp_path):
    api = API()
    update(tmp_path, api, [example(1)])
    saved = receipt(tmp_path)
    saved["dataset_id"] = uid(999)
    (tmp_path / "receipt.json").write_text(json.dumps(saved))
    api.calls.clear()
    with pytest.raises(PipelineError, match="destination or selection changed"):
        update(tmp_path, api, [example(1)])
    assert not api.calls

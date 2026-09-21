import copy
import json

import pytest

from smithtune import checkpoint, cli, dataset_workflow as workflow, triage, triage_source
from smithtune.artifacts import output_lock
from smithtune.providers.base import PipelineError
from test_triage import API, judge_call, uid


SOURCE = {"workspace_id": uid(100), "project_id": uid(101),
          "start_time": "2026-09-02T00:00:00Z", "end_time": "2026-09-03T00:00:00Z"}
FILTER = 'and(eq(feedback_key,"correctness"),gte(feedback_score,0.9))'


def run(directory, api, command="create", *, confirm=False, judge=judge_call, **options):
    if command in {"create", "pull"} and not (directory / "checkpoint.json").exists():
        options = {**SOURCE, **options}
    return workflow.run(command, directory, confirm=confirm, runner=api, judge_call=judge, **options)


def no_judge(*_args):
    pytest.fail("unexpected paid judge call")


@pytest.mark.parametrize("concurrency", [None, 16])
def test_filter_only_create_preview_and_flagless_resume(tmp_path, concurrency):
    api = API()
    preview = run(tmp_path, api, name="selected", filter=FILTER, concurrency=concurrency, judge=no_judge)
    assert preview["status"] == "preview" and preview["selection"]["mode"] == "filters"
    assert preview["created"] == preview["eligible"] == 1
    assert preview["pending_stages"] == ["push"]
    assert api.datasets == {} and api.imported == []
    root_query = next(json.loads(body) for command, body in api.calls if command[2] == "/api/v2/runs/query" and body)
    assert FILTER in root_query["filter"]
    assert not (tmp_path / "plan.json").exists()
    api.calls.clear()
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["status"] == "complete" and result["created"] == 1
    assert result["pending_stages"] == []
    assert all(command[2] in {"/api/v1/datasets", "/api/v1/examples"} for command, _ in api.calls)
    assert "smithtune_triage" not in api.imported[0]["metadata"]
    api.calls.clear()
    assert run(tmp_path, api, "resume", confirm=True, judge=no_judge)["status"] == "complete"
    assert api.calls == []


@pytest.mark.parametrize("options", [{}, {"filter": FILTER, "rules": ["Keep solutions grounded in documentation."]},
                                     {"filter": FILTER, "judges": ["gpt-5.6-terra"]}])
def test_create_council_preview_never_judges_or_uploads(tmp_path, options):
    api = API()
    preview = run(tmp_path, api, name="selected", judge=no_judge, **options)
    assert preview["status"] == "preview" and preview["selection"]["mode"] == "council"
    assert preview["pending_stages"] == ["triage", "push"]
    assert not api.datasets and not api.imported
    preview_settings = copy.deepcopy(checkpoint.load(tmp_path)["workflow"]["council"])
    result = run(tmp_path, api, "create", confirm=True)
    assert result["status"] == "complete" and result["pending_stages"] == []
    assert "smithtune_triage" in api.imported[0]["metadata"]
    assert checkpoint.load(tmp_path)["workflow"]["council"] == preview_settings
    api.calls.clear()
    run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert api.calls == []


def test_staged_pull_triage_push_uses_only_local_evidence(tmp_path):
    api = API()
    pulled = run(tmp_path, api, "pull", judge=no_judge)
    assert pulled["status"] == "complete" and pulled["downloaded"] == 1
    assert pulled["pending_stages"] == []
    api.calls.clear()
    run(tmp_path, api, "triage", judge=no_judge, rules=["The answer completes the request."])
    assert api.calls == []
    run(tmp_path, api, "triage", confirm=True)
    assert api.calls == []
    preview = run(tmp_path, api, "push", name="staged", judge=no_judge)
    assert preview["status"] == "preview" and not api.imported
    result = run(tmp_path, api, "push", confirm=True, judge=no_judge)
    assert result["created"] == 1 and len(api.datasets) == len(api.imported) == 1


def test_pull_then_push_does_not_require_council(tmp_path):
    api = API()
    run(tmp_path, api, "pull", judge=no_judge)
    result = run(tmp_path, api, "push", name="raw", confirm=True, judge=no_judge)
    assert result["selection"]["mode"] == "unreviewed"
    assert result["created"] == 1 and not (tmp_path / "judgments.jsonl").exists()


def test_no_triage_is_explicit_and_saved(tmp_path):
    api = API()
    preview = run(tmp_path, api, name="raw", no_triage=True, judge=no_judge)
    assert preview["selection"]["mode"] == "unreviewed" and preview["pending_stages"] == ["push"]
    assert run(tmp_path, api, confirm=True, judge=no_judge)["created"] == 1


@pytest.mark.parametrize("option", [{"rules": ["good"]}, {"judges": ["gpt-5.6-terra"]}, {"rubric_path": "unused.md"}])
def test_no_triage_cannot_discard_judging_criteria(tmp_path, option):
    api = API()
    with pytest.raises(PipelineError, match="no-triage conflicts"):
        run(tmp_path, api, no_triage=True, confirm=True, **option)
    assert api.calls == []


def test_existing_council_plan_cannot_be_bypassed_by_push(tmp_path):
    api = API()
    run(tmp_path, api, name="reviewed")
    with pytest.raises(PipelineError, match="judging is incomplete"):
        run(tmp_path, api, "push", confirm=True)
    with pytest.raises(PipelineError, match="no-triage conflicts"):
        run(tmp_path, api, no_triage=True, confirm=True)
    assert not api.datasets


@pytest.mark.parametrize("command", ["create", "triage"])
def test_cli_rubric_is_frozen_for_partial_resume_and_upload(tmp_path, monkeypatch, capsys, command):
    api, seen = API(), []
    directory = tmp_path / "run"
    rubric_path = tmp_path / "rubric.md"
    rubric = '# Selection\r\nKeep supported answers, including “no”.\r\n\r\nDrop invented results.\r\n'
    rules = ["Keep useful outcomes.", "Require evidence for claimed success."]
    rubric_path.write_text("Draft criteria", encoding="utf-8")

    def judge(slot, prompt, tokens):
        seen.append((slot["name"], copy.deepcopy(prompt)))
        if slot["name"] == "judge-2" and sum(name == "judge-2" for name, _ in seen) == 1:
            return {"keep": 1}  # Leave one vote incomplete.
        return judge_call(slot, prompt, tokens)

    original = workflow.run
    monkeypatch.setattr(workflow, "run", lambda *args, **kwargs: original(*args, **kwargs, runner=api, judge_call=judge))
    source_args = [item for key, value in SOURCE.items() for item in ("--" + key.replace("_", "-"), value)]
    if command == "triage":
        cli.main(["dataset", "pull", str(directory), *source_args])
        capsys.readouterr()
    args = ["dataset", command, str(directory), "--rubric", str(rubric_path), "--attempts", "1", "--concurrency", "1"]
    if command == "create":
        args += [*source_args, "--name", "selected", "--filter", FILTER]
    cli.main(args)
    assert json.loads(capsys.readouterr().out)["selection"]["mode"] == "council"
    # A preview can replace criteria before votes, without changing the source.
    rubric_path.write_bytes(rubric.encode("utf-8"))
    cli.main(["dataset", command, str(directory), "--rubric", str(rubric_path),
              *[item for rule in rules for item in ("--rule", rule)]])
    plan = json.loads(capsys.readouterr().out)["triage"]
    assert plan["selection_rubric"] == rubric and not seen and not api.imported
    assert checkpoint.load(directory)["workflow"]["council"]["selection_rubric"] == rubric
    frozen = triage_source.load_snapshot(directory)
    trajectory, = triage_source.conversation_trajectories(frozen)
    expected = triage.judge_messages(trajectory, triage.rubric_text() + "\nTask-specific selection rubric:\n" + rubric, rules)
    assert expected[0]["content"].startswith(triage.rubric_text())
    assert json.loads(expected[1]["content"])["untrusted_trajectory"] == trajectory["messages"]
    rubric_path.write_text("Changed criteria", encoding="utf-8")
    api.calls.clear()
    with pytest.raises(SystemExit) as incomplete:
        cli.main(["dataset", command, str(directory), "--confirm"])
    assert incomplete.value.code == 1
    assert json.loads(capsys.readouterr().out)["triage"]["incomplete"] == 1
    assert len(seen) == 3 and not api.calls and not api.imported
    saved = {name: (directory / name).read_bytes() for name in
             ("checkpoint.json", "plan.json", "snapshot.json", "triage-config.json", "judgments.jsonl")}
    completed = {vote["judge"]: vote for vote in map(json.loads, saved["judgments.jsonl"].splitlines()) if vote["status"] == "complete"}
    for override in (["--rubric", str(rubric_path)], ["--rule", "Different policy"]):
        for confirm in ([], ["--confirm"]):
            with pytest.raises(SystemExit) as changed:
                cli.main(["dataset", command, str(directory), *override, *confirm])
            assert changed.value.code == 2
            assert "conflict" in capsys.readouterr().err
            assert len(seen) == 3 and not api.calls
            assert all((directory / name).read_bytes() == content for name, content in saved.items())
    rubric_path.unlink()
    cli.main(["dataset", "resume", str(directory), "--confirm"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert len(seen) == 4 and seen[-1][0] == "judge-2"
    assert all(prompt == expected for _, prompt in seen)
    votes = {vote["judge"]: vote for vote in map(json.loads, (directory / "judgments.jsonl").read_text().splitlines())}
    assert all(votes[name] == vote for name, vote in completed.items())
    if command == "triage":
        cli.main(["dataset", "push", str(directory), "--name", "selected", "--confirm"])
        capsys.readouterr()
    assert api.imported[0]["inputs"]["messages"] == trajectory["messages"]
    api.calls.clear()
    cli.main(["dataset", "resume", str(directory), "--confirm"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert len(seen) == 4 and not api.calls


@pytest.mark.parametrize("content", [None, b"\xff", b"", b" \n\t", "directory"])
def test_invalid_rubric_fails_before_download(tmp_path, content):
    api = API()
    rubric_path = tmp_path / "rubric.md"
    if content == "directory":
        rubric_path.mkdir()
    elif content is not None:
        rubric_path.write_bytes(content)
    with pytest.raises(PipelineError, match="rubric"):
        run(tmp_path / "run", api, rubric_path=rubric_path, filter=FILTER, confirm=True, judge=no_judge)
    assert not api.calls
    assert not (tmp_path / "run" / "plan.json").exists()


def test_judge_failure_blocks_push_and_resume_reuses_votes(tmp_path):
    api = API()
    calls = []

    def fail_one(judge, *args):
        calls.append(judge["name"])
        if judge["name"] == "judge-2":
            raise PipelineError("transient failure")
        return judge_call(judge, *args)

    result = run(tmp_path, api, name="reviewed", confirm=True, judge=fail_one, attempts=1)
    assert result["status"] == "incomplete" and result["pending_stages"] == ["triage", "push"]
    assert not api.datasets
    calls.clear()

    def fixed(judge, *args):
        calls.append(judge["name"])
        return judge_call(judge, *args)

    assert run(tmp_path, api, "resume", confirm=True, judge=fixed)["created"] == 1
    assert calls == ["judge-2"]


def test_interrupted_pull_can_resume_without_source_flags(tmp_path):
    api = API()
    fired = False

    def interrupted(command):
        nonlocal fired
        if command[2] == "/v1/trajectory" and not fired:
            fired = True
            raise KeyboardInterrupt()

    api.failure = interrupted
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, api, filter=FILTER, name="resume-me")
    api.calls.clear()
    preview = run(tmp_path, api, "resume", judge=no_judge)
    assert preview["pending_stages"] == ["pull", "push"] and api.calls == []
    api.failure = None
    assert run(tmp_path, api, "resume", confirm=True, judge=no_judge)["created"] == 1


def test_upload_response_loss_uses_existing_recovery(tmp_path):
    api = API()
    run(tmp_path, api, name="saved", filter=FILTER)
    failed = False

    def lost_response(command, **kwargs):
        nonlocal failed
        response = api(command, **kwargs)
        if not failed and command[2] == "/api/v1/examples":
            failed = True
            raise PipelineError("HTTP 504")
        return response

    with pytest.raises(PipelineError, match="dataset resume"):
        run(tmp_path, lost_response, "resume", confirm=True, judge=no_judge)
    assert run(tmp_path, api, "resume", confirm=True, judge=no_judge)["created"] == 1
    assert len(api.imported) == 1 and len(api.datasets) == 1


@pytest.mark.parametrize("change", [{"project_id": uid(999)}, {"filter": "eq(error,true)"}, {"limit": 5},
                                    {"name": "different"}])
def test_saved_selection_and_destination_are_fixed(tmp_path, change):
    api = API()
    run(tmp_path, api, name="original", filter=FILTER)
    api.calls.clear()
    with pytest.raises(PipelineError, match="conflicts"):
        run(tmp_path, api, **change)
    assert api.calls == []


def test_council_rules_can_change_in_preview_but_not_after_votes(tmp_path):
    api = API()
    run(tmp_path, api, name="reviewed")
    run(tmp_path, api, "triage", rules=["Keep accurate answers."])
    run(tmp_path, api, "triage", confirm=True)
    with pytest.raises(PipelineError, match="saved votes"):
        run(tmp_path, api, "triage", rules=["Keep different answers."], confirm=True)


def test_no_council_can_be_added_after_upload_started(tmp_path):
    api = API()
    run(tmp_path, api, name="filtered", filter=FILTER, confirm=True, judge=no_judge)
    with pytest.raises(PipelineError, match="after upload started"):
        run(tmp_path, api, "triage", confirm=True)


def test_all_dropped_avoids_empty_dataset_and_completes(tmp_path):
    api = API()
    result = run(tmp_path, api, name="empty", confirm=True, judge=lambda *_: {"keep": 0, "reason": "not useful"})
    assert result["eligible"] == result["example_count"] == 0
    assert result["pending_stages"] == [] and not api.datasets
    assert run(tmp_path, api, "resume", judge=no_judge)["status"] == "complete"


def test_all_invalid_filters_avoid_judging_and_upload(tmp_path):
    api = API()
    api.trajectory_pages[None]["messages"].append({"role": "system", "content": "misplaced"})
    result = run(tmp_path, api, name="invalid", filter=FILTER, confirm=True, judge=no_judge)
    assert result["rejected"] == result["downloaded"] == 1
    assert result["example_count"] == 0 and not api.datasets


def test_distinct_trajectory_limit_stops_query_pagination(tmp_path):
    api = API()
    api.root_pages = [[{"trace_id": uid(1), "thread_id": "conversation-a"},
                       {"trace_id": uid(2), "thread_id": "conversation-a"},
                       {"trace_id": uid(3), "thread_id": None}],
                      [{"trace_id": uid(4), "thread_id": None}]]
    for page in api.root_pages:
        for root in page:
            root["start_time"] = "2026-09-02T00:00:00Z"
    result = run(tmp_path, api, "pull", limit=2)
    assert result["downloaded"] == 2
    root_queries = [json.loads(body) for command, body in api.calls if command[2] == "/api/v2/runs/query" and body]
    assert len(root_queries) == 1 and "cursor" not in root_queries[0]


def test_partial_legacy_direct_import_can_use_resume(tmp_path):
    from test_curation import API as DirectAPI, create, root

    api = DirectAPI([[root(1, "a")]])
    api.failure = lambda path, _: (_ for _ in ()).throw(PipelineError("HTTP 504")) if path == "/api/v1/examples" else None
    with pytest.raises(PipelineError):
        create(tmp_path, api)
    api.failure = None
    assert run(tmp_path, api, "resume", confirm=True, judge=no_judge)["created"] == 1
    assert len(api.examples) == 1 and len(api.datasets) == 1


def test_pre_workflow_triage_snapshot_can_be_pushed(tmp_path):
    from test_triage import run as old_triage

    api = API()
    old_triage(tmp_path, api)
    api.calls.clear()
    result = run(tmp_path, api, "push", name="legacy", confirm=True, judge=no_judge)
    assert result["created"] == 1
    assert all(command[2] in {"/api/v1/datasets", "/api/v1/examples"} for command, _ in api.calls)


def test_same_directory_lock_blocks_all_stages(tmp_path):
    api = API()
    with output_lock(tmp_path), pytest.raises(PipelineError, match="another smithtune operation"):
        run(tmp_path, api)
    assert not api.calls


@pytest.mark.parametrize("args,hint", [
    (["create", "--run-dir", "old"], "dataset create DIR"),
    (["create", "--output=old.json"], "dataset pull DIR"),
    (["create", "--triage-dir", "old"], "dataset push DIR"),
    (["triage", "--workspace-id", uid(100)], "dataset pull DIR"),
])
def test_removed_flags_show_replacement(args, hint, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["dataset", *args])
    assert exc.value.code == 2
    assert hint in capsys.readouterr().err


@pytest.mark.parametrize("command", ["pull", "triage", "push", "create", "resume"])
def test_command_help_is_available(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["dataset", command, "--help"])
    assert exc.value.code == 0 and "directory" in capsys.readouterr().out


def test_pre_workflow_interrupted_triage_upload_is_resumed(tmp_path):
    from test_triage import run as old_triage

    api = API()
    old_triage(tmp_path, api)
    api.failure = lambda command: (_ for _ in ()).throw(PipelineError("HTTP 504")) if command[2] == "/api/v1/examples" else None
    with pytest.raises(PipelineError):
        triage.create_triaged_dataset(tmp_path, "old-upload", confirm=True, runner=api)
    api.failure = None
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["created"] == 1 and result["pending_stages"] == []
    assert len(api.datasets) == len(api.imported) == 1


def test_operator_skill_update_preserves_partial_council_votes(tmp_path):
    from smithtune.inference_contract import json_sha256

    api = API()
    def fail_one(judge, *args):
        return {} if judge["name"] == "judge-2" else judge_call(judge, *args)
    run(tmp_path, api, name="legacy-votes", confirm=True, judge=fail_one, attempts=1)
    # Reproduce the previous release's identity, which hashed the whole skill.
    path = tmp_path / "triage-config.json"
    identity = json.loads(path.read_text())
    identity["skill_sha256"] = next(iter(triage.LEGACY_COORDINATOR_SKILLS))
    path.write_text(json.dumps(identity))
    digest = json_sha256(identity)
    records = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    for record in records:
        record["identity_sha256"] = digest
    (tmp_path / "judgments.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["identity_sha256"] = digest
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    calls = []
    def fixed(judge, *args):
        calls.append(judge["name"])
        return judge_call(judge, *args)
    result = run(tmp_path, api, "resume", confirm=True, judge=fixed)
    assert result["created"] == 1 and calls == ["judge-2"]
    assert json.loads(path.read_text()) == identity


def test_parallel_pull_preserves_inflight_downloads_and_order_on_resume(tmp_path, monkeypatch):
    from threading import Barrier, Lock
    from smithtune import triage_source

    api = API()
    api.root_pages = [[{"trace_id": uid(n), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"}
                       for n in range(1, 9)]]
    fetch = triage_source._fetch_trajectory
    barrier, lock = Barrier(4), Lock()
    active = peak = 0
    seen = []

    def interrupted(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(args[2]["id"])
        try:
            barrier.wait(timeout=5)
            if args[2]["id"] == uid(1):
                raise PipelineError("source read interrupted")
            return fetch(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(triage_source, "_fetch_trajectory", interrupted)
    with pytest.raises(PipelineError, match="source read interrupted"):
        run(tmp_path, api, "pull", concurrency=16)
    assert peak == 4 and len(seen) == 4
    assert len(checkpoint.load(tmp_path)["downloads"]) == 3
    assert not (tmp_path / "snapshot.json").exists()
    monkeypatch.setattr(triage_source, "_fetch_trajectory", fetch)
    api.calls.clear()
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["status"] == "complete" and result["downloaded"] == 8
    fetched = [json.loads(body)["trace_id"] for command, body in api.calls if command[2] == "/v1/trajectory"]
    assert set(fetched) == {uid(n) for n in (1, 5, 6, 7, 8)}
    assert not any(command[2] == "/api/v2/runs/query" for command, _ in api.calls)
    frozen = triage_source.load_snapshot(tmp_path)
    assert [unit["example"]["metadata"]["source_trace_id"] for unit in frozen["units"]] == [uid(n) for n in range(1, 9)]


def test_legacy_interrupted_triage_download_keeps_council_intent(tmp_path):
    from test_triage import source

    api = API()
    api.failure = lambda command: (_ for _ in ()).throw(KeyboardInterrupt()) if command[2] == "/v1/trajectory" else None
    with pytest.raises(KeyboardInterrupt):
        triage.run_triage(source(), tmp_path, runner=api, judge_call=no_judge, dry_run=True)
    assert not (tmp_path / "plan.json").exists()
    api.calls.clear()
    preview = run(tmp_path, api, "resume", judge=no_judge)
    assert preview["pending_stages"] == ["pull", "triage"] and not api.calls
    api.failure = None
    result = run(tmp_path, api, "resume", confirm=True)
    assert result["status"] == "complete" and result["triage"]["status"] == "complete"
    assert result["pending_stages"] == [] and not api.datasets

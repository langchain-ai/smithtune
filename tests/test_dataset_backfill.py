import json
from threading import Event
from uuid import UUID

import pytest

from smithtune import cli, curation, dataset_workflow, triage_source
from smithtune.providers.base import PipelineError
from test_dataset_workflow import SOURCE, no_judge, run
from test_triage import API, uid


class Candidates(API):
    def __init__(self, pages, *, invalid=()):
        super().__init__()
        self.root_pages = [[{"trace_id": uid(n), "thread_id": None,
                            "start_time": "2026-09-02T00:00:00Z"} for n in page] for page in pages]
        self.invalid = set(invalid)

    def __call__(self, command, **kwargs):
        response = super().__call__(command, **kwargs)
        body = json.loads(kwargs.get("input") or "{}")
        if command[2] == "/v1/trajectory" and UUID(body["trace_id"]).int in self.invalid:
            response.stdout = json.dumps({"items": [], "next_cursor": None})
        return response

    def reads(self, endpoint):
        return [json.loads(body) for command, body in self.calls if command[2] == endpoint]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_backfill_reaches_target_without_extra_downloads(tmp_path, concurrency):
    api = Candidates([[10, 11], [12, 13], [14, 15]], invalid=[10, 12])
    result = run(tmp_path, api, no_triage=True, target_count=3, max_candidates=10, concurrency=concurrency,
                 filter='eq(name,"reviewer")', judge=no_judge)
    summary = result["download_summary"]
    assert summary["examined"] == 5 and summary["usable"] == 3 and summary["excluded"] == 2
    assert summary["target_met"] and summary["stop_reason"] == "target_reached"
    # Empty trajectories are re-read in case indexing lags ingestion; no extra candidates are fetched.
    reads = [body["trace_id"] for body in api.reads("/v1/trajectory")]
    assert sorted(set(reads)) == [uid(n) for n in range(10, 15)]
    assert {trace: reads.count(trace) for trace in (uid(10), uid(12))} == {
        uid(10): curation.EMPTY_TRAJECTORY_ATTEMPTS, uid(12): curation.EMPTY_TRAJECTORY_ATTEMPTS}
    assert all(reads.count(uid(n)) == 1 for n in (11, 13, 14))
    queries = api.reads("/api/v2/runs/query")
    assert [body.get("cursor") for body in queries] == [None, "1", "2"]
    assert all('eq(name,"reviewer")' in body["filter"] and
               body["min_start_time"] == SOURCE["start_time"].replace("Z", "+00:00") and
               body["max_start_time"] == SOURCE["end_time"].replace("Z", "+00:00") for body in queries)
    assert not api.imported and not (tmp_path / "judgments.jsonl").exists()
    run(tmp_path, api, "push", name="usable", confirm=True, judge=no_judge)
    assert len(api.imported) == 3


@pytest.mark.parametrize("cap,reason,examined", [(3, "candidate_cap", 3), (10, "source_exhausted", 4)])
def test_backfill_reports_shortfall(tmp_path, capsys, cap, reason, examined):
    api = Candidates([[10, 11], [12, 13]], invalid=[10, 11, 12])
    result = run(tmp_path, api, no_triage=True, target_count=3, max_candidates=cap, judge=no_judge)
    summary = result["download_summary"]
    assert result["status"] == "complete" and not result["pending_stages"]
    assert summary["examined"] == examined and summary["stop_reason"] == reason
    assert summary["usable"] == (1 if examined == 4 else 0) and not summary["target_met"]
    output = capsys.readouterr().err
    assert f"Examined {examined} candidates" in output and "requested 3" in output
    api.calls.clear()
    assert run(tmp_path, api, "resume")["collection"]["eligible"] == summary["usable"]
    assert not api.calls


def test_duplicate_threads_do_not_consume_candidate_cap(tmp_path):
    api = API()
    api.root_pages = [
        [{"trace_id": uid(1), "thread_id": "conversation-a", "start_time": "2026-09-02T00:00:00Z"}],
        [{"trace_id": uid(2), "thread_id": "conversation-a", "start_time": "2026-09-02T00:00:00Z"},
         {"trace_id": uid(3), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"}],
    ]
    api.trajectory_pages = {None: {"messages": [], "next_cursor": None}}
    result = run(tmp_path, api, no_triage=True, target_count=1, max_candidates=2, judge=no_judge)
    summary = result["download_summary"]
    assert summary["examined"] == 2 and summary["usable"] == 1 and summary["target_met"]
    assert summary["threads"] == 1 and summary["standalone_traces"] == 1


@pytest.mark.parametrize("failure", ["next_page", "trajectory"])
def test_backfill_resume_reuses_pages_and_completed_downloads(tmp_path, failure):
    api = Candidates([[10, 11], [12, 13]], invalid=[10])
    interrupted = False

    def runner(command, **kwargs):
        nonlocal interrupted
        body = json.loads(kwargs.get("input") or "{}")
        should_fail = ((failure == "next_page" and command[2] == "/api/v2/runs/query" and body.get("cursor") == "1") or
                       (failure == "trajectory" and command[2] == "/v1/trajectory" and body.get("trace_id") == uid(11)))
        if should_fail and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return api(command, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, runner, no_triage=True, target_count=2, max_candidates=4, concurrency=1, judge=no_judge)
    assert not (tmp_path / "snapshot.json").exists()
    api.calls.clear()
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["download_summary"]["target_met"]
    assert result["download_summary"]["examined"] == 3
    assert all(body.get("cursor") == "1" for body in api.reads("/api/v2/runs/query"))
    reads = {body["trace_id"] for body in api.reads("/v1/trajectory")}
    assert uid(10) not in reads  # The saved exclusion is not downloaded again.
    assert (uid(11) in reads) == (failure == "trajectory")
    assert uid(12) in reads and uid(13) not in reads
    assert len(triage_source.load_snapshot(tmp_path)["units"]) == 3


def test_resume_keeps_other_workers_completed_downloads(tmp_path):
    api = Candidates([[10, 11, 12, 13]])
    neighbor_finished = Event()

    def runner(command, **kwargs):
        body = json.loads(kwargs.get("input") or "{}")
        if command[2] == "/v1/trajectory" and body.get("trace_id") == uid(10):
            assert neighbor_finished.wait(5)
            raise PipelineError("source temporarily unavailable")
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory" and body.get("trace_id") == uid(12):
            neighbor_finished.set()
        return response

    with pytest.raises(PipelineError, match="source temporarily unavailable"):
        run(tmp_path, runner, no_triage=True, target_count=3, max_candidates=4, concurrency=3, judge=no_judge)
    api.calls.clear()
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["download_summary"]["target_met"] and result["download_summary"]["examined"] == 3
    assert not api.reads("/api/v2/runs/query")
    assert [body["trace_id"] for body in api.reads("/v1/trajectory")] == [uid(10)]


@pytest.mark.parametrize("bad_content", ["multimodal", "system", "missing_tools"])
def test_backfill_replaces_structurally_unusable_content(tmp_path, bad_content):
    api = Candidates([[10, 11]])

    def runner(command, **kwargs):
        response = api(command, **kwargs)
        body = json.loads(kwargs.get("input") or "{}")
        if command[2] == "/v1/trajectory" and body.get("trace_id") == uid(10):
            page = json.loads(response.stdout)
            assistant = page["items"][-1]["message"]
            if bad_content == "missing_tools":
                assistant.pop("available_tools")
            elif bad_content == "system":
                page["items"][1]["message"]["role"] = "system"
            else:
                assistant["content"] = [{"type": "image", "url": "https://example.invalid/image.png"}]
            response.stdout = json.dumps(page)
        return response

    result = run(tmp_path, runner, no_triage=True, target_count=1, max_candidates=2, judge=no_judge)
    assert result["download_summary"]["usable"] == 1 and result["download_summary"]["excluded"] == 1
    assert result["download_summary"]["examined"] == 2
    pushed = run(tmp_path, api, "push", name="usable", confirm=True, judge=no_judge)
    assert pushed["rejected"] == 1 and len(api.imported) == 1
    assert api.imported[0]["metadata"]["source_trace_id"] == uid(11)


@pytest.mark.parametrize("options", [{"target_count": 0}, {"max_candidates": 0},
                                     {"target_count": -1}, {"max_candidates": 2001}])
def test_invalid_bounds_fail_before_remote_reads(tmp_path, options):
    api = API()
    with pytest.raises(PipelineError, match="target count|max candidates"):
        run(tmp_path, api, **options)
    assert not api.calls


def test_cli_passes_explicit_bounds_and_rejects_old_limit(tmp_path, monkeypatch, capsys):
    api = Candidates([[10, 11]], invalid=[10])
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(*args, **kwargs, runner=api))
    args = ["dataset", "pull", str(tmp_path), "--workspace-id", uid(100), "--project-id", uid(101),
            "--start-time", SOURCE["start_time"], "--end-time", SOURCE["end_time"]]
    cli.main([*args, "--no-triage", "--target-count", "1", "--max-candidates", "2"])
    result = json.loads(capsys.readouterr().out)
    assert result["source"]["target_count"] == 1 and result["source"]["max_candidates"] == 2
    assert result["download_summary"]["target_met"]
    with pytest.raises(SystemExit):
        cli.main([*args, "--limit", "2"])


@pytest.mark.parametrize("approved", [0, 1, 99, 100])
def test_small_reviewed_dataset_advisory_does_not_block_upload(tmp_path, capsys, approved):
    count = max(approved, 1)
    api = Candidates([list(range(10, 10 + count))])
    run(tmp_path, api, target_count=count, max_candidates=count)
    preview = run(tmp_path, api, "triage", judges=["gpt-5.6-terra"])
    assert "advisories" not in preview
    api.calls.clear()
    reviewed = run(tmp_path, api, "triage", confirm=True,
                   judge=lambda *_: {"keep": int(approved > 0), "reason": "Recorded evidence supports the decision."})
    assert not api.calls  # Council decisions never trigger backfill.
    capsys.readouterr()
    preview = run(tmp_path, api, "push", name="reviewed")
    expected = 0 < approved < 100
    assert ("advisories" in reviewed) == expected and ("advisories" in preview) == expected
    if expected:
        message, = preview["advisories"]
        assert f"Council approved {approved} " in message
        assert "We recommend collecting more trajectories to improve reliability." in message
        assert "100" not in message and message in capsys.readouterr().err
    pushed = run(tmp_path, api, "push", confirm=True)
    assert pushed["status"] == "complete" and len(api.imported) == approved


def test_unreviewed_dataset_does_not_claim_council_approval(tmp_path):
    api = Candidates([[10]])
    run(tmp_path, api, no_triage=True, target_count=1, max_candidates=1)
    result = run(tmp_path, api, "push", name="unreviewed")
    assert "advisories" not in result

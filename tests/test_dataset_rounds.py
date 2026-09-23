import json

import pytest

from smithtune import triage_source
from smithtune.providers.base import PipelineError
from test_dataset_backfill import Candidates
from test_dataset_workflow import no_judge, run
from test_triage import uid


def judge_for(kept, calls):
    def judge(_slot, prompt, _tokens):
        messages = json.loads(prompt[-1]["content"])["untrusted_trajectory"]
        n = int(messages[-1]["id"].removeprefix("ai-"))
        calls.append(n)
        return {"keep": int(n in kept), "reason": "Decision supported by the recorded evidence."}
    return judge


def review(directory, api, kept, calls):
    return run(directory, api, "triage", confirm=True, judges=["gpt-5.6-terra"], judge=judge_for(kept, calls))


def test_council_rounds_preserve_votes_and_stop_at_approved_target(tmp_path):
    api = Candidates([[10, 11, 12, 13, 14]])
    pulled = run(tmp_path, api, target_count=2, max_candidates=2)
    assert pulled["downloaded"] == 2 and pulled["selection"]["mode"] == "council"
    assert pulled["collection"]["status"] == "needs_review"
    calls = []
    reviewed = review(tmp_path, api, {11, 13}, calls)
    assert reviewed["collection"]["eligible"] == 1 and reviewed["collection"]["status"] == "needs_candidates"
    votes = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    original = list(triage_source.load_snapshot(tmp_path)["units"])
    api.calls.clear()
    pulled = run(tmp_path, api)
    assert pulled["downloaded"] == 4 and pulled["collection"]["round"] == 2
    assert {body["trace_id"] for body in api.reads("/v1/trajectory")} == {uid(12), uid(13)}
    assert not api.reads("/api/v2/runs/query")  # Remaining roots came from the saved page.
    assert list(triage_source.load_snapshot(tmp_path)["units"])[:2] == original
    reviewed = review(tmp_path, api, {11, 13}, calls)
    assert sorted(calls) == [10, 11, 12, 13]
    assert reviewed["collection"]["eligible"] == 2 and reviewed["collection"]["status"] == "target_reached"
    current_votes = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    assert all(vote in current_votes for vote in votes)
    api.calls.clear()
    run(tmp_path, api)
    assert not api.calls
    pushed = run(tmp_path, api, "push", name="approved", confirm=True)
    assert pushed["created"] == 2
    assert {item["metadata"]["source_trace_id"] for item in api.imported} == {uid(11), uid(13)}


def test_council_leaves_unreviewed_candidates_when_target_is_reached(tmp_path):
    api = Candidates([[10, 11, 12]])
    run(tmp_path, api, target_count=1, max_candidates=3)
    calls = []
    result = review(tmp_path, api, {10, 11, 12}, calls)
    assert calls == [10]
    assert result["triage"]["status"] == "complete" and result["triage"]["unreviewed"] == 2
    assert len(triage_source.load_snapshot(tmp_path)["units"]) == 3
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert all("rerun" not in label["reason"] for label in labels)
    run(tmp_path, api, "push", name="one-approved", confirm=True)
    assert len(api.imported) == 1


@pytest.mark.parametrize("no_triage", [False, True])
def test_three_round_limit_keeps_eligible_subset_and_stops_downloads(tmp_path, no_triage):
    api = Candidates([[10, 11, 12, 13]], invalid=[11, 12] if no_triage else [])
    calls = []
    for expected_round in range(1, 4):
        result = run(tmp_path, api, target_count=2, max_candidates=1, no_triage=no_triage) if expected_round == 1 else run(tmp_path, api)
        if not no_triage:
            result = review(tmp_path, api, {10}, calls)
        assert result["collection"]["round"] == expected_round
        assert result["collection"]["status"] == ("round_limit" if expected_round == 3 else "needs_candidates")
    assert "collection limit" in result["message"] and "3 rounds" in result["message"]
    api.calls.clear()
    result = run(tmp_path, api)
    assert not api.calls and result["collection"]["round"] == 3
    run(tmp_path, api, "push", name="subset", confirm=True)
    assert len(api.imported) == 1


def test_no_triage_stops_at_cumulative_structural_target(tmp_path):
    api = Candidates([[10, 11], [12, 13, 14]], invalid=[10])
    result = run(tmp_path, api, target_count=2, max_candidates=2, no_triage=True, judge=no_judge)
    assert result["collection"]["eligible"] == 1 and result["collection"]["status"] == "needs_candidates"
    result = run(tmp_path, api, judge=no_judge)
    assert result["collection"]["round"] == 2 and result["collection"]["status"] == "target_reached"
    assert [body["trace_id"] for body in api.reads("/v1/trajectory")] == [uid(10), uid(11), uid(12)]
    assert "No council review was performed" in result["message"]
    run(tmp_path, api, "push", name="structural", confirm=True, judge=no_judge)
    assert len(api.imported) == 2
    assert not (tmp_path / "judgments.jsonl").exists()
    with pytest.raises(PipelineError, match="no-triage"):
        run(tmp_path, api, "triage")


def test_unfinished_review_cannot_be_bypassed_by_pull_or_mode_change(tmp_path):
    api = Candidates([[10, 11, 12]])
    run(tmp_path, api, target_count=2, max_candidates=1)
    api.calls.clear()
    assert run(tmp_path, api)["collection"]["round"] == 1
    assert not api.calls
    with pytest.raises(PipelineError, match="incomplete"):
        run(tmp_path, api, "push", name="unreviewed", confirm=True)
    with pytest.raises(PipelineError, match="saved review mode"):
        run(tmp_path, api, no_triage=True)
    assert not api.imported


def test_source_exhaustion_stops_before_three_rounds(tmp_path):
    api = Candidates([[10]])
    run(tmp_path, api, target_count=3, max_candidates=2)
    result = review(tmp_path, api, {10}, [])
    assert result["collection"]["status"] == "source_exhausted"
    assert "No unseen matching candidates" in result["message"]
    api.calls.clear()
    run(tmp_path, api)
    assert not api.calls


def test_interrupted_second_pull_resumes_same_round(tmp_path):
    api = Candidates([[10, 11], [12, 13, 14]], invalid=[10])
    run(tmp_path, api, target_count=3, max_candidates=2, no_triage=True)
    def fail(command, **kwargs):
        if command[2] == "/v1/trajectory":
            raise PipelineError("temporary source failure")
        return api(command, **kwargs)
    with pytest.raises(PipelineError, match="temporary source failure"):
        run(tmp_path, fail)
    status = run(tmp_path, api, "resume", judge=no_judge)
    assert status["collection"]["status"] == "downloading" and status["collection"]["round"] == 2
    with pytest.raises(PipelineError, match="download is incomplete"):
        run(tmp_path, api, "push", name="partial", confirm=True, judge=no_judge)
    assert not api.imported
    resumed = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert resumed["collection"]["round"] == 2 and resumed["collection"]["status"] == "target_reached"
    assert len(triage_source.load_snapshot(tmp_path)["units"]) == 4


def test_failed_judge_resumes_without_spending_collection_round(tmp_path):
    api = Candidates([[10, 11]])
    run(tmp_path, api, target_count=2, max_candidates=1)
    def fail(*_):
        raise RuntimeError("temporary judge failure")
    result = run(tmp_path, api, "triage", confirm=True, judges=["gpt-5.6-terra"], attempts=1, judge=fail)
    assert result["status"] == "incomplete" and result["collection"]["status"] == "needs_review"
    api.calls.clear()
    assert run(tmp_path, api)["collection"]["round"] == 1 and not api.calls
    result = run(tmp_path, api, "triage", confirm=True, judge=judge_for({10}, []))
    assert result["collection"]["round"] == 1 and result["collection"]["status"] == "needs_candidates"


def test_all_invalid_first_round_can_add_valid_candidates_and_review(tmp_path):
    api = Candidates([[10, 11]], invalid=[10])
    run(tmp_path, api, target_count=1, max_candidates=1)
    first = review(tmp_path, api, {11}, [])
    assert first["triage"]["kept"] == 0 and first["collection"]["status"] == "needs_candidates"
    run(tmp_path, api)
    calls = []
    result = review(tmp_path, api, {11}, calls)
    assert calls == [11] and result["collection"]["status"] == "target_reached"
    run(tmp_path, api, "push", name="recovered", confirm=True)
    assert len(api.imported) == 1


@pytest.mark.parametrize("changed", ["messages", "tools"])
def test_saved_votes_remain_bound_to_original_trajectory_evidence(tmp_path, changed):
    import copy
    from smithtune import checkpoint, triage

    api = Candidates([[10, 11]])
    run(tmp_path, api, target_count=2, max_candidates=1)
    review(tmp_path, api, {10}, [])
    run(tmp_path, api)
    frozen = triage_source.load_snapshot(tmp_path)
    units = copy.deepcopy(list(frozen["units"]))
    if changed == "messages":
        units[0]["example"]["inputs"]["messages"][-1]["content"] = "Changed evidence"
    else:
        units[0]["example"]["metadata"]["smithtune_source"]["assistant_runs"][0]["tools"] = [
            {"type": "function", "function": {"name": "new_tool", "parameters": {"type": "object", "properties": {}}}},
        ]
    with pytest.raises(PipelineError, match="different trajectory content or tool evidence"):
        triage._run_triage(frozen["source"], tmp_path, dry_run=True, frozen={**frozen, "units": units},
                           **checkpoint.load(tmp_path)["workflow"]["council"])


def test_target_reached_preview_and_resume_need_no_new_votes(tmp_path):
    api = Candidates([[10, 11]])
    run(tmp_path, api, target_count=1, max_candidates=2)
    review(tmp_path, api, {10}, [])
    api.calls.clear()
    preview = run(tmp_path, api, "triage", judge=no_judge)
    assert preview["triage"]["pending_judge_tasks"] == 0
    result = run(tmp_path, api, "resume", confirm=True, judge=no_judge)
    assert result["collection"]["status"] == "target_reached" and not api.calls

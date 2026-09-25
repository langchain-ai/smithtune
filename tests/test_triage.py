import copy
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
from urllib.error import HTTPError

import pytest

from smithtune import cli, curation, dataset, dataset_workflow, triage, triage_judges, triage_source
from smithtune.bindings import evidence_hash, read_bindings, tool_contract
from smithtune.dataset_artifacts import load_conversation
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL
from trajectory_fixtures import items


def uid(n):
    return str(UUID(int=n))


SYSTEM = [{"role": "system", "content": "Answer each request accurately."}]


def messages(n):
    return [{"role": "human", "content": f"question-{n}", "id": f"user-{n}"},
            {"role": "ai", "content": f"answer-{n}", "id": f"ai-{n}"}]


class API:
    """LangSmith trajectory pages and complete source-run evidence."""
    def __init__(self):
        self.calls = []
        self.imported = []
        self.failure = None
        self.datasets = {}
        self.root_pages = [[{"trace_id": uid(2), "thread_id": "conversation-a", "start_time": "2026-09-02T00:00:00Z", "feedback_stats": None}]]
        self.trajectory_pages = {
            None: {"messages": SYSTEM + messages(1), "next_cursor": "next"},
            "next": {"messages": messages(2), "next_cursor": None},
        }
        self.thread_roots = [
            {"id": uid(n), "trace_id": uid(n), "thread_id": "conversation-a",
             "project_id": uid(101), "start_time": f"2026-09-0{n}T00:00:00Z"}
            for n in (1, 2)
        ]

    def __call__(self, command, *, capture=False, input=None):
        self.calls.append((command, input))
        if self.failure:
            self.failure(command)
        assert command[command.index("--workspace") + 1] == uid(100)
        path = command[2]
        body = json.loads(input) if input else json.loads(command[command.index("--body") + 1]) if "--body" in command else None
        if path == "/api/v2/runs/query":
            if body.get("filter", "").startswith("eq(thread_id,"):
                assert body["min_start_time"] == "2026-08-01T00:00:00+00:00"
                return SimpleNamespace(stdout=json.dumps({"items": self.thread_roots, "next_cursor": None}))
            index = int(body.get("cursor", 0))
            value = {"items": self.root_pages[index], "next_cursor": str(index + 1) if index + 1 < len(self.root_pages) else None}
        elif path == "/v1/trajectory":
            assert body["include"] == {"system_messages": True, "tool_definitions": True}
            assert body["format"] == "ui"
            if "thread_id" in body:
                value = self.trajectory_pages[body.get("cursor")]
            else:
                value = {"messages": SYSTEM + messages(UUID(body["trace_id"]).int), "next_cursor": None}
            page = value
            wire = []
            for message in page["messages"]:
                mid = message.get("id", "")
                n = int(mid.split("-")[-1]) if mid.startswith(("ai-", "user-")) else UUID(body.get("trace_id", uid(1))).int
                wire.extend(items([message], trace_id=uid(n), run_id=uid(n + 1000)))
            value = {"items": wire, "next_cursor": page.get("next_cursor")}
        elif path.startswith("/api/v1/sessions/"):
            value = {"id": uid(101), "start_time": "2026-08-01T00:00:00+00:00"}
        elif path == "/api/v1/datasets":
            value = copy.deepcopy(body)
            self.datasets[body["id"]] = value
        elif path == "/api/v1/examples":
            self.imported.append(body)
            value = {"id": body["id"]}
        elif path.endswith("/versions?limit=1"):
            value = [{"as_of": "2026-09-15T00:00:00+00:00"}] if self.imported else []
        elif path.startswith("/api/v1/datasets/"):
            value = self.datasets.get(path.rsplit("/", 1)[1])
            if value is None:
                raise PipelineError("HTTP 404")
        elif path.startswith("/api/v1/examples?"):
            value = self.imported
        elif path.startswith("/api/v1/examples/"):
            value = next((ex for ex in self.imported if ex["id"] == path.rsplit("/", 1)[1]), None)
            if value is None:
                raise PipelineError("HTTP 404")
        else:
            raise AssertionError(path)
        return SimpleNamespace(stdout=json.dumps(value))


def source():
    return triage_source.source_options(uid(100), uid(101), "2026-09-02T00:00:00Z", "2026-09-03T00:00:00Z")


def judge_call(judge, messages_, max_tokens):
    return {"keep": 1, "reason": "The answer completes the request."}


def run(tmp_path, api=None, **kwargs):
    return triage.run_triage(source(), tmp_path, runner=api or API(), judge_call=judge_call, confirm=True, sleeper=lambda _: None, **kwargs)


def test_snapshot_expands_to_earlier_turns_without_persisting_raw_trees(tmp_path):
    api = API()
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    assert frozen["selected_trace_ids"] == [uid(2)]
    assert [trace["trace_id"] for trace in frozen["traces"]] == [uid(1), uid(2)]
    assert all("runs" not in trace for trace in frozen["traces"])
    assert frozen["units"][0]["example"]["inputs"]["messages"] == SYSTEM + messages(1) + messages(2)
    conversation, = (tmp_path / "conversations").glob("*.json")
    assert load_conversation(conversation) == frozen["units"][0]
    api.calls.clear()
    assert triage_source.snapshot(source(), tmp_path, runner=api) == frozen
    assert api.calls == []


def test_empty_trajectory_is_rejected_without_stopping_pull_or_resume(tmp_path):
    api = API()
    api.trajectory_pages = {None: {"messages": [], "next_cursor": None}}
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None,
                             "start_time": "2026-09-02T01:00:00Z"})
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    empty, valid = frozen["units"]
    assert empty["example"]["inputs"]["messages"] == []
    assert empty["example"]["metadata"]["source_scope_id"] == "conversation-a"
    assert empty["training_error"] == "thread_id conversation-a returned no messages"
    assert valid["training_error"] is None

    # Completed empty and valid units both survive an interrupted pull.
    (tmp_path / "snapshot.json").unlink()
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("completed trajectory was fetched again")

    resumed = triage_source.snapshot(source(), tmp_path, runner=unexpected_read)
    assert list(resumed["units"]) == [empty, valid]

    def judge_nonempty(judge, messages_, max_tokens):
        evidence = json.loads(messages_[1]["content"])
        assert evidence["untrusted_trajectory"] == valid["example"]["inputs"]["messages"]
        return judge_call(judge, messages_, max_tokens)

    result = triage.run_triage(source(), tmp_path, runner=unexpected_read, confirm=True,
                              judge_call=judge_nonempty)
    assert result["filtered_training"] == 1
    assert result["kept"] == 1 and result["incomplete"] == 0
    labels = {row["trajectory_id"]: row for row in map(json.loads, (tmp_path / "labels.jsonl").read_text().splitlines())}
    assert labels[empty["example"]["id"]]["keep"] == 0
    assert "returned no messages" in labels[empty["example"]["id"]]["reason"]
    selected, = triage.selected_examples(tmp_path)
    assert selected["id"] == valid["example"]["id"]
    assert selected["inputs"] == valid["example"]["inputs"]


@pytest.mark.parametrize("oversized_cursor", [None, "next"])
def test_fetch_limit_excludes_whole_trajectory_and_survives_resume(tmp_path, oversized_cursor):
    api = API()
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None,
                             "start_time": "2026-09-02T01:00:00Z"})
    requests = []

    def runner(command, **kwargs):
        body = json.loads(kwargs.get("input") or "{}")
        if command[2] == "/v1/trajectory" and body.get("thread_id") == "conversation-a":
            requests.append(body)
            if body.get("cursor") == oversized_cursor:
                raise subprocess.CalledProcessError(1, command, output=
                    "HTTP 400: trajectory view exceeded response data size limit. "
                    "Narrow the requested trajectory page; private source content")
        return api(command, **kwargs)

    frozen = triage_source.snapshot(source(), tmp_path, runner=runner)
    excluded, valid = frozen["units"]
    assert excluded["example"]["inputs"]["messages"] == []
    assert "smithtune_source" not in excluded["example"]["metadata"]
    assert "trajectory fetch limit" in excluded["training_error"]
    assert "page_size=1" in excluded["training_error"]
    assert "private source content" not in excluded["training_error"]
    assert valid["training_error"] is None
    failed = [request for request in requests if request.get("cursor") == oversized_cursor]
    assert len(failed) == 2 and failed[1] == {**failed[0], "page_size": 1}
    assert dataset_workflow._download_summary(frozen)["exclusion_reasons"] == {"trajectory_fetch_limit": 1}

    # An interruption before snapshot completion must not retry the permanent exclusion.
    (tmp_path / "snapshot.json").unlink()
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("saved exclusion or completed trajectory was fetched again")
    resumed = triage_source.snapshot(source(), tmp_path, runner=unexpected_read)
    assert list(resumed["units"]) == [excluded, valid]

    def judge_valid(slot, prompt, tokens):
        assert json.loads(prompt[1]["content"])["untrusted_trajectory"] == valid["example"]["inputs"]["messages"]
        return judge_call(slot, prompt, tokens)
    result = triage.run_triage(source(), tmp_path, runner=unexpected_read, judge_call=judge_valid, confirm=True)
    assert result["filtered_training"] == 1 and result["kept"] == 1
    dataset_workflow.run("push", tmp_path, name="within-limits", confirm=True, runner=api)
    assert len(api.imported) == 1
    assert api.imported[0]["inputs"] == valid["example"]["inputs"]


def test_snapshot_retries_failed_reads(tmp_path, monkeypatch):
    api = API()
    failures = []
    delays = []
    monkeypatch.setattr(triage_source.time, "sleep", delays.append)

    def flaky(command, **kwargs):
        if command[2] == "/v1/trajectory" and not failures:
            failures.append(command)
            raise subprocess.CalledProcessError(1, command, output="429: rate limit exceeded")
        return api(command, **kwargs)

    frozen = triage_source.snapshot(source(), tmp_path, runner=flaky)
    assert len(failures) == 1 and len(frozen["traces"]) == 2
    assert delays == [30]
    # A retry after an interrupted download reuses completed reads.
    (tmp_path / "snapshot.json").unlink()
    def only_membership(command, **kwargs):
        assert command[2] == "/api/v2/runs/query"
        assert json.loads(command[command.index("--body") + 1])["filter"].startswith("eq(thread_id,")
        return api(command, **kwargs)

    assert triage_source.snapshot(source(), tmp_path, runner=only_membership) == frozen


def test_oversized_trajectory_pages_bypass_read_retries_and_cache_smaller_pages(tmp_path, monkeypatch):
    from functools import partial

    api = API()
    attempts, delays = [], []
    monkeypatch.setattr(triage_source.time, "sleep", delays.append)
    monkeypatch.setattr(curation, "_sleep", delays.append)

    def runner(command, **kwargs):
        body = json.loads(kwargs["input"])
        attempts.append(body)
        if body.get("cursor") == "next" and body.get("page_size") != 1:
            raise subprocess.CalledProcessError(1, command, output=json.dumps({
                "status": 400, "detail": "Narrow the requested trajectory page and retry: "
                "trajectory view exceeded response data size limit"}) + "\nHTTP 400")
        return api(command, **kwargs)

    cached = partial(triage_source._fetch, runner=runner, cache_dir=tmp_path)
    for _ in range(2):
        result = curation._fetch_trajectory(uid(100), uid(101), {
            "key": "thread_id", "id": "conversation-a"}, runner=cached)
        assert result["messages"] == SYSTEM + messages(1) + messages(2)
        assert result["training_error"] is None
        assert len(result["source"]["assistant_runs"]) == 2
    assert [(body.get("cursor"), body.get("page_size")) for body in attempts] == [
        (None, None), ("next", None), ("next", 1), ("next", None)]
    assert len(list(tmp_path.glob("*.json"))) == 2  # Only successful responses are cached.
    assert delays == []


def test_snapshot_preserves_content_unsupported_for_training(tmp_path):
    api = API()
    image = {"type": "image", "url": "https://example.invalid/image.png"}
    api.trajectory_pages[None]["messages"][1]["content"] = [image]
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    assert frozen["units"][0]["example"]["inputs"]["messages"][1]["content"] == [image]
    assert frozen["units"][0]["training_error"]
    result = triage.run_triage(source(), tmp_path, runner=api, confirm=True,
                              judge_call=lambda *_: pytest.fail("multimodal trace reached a judge"))
    assert result["filtered_multimodal"] == 1 and result["incomplete"] == 0
    assert (tmp_path / "judgments.jsonl").read_text() == ""
    assert all(json.loads(line)["keep"] == 0 and "Filtered before judging" in json.loads(line)["reason"]
               for line in (tmp_path / "labels.jsonl").read_text().splitlines())


def tool_conversation(name="lookup", args=None, result=True, repeat=False):
    api = API()
    conversation = [
        {"role": "ai", "id": "tool-output", "content": [{"type": "text", "text": "Looking it up."},
            {"type": "tool_call", "name": name, "args": args or {}, "id": "call-1"}]},
    ]
    if repeat:
        conversation[0]["content"].append(copy.deepcopy(conversation[0]["content"][-1]))
    if result:
        conversation.append({"role": "tool", "content": "done", "tool_call_id": "call-1"})
    api.trajectory_pages[None]["messages"][2:2] = conversation
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a record.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}}

    def runner(command, **kwargs):
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory":
            page = json.loads(response.stdout)
            for item in page["items"]:
                if item["message"].get("role") == "ai":
                    item["message"]["available_tools"] = [tool]
                if item["message"].get("id") == "tool-output":
                    item["metadata"]["run_id"] = uid(9999)
            response.stdout = json.dumps(page)
        return response

    return runner


@pytest.mark.parametrize("kwargs,error", [
    ({"name": "missing_tool"}, "unknown tool in recorded tool call"),
    ({"args": {"limit": "many"}}, "do not match its JSON Schema"),
    ({"result": False}, "unmatched tool calls or results"),
    ({"repeat": True}, "repeats tool call id"),
])
@pytest.mark.parametrize("cached", [False, True])
def test_triage_checks_preparation_compatibility(tmp_path, monkeypatch, kwargs, error, cached):
    api = tool_conversation(**kwargs)
    if cached:
        # Simulate a snapshot saved before tool-call validation was introduced.
        with monkeypatch.context() as patch:
            patch.setattr(triage_source, "training_error", lambda _: None)
            patch.setattr(triage, "training_error", lambda _: None)
            assert run(tmp_path, api)["eligible_conversations"] == 1
        original = (tmp_path / "snapshot.json").read_bytes()
        with pytest.raises(PipelineError, match="no complete, kept"):
            triage.selected_examples(tmp_path)

        def api(*_a, **_kw):
            pytest.fail("cached source was fetched again")
    def unexpected_judge(*args):
        pytest.fail("invalid trajectory reached the council")

    monkeypatch.setattr(triage, "check_credentials", unexpected_judge)
    summary = triage.run_triage(source(), tmp_path, runner=api, judge_call=unexpected_judge, confirm=True)
    frozen = triage_source.load_snapshot(tmp_path)
    assert error in triage_source.training_error(frozen["units"][0])
    assert summary["kept"] == 0
    assert summary["filtered_training"] == 1
    assert summary["incomplete"] == 0
    assert error in json.loads((tmp_path / "labels.jsonl").read_text())["reason"]
    assert summary["eligible_conversations"] == 0
    assert summary["unsupported_training_conversations"] == 1
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)
    if cached:
        assert (tmp_path / "snapshot.json").read_bytes() == original


@pytest.mark.parametrize("existing", [False, True])
def test_cached_triage_prunes_misplaced_system_messages_before_upload(tmp_path, monkeypatch, existing):
    api = API()
    api.trajectory_pages["next"]["messages"].insert(0, {"role": "system", "content": "Late instructions."})
    with monkeypatch.context() as patch:
        patch.setattr(triage_source, "training_error", lambda _: None)
        patch.setattr(triage, "training_error", lambda _: None)
        assert run(tmp_path, api)["eligible_conversations"] == 1
    original = (tmp_path / "snapshot.json").read_bytes()
    api.calls.clear()
    destination = {"dataset_id": uid(200)} if existing else {"name": "new"}
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.create_triaged_dataset(tmp_path, confirm=True, runner=api, **destination)
    assert api.calls == []
    assert (tmp_path / "snapshot.json").read_bytes() == original


@pytest.mark.parametrize("runner_mode", ["api", "deepagent"])
@pytest.mark.parametrize("issue,reason", [
    ("system", "misplaced system message"),
    ("builtin", "provider built-in tool"),
    ("missing_tools", "unknown tool availability"),
])
def test_invalid_trajectories_skip_all_paid_work(tmp_path, monkeypatch, runner_mode, issue, reason):
    api = API()
    if issue == "system":
        api.trajectory_pages["next"]["messages"].insert(0, {"role": "system", "content": "New instructions."})

    def runner(command, **kwargs):
        response = api(command, **kwargs)
        if command[2] == "/v1/trajectory" and issue != "system":
            page = json.loads(response.stdout)
            for item in page["items"]:
                message = item["message"]
                if message.get("role") != "ai":
                    continue
                if issue == "builtin":
                    message["available_tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
                else:
                    message.pop("available_tools")
            response.stdout = json.dumps(page)
        return response

    def unexpected(*args, **kwargs):
        pytest.fail("ineligible trajectories must not require credentials or inference dependencies")

    monkeypatch.setattr(triage, "check_credentials", unexpected)
    monkeypatch.setattr(triage, "api_judge", unexpected)
    monkeypatch.setattr(triage, "deepagent_judge", unexpected)
    preview = triage.run_triage(source(), tmp_path, runner=runner, runner_mode=runner_mode, dry_run=True)
    assert preview["filtered_training"] == 1
    assert preview["judge_tasks"] == 0
    assert reason in preview["rejections"][0]["reason"]
    summary = triage.run_triage(source(), tmp_path, runner=unexpected, runner_mode=runner_mode, confirm=True)
    assert summary["status"] == "complete" and summary["filtered_training"] == 1
    assert (tmp_path / "judgments.jsonl").read_text() == ""
    assert reason in json.loads((tmp_path / "labels.jsonl").read_text())["reason"]
    assert triage.selected_examples(tmp_path, require_complete=True, allow_empty=True) == []


@pytest.mark.parametrize("cached", [False, True])
def test_prefilter_mixed_trajectories_preserves_completed_votes(tmp_path, monkeypatch, cached):
    api = API()
    api.trajectory_pages["next"]["messages"].insert(0, {"role": "system", "content": "Late instructions."})
    api.root_pages[0].append({"trace_id": uid(3), "thread_id": None, "start_time": "2026-09-02T00:00:00Z"})
    saved_votes = []
    if cached:
        def old_judge(slot, prompt, tokens):
            if len(json.loads(prompt[-1]["content"])["untrusted_trajectory"]) > 3:
                raise PipelineError("temporary failure")
            return judge_call(slot, prompt, tokens)

        with monkeypatch.context() as patch:
            patch.setattr(triage_source, "training_error", lambda _: None)
            patch.setattr(triage, "training_error", lambda _: None)
            summary = triage.run_triage(source(), tmp_path, runner=api, judge_call=old_judge, confirm=True, attempts=1)
            assert summary["incomplete"] == 1 and summary["kept"] == 1
        saved_votes = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
        # Historical incomplete labels are validated, then invalid trajectories
        # are excluded without forcing more votes before upload.
        assert len(triage.selected_examples(tmp_path, require_complete=True)) == 1
        identity = (tmp_path / "triage-config.json").read_bytes()
        snapshot = (tmp_path / "snapshot.json").read_bytes()
        def api(*args, **kwargs):
            pytest.fail("resume must not fetch saved trajectories")

    seen = []
    def judge(slot, prompt, tokens):
        assert not cached, "completed valid votes must be reused"
        assert len(json.loads(prompt[-1]["content"])["untrusted_trajectory"]) == 3
        seen.append(slot["name"])
        return judge_call(slot, prompt, tokens)

    plan = triage.run_triage(source(), tmp_path, runner=api, dry_run=True)
    assert plan["filtered_training"] == 1 and plan["judge_tasks"] == plan["judges"]
    summary = triage.run_triage(source(), tmp_path, runner=api, judge_call=judge, confirm=True)
    assert summary["status"] == "complete" and summary["kept"] == 1 and summary["filtered_training"] == 1
    assert len(triage.selected_examples(tmp_path, require_complete=True)) == 1
    assert len(seen) == (0 if cached else plan["judges"])
    if cached:
        assert [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()] == saved_votes
        assert (tmp_path / "triage-config.json").read_bytes() == identity
        assert (tmp_path / "snapshot.json").read_bytes() == snapshot


def test_triage_accepts_valid_tool_calls_and_keeps_saved_bindings(tmp_path):
    assert run(tmp_path, tool_conversation())["eligible_conversations"] == 1
    frozen = triage_source.load_snapshot(tmp_path)
    example, = triage.selected_examples(tmp_path)
    assert example["inputs"] == frozen["units"][0]["example"]["inputs"]
    assert example["metadata"]["smithtune_source"] == frozen["units"][0]["example"]["metadata"]["smithtune_source"]
    assert example["metadata"]["smithtune_triage"]["evidence_sha256"] == evidence_hash(example)
    assert dataset.prepare_sft_rows([example])


def test_triage_defers_reasoning_policy_to_preparation(tmp_path):
    frozen = triage_source.snapshot(source(), tmp_path, runner=API())
    unit = copy.deepcopy(frozen["units"][0])
    for message in unit["example"]["inputs"]["messages"]:
        if message["role"] == "ai":
            message["content"] = [{"type": "reasoning", "reasoning": "Working through the answer."}]
    assert triage_source.training_error(unit) is None
    assert dataset.prepare_sft_rows([unit["example"]], model=DEFAULT_MODEL, reasoning_policy="preserve")


@pytest.mark.parametrize("evidence,expected", [
    ({"messages": [{"content": [{"type": "input_audio", "data": "recorded"}]}]}, ["input_audio"]),
    ({"runs": [{"outputs": {"content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]}}]}, ["image_url"]),
    ({"runs": [{"attachments": {"photo.png": "https://example.invalid/a.png"}}]}, ["attachment"]),
    ({"runs": [{"outputs": {"message": {"role": "assistant", "audio": {"data": "recorded"}}}}]}, ["audio"]),
    ({"messages": [{"content": "Explain image and audio formats."}], "runs": [{"outputs": {"type": "file", "path": "main.py", "url": "https://example.invalid/main.py"}}]}, []),
    ({"runs": [{"inputs": {"schema": {"type": ["string", "null"]}}, "attachments": {"readme.md": "https://example.invalid/readme.md"}}]}, []),
])
def test_multimodal_prefilter(evidence, expected):
    assert triage_source.multimodal_types(evidence) == expected


def test_standalone_traces_are_labeled(tmp_path):
    api = API()
    api.root_pages[0][0]["thread_id"] = None
    result = run(tmp_path, api)
    assert result["kept"] == 1
    assert json.loads((tmp_path / "labels.jsonl").read_text()) == {
        "trajectory_id": triage_source.load_snapshot(tmp_path)["units"][0]["example"]["id"], "keep": 1, "reason": "2/2 judges voted 1. The answer completes the request.",
    }


def test_dry_run_does_not_judge_and_single_judge_is_supported(tmp_path):
    def no_judge(*args):
        pytest.fail("dry-run must not judge")
    path = config(tmp_path, count=1)
    plan = triage.run_triage(source(), tmp_path, runner=API(), dry_run=True, judge_call=no_judge, config_path=path)
    assert plan["judges"] == 1 and plan["judge_tasks"] == 1
    assert not (tmp_path / "judgments.jsonl").exists()
    result = run(tmp_path, config_path=path)
    assert result["kept"] == 1 and result["status"] == "complete"


def test_rerun_retains_one_successful_vote_per_slot(tmp_path):
    run(tmp_path)
    before = (tmp_path / "judgments.jsonl").read_text()
    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=lambda *_: pytest.fail("completed vote repeated"), confirm=True)
    assert result["kept"] == 1
    assert (tmp_path / "judgments.jsonl").read_text() == before


def test_failed_judge_is_incomplete_then_retried(tmp_path):
    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=lambda *_: {"keep": 1}, confirm=True, attempts=1)
    assert result["incomplete"] == 1 and result["kept"] == 0
    assert run(tmp_path)["kept"] == 1
    seen = {}

    def fix_result(judge, messages_, max_tokens):
        value = judge_call(judge, messages_, max_tokens)
        key = judge["name"]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            value["reason"] = ""
        else:
            assert "previous attempt failed validation" in messages_[0]["content"]
        return value

    result = triage.run_triage(source(), tmp_path / "retry", runner=API(), judge_call=fix_result,
                               confirm=True, attempts=2, sleeper=lambda _: None)
    assert result["kept"] == 1 and all(count == 2 for count in seen.values())


def config(tmp_path, count=2):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"judges": [{"name": f"judge-{i}", "provider": "fireworks", "model": "accounts/fireworks/models/example"} for i in range(count)],
                               "rules": ["Keep trajectories that complete the request."]}))
    return path


def test_multiple_judges_ties_drop(tmp_path):
    path = config(tmp_path)

    def disagree(judge, *args):
        result = judge_call(judge, *args)
        result["keep"] = int(judge["name"] == "judge-0")
        return result

    result = triage.run_triage(source(), tmp_path / "work", runner=API(), judge_call=disagree, confirm=True, config_path=path)
    assert result["kept"] == 0 and result["disagreement"] == 1 and result["incomplete"] == 0
    labels = [json.loads(line) for line in (tmp_path / "work/labels.jsonl").read_text().splitlines()]
    assert all(label["keep"] == 0 and label["reason"].startswith("Tied vote") for label in labels)


def test_majority_drop_blocks_whole_conversation(tmp_path):
    def drop(judge, *args):
        result = judge_call(judge, *args)
        result["keep"] = int(judge["name"] == "judge-1")
        return result

    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=drop, confirm=True)
    assert result["kept"] == 0 and result["dropped"] == 1
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)


def test_frozen_dataset_import_and_prepare_use_judged_messages_and_tools(tmp_path, monkeypatch):
    api = API()
    run(tmp_path / "triage", api)
    # The live source changes. Import must not read it again.
    api.calls.clear()
    api.trajectory_pages["next"]["messages"][0]["content"] = "different live message"
    imported = triage.create_triaged_dataset(tmp_path / "triage", "selected", confirm=True, runner=api)
    assert imported["example_count"] == 1
    assert all(command[2] in {"/api/v1/datasets", "/api/v1/examples"} for command, _ in api.calls)
    example = api.imported[0]
    assert example["inputs"]["messages"] == SYSTEM + messages(1) + messages(2)
    bindings = read_bindings(example["metadata"], example["inputs"]["messages"])
    assert all(binding["tools"] == [] for binding in bindings.values())
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    # Add independent source groups to exercise the normal 80/10/10 path.
    examples = []
    for n in range(3):
        item = copy.deepcopy(example)
        item["id"] = uid(500 + n)
        item["metadata"]["source_scope_id"] = f"independent-{n}"
        item["inputs"]["messages"][-1]["content"] += f" variant-{n}"
        item["metadata"]["smithtune_triage"]["messages_sha256"] = json_sha256(item["inputs"]["messages"])
        item["metadata"]["smithtune_triage"]["evidence_sha256"] = evidence_hash(item)
        examples.append(item)
    monkeypatch.setattr(dataset, "_load_dataset_source", lambda *_a, **_kw: ({"name": "selected"}, examples, "snapshot"))
    manifest = dataset.prepare_dataset(uid(100), uid(200), DEFAULT_MODEL, tmp_path / "data", fetch=True, check_render=False)
    assert manifest["prepared"]["accepted"] == 3
    assert sum(manifest["split"][name] for name in dataset.SPLIT_NAMES) == 3


def test_changed_triaged_messages_are_rejected(tmp_path):
    run(tmp_path)
    examples = list(triage.selected_examples(tmp_path))
    examples[0]["inputs"]["messages"][-1]["content"] = "unjudged change"
    with pytest.raises(PipelineError, match="changed after judging"):
        dataset.validate_trajectories(examples, len(examples))


def test_changed_conversation_file_blocks_import_before_writes(tmp_path):
    run(tmp_path)
    path, = (tmp_path / "conversations").glob("*.json")
    example = load_conversation(path)
    example["example"]["inputs"]["messages"][-1]["content"] = "changed since judging"
    path.write_text(json.dumps(example))
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=True,
            runner=lambda *_a, **_kw: pytest.fail("changed conversation reached upload"))


def test_old_snapshot_materializes_conversation_without_refetching(tmp_path):
    frozen = triage_source.snapshot(source(), tmp_path, runner=API())
    frozen["units"] = list(frozen["units"])
    frozen.pop("unit_files")
    frozen.pop("snapshot_sha256")
    frozen["schema_version"] = 2
    for unit in frozen["units"]:
        unit.pop("traces")
        unit.pop("multimodal_types")
    for trace in frozen["traces"]:
        trace["runs"] = []
    frozen["snapshot_sha256"] = json_sha256(frozen)
    (tmp_path / "snapshot.json").write_text(json.dumps(frozen))
    run(tmp_path, api=lambda *_a, **_kw: pytest.fail("legacy snapshot must not refetch"))
    snapshot_bytes = (tmp_path / "snapshot.json").read_bytes()
    for path in (tmp_path / "conversations").glob("*.json"):
        path.unlink()
    example, = triage.selected_examples(tmp_path)
    saved, = (tmp_path / "conversations").glob("*.json")
    assert load_conversation(saved)["inputs"] == example["inputs"]
    assert (tmp_path / "snapshot.json").read_bytes() == snapshot_bytes


def test_cli_default_run_directories_are_unique_and_can_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curation, "_utc_now", lambda: "2026-09-03T00:00:00+00:00")
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(
        *args, **kwargs, runner=API(), judge_call=judge_call))
    args = ["dataset", "pull", "--workspace-id", uid(100), "--project-id", uid(101)]
    directories = []
    for _ in range(2):
        cli.main(args)
        run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
        assert run_dir.parent == Path("data/datasets")
        assert (run_dir / "snapshot.json").exists()
        saved_source = json.loads((run_dir / "snapshot.json").read_text())["source"]
        assert saved_source["start_time"] == "2026-09-02T00:00:00+00:00"
        assert saved_source["end_time"] == "2026-09-03T00:00:00+00:00"
        assert len(list((run_dir / "conversations").glob("*.json"))) == 1
        directories.append(run_dir)
    assert directories[0] != directories[1]
    cli.main(["dataset", "triage", str(directories[0]), "--confirm"])
    assert json.loads(capsys.readouterr().out)["run_dir"] == str(directories[0])
    with pytest.raises(SystemExit):
        cli.main(["dataset", "triage", "--confirm"])
    assert "requires a saved directory" in capsys.readouterr().err


def test_coordinator_prompt_changes_require_a_new_run(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(triage.load_config(None)))
    run_dir = tmp_path / "run"
    run(run_dir, runner_mode="deepagent", config_path=config_path)
    saved_votes = (run_dir / "judgments.jsonl").read_bytes()
    assert run(run_dir, runner_mode="deepagent", config_path=config_path)["status"] == "complete"
    assert (run_dir / "judgments.jsonl").read_bytes() == saved_votes
    monkeypatch.setattr(triage, "COORDINATOR_PROMPT", triage.COORDINATOR_PROMPT.replace("strict majority", "unanimous vote"))
    with pytest.raises(PipelineError, match="different input.*new output directory"):
        run(run_dir, runner_mode="deepagent", config_path=config_path)


@pytest.mark.parametrize("fetch", [False, True])
@pytest.mark.parametrize("global_contract", [False, True])
def test_prepare_always_checks_triaged_message_integrity(tmp_path, monkeypatch, fetch, global_contract):
    run(tmp_path)
    examples = list(triage.selected_examples(tmp_path))
    contract = tool_contract([]) if global_contract else None
    examples[0]["inputs"]["messages"][-1]["content"] = "unjudged change"
    monkeypatch.setattr(dataset, "_load_dataset_source", lambda *_a, **_kw: ({"name": "selected"}, examples, "snapshot"))
    with pytest.raises(PipelineError, match="changed after judging"):
        dataset.prepare_dataset(uid(100), uid(200), DEFAULT_MODEL, tmp_path / "data", fetch=fetch,
                                inference_contract=contract, check_render=False)


@pytest.mark.parametrize("changed", ["config", "source", "labels", "votes", "snapshot"])
def test_resume_and_import_reject_mixed_or_tampered_artifacts(tmp_path, changed):
    run(tmp_path)
    if changed == "config":
        with pytest.raises(PipelineError, match="different input"):
            run(tmp_path, max_output_tokens=1024)
    elif changed == "source":
        with pytest.raises(PipelineError, match="different source"):
            triage_source.snapshot({**source(), "seed": 5}, tmp_path, runner=API())
    elif changed == "labels":
        (tmp_path / "labels.jsonl").write_text('{}\n')
        with pytest.raises(PipelineError, match="changed|invalid"):
            triage.selected_examples(tmp_path)
    elif changed == "snapshot":
        path = tmp_path / "snapshot.json"
        value = json.loads(path.read_text())
        value["source"]["project_id"] = uid(999)
        path.write_text(json.dumps(value))
        with pytest.raises(PipelineError, match="hash mismatch"):
            run(tmp_path)
    else:
        path = tmp_path / "judgments.jsonl"
        path.write_text(path.read_text() * 2)
        with pytest.raises(PipelineError, match="duplicate"):
            run(tmp_path)


@pytest.mark.parametrize("change", [{"keep": True}, {"keep": 2}, {"reason": ""}, {"extra": "wrong"}])
def test_judge_output_is_a_score_and_reason(change):
    with pytest.raises(PipelineError):
        triage_judges.validate_judgment({"keep": 1, "reason": "complete", **change})


@pytest.mark.parametrize("body", [
    {"error": {"code": "context_length_exceeded"}},
    {"error": {"message": "Input length 880063 exceeds maximum allowed token limit 262112"}},
    {"detail": "Input is too long: 880063 tokens, limit 262112"},
    {"error": "Input token count 880063 exceeds the limit of 262112 tokens"},
    {"detail": [{"msg": "The input (880063 tokens) is longer than the model's context length (262112 tokens)."}]},
])
@pytest.mark.parametrize("http_error", [False, True])
def test_provider_context_rejection_filters_without_shortening_or_retrying(tmp_path, body, http_error):
    api = API()
    api.trajectory_pages[None]["messages"][1]["content"] = "x" * 20_000
    calls = []

    def reject(judge, prompt, tokens):
        assert json.loads(prompt[-1]["content"])["untrusted_trajectory"][1]["content"] == "x" * 20_000
        calls.append(judge["name"])
        if http_error:
            error = HTTPError("https://example.invalid/chat/completions", 400, "Bad Request", {},
                              io.BytesIO(json.dumps(body).encode()))
            raise PipelineError("triage judge failed with HTTP 400") from error
        error = RuntimeError("provider error")
        error.status_code, error.body = 400, body
        raise error

    result = triage.run_triage(source(), tmp_path, runner=api, judge_call=reject, confirm=True,
                              sleeper=lambda _: pytest.fail("context rejection retried"))
    assert result["incomplete"] == 0 and result["filtered_context"] == 1
    assert len(calls) == 2
    label = json.loads((tmp_path / "labels.jsonl").read_text())
    assert label["keep"] == 0 and "context window" in label["reason"]
    assert triage.run_triage(source(), tmp_path, confirm=True,
        judge_call=lambda *_: pytest.fail("filtered trajectory retried"))["filtered_context"] == 1
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)


def test_default_council_is_packaged_without_selection_criteria(tmp_path):
    config = triage.DEFAULT_COUNCIL
    assert config["rules"] == []
    assert [(j["provider"], j["model"]) for j in config["judges"]] == [
        ("baseten", "deepseek-ai/DeepSeek-V4.1-Flash"),
        ("baseten", "zai-org/GLM-5.3-Flash"),
    ]


@pytest.mark.no_default_rule
def test_council_review_requires_selection_criteria(tmp_path):
    with pytest.raises(PipelineError, match="--rubric FILE or --rule TEXT"):
        run(tmp_path / "none")
    assert run(tmp_path / "rule", config={**triage.load_config(None), "rules": ["Keep complete answers."]})["status"] == "complete"
    assert run(tmp_path / "rubric", selection_rubric="Keep complete answers.")["status"] == "complete"


def test_judge_prompt_ships_no_default_quality_criteria():
    prompt = triage_judges.JUDGE_PROMPT
    assert "untrusted data" in prompt and "selection criteria supplied below" in prompt
    assert "Use 1 when" not in prompt and "Use 0 for" not in prompt


def test_cli_triage_and_dataset_handoff(tmp_path, monkeypatch, capsys):
    api = API()
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(*args, **kwargs, runner=api, judge_call=judge_call))
    cli.main(["dataset", "pull", str(tmp_path), "--workspace-id", uid(100), "--project-id", uid(101),
              "--start-time", source()["start_time"], "--end-time", source()["end_time"]])
    capsys.readouterr()
    cli.main(["dataset", "triage", str(tmp_path), "--judges",
              "baseten:deepseek-ai/DeepSeek-V4.1-Flash,baseten:zai-org/GLM-5.3-Flash", "--rule", "Keep supported answers."])
    plan = json.loads(capsys.readouterr().out)["triage"]
    assert plan["judges"] == 2 and plan["runner"] == "deepagent"
    assert plan["config"]["rules"] == ["Keep supported answers."]
    assert plan["config"]["judges"] == triage.load_config(None)["judges"]
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["triage"]["kept"] == 1
    saved = (tmp_path / "judgments.jsonl").read_bytes()
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["triage"]["kept"] == 1
    assert (tmp_path / "judgments.jsonl").read_bytes() == saved
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert all(set(label) == {"trajectory_id", "keep", "reason"} for label in labels)
    assert all(label["reason"] in (tmp_path / "report.md").read_text() for label in labels)
    # A changed package default must not replace a saved council.
    monkeypatch.setattr(triage, "load_config", lambda *_: pytest.fail("saved council ignored"))
    assert triage.council_settings(tmp_path)["config"] == plan["config"]
    cli.main(["dataset", "push", str(tmp_path), "--name", "selected", "--confirm"])
    assert json.loads(capsys.readouterr().out)["example_count"] == 1


@pytest.mark.parametrize("judges,expected", [
    ([" GPT-5.6-Terra ", "gpt-5.6-terra"], [("openai", "gpt-5.6-terra")] * 2),
    (["openai:custom-model", "fireworks:accounts/fireworks/models/custom"],
     [("openai", "custom-model"), ("fireworks", "accounts/fireworks/models/custom")]),
    ([""], None), (["gpt-5.6-terra", ""], None), (["unknown"], None), (["openai:"], None),
])
def test_council_model_selection(tmp_path, judges, expected):
    if expected is None:
        with pytest.raises(PipelineError, match="--judges"):
            triage.council_settings(tmp_path, judges=judges)
    else:
        slots = triage.council_settings(tmp_path, judges=judges)["config"]["judges"]
        assert [(j["provider"], j["model"]) for j in slots] == expected
        assert len({j["name"] for j in slots}) == len(judges)


def test_cli_incomplete_triage_prints_summary_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(*args, **kwargs, runner=API(), judge_call=lambda *_: {}))
    triage_source.snapshot(source(), tmp_path, runner=API())
    args = ["dataset", "triage", str(tmp_path), "--confirm", "--attempts", "1"]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["triage"]["incomplete"] == 1


def test_dataset_import_failure_resumes_with_saved_destination(tmp_path):
    run(tmp_path)
    api = API()

    def fail(command):
        if command[2] == "/api/v1/examples":
            raise PipelineError("connection lost")

    api.failure = fail
    with pytest.raises(PipelineError, match="import incomplete"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=True, runner=api)
    receipt = json.loads((tmp_path / "dataset-import.json").read_text())
    assert receipt["dataset_id"] == next(iter(api.datasets))
    assert receipt["status"] == "incomplete" and receipt["pending_write"]
    api.failure = None
    result = triage.create_triaged_dataset(tmp_path, "selected", confirm=True, runner=api)
    assert result["example_count"] == 1
    assert len(api.imported) == 1
    assert len(api.datasets) == 1


def test_concurrent_output_use_is_rejected_before_fetch_or_inference(tmp_path):
    from smithtune.artifacts import output_lock
    api = API()
    with output_lock(tmp_path), pytest.raises(PipelineError, match="another smithtune operation"):
        run(tmp_path, api)
    assert api.calls == []
    assert run(tmp_path, api)["status"] == "complete"


def test_source_window_compares_fractional_timestamps_as_times():
    value = triage_source.source_options(uid(100), uid(101), "2026-09-02T00:00:00Z", "2026-09-02T00:00:00.1Z")
    assert value["end_time"].endswith(".100000+00:00")


def test_source_pagination_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(triage_source, "MAX_SOURCE_PAGES", 1)
    api = API()
    api.root_pages.append([])
    with pytest.raises(PipelineError, match="page limit"):
        triage_source.snapshot(source(), tmp_path, runner=api)
    assert not (tmp_path / "snapshot.json").exists()


@pytest.mark.parametrize("provider,url,key", [
    ("fireworks", "https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY"),
    ("openai", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    ("baseten", "https://inference.baseten.co/v1/chat/completions", "BASETEN_API_KEY"),
    ("anthropic", "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"),
])
def test_judge_transport_routes_credentials_to_the_selected_provider(monkeypatch, provider, url, key):
    from smithtune import inference
    monkeypatch.setenv(key, "test-credential")
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    requests = []

    def post(request, label):
        requests.append(request)
        content = json.dumps({"keep": 1})
        return {"content": [{"type": "text", "text": content}]} if provider.startswith("anthropic") else {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(triage_judges, "_post_json", post)
    monkeypatch.setattr(inference, "_post_json", post)
    result = triage_judges.api_judge({"provider": provider, "model": "example"},
                                   [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "evidence"}], 128)
    assert result == {"keep": 1}
    assert requests[0].full_url == url
    header = "X-api-key" if provider.startswith("anthropic") else "Authorization"
    assert requests[0].get_header(header) == ("test-credential" if provider.startswith("anthropic") else "Bearer test-credential")
    if provider in {"fireworks", "baseten"}:
        body = json.loads(requests[0].data)
        assert body["reasoning_effort"] == "none"
        assert body["max_tokens"] == 128 and "max_completion_tokens" not in body


def test_missing_selected_thread_root_fails_closed(tmp_path):
    api = API()
    api.thread_roots = api.thread_roots[:1]
    with pytest.raises(PipelineError, match="selected root was not found"):
        triage_source.snapshot(source(), tmp_path, runner=api)
    assert not (tmp_path / "snapshot.json").exists()


def test_paid_and_remote_writes_require_confirmation(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="incurs cost"):
        triage.run_triage(source(), tmp_path, runner=API())
    with pytest.raises(PipelineError, match="requires --confirm"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="BASETEN_API_KEY"):
        triage.run_triage(source(), tmp_path, runner=API(), confirm=True)
    assert not (tmp_path / "triage-config.json").exists()
    assert not (tmp_path / "judgments.jsonl").exists()
    # No paid work happened, so choosing another council remains possible.
    assert run(tmp_path, config_path=config(tmp_path, count=1))["status"] == "complete"


def test_judge_can_drop_a_trajectory_with_missing_evidence(tmp_path):
    result = triage.run_triage(source(), tmp_path, runner=API(), confirm=True,
        judge_call=lambda *_: {"keep": 0, "reason": "The recorded outcome is missing."})
    assert result["incomplete"] == 0 and result["dropped"] == 1
    label = json.loads((tmp_path / "labels.jsonl").read_text())
    assert label["keep"] == 0 and "recorded outcome is missing" in label["reason"]


def test_cli_labels_local_snapshot_without_source_query(tmp_path, monkeypatch, capsys):
    triage_source.snapshot(source(), tmp_path, runner=API())
    original = dataset_workflow.run
    monkeypatch.setattr(dataset_workflow, "run", lambda *args, **kwargs: original(
        *args, **kwargs, judge_call=judge_call,
        runner=lambda *_a, **_kw: pytest.fail("local snapshot must not query LangSmith")))
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["triage"]["kept"] == 1


def test_cli_requires_source_when_no_snapshot_exists(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["dataset", "triage", str(tmp_path)])
    assert "start with dataset pull" in capsys.readouterr().err


def test_empty_source_does_not_create_a_misleading_completed_run(tmp_path):
    api = API()
    api.root_pages = [[]]
    with pytest.raises(PipelineError, match="no traces match"):
        triage.run_triage(source(), tmp_path, runner=api, dry_run=True)
    assert not (tmp_path / "snapshot.json").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("media_turn", [None, 1, 2])
def test_council_judges_distinct_full_conversations_and_filters_any_turn(tmp_path, media_turn):
    api = API()
    api.root_pages[0] += [
        {**api.root_pages[0][0], "trace_id": uid(1)},
        {**api.root_pages[0][0], "trace_id": uid(3), "thread_id": None},
    ]

    def source_api(command, **kwargs):
        response = api(command, **kwargs)
        if media_turn and command[2] == "/v1/trajectory":
            page = json.loads(response.stdout)
            for item in page["items"]:
                if item["message"].get("id") == f"user-{media_turn}":
                    item["message"]["content"] = [{"type": "image", "url": "https://example.invalid/image.png"}]
            response.stdout = json.dumps(page)
        return response

    seen = []

    def judge(slot, prompt, tokens):
        evidence = json.loads(prompt[-1]["content"])["untrusted_trajectory"]
        seen.append((slot["name"], evidence))
        assert evidence[0] == SYSTEM[0]
        if len(evidence) == 5:
            assert media_turn is None
            assert [m["content"] for m in evidence[1:]] == ["question-1", "answer-1", "question-2", "answer-2"]
        return judge_call(slot, prompt, tokens)

    summary = triage.run_triage(source(), tmp_path, runner=source_api, judge_call=judge, confirm=True)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["selected_traces"] == 3 and plan["source_traces"] == 3
    assert plan["trajectories"] == 2
    assert plan["judge_tasks"] == len(seen) == (2 if media_turn else 4)
    assert summary["filtered_multimodal"] == int(media_turn is not None)
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 2
    examples = triage.selected_examples(tmp_path)
    assert {e["id"] for e in examples} == {label["trajectory_id"] for label in labels if label["keep"]}
    for example in examples:
        judged = [e for _, e in seen if e == example["inputs"]["messages"]]
        assert len(judged) == 2
        assert all(e == example["inputs"]["messages"] for e in judged)


def test_old_trace_votes_cannot_be_reused_or_imported(tmp_path):
    run(tmp_path)
    manifest = tmp_path / "triage-config.json"
    identity = json.loads(manifest.read_text())
    identity.pop("judging_unit")
    manifest.write_text(json.dumps(identity))
    saved = (tmp_path / "judgments.jsonl").read_bytes()
    with pytest.raises(PipelineError, match="new output directory"):
        triage.run_triage(source(), tmp_path, confirm=True, judge_call=lambda *_: pytest.fail("old votes triggered inference"))
    with pytest.raises(PipelineError, match="per-trace votes"):
        triage.selected_examples(tmp_path)
    assert (tmp_path / "judgments.jsonl").read_bytes() == saved


@pytest.mark.parametrize("status,body,expected", [
    (400, {"code": "context_length_exceeded"}, True),
    (400, {"error": {"message": "prompt is too long: 120000 tokens > 100000 maximum"}}, True),
    (429, {"message": "Rate limit reached"}, False),
    (400, {"message": "max_tokens exceeds the output limit"}, False),
    (422, {"detail": "Input length 880063 exceeds maximum allowed token limit 262112"}, True),
    (400, {"detail": "Output token count exceeds the limit of 4096 tokens"}, False),
    (400, {"detail": "Input token budget exceeded for this account"}, False),
    (400, {"detail": "Input token budget exceeded the account limit"}, False),
    (400, {"detail": "Input token count 880063 exceeds the account limit of 262112"}, False),
    (400, {"detail": "Input validation error: requested output tokens exceed the maximum allowed output length"}, False),
    (429, {"error": "Input token count 880063 exceeds the limit of 262112 tokens"}, False),
    (503, {"message": "prompt is too long"}, False),
    (401, {"error": {"code": "context_length_exceeded"}}, False),
    (400, {"error": {"code": {}, "message": "Invalid request"}}, False),
])
def test_only_context_rejections_filter_trajectories(status, body, expected):
    error = RuntimeError("provider error")
    error.status_code, error.body = status, body
    assert triage_judges.context_window_exceeded(error) is expected


def test_old_snapshot_requires_fresh_system_message_capture(tmp_path):
    triage_source.snapshot(source(), tmp_path, runner=API())
    path = tmp_path / "snapshot.json"
    value = json.loads(path.read_text())
    value["schema_version"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(PipelineError, match="predates system-message capture"):
        triage_source.load_snapshot(tmp_path)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("growth", ["before_read", "between_pages", "after_read"])
def test_snapshot_keeps_growing_threads_and_resumes_saved_content(tmp_path, resume, growth):
    api = API()
    api.root_pages[0].append({"trace_id": uid(4), "thread_id": None,
                             "start_time": "2026-09-02T01:00:00Z"})
    changed = interrupted = False

    def grow():
        nonlocal changed
        changed = True
        api.thread_roots.append({**api.thread_roots[-1], "id": uid(3), "trace_id": uid(3)})
        api.trajectory_pages["next"]["messages"] += messages(3)

    def runner(command, **kwargs):
        nonlocal interrupted
        if command[2] != "/v1/trajectory":
            return api(command, **kwargs)
        body = json.loads(kwargs["input"])
        if "thread_id" not in body:
            return api(command, **kwargs)
        if resume and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        if not changed and growth == "before_read":
            grow()
        response = api(command, **kwargs)
        if not changed and (growth == "between_pages" or body.get("cursor") == "next"):
            grow()
        return response

    if resume:
        with pytest.raises(KeyboardInterrupt):
            triage_source.snapshot(source(), tmp_path, runner=runner)
    frozen = triage_source.snapshot(source(), tmp_path, runner=runner)
    growing, stable = frozen["units"]
    included = growth != "after_read"
    expected_messages = SYSTEM + messages(1) + messages(2) + (messages(3) if included else [])
    expected_traces = [uid(1), uid(2)] + ([uid(3)] if included else [])
    assert growing["example"]["inputs"]["messages"] == expected_messages
    assert growing["trace_ids"] == expected_traces
    assert growing["example"]["metadata"]["triage_trace_ids"] == expected_traces
    assert [run["trace_id"] for run in growing["example"]["metadata"]["smithtune_source"]["assistant_runs"]] == expected_traces
    assert [record["trace_id"] for record in growing["traces"]] == expected_traces
    assert growing["training_error"] is None and stable["training_error"] is None
    assert dataset_workflow._download_summary(frozen)["excluded"] == 0

    # Completed downloads survive interruption and further changes to the live thread.
    (tmp_path / "snapshot.json").unlink()
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("completed trajectory was fetched again")
    resumed = triage_source.snapshot(source(), tmp_path, runner=unexpected_read)
    assert list(resumed["units"]) == [growing, stable]
    result = triage.run_triage(source(), tmp_path, runner=unexpected_read, judge_call=judge_call, confirm=True)
    assert result["filtered_training"] == 0 and result["kept"] == 2
    dataset_workflow.run("push", tmp_path, name="growing-trajectories", confirm=True, runner=api)
    assert len(api.imported) == 2
    assert growing["example"]["inputs"] in [example["inputs"] for example in api.imported]
    assert stable["example"]["inputs"] in [example["inputs"] for example in api.imported]


def test_growing_thread_with_incomplete_tool_exchange_is_filtered_before_judging(tmp_path):
    api = API()
    api.root_pages[0].append({"trace_id": uid(4), "thread_id": None,
                             "start_time": "2026-09-02T01:00:00Z"})
    def grow(command):
        if command[2] == "/v1/trajectory" and len(api.thread_roots) == 2:
            api.thread_roots.append({**api.thread_roots[-1], "id": uid(3), "trace_id": uid(3)})
            api.trajectory_pages["next"]["messages"] += [
                {"role": "tool", "content": "result without its call", "tool_call_id": "missing"},
                *messages(3),
            ]
    api.failure = grow
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    invalid, valid = frozen["units"]
    assert "unmatched tool calls or results" in invalid["training_error"]
    assert invalid["example"]["inputs"]["messages"]
    def judge_valid(slot, prompt, tokens):
        assert json.loads(prompt[1]["content"])["untrusted_trajectory"] == valid["example"]["inputs"]["messages"]
        return judge_call(slot, prompt, tokens)
    result = triage.run_triage(source(), tmp_path, runner=api, judge_call=judge_valid, confirm=True)
    assert result["filtered_training"] == 1 and result["kept"] == 1


@pytest.mark.parametrize("foreign_source", ["trace", "thread"])
def test_snapshot_still_rejects_foreign_source_evidence(tmp_path, foreign_source):
    api = API()
    if foreign_source == "trace":
        api.trajectory_pages["next"]["messages"] += messages(3)
        error = "trajectory evidence belongs to another trace"
    else:
        api.thread_roots[0]["thread_id"] = "another-thread"
        error = "thread source query returned another thread"
    with pytest.raises(PipelineError, match=error):
        triage_source.snapshot(source(), tmp_path, runner=api)
    assert not (tmp_path / "snapshot.json").exists()


def test_baseten_council_settings_record_models_and_reasoning(tmp_path):
    settings = triage.council_settings(tmp_path, judges=["baseten:zai-org/GLM-5.3-Flash", "glm-5.3-flash", "gpt-5.6-terra"])
    plan = triage.run_triage(source(), tmp_path, runner=API(), dry_run=True, **settings)
    assert plan["coordinator"] == {"name": "judge-1", "provider": "baseten", "model": "zai-org/GLM-5.3-Flash"}
    assert plan["reasoning"]["baseten"] == "none"
    assert plan["reasoning"]["zai-org/GLM-5.3-Flash"] == "low"
    assert triage.council_settings(tmp_path)["config"] == settings["config"]


def test_baseten_aliases_match_the_default_council(tmp_path):
    settings = triage.council_settings(tmp_path, judges=["baseten-deepseek-v4.1-flash", "baseten-glm-5.3-flash"])
    assert [(j["provider"], j["model"]) for j in settings["config"]["judges"]] == [
        (j["provider"], j["model"]) for j in triage.DEFAULT_COUNCIL["judges"]]

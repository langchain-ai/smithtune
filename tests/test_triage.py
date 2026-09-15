import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from uuid import UUID

import pytest

from smithtune import cli, dataset, triage, triage_judges, triage_source
from smithtune.dataset_artifacts import load_conversation
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL


def uid(n):
    return str(UUID(int=n))


def messages(n):
    return [{"role": "human", "content": f"question-{n}", "id": f"user-{n}"},
            {"role": "ai", "content": f"answer-{n}", "id": f"ai-{n}"}]


class API:
    """Documented LangSmith CLI/API shapes, including backwards pagination."""
    def __init__(self):
        self.calls = []
        self.imported = []
        self.failure = None
        self.root_pages = [[{"trace_id": uid(2), "thread_id": "conversation-a", "start_time": "2026-09-02T00:00:00Z", "feedback_stats": None}]]
        self.thread_pages = {
            None: {"thread_id": "conversation-a", "groups": self.turn(1, 2), "cursors": {"prev": "older", "next": None}},
            "older": {"thread_id": "conversation-a", "groups": self.turn(0, 1), "cursors": {"prev": None, "next": "newer"}},
            "newer": {"thread_id": "conversation-a", "groups": self.turn(1, 2), "cursors": {"prev": "older", "next": None}},
        }

    def turn(self, index, n):
        return [{"type": "turn_boundary", "turnBoundary": {"trace_id": uid(n), "turn_index": index}},
                *[{"type": "message", "message": m} for m in messages(n)]]

    def __call__(self, command, *, capture=False, input=None):
        self.calls.append((command, input))
        if self.failure:
            self.failure(command)
        assert command[command.index("--workspace") + 1] == uid(100)
        if command[1:3] == ["thread", "messages"]:
            cursor = command[command.index("--cursor") + 1] if "--cursor" in command else None
            return SimpleNamespace(stdout=json.dumps(self.thread_pages[cursor]))
        path = command[2]
        body = json.loads(input) if input else None
        if path == "/api/v2/runs/query":
            index = int(body.get("cursor", 0))
            value = {"items": self.root_pages[index], "next_cursor": str(index + 1) if index + 1 < len(self.root_pages) else None}
        elif path.startswith("/api/v2/traces/") and "/runs?" in path:
            tid = path.split("/")[4]
            n = UUID(tid).int
            value = {"items": [{"id": tid, "trace_id": tid, "parent_run_ids": [], "is_root": True,
                               "project_id": uid(101), "run_type": "chain", "end_time": "2026-09-02T00:01:00Z",
                               "inputs": {"messages": messages(n)[:1]}, "outputs": {"messages": messages(n)}, "extra": {}},
                              {"id": uid(n + 1000), "trace_id": tid, "parent_run_ids": [tid], "is_root": False, "project_id": uid(101),
                               "run_type": "llm", "inputs": {"messages": messages(n)[:1]}, "outputs": {"messages": messages(n)[1:]},
                               "extra": {"invocation_params": {"tools": []}}}], "cursors": {}}
        elif path == "/api/v2/traces/messages":
            value = {"items": [{"trace_id": body["ids"][0], "groups": [{"type": "message", "message": m} for m in messages(UUID(body["ids"][0]).int)]}], "next_cursor": None}
        elif path == "/api/v1/datasets":
            value = {"id": uid(200)}
        elif path == "/api/v1/examples":
            self.imported.append(body)
            value = {"id": body["id"]}
        else:
            raise AssertionError(path)
        return SimpleNamespace(stdout=json.dumps(value))


def source():
    return triage_source.source_options(uid(100), uid(101), "2026-09-02T00:00:00Z", "2026-09-03T00:00:00Z")


def judge_call(judge, messages_, max_tokens):
    return {"keep": 1, "reason": "The answer completes the request."}


def run(tmp_path, api=None, **kwargs):
    return triage.run_triage(source(), tmp_path, runner=api or API(), judge_call=judge_call, confirm=True, sleeper=lambda _: None, **kwargs)


def test_snapshot_expands_to_earlier_turns_and_preserves_tree(tmp_path):
    api = API()
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    assert frozen["selected_trace_ids"] == [uid(2)]
    assert [trace["trace_id"] for trace in frozen["traces"]] == [uid(1), uid(2)]
    assert frozen["traces"][1]["turn_start"] == 2
    assert frozen["traces"][1]["messages"] == messages(1) + messages(2)
    assert frozen["traces"][1]["runs"][1]["parent_run_id"] == uid(2)
    assert frozen["units"][0]["example"]["inputs"]["messages"] == messages(1) + messages(2)
    conversation, = (tmp_path / "conversations").glob("*.json")
    assert load_conversation(conversation) == frozen["units"][0]["example"]
    api.calls.clear()
    assert triage_source.snapshot(source(), tmp_path, runner=api) == frozen
    assert api.calls == []


def test_snapshot_retries_failed_reads(tmp_path, monkeypatch):
    api = API()
    failures = []
    delays = []
    monkeypatch.setattr(triage_source.time, "sleep", delays.append)

    def flaky(command, **kwargs):
        if "/runs?" in command[2] and not failures:
            failures.append(command)
            raise subprocess.CalledProcessError(1, command, output="429: rate limit exceeded")
        return api(command, **kwargs)

    frozen = triage_source.snapshot(source(), tmp_path, runner=flaky)
    assert len(failures) == 1 and len(frozen["traces"]) == 2
    assert delays == [30]
    # A retry after an interrupted download reuses completed reads.
    (tmp_path / "snapshot.json").unlink()
    assert triage_source.snapshot(source(), tmp_path, runner=lambda *_a, **_k: pytest.fail("read repeated")) == frozen


def test_snapshot_retains_missing_root_for_judging_but_blocks_import(tmp_path):
    api = API()

    def missing_root(command, **kwargs):
        response = api(command, **kwargs)
        if "/runs?" in command[2]:
            page = json.loads(response.stdout)
            page["items"] = page["items"][1:]
            response.stdout = json.dumps(page)
        return response

    result = run(tmp_path, missing_root)
    frozen = triage_source.load_snapshot(tmp_path)
    assert all(trace["root_run_id"] is None and trace["source_warnings"] for trace in frozen["traces"])
    assert result["kept"] == 1 and result["eligible_conversations"] == 0
    assert frozen["units"][0]["training_error"] == "conversation source has a missing root run"
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)


def test_snapshot_preserves_content_unsupported_for_training(tmp_path):
    api = API()
    image = {"type": "image", "url": "https://example.invalid/image.png"}
    api.thread_pages["older"]["groups"][1]["message"]["content"] = [image]
    frozen = triage_source.snapshot(source(), tmp_path, runner=api)
    assert frozen["traces"][0]["messages"][0]["content"] == [image]
    assert frozen["units"][0]["training_error"]
    result = triage.run_triage(source(), tmp_path, runner=api, confirm=True,
                              judge_call=lambda *_: pytest.fail("multimodal trace reached a judge"))
    assert result["filtered_multimodal"] == 1 and result["incomplete"] == 0
    assert (tmp_path / "judgments.jsonl").read_text() == ""
    assert all(json.loads(line)["keep"] == 0 and "Filtered before judging" in json.loads(line)["reason"]
               for line in (tmp_path / "labels.jsonl").read_text().splitlines())


def tool_conversation(name="lookup", args=None, result=True):
    api = API()
    conversation = [
        {"role": "ai", "content": [{"type": "text", "text": "Looking it up."},
            {"type": "tool_call", "name": name, "args": args or {}, "id": "call-1"}]},
    ]
    if result:
        conversation.append({"role": "tool", "content": "done", "tool_call_id": "call-1"})
    api.thread_pages["older"]["groups"][2:2] = [
        {"type": "message", "message": message} for message in conversation
    ]
    tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a record.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}}

    def runner(command, **kwargs):
        response = api(command, **kwargs)
        if "/runs?" in command[2]:
            page = json.loads(response.stdout)
            page["items"][1]["extra"]["invocation_params"]["tools"] = [tool]
            response.stdout = json.dumps(page)
        return response

    return runner


@pytest.mark.parametrize("kwargs,error", [
    ({"name": "missing_tool"}, "unknown tool missing_tool"),
    ({"args": {"limit": "many"}}, "do not match its JSON Schema"),
    ({"result": False}, "unmatched tool calls or results"),
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
    summary = run(tmp_path, api)
    frozen = triage_source.load_snapshot(tmp_path)
    assert error in triage_source.training_error(frozen["units"][0])
    assert summary["kept"] == 1  # Compatibility does not change the quality votes.
    assert summary["eligible_conversations"] == 0
    assert summary["unsupported_training_conversations"] == 1
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)
    if cached:
        assert (tmp_path / "snapshot.json").read_bytes() == original


def test_triage_accepts_valid_tool_calls_and_keeps_saved_contract(tmp_path):
    assert run(tmp_path, tool_conversation())["eligible_conversations"] == 1
    frozen = triage_source.load_snapshot(tmp_path)
    example, = triage.selected_examples(tmp_path)
    assert example["inputs"] == frozen["units"][0]["example"]["inputs"]
    assert example["metadata"]["smithtune_triage"]["contract"] == frozen["units"][0]["contract"]
    contract = parse_inference_contract(example["metadata"]["smithtune_triage"]["contract"])
    assert dataset.prepare_sft_rows([example], contract=contract)


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
        "trajectory_id": triage_source.load_snapshot(tmp_path)["units"][0]["example"]["id"], "keep": 1, "reason": "3/3 judges voted 1. The answer completes the request.",
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
    path.write_text(json.dumps({"judges": [{"name": f"judge-{i}", "provider": "fireworks", "model": "accounts/fireworks/models/example"} for i in range(count)]}))
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
    api.thread_pages[None]["groups"][1]["message"]["content"] = "different live message"
    imported = triage.create_triaged_dataset(tmp_path / "triage", "selected", confirm=True, runner=api)
    assert imported["example_count"] == 1
    assert all(command[2] in {"/api/v1/datasets", "/api/v1/examples"} for command, _ in api.calls)
    example = api.imported[0]
    assert example["inputs"]["messages"] == messages(1) + messages(2)
    contracts = dataset.capture_example_contracts(uid(100), [example], runner=lambda *_args, **_kw: pytest.fail("must not fetch changed tools"))
    assert list(contracts[example["id"]].tools) == []
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
        examples.append(item)
    monkeypatch.setattr(dataset, "_load_dataset_source", lambda *_a, **_kw: ({"name": "selected"}, examples, "snapshot"))
    manifest = dataset.prepare_dataset(uid(100), uid(200), DEFAULT_MODEL, tmp_path / "data", fetch=True, check_render=False)
    assert manifest["prepared"]["accepted"] == 3
    assert manifest["split"]["train"] == manifest["split"]["validation"] == manifest["split"]["test"] == 1


def test_changed_triaged_messages_are_rejected(tmp_path):
    run(tmp_path)
    examples = triage.selected_examples(tmp_path)
    examples[0]["inputs"]["messages"][-1]["content"] = "unjudged change"
    with pytest.raises(PipelineError, match="changed after judging"):
        dataset.capture_example_contracts(uid(100), examples)


def test_changed_conversation_file_blocks_import_before_writes(tmp_path):
    run(tmp_path)
    path, = (tmp_path / "conversations").glob("*.json")
    example = load_conversation(path)
    example["inputs"]["messages"][-1]["content"] = "changed since judging"
    path.write_text(json.dumps(example))
    with pytest.raises(PipelineError, match="saved conversation has changed"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=True,
            runner=lambda *_a, **_kw: pytest.fail("changed conversation reached upload"))


def test_old_snapshot_materializes_conversation_without_refetching(tmp_path):
    run(tmp_path)
    snapshot_bytes = (tmp_path / "snapshot.json").read_bytes()
    for path in (tmp_path / "conversations").glob("*.json"):
        path.unlink()
    example, = triage.selected_examples(tmp_path)
    saved, = (tmp_path / "conversations").glob("*.json")
    assert load_conversation(saved)["inputs"] == example["inputs"]
    assert (tmp_path / "snapshot.json").read_bytes() == snapshot_bytes


def test_cli_default_run_directories_are_unique_and_can_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    original = triage.run_triage
    monkeypatch.setattr(triage, "run_triage", lambda *args, **kwargs: original(
        *args, **kwargs, runner=API(), judge_call=judge_call))
    args = ["dataset", "triage", "--workspace-id", uid(100), "--project-id", uid(101),
            "--start-time", source()["start_time"], "--end-time", source()["end_time"]]
    directories = []
    for _ in range(2):
        cli.main(args)
        run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
        assert run_dir.parent == Path("data/datasets")
        assert (run_dir / "snapshot.json").exists()
        assert len(list((run_dir / "conversations").glob("*.json"))) == 1
        directories.append(run_dir)
    assert directories[0] != directories[1]
    cli.main(["dataset", "triage", str(directories[0]), "--confirm"])
    assert json.loads(capsys.readouterr().out)["run_dir"] == str(directories[0])
    with pytest.raises(SystemExit):
        cli.main(["dataset", "triage", "--confirm"])
    assert "supply a saved run directory" in capsys.readouterr().err


def test_coordinator_skill_changes_require_a_new_run(tmp_path, monkeypatch):
    resource_dir = tmp_path / "resources"
    skill_path = resource_dir / "skills/sft-trace-triage/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    current = triage.files("smithtune").joinpath("skills/sft-trace-triage/SKILL.md").read_text()
    skill_path.write_text(current)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(triage.load_config(None)))
    run_dir = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(triage, "files", lambda _: resource_dir)
        run(run_dir, runner_mode="deepagent", config_path=config_path)
    saved_votes = (run_dir / "judgments.jsonl").read_bytes()
    assert run(run_dir, runner_mode="deepagent", config_path=config_path)["status"] == "complete"
    assert (run_dir / "judgments.jsonl").read_bytes() == saved_votes
    skill_path.write_text(current.replace("strict majority", "unanimous vote"))
    monkeypatch.setattr(triage, "files", lambda _: resource_dir)
    with pytest.raises(PipelineError, match="different input.*new output directory"):
        run(run_dir, runner_mode="deepagent", config_path=config_path)


@pytest.mark.parametrize("fetch", [False, True])
@pytest.mark.parametrize("global_contract", [False, True])
def test_prepare_always_checks_triaged_message_integrity(tmp_path, monkeypatch, fetch, global_contract):
    run(tmp_path)
    examples = triage.selected_examples(tmp_path)
    contract = parse_inference_contract(examples[0]["metadata"]["smithtune_triage"]["contract"]) if global_contract else None
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
        value["traces"][0]["messages"][0]["content"] = "edited"
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


def test_provider_context_rejection_filters_without_shortening_or_retrying(tmp_path):
    api = API()
    api.thread_pages["older"]["groups"][1]["message"]["content"] = "x" * 20_000
    calls = []

    class ContextError(Exception):
        status_code = 400
        body: ClassVar[dict] = {"error": {"code": "context_length_exceeded"}}

    def reject(judge, prompt, tokens):
        assert json.loads(prompt[-1]["content"])["untrusted_trajectory"][0]["content"] == "x" * 20_000
        calls.append(judge["name"])
        raise ContextError()

    result = triage.run_triage(source(), tmp_path, runner=api, judge_call=reject, confirm=True)
    assert result["incomplete"] == 0 and result["filtered_context"] == 1
    assert len(calls) == 3
    label = json.loads((tmp_path / "labels.jsonl").read_text())
    assert label["keep"] == 0 and "context window" in label["reason"]
    assert triage.run_triage(source(), tmp_path, confirm=True,
        judge_call=lambda *_: pytest.fail("filtered trajectory retried"))["filtered_context"] == 1
    with pytest.raises(PipelineError, match="no complete, kept"):
        triage.selected_examples(tmp_path)


def test_skill_export_works_outside_checkout(tmp_path):
    result = triage.export_skill(tmp_path)
    assert Path(result["skill"]).read_text().startswith("---\nname: sft-trace-triage")
    assert (tmp_path / "sft-trace-triage/judge.md").exists()
    exported = json.loads((tmp_path / "sft-trace-triage/config.example.json").read_text())
    assert exported == triage.load_config(None) == triage.council_settings(tmp_path / "new")["config"]
    assert [(j["provider"], j["model"]) for j in exported["judges"]] == [
        ("fireworks", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("fireworks", "accounts/fireworks/models/muse-glimmer-30b"),
        ("openai", "gpt-5.6-terra"),
    ]
    with pytest.raises(PipelineError, match="already exists"):
        triage.export_skill(tmp_path)


def test_cli_triage_and_dataset_handoff(tmp_path, monkeypatch, capsys):
    api = API()
    original = triage.run_triage
    monkeypatch.setattr(triage, "run_triage", lambda *args, **kwargs: original(*args, **kwargs, runner=api, judge_call=judge_call))
    args = ["dataset", "triage", "--workspace-id", uid(100), "--project-id", uid(101),
            "--start-time", source()["start_time"], "--end-time", source()["end_time"], str(tmp_path),
            "--judges", "deepseek-v4.1-flash,muse-glimmer-30b,gpt-5.6-terra", "--rule", "Keep supported answers."]
    cli.main(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["judges"] == 3 and plan["runner"] == "deepagent"
    assert plan["config"]["rules"] == ["Keep supported answers."]
    assert plan["config"]["judges"] == triage.load_config(None)["judges"]
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 1
    saved = (tmp_path / "judgments.jsonl").read_bytes()
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 1
    assert (tmp_path / "judgments.jsonl").read_bytes() == saved
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert all(set(label) == {"trajectory_id", "keep", "reason"} for label in labels)
    assert all(label["reason"] in (tmp_path / "report.md").read_text() for label in labels)
    # A changed package default must not replace a saved council.
    monkeypatch.setattr(triage, "load_config", lambda *_: pytest.fail("saved council ignored"))
    assert triage.council_settings(tmp_path)["config"] == plan["config"]
    create = triage.create_triaged_dataset
    monkeypatch.setattr(triage, "create_triaged_dataset", lambda *args, **kwargs: create(*args, **kwargs, runner=api))
    cli.main(["dataset", "create", "--triage-dir", str(tmp_path), "--name", "selected", "--confirm"])
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
    original = triage.run_triage
    monkeypatch.setattr(triage, "run_triage", lambda *args, **kwargs: original(*args, **kwargs, runner=API(), judge_call=lambda *_: {}))
    args = ["dataset", "triage", "--workspace-id", uid(100), "--project-id", uid(101),
            "--start-time", source()["start_time"], "--end-time", source()["end_time"], "--output-dir", str(tmp_path), "--confirm", "--attempts", "1"]
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["incomplete"] == 1


def test_dataset_import_failure_has_receipt_and_cannot_repeat_writes(tmp_path):
    run(tmp_path)
    api = API()

    def fail(command):
        if command[2] == "/api/v1/examples":
            raise PipelineError("connection lost")

    api.failure = fail
    with pytest.raises(PipelineError, match="import incomplete"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=True, runner=api)
    receipt = json.loads((tmp_path / "dataset-import.json").read_text())
    assert receipt["dataset_id"] == uid(200)
    assert receipt["status"] == "incomplete" and receipt["pending_write"]
    count = len(api.calls)
    with pytest.raises(PipelineError, match="cannot create"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=True, runner=api)
    assert len(api.calls) == count


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
    with pytest.raises(PipelineError, match="page limit"):
        triage_source.snapshot(source(), tmp_path, runner=API())
    assert not (tmp_path / "snapshot.json").exists()


@pytest.mark.parametrize("provider,url,key", [
    ("fireworks", "https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY"),
    ("openai", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    ("anthropic", "https://api.anthropic.com/v1/messages", "SMITHTUNE_ANTHROPIC_API_KEY"),
    ("anthropic-gateway", "https://gateway.smith.langchain.com/anthropic/v1/messages", "ANTHROPIC_API_KEY"),
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
    if provider == "fireworks":
        assert json.loads(requests[0].data)["reasoning_effort"] == "none"


def test_missing_thread_pages_and_changed_turns_fail_closed(tmp_path):
    api = API()
    api.thread_pages[None]["cursors"] = {"next": None, "prev": None}
    with pytest.raises(PipelineError, match="incomplete"):
        triage_source.snapshot(source(), tmp_path, runner=api)
    assert not (tmp_path / "snapshot.json").exists()


def test_paid_and_remote_writes_require_confirmation(tmp_path, monkeypatch):
    with pytest.raises(PipelineError, match="incurs cost"):
        triage.run_triage(source(), tmp_path, runner=API())
    with pytest.raises(PipelineError, match="requires --confirm"):
        triage.create_triaged_dataset(tmp_path, "selected", confirm=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="OPENAI_API_KEY"):
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
    original = triage.run_triage
    monkeypatch.setattr(triage, "run_triage", lambda *args, **kwargs: original(
        *args, **kwargs, judge_call=judge_call,
        runner=lambda *_a, **_kw: pytest.fail("local snapshot must not query LangSmith")))
    cli.main(["dataset", "triage", "--output-dir", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 1


def test_cli_requires_source_when_no_snapshot_exists(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["dataset", "triage", "--output-dir", str(tmp_path), "--dry-run"])
    assert "no local snapshot" in capsys.readouterr().err


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
        if media_turn and f"/traces/{uid(media_turn)}/runs?" in command[2]:
            page = json.loads(response.stdout)
            page["items"][-1]["attachments"] = {"image.png": "recorded"}
            response.stdout = json.dumps(page)
        return response

    seen = []

    def judge(slot, prompt, tokens):
        evidence = json.loads(prompt[-1]["content"])["untrusted_trajectory"]
        seen.append((slot["name"], evidence))
        if len(evidence) == 4:
            assert media_turn is None
            assert [m["content"] for m in evidence] == ["question-1", "answer-1", "question-2", "answer-2"]
        return judge_call(slot, prompt, tokens)

    summary = triage.run_triage(source(), tmp_path, runner=source_api, judge_call=judge, confirm=True)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["selected_traces"] == 3 and plan["source_traces"] == 3
    assert plan["trajectories"] == 2
    assert plan["judge_tasks"] == len(seen) == (3 if media_turn else 6)
    assert summary["filtered_multimodal"] == int(media_turn is not None)
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert len(labels) == 2
    examples = triage.selected_examples(tmp_path)
    assert {e["id"] for e in examples} == {label["trajectory_id"] for label in labels if label["keep"]}
    for example in examples:
        judged = [e for _, e in seen if e == example["inputs"]["messages"]]
        assert len(judged) == 3
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
])
def test_only_context_rejections_filter_trajectories(status, body, expected):
    error = RuntimeError("provider error")
    error.status_code, error.body = status, body
    assert triage_judges.context_window_exceeded(error) is expected

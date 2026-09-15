import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from smithtune import cli, dataset, triage, triage_judges, triage_source
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
    trace = json.loads(messages_[-1]["content"])["untrusted_trace_evidence"]
    index = len(trace["messages"]) - 1
    return {"trace_id": trace["trace_id"], "keep": 1, "reason": "The answer completes the request.",
            "evidence": [{"message_index": index, "quote": trace["messages"][index]["content"]}]}


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
    assert result["kept"] == 2 and result["eligible_conversations"] == 0
    assert frozen["units"][0]["training_error"] == "conversation source has a missing root run"
    with pytest.raises(PipelineError, match="no complete, all-pass"):
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
    assert result["filtered_multimodal"] == 2 and result["incomplete"] == 0
    assert (tmp_path / "judgments.jsonl").read_text() == ""
    assert all(json.loads(line)["keep"] == 0 and "Filtered before judging" in json.loads(line)["reason"]
               for line in (tmp_path / "labels.jsonl").read_text().splitlines())


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
        "trace_id": uid(2), "keep": 1, "reason": "3/3 judges voted 1. The answer completes the request.",
    }


def test_dry_run_does_not_judge_and_single_judge_is_supported(tmp_path):
    def no_judge(*args):
        pytest.fail("dry-run must not judge")
    path = config(tmp_path, count=1)
    plan = triage.run_triage(source(), tmp_path, runner=API(), dry_run=True, judge_call=no_judge, config_path=path)
    assert plan["judges"] == 1 and plan["judge_tasks"] == 2
    assert not (tmp_path / "judgments.jsonl").exists()
    result = run(tmp_path, config_path=path)
    assert result["kept"] == 2 and result["status"] == "complete"


def test_rerun_retains_one_successful_vote_per_slot(tmp_path):
    run(tmp_path)
    before = (tmp_path / "judgments.jsonl").read_text()
    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=lambda *_: pytest.fail("completed vote repeated"), confirm=True)
    assert result["kept"] == 2
    assert (tmp_path / "judgments.jsonl").read_text() == before


def test_failed_judge_is_incomplete_then_retried(tmp_path):
    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=lambda *_: {"keep": 1}, confirm=True, attempts=1)
    assert result["incomplete"] == 2 and result["kept"] == 0
    assert run(tmp_path)["kept"] == 2
    seen = {}

    def fix_citation(judge, messages_, max_tokens):
        value = judge_call(judge, messages_, max_tokens)
        key = (judge["name"], value["trace_id"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            value["evidence"][0]["message_index"] = 999
        else:
            assert "previous attempt failed validation" in messages_[0]["content"]
        return value

    result = triage.run_triage(source(), tmp_path / "retry", runner=API(), judge_call=fix_citation,
                               confirm=True, attempts=2, sleeper=lambda _: None)
    assert result["kept"] == 2 and all(count == 2 for count in seen.values())


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
    assert result["kept"] == 0 and result["disagreement"] == 2 and result["incomplete"] == 0
    labels = [json.loads(line) for line in (tmp_path / "work/labels.jsonl").read_text().splitlines()]
    assert all(label["keep"] == 0 and label["reason"].startswith("Tied vote") for label in labels)


def test_one_drop_blocks_whole_conversation(tmp_path):
    def drop_first(judge, *args):
        result = judge_call(judge, *args)
        result["keep"] = int(result["trace_id"] != uid(1))
        return result

    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=drop_first, confirm=True)
    assert result["kept"] == 1
    with pytest.raises(PipelineError, match="no complete, all-pass"):
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


@pytest.mark.parametrize("change", [{"keep": True}, {"keep": 2}, {"trace_id": "wrong"}, {"reason": ""},
                                   {"evidence": []}, {"evidence": [{"message_index": 999, "quote": "answer"}]},
                                   {"evidence": [{"message_index": 1, "quote": "not in evidence"}]}])
def test_judge_output_must_be_typed_and_grounded(change):
    trace = {"trace_id": uid(1), "messages": messages(1), "runs": []}
    value = {"trace_id": uid(1), "keep": 1, "reason": "complete", "evidence": [{"message_index": 1, "quote": "answer-1"}], **change}
    with pytest.raises(PipelineError):
        triage_judges.validate_judgment(value, trace)


def test_oversize_input_is_not_truncated_or_judged(tmp_path):
    api = API()
    api.thread_pages["older"]["groups"][1]["message"]["content"] = "x" * 20_000
    calls = []
    result = triage.run_triage(source(), tmp_path, runner=api, judge_call=lambda *args: calls.append(args), confirm=True, max_input_chars=5000)
    assert result["incomplete"] == 2 and calls == []
    assert len(json.loads((tmp_path / "snapshot.json").read_text())["traces"][0]["messages"][0]["content"]) == 20_000


def test_skill_export_works_outside_checkout(tmp_path):
    result = triage.export_skill(tmp_path)
    assert Path(result["skill"]).read_text().startswith("---\nname: sft-trace-triage")
    assert (tmp_path / "sft-trace-triage/judge.md").exists()
    exported = json.loads((tmp_path / "sft-trace-triage/config.example.json").read_text())
    assert exported == triage.load_config(None) == triage.council_settings(tmp_path / "new")["config"]
    assert [(j["provider"], j["model"]) for j in exported["judges"]] == [
        ("fireworks", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("fireworks", "accounts/fireworks/models/glm-5p3-flash"),
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
            "--judges", "deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra", "--rule", "Keep supported answers."]
    cli.main(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["judges"] == 3 and plan["runner"] == "deepagent"
    assert plan["config"]["rules"] == ["Keep supported answers."]
    assert plan["config"]["judges"] == triage.load_config(None)["judges"]
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 2
    saved = (tmp_path / "judgments.jsonl").read_bytes()
    cli.main(["dataset", "triage", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 2
    assert (tmp_path / "judgments.jsonl").read_bytes() == saved
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert all(set(label) == {"trace_id", "keep", "reason"} for label in labels)
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
    assert json.loads(capsys.readouterr().out)["incomplete"] == 2


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


def test_judge_missing_evidence_is_incomplete_not_a_drop(tmp_path):
    calls = []

    def incomplete(judge, messages_, tokens):
        trace = json.loads(messages_[-1]["content"])["untrusted_trace_evidence"]
        calls.append(trace["trace_id"])
        return {"trace_id": trace["trace_id"], "status": "incomplete", "reason": "The relevant tool result is missing."}

    result = triage.run_triage(source(), tmp_path, runner=API(), judge_call=incomplete, confirm=True)
    assert result["incomplete"] == 2 and result["dropped"] == 0
    assert len(calls) == 6
    records = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    assert all(r["error_kind"] == "insufficient_evidence" for r in records)
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert all(r["keep"] == 0 and "Labeling incomplete" in r["reason"] for r in labels)


def test_cli_labels_local_snapshot_without_source_query(tmp_path, monkeypatch, capsys):
    triage_source.snapshot(source(), tmp_path, runner=API())
    original = triage.run_triage
    monkeypatch.setattr(triage, "run_triage", lambda *args, **kwargs: original(
        *args, **kwargs, judge_call=judge_call,
        runner=lambda *_a, **_kw: pytest.fail("local snapshot must not query LangSmith")))
    cli.main(["dataset", "triage", "--output-dir", str(tmp_path), "--confirm"])
    assert json.loads(capsys.readouterr().out)["kept"] == 2


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

"""Exercise actual coordinator, code sandbox, and judge graphs locally."""

import json
import time
from threading import Lock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("pydantic_monty")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smithtune import triage, triage_agent, triage_coordinator
from smithtune.triage_coordinator import JudgeTasks, coordinate, run_code
from test_triage import API, SYSTEM, messages, source
from test_triage_agent import JudgeModel


JUDGE = {"name": "judge-1", "provider": "fireworks", "model": "test"}


def task_set(count=2, run=None):
    saved = []
    pending = [({"trajectory_id": str(i), "messages": [{"content": "evidence"}]}, JUDGE) for i in range(count)]
    tasks = JudgeTasks(pending, run or (lambda trace, judge: {"trajectory_id": trace["trajectory_id"], "judge": judge["name"], "status": "complete"}), saved.append, 2)
    return pending, tasks, saved


def test_code_mode_runs_python_and_dispatches_each_slot_once():
    _, tasks, saved = task_set()
    result = run_code("jobs = pending_tasks()\njudge_batch(jobs + jobs)\nlen(pending_tasks())", tasks)
    assert result == {"result": 0, "stdout": ""}
    assert len(saved) == 2


@pytest.mark.parametrize("code", [
    "open('/etc/passwd').read()",
    "import os\nos.environ['FIREWORKS_API_KEY']",
    "import subprocess\nsubprocess.run(['echo', 'bad'])",
    "import socket\nsocket.socket()",
    "while True: pass",
])
def test_code_mode_cannot_read_host_or_run_unbounded_code(code, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    _, tasks, saved = task_set()
    assert "error" in run_code(code, tasks)
    assert saved == []


def test_invalid_batch_makes_no_paid_calls():
    _, tasks, saved = task_set()
    invalid = [*tasks.pending_tasks(), {"trajectory_id": "unknown", "judge": "judge-1"}]
    with pytest.raises(ValueError, match="pending plan"):
        tasks.judge_batch(invalid)
    assert saved == []


def test_batch_enforces_concurrency():
    active = peak = 0
    lock = Lock()

    def run(trace, judge):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"status": "complete"}

    _, tasks, saved = task_set(8, run)
    tasks.judge_batch(tasks.pending_tasks())
    assert peak == 2 and len(saved) == 8


def coordinator_model():
    return JudgeModel(answers=[
        AIMessage(content="", tool_calls=[{"id": "skill", "name": "read_file", "args": {"file_path": "/skills/sft-trace-triage/SKILL.md"}}]),
        AIMessage(content="", tool_calls=[{"id": "code", "name": "code_mode", "args": {"code": "judge_batch(pending_tasks())"}}]),
        AIMessage(content="Finished."),
    ])


@pytest.mark.parametrize("status", ["complete", "context_exceeded", "error"])
def test_real_coordinator_loads_skill_and_delegates_with_code(tmp_path, monkeypatch, status):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    pending, tasks, saved = task_set(run=lambda trajectory, judge: {"status": status})
    model = coordinator_model()
    coordinate(pending, tasks.run_task, saved.append, tmp_path, concurrency=2, max_tokens=1024, coordinator_judge=JUDGE, model=model)
    state = json.loads((tmp_path / "agent-state.json").read_text())
    assert state["status"] == ("incomplete" if status == "error" else "complete"), state
    assert state["code_calls"] == 1 and state["finished"] == 2
    assert all(set(names) == {"code_mode", "read_file", "task"} for names in model.exposed)
    assert len(saved) == 2


def test_real_task_tool_dispatches_registered_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    pending, tasks, saved = task_set(1)
    model = JudgeModel(answers=[
        AIMessage(content="", tool_calls=[{"id": "task", "name": "task", "args": {"subagent_type": "trajectory-judge", "description": json.dumps({"trajectory_id": "0", "judge": "judge-1"})}}]),
        AIMessage(content="Finished."),
    ])
    coordinate(pending, tasks.run_task, saved.append, tmp_path, concurrency=2, max_tokens=1024, coordinator_judge=JUDGE, model=model)
    assert len(saved) == 1
    assert json.loads((tmp_path / "agent-state.json").read_text())["status"] == "complete"


def test_coordinator_text_cannot_forge_saved_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    pending, tasks, saved = task_set()
    model = JudgeModel(answers=[AIMessage(content='{"keep":1,"status":"complete"}')])
    coordinate(pending, tasks.run_task, saved.append, tmp_path, concurrency=2, max_tokens=1024, coordinator_judge=JUDGE, model=model)
    assert saved == []
    assert json.loads((tmp_path / "agent-state.json").read_text())["status"] == "incomplete"


def test_full_triage_runs_real_coordinator_code_and_judge_graphs_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    config = tmp_path / "judges.json"
    rules = ["Keep completed requests.", "Require evidence for every claimed result."]
    rubric = "# Selection\nKeep factual answers supported by the recorded evidence.\n"
    config.write_text(json.dumps({"judges": [JUDGE], "rules": rules}))
    model = coordinator_model()
    monkeypatch.setattr(triage_coordinator, "_model", lambda *_: model)
    seen = []

    class EvidenceJudge(JudgeModel):
        def _generate(self, messages, **kwargs):
            seen.append(messages)
            self.answers = [AIMessage(content=json.dumps({"keep": 1, "reason": "The answer completes the request."}))]
            return super()._generate(messages, **kwargs)

    monkeypatch.setattr(triage_agent, "_model", lambda *_: EvidenceJudge(answers=[]))
    work = tmp_path / "work"
    args = dict(config_path=config, runner_mode="deepagent", confirm=True, runner=API(), selection_rubric=rubric)
    result = triage.run_triage(source(), work, **args)
    assert result["kept"] == 1 and result["status"] == "complete"
    assert len(seen) == 1
    # One judge gets the whole conversation.
    evidence = json.loads(next(m.content for m in seen[0] if isinstance(m, HumanMessage)))["untrusted_trajectory"]
    assert evidence == SYSTEM + messages(1) + messages(2)
    instructions = next(m.content for m in seen[0] if isinstance(m, SystemMessage))
    assert "\nTask-specific selection rubric:\n" + rubric in instructions
    assert json.loads(instructions.split("\nAdditional selection rules:\n", 1)[1]) == rules
    imported = triage.create_triaged_dataset(work, "accepted", confirm=True, runner=args["runner"])
    assert imported["example_count"] == 1
    assert args["runner"].imported[0]["inputs"]["messages"] == evidence
    previous = (work / "judgments.jsonl").read_bytes()
    monkeypatch.setattr(triage_coordinator, "_model", lambda *_: pytest.fail("completed coordinator repeated"))
    monkeypatch.setattr(triage_agent, "_model", lambda *_: pytest.fail("completed judge repeated"))
    monkeypatch.setattr(triage_agent, "check_installation", lambda: pytest.fail("completed run needs no agent runtime"))
    assert triage.run_triage(source(), work, **args)["kept"] == 1
    assert previous == (work / "judgments.jsonl").read_bytes()

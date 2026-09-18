"""Local council review, confirmation and evidence-bound resumability."""
import copy
import json

import pytest

from smithtune import checkpoint, dataset_workflow as workflow, triage
from smithtune.artifacts import _json_dump
from smithtune.dataset_artifacts import save_conversation
from smithtune.providers.base import PipelineError
from curation_fakes import API, SOURCE


COUNCIL = ['fireworks:test-one', 'openai:test-two', 'fireworks:test-three']


def pulled(tmp_path):
    api = API()
    workflow.run('pull', tmp_path, runner=api, **SOURCE)
    return api


def keep(*_):
    return {'keep': 1, 'reason': 'Completes the whole request.'}


def test_preview_confirm_and_flagless_resume_are_local(tmp_path):
    api = pulled(tmp_path)
    api.calls.clear()
    preview = workflow.run('triage', tmp_path, judges=COUNCIL, runner=api)
    assert preview['triage']['pending_votes'] == 3 and preview['triage']['accepted'] is None
    assert not (tmp_path / 'triage.jsonl').exists()
    prompts = []
    def judge(slot, messages, tokens):
        prompts.append(messages)
        evidence = json.loads(messages[-1]['content'])
        assert evidence['untrusted_trajectory'] == api.messages
        assert [len(b['tools']) for b in evidence['untrusted_assistant_tool_bindings']] == [1, 2, 0]
        return keep()
    result = workflow.run('triage', tmp_path, confirm=True, runner=api, judge_call=judge)
    assert result['triage']['kept'] == 1
    assert len(prompts) == 3 and not api.calls
    assert len((tmp_path / 'triage.jsonl').read_text().splitlines()) == 3
    assert {p.name for p in tmp_path.iterdir()} == {'.smithtune.lock', 'checkpoint.json', 'triage.jsonl', 'conversations'}
    again = workflow.run('resume', tmp_path, confirm=True, runner=lambda *_a, **_k: pytest.fail('source read'), judge_call=lambda *_: pytest.fail('repeat'))
    assert again['status'] == 'complete'


def test_successful_votes_survive_errors_and_software_changes(tmp_path, monkeypatch):
    pulled(tmp_path)
    called = []
    def partial(slot, *_):
        called.append(slot['name'])
        if slot['name'] == 'judge-2':
            raise RuntimeError('temporary')
        return keep()
    value = workflow.run('triage', tmp_path, judges=COUNCIL, attempts=1, confirm=True, judge_call=partial)
    assert value['status'] == 'incomplete'
    with pytest.raises(PipelineError, match='incomplete'):
        triage.selected_examples(tmp_path, checkpoint.load(tmp_path))
    monkeypatch.setattr(triage, 'rubric_text', lambda: 'NEW SOFTWARE RUBRIC')
    called.clear()
    value = workflow.run('triage', tmp_path, concurrency=1, confirm=True, judge_call=lambda slot, *_: (called.append(slot['name']), keep())[1])
    assert called == ['judge-2'] and value['triage']['kept'] == 1
    assert checkpoint.load(tmp_path)['council']['rubric'] != 'NEW SOFTWARE RUBRIC'
    with pytest.raises(PipelineError, match='frozen'):
        workflow.run('triage', tmp_path, rules=['new rule'])


def test_tie_drops_and_context_rejection_is_terminal(tmp_path):
    pulled(tmp_path)
    value = workflow.run('triage', tmp_path, judges=COUNCIL[:2], confirm=True,
                         judge_call=lambda slot, *_: {'keep': int(slot['name'] == 'judge-1'), 'reason': 'judgment'})
    assert value['triage']['kept'] == 0 and 'Tied vote' in value['triage']['results'][0]['reason']
    directory = tmp_path / 'context'
    pulled(directory)
    def reject(*_):
        error = RuntimeError('maximum context length exceeded')
        error.status_code = 400
        error.body = {'error': {'code': 'context_length_exceeded'}}
        raise error
    result = workflow.run('triage', directory, judges=COUNCIL, confirm=True, judge_call=reject)
    assert result['triage']['filtered_context'] == 1
    assert workflow.run('triage', directory, confirm=True, judge_call=lambda *_: pytest.fail('repeat'))['status'] == 'complete'


@pytest.mark.parametrize('change', ['messages', 'tools'])
def test_changed_messages_or_bindings_invalidate_votes(tmp_path, change):
    pulled(tmp_path)
    workflow.run('triage', tmp_path, confirm=True, judge_call=keep)
    cp = checkpoint.load(tmp_path)
    ex, _ = next(iter(checkpoint.conversations(tmp_path, cp).values()))
    if change == 'messages':
        ex['inputs']['messages'][1]['content'] = 'new'
    else:
        ex['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = []
    path = save_conversation(tmp_path, ex)
    cp['selection'][0]['file'] = str(path.relative_to(tmp_path))
    _json_dump(tmp_path / 'checkpoint.json', cp)
    with pytest.raises(PipelineError, match='changed after judging'):
        workflow.run('resume', tmp_path, confirm=True)


def test_saved_triage_provenance_covers_tools(tmp_path):
    pulled(tmp_path)
    workflow.run('triage', tmp_path, confirm=True, judge_call=keep)
    selected, = triage.selected_examples(tmp_path, checkpoint.load(tmp_path))
    assert len(selected['metadata']['smithtune_triage']['votes']) == 3
    from smithtune.dataset import validate_trajectories
    modified = copy.deepcopy(selected)
    modified['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = []
    with pytest.raises(PipelineError, match='evidence changed'):
        validate_trajectories([modified], 1)

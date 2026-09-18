"""Resumable push reconciles server state without a destination cache or log."""
import copy
import subprocess

import pytest

from smithtune import checkpoint, dataset_workflow as workflow, dataset_import
from smithtune.providers.base import PipelineError
from curation_fakes import API, SOURCE, DATASET


def pulled(tmp_path):
    api = API()
    workflow.run('pull', tmp_path, runner=api, **SOURCE)
    return api


def test_preview_then_push_and_network_free_completed_resume(tmp_path):
    api = pulled(tmp_path)
    result = workflow.run('push', tmp_path, name='destination', runner=api)
    assert result['status'] == 'preview' and result['created'] == 1 and not api.writes
    result = workflow.run('push', tmp_path, confirm=True, runner=api)
    assert result['status'] == 'complete' and result['created'] == result['final_size'] == 1
    example, = api.examples.values()
    assert example['inputs']['messages'] == api.messages and example['outputs'] is None
    assert len(example['metadata']['smithtune_source']['assistant_runs']) == 3
    api.calls.clear()
    assert workflow.run('resume', tmp_path, confirm=True, runner=api)['status'] == 'complete'
    assert not api.calls
    result = workflow.run('push', tmp_path, confirm=True, runner=api)
    assert result['skipped'] == result['existing_before'] == result['final_size'] == 1
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint.json', 'conversations', '.smithtune.lock'}


@pytest.mark.parametrize('stage', ['dataset', 'example'])
@pytest.mark.parametrize('persist', [False, True])
def test_ambiguous_writes_reconcile_and_preserve_ids(tmp_path, stage, persist):
    api = pulled(tmp_path)
    original = api.__call__
    failed = False
    def ambiguous(command, **kwargs):
        nonlocal failed
        path = command[2]
        method = command[command.index('--method') + 1]
        route = '/api/v1/datasets' if stage == 'dataset' else '/api/v1/examples'
        if not failed and method == 'POST' and path == route:
            failed = True
            if persist:
                original(command, **kwargs)
            raise subprocess.CalledProcessError(1, command, stderr='HTTP 409' if persist else 'timeout')
        return original(command, **kwargs)
    first = workflow.run('push', tmp_path, name='destination', confirm=True, runner=ambiguous)
    if stage == 'dataset' or not persist:
        assert first['status'] == 'incomplete'
        pending = checkpoint.load(tmp_path)['pending_write']
        assert pending['kind'] == stage
        second = workflow.run('resume', tmp_path, confirm=True, runner=api)
        assert second['status'] == 'complete'
        values = api.datasets if stage == 'dataset' else api.examples
        assert pending['id'] in values
    else:
        assert first['status'] == 'complete'
    assert len(api.datasets) == len(api.examples) == 1


def test_zero_eligible_does_not_create_remote_dataset(tmp_path):
    api = pulled(tmp_path)
    workflow.run('triage', tmp_path, confirm=True, judge_call=lambda *_: {'keep': 0, 'reason': 'wrong answer'})
    result = workflow.run('push', tmp_path, name='empty', confirm=True, runner=api)
    assert result['eligible'] == 0 and not api.writes


def test_name_collision_requires_explicit_id_and_destination_is_bound(tmp_path):
    api = pulled(tmp_path)
    api.datasets[DATASET] = {'id': DATASET, 'name': 'collision', 'data_type': 'kv'}
    first = workflow.run('push', tmp_path, name='collision', confirm=True, runner=api)
    assert first['status'] == 'incomplete' and 'collision' in first['error'] and not api.writes
    assert workflow.run('push', tmp_path, dataset_id=DATASET, confirm=True, runner=api)['status'] == 'complete'
    with pytest.raises(PipelineError, match='already bound'):
        workflow.run('push', tmp_path, name='different', runner=api)


def test_changed_tools_conflict_and_extension_preserves_remote_metadata(tmp_path):
    api = pulled(tmp_path)
    workflow.run('push', tmp_path, name='destination', confirm=True, runner=api)
    _eid, remote = next(iter(api.examples.items()))
    remote['metadata']['unrelated'] = 'keep'
    incoming = copy.deepcopy(remote)
    incoming['inputs']['messages'] += [{'role': 'human', 'content': 'more'}, {'role': 'ai', 'content': 'new'}]
    incoming['metadata']['smithtune_source']['assistant_runs'].append({'message_index': 7, 'run_id': 'new', 'trace_id': 'trace', 'tools': []})
    action, body = dataset_import._action(incoming, remote, triaged=False)
    assert action == 'updated' and body['metadata']['unrelated'] == 'keep'
    incoming['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = []
    with pytest.raises(PipelineError, match='prior tool bindings changed'):
        dataset_import._action(incoming, remote, triaged=False)
    del remote['metadata']['smithtune_source']
    with pytest.raises(PipelineError, match='legacy union'):
        dataset_import._action(incoming, remote, triaged=False)


def test_interruption_after_remote_write_resumes_as_skipped(tmp_path, monkeypatch):
    api = pulled(tmp_path)
    original = checkpoint.save
    def interrupt(directory, value):
        if any(s.get('upload') for s in value['selection']):
            raise KeyboardInterrupt
        original(directory, value)
    monkeypatch.setattr(dataset_import, 'save', interrupt)
    with pytest.raises(KeyboardInterrupt):
        workflow.run('push', tmp_path, name='destination', confirm=True, runner=api)
    monkeypatch.setattr(dataset_import, 'save', original)
    result = workflow.run('resume', tmp_path, confirm=True, runner=api)
    assert result['skipped'] == 1 and result['created'] == 0


def test_incomplete_council_never_pushes(tmp_path):
    api = pulled(tmp_path)
    workflow.run('triage', tmp_path)
    result = workflow.run('push', tmp_path, name='destination', confirm=True, runner=api)
    assert result['status'] == 'incomplete' and not api.writes


def test_full_destination_count_includes_unrelated_examples(tmp_path):
    api = pulled(tmp_path)
    workflow.run('push', tmp_path, name='destination', confirm=True, runner=api)
    destination = next(iter(api.datasets))
    from curation_fakes import uid
    for i in range(200, 301):
        api.examples[uid(i)] = {'id': uid(i), 'dataset_id': destination, 'inputs': {'unrelated': True}, 'metadata': {}}
    result = workflow.run('push', tmp_path, confirm=True, runner=api)
    assert result['existing_before'] == result['final_size'] == 102
    assert 'exceeds' in result['warning']
    assert not (tmp_path / 'destination').exists()

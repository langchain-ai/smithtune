"""Pull checkpoints replace selections, run caches, and duplicate snapshots."""
import json

import pytest

from smithtune import checkpoint, dataset_workflow as workflow, triage_source
from smithtune.artifacts import output_lock
from smithtune.providers.base import PipelineError
from curation_fakes import API, SOURCE, uid


def test_seeded_distinct_selection_and_small_storage(tmp_path):
    api = API()
    api.roots += [{**api.roots[0], 'id': uid(2), 'trace_id': uid(2)}]
    result = workflow.run('pull', tmp_path, runner=api, limit=1, **SOURCE)
    assert result['selected'] == result['downloaded'] == 1
    assert {p.name for p in tmp_path.iterdir()} == {'checkpoint.json', 'conversations', '.smithtune.lock'}
    files = list((tmp_path / 'conversations').iterdir())
    assert len(files) == 1
    value = json.loads(files[0].read_text())
    assert value['inputs']['messages'] == api.messages
    assert [len(b['tools']) for b in value['metadata']['smithtune_source']['assistant_runs']] == [1, 2, 0]
    assert 'NEVER_PERSIST_RUN_TREE' not in ''.join(p.read_text() for p in tmp_path.rglob('*') if p.is_file())
    assert 'messages' not in (tmp_path / 'checkpoint.json').read_text()
    api.calls.clear()
    assert workflow.run('resume', tmp_path, runner=api, confirm=True)['status'] == 'complete'
    assert workflow.run('pull', tmp_path, runner=api)['status'] == 'complete'
    assert api.calls == []


def test_interrupted_download_uses_frozen_selection_and_restarts_pages(tmp_path, monkeypatch):
    api = API()
    monkeypatch.setattr(triage_source.time, 'sleep', lambda _: None)
    def fail(method, path, body):
        if path == '/v1/trajectory' and body.get('cursor') == '2':
            raise OSError('interrupted')
    api.failure = fail
    result = workflow.run('pull', tmp_path, runner=api, **SOURCE)
    assert result['status'] == 'incomplete'
    selected = checkpoint.load(tmp_path)['selection']
    api.roots.append({**api.roots[0], 'trace_id': uid(9), 'thread_id': 'new'})
    api.failure = None
    api.calls.clear()
    assert workflow.run('resume', tmp_path, runner=api, confirm=True)['status'] == 'complete'
    assert [(s['scope'], s['scope_id']) for s in checkpoint.load(tmp_path)['selection']] == [(s['scope'], s['scope_id']) for s in selected]
    assert not any(path == '/api/v2/runs/query' and not body['filter'].startswith('eq(thread_id,') for _, path, body in api.calls)
    assert next(body for _, path, body in api.calls if path == '/v1/trajectory').get('cursor') is None


def test_interruption_between_file_and_checkpoint_is_recoverable(tmp_path, monkeypatch):
    api = API()
    original = checkpoint.save
    def interrupted(directory, value):
        if any(s.get('file') for s in value['selection'] or []):
            raise KeyboardInterrupt
        original(directory, value)
    monkeypatch.setattr(checkpoint, 'save', interrupted)
    with pytest.raises(KeyboardInterrupt):
        workflow.run('pull', tmp_path, runner=api, **SOURCE)
    assert checkpoint.load(tmp_path)['selection'][0]['status'] == 'pending'
    assert len(list((tmp_path / 'conversations').iterdir())) == 1
    monkeypatch.setattr(checkpoint, 'save', original)
    workflow.run('resume', tmp_path, runner=api, confirm=True)
    assert len(list((tmp_path / 'conversations').iterdir())) == 1


@pytest.mark.parametrize('change', ['history', 'media', 'unknown_tools', 'ambiguous', 'thread'])
def test_whole_conversation_exclusions_are_terminal(tmp_path, change):
    api = API()
    if change == 'history':
        api.messages[1]['id'] = 'input-only-history'
        api.runs[1]['outputs']['messages'] = []
    elif change == 'media':
        api.runs[1]['inputs']['attachment'] = {'type': 'image', 'url': 'private'}
    elif change == 'unknown_tools':
        api.runs[1]['extra'] = {}
    elif change == 'ambiguous':
        api.runs.append({**api.runs[1], 'id': uid(999)})
    else:
        def changed(method, path, body):
            if path == '/v1/trajectory':
                api.membership = [*api.roots, {**api.roots[0], 'trace_id': uid(2), 'id': uid(2)}]
        api.failure = changed
    result = workflow.run('pull', tmp_path, runner=api, **SOURCE)
    assert result['downloaded'] == 0 and len(result['rejections']) == 1
    api.calls.clear()
    workflow.run('pull', tmp_path, runner=api)
    assert not api.calls


def test_source_mismatch_lock_and_legacy_hint(tmp_path):
    workflow.run('pull', tmp_path, runner=API(), **SOURCE)
    with pytest.raises(PipelineError, match='conflicts'):
        workflow.run('pull', tmp_path, limit=10)
    with output_lock(tmp_path), pytest.raises(PipelineError, match='another smithtune'):
        workflow.run('resume', tmp_path)
    (tmp_path / 'snapshot.json').write_text('{}')
    with pytest.raises(PipelineError, match='NEW_DIR.*--dataset-id'):
        workflow.run('resume', tmp_path)


@pytest.mark.parametrize('limit', [0, -1, 2001])
def test_limits_are_bounded(limit):
    with pytest.raises(PipelineError, match='between 1 and 2000'):
        triage_source.source_options(SOURCE['workspace_id'], SOURCE['project_id'], limit=limit)


def test_selection_samples_distinct_identities_after_all_root_pages():
    import random
    api = API()
    api.roots = [{**api.roots[0], 'id': uid(i), 'trace_id': uid(i), 'thread_id': f'thread-{i // 2}'}
                 for i in range(2, 22)]
    source = triage_source.source_options(**SOURCE, limit=4, seed=9)
    roots = triage_source._root_selection(source, runner=api)
    groups = {r['thread_id']: r for r in reversed(api.roots)}
    expected = random.Random(9).sample([groups[k] for k in sorted(groups)], 4)
    assert {r['thread_id'] for r in roots} == {r['thread_id'] for r in expected}
    assert len(api.calls) == len(api.roots)


def test_empty_conversation_is_a_terminal_exclusion(tmp_path):
    api = API()
    api.messages = []
    result = workflow.run('pull', tmp_path, runner=api, **SOURCE)
    assert result['status'] == 'complete' and 'no messages' in result['rejections'][0]['reason']
    api.calls.clear()
    workflow.run('pull', tmp_path, runner=api)
    assert not api.calls


def test_root_page_bound_explains_how_to_narrow_selection(monkeypatch):
    api = API()
    api.roots *= 4
    monkeypatch.setattr(triage_source, 'MAX_SOURCE_PAGES', 2)
    with pytest.raises(PipelineError, match='narrow the time window or filter'):
        triage_source._root_selection(triage_source.source_options(**SOURCE), runner=api)


@pytest.mark.parametrize('endpoint', ['project', 'membership'])
def test_supporting_source_reads_have_one_retry_layer(tmp_path, monkeypatch, endpoint):
    api = API()
    monkeypatch.setattr(triage_source.time, 'sleep', lambda _: None)
    attempts = []
    def fail(method, path, body):
        selected = path.startswith('/api/v1/sessions/') if endpoint == 'project' else (
            path == '/api/v2/runs/query' and body.get('filter', '').startswith('eq(thread_id,'))
        if selected:
            attempts.append(path)
            raise PipelineError('HTTP 429')
    api.failure = fail
    result = workflow.run('pull', tmp_path, runner=api, **SOURCE)
    assert result['status'] == 'incomplete' and len(attempts) == 3

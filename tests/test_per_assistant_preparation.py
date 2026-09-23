"""Self-contained examples replace conversation-wide automatic contracts."""
import copy
import hashlib

import pytest

from smithtune import dataset
from smithtune.artifacts import _json_dump, _load_jsonl
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL
from binding_fixtures import bound_example
from test_assistant_bindings import example as changing_example


def example(index, *, thread=None, project='project-1'):
    return bound_example({'id': f'example-{index}', 'inputs': {'messages': [
        {'role': 'human', 'content': f'question {index}'}, {'role': 'ai', 'content': f'answer {index}'}]},
        'outputs': None, 'metadata': {'source_scope': 'thread', 'source_scope_id': thread or f'thread-{index}',
        'source_project_id': project, 'trajectory_format': 'messages', 'conversation_scope': 'root'}}, tools=[])


def write_empty_tool_snapshot(root, examples, dataset_id='dataset-id', workspace_id='workspace-id'):
    """A previously captured per-message snapshot for old-export test fixtures."""
    _json_dump(root / 'raw/bound_examples.json', {
        'workspace_id': workspace_id, 'source_sha256': hashlib.sha256((root / 'raw/examples.json').read_bytes()).hexdigest(),
        'examples': [bound_example(copy.deepcopy(e), tools=[]) for e in examples], 'exclusions': []})


def test_prepare_exported_bindings_offline_and_override_preserves_evidence(tmp_path, monkeypatch):
    from test_pipeline import write_raw
    from smithtune.bindings import tool_contract
    value, _ = changing_example()
    write_raw(tmp_path, [value])
    monkeypatch.setattr(dataset, 'capture_example_bindings', lambda *_a, **_k: pytest.fail('source read'))
    manifest = dataset.prepare_dataset('workspace-id', 'dataset-id', DEFAULT_MODEL, tmp_path,
                                       fetch=False, check_render=False, test_fraction=1, validation_fraction=0)
    row, = _load_jsonl(tmp_path / 'prepared/test.jsonl')
    assert manifest['schema_version'] == 2 and manifest['tool_policy'] == 'per_assistant'
    assert 'tools' not in row
    assert row['_source']['metadata']['smithtune_source'] == value['metadata']['smithtune_source']
    overridden, = dataset.prepare_sft_rows([value], tool_contract([]))
    assert overridden['tool_policy'] == 'global_override'
    assert overridden['_source']['metadata']['smithtune_source'] == value['metadata']['smithtune_source']


def test_missing_or_invalid_bindings_do_not_fall_back_to_union():
    value, _ = changing_example()
    del value['metadata']['smithtune_source']
    with pytest.raises(PipelineError, match='provenance'):
        dataset.prepare_sft_rows([value])
    with pytest.raises(PipelineError, match='legacy conversation-wide'):
        dataset.prepare_sft_rows([value], example_contracts={})


def test_invalid_target_schema_excludes_whole_conversation():
    value, _ = changing_example()
    value['inputs']['messages'][1]['content'] = [{'type': 'tool_call', 'id': 'call', 'name': 'search', 'args': {'query': 4}}]
    value['inputs']['messages'].insert(2, {'role': 'tool', 'tool_call_id': 'call', 'content': 'result'})
    for binding in value['metadata']['smithtune_source']['assistant_runs'][1:]:
        binding['message_index'] += 1
    warnings = []
    assert dataset.prepare_sft_rows([value], exclusion_warnings=warnings) == []
    assert 'JSON Schema' in warnings[0]['reason']


def test_automatic_capture_uses_same_verified_mapping(tmp_path):
    from curation_fakes import API, SOURCE
    api = API()
    value, _ = changing_example()
    del value['metadata']['smithtune_source']
    value['metadata'].update(source_workspace_id=SOURCE['workspace_id'], source_project_id=SOURCE['project_id'])
    enriched, errors = dataset.capture_example_bindings([value], SOURCE['workspace_id'], runner=api)
    assert not errors and len(enriched[0]['metadata']['smithtune_source']['assistant_runs']) == 3
    assert 'smithtune_source' not in value['metadata']


def test_old_prepared_meaning_requires_repreparation():
    with pytest.raises(PipelineError, match='prepare again'):
        dataset.require_current_preparation({'schema_version': 1})


def test_unbound_export_reports_mapping_failures_and_preserves_other_examples():
    from curation_fakes import API, SOURCE
    api = API()
    value, _ = changing_example()
    del value['metadata']['smithtune_source']
    value['metadata'].update(source_workspace_id=SOURCE['workspace_id'], source_project_id=SOURCE['project_id'])
    api.runs[1]['extra'] = {}  # Unknown availability is never converted to [].
    enriched, errors = dataset.capture_example_bindings([value, example(99)], 'destination-workspace', runner=api)
    assert [e['id'] for e in enriched] == ['example-99']
    assert errors[0]['example_id'] == 'example' and 'unknown tool availability' in errors[0]['reason']
    assert 'message 1' in errors[0]['reason'] and api.runs[1]['id'] in errors[0]['reason']


def test_interrupted_binding_capture_reuses_completed_examples(tmp_path, monkeypatch):
    from smithtune import curation
    from test_assistant_bindings import example as source_example
    first, _ = source_example()
    source = first['metadata']['smithtune_source']
    second = copy.deepcopy(first)
    second['id'] = 'second'
    for value in (first, second):
        del value['metadata']['smithtune_source']
        value['metadata'].update(source_scope='trace', source_scope_id=value['id'])
    calls = []
    def fetch(workspace, project, query, **kwargs):
        calls.append(query['id'])
        if query['id'] == 'second' and calls.count('second') == 1:
            raise PipelineError('source read interrupted')
        return {'messages': first['inputs']['messages'], 'source': source,
                'trace_ids': [query['id']], 'training_error': None}
    monkeypatch.setattr(curation, '_fetch_trajectory', fetch)
    path = tmp_path / 'bindings.partial.json'
    with pytest.raises(PipelineError, match='source read interrupted'):
        dataset.capture_example_bindings([first, second], 'workspace', checkpoint_path=path)
    enriched, errors = dataset.capture_example_bindings([first, second], 'workspace', checkpoint_path=path)
    assert len(enriched) == 2 and not errors
    assert calls == ['example', 'second', 'second']
    import json
    saved = json.loads(path.read_text())
    saved['bindings']['example']['assistant_runs'][0]['tools'] = []
    path.write_text(json.dumps(saved))
    with pytest.raises(PipelineError, match='hash mismatch'):
        dataset.capture_example_bindings([first, second], 'workspace', checkpoint_path=path)


@pytest.mark.parametrize('triaged', [False, True])
def test_pull_upload_prepare_and_replay_preserve_changing_tools(tmp_path, monkeypatch, triaged):
    from smithtune import dataset_workflow, triage_source
    from smithtune.evaluation.replay import build_replay_cases
    from curation_fakes import API, SOURCE
    from test_pipeline import write_raw
    api = API()
    def judge(_judge, messages, _tokens):
        import json
        evidence = json.loads(messages[1]['content'])
        assert [len(b['tools']) for b in evidence['untrusted_assistant_tool_bindings']] == [1, 2, 0]
        return {'keep': 1, 'reason': 'Complete answer.'}
    directory = tmp_path / 'curation'
    dataset_workflow.run('pull', directory, runner=api, filter='eq(name,"agent")', **SOURCE)
    if triaged:
        dataset_workflow.run('triage', directory, runner=api, judge_call=judge, confirm=True,
                             rules=['Keep complete answers.'])
    result = dataset_workflow.run('push', directory, runner=api, confirm=True, name='changing-tools')
    assert result['created'] == 1
    frozen = triage_source.load_snapshot(directory)
    uploaded, = api.examples.values()
    assert uploaded['inputs']['messages'] == api.messages
    assert uploaded['metadata']['smithtune_source'] == frozen['units'][0]['example']['metadata']['smithtune_source']
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    write_raw(data_dir, [uploaded])
    monkeypatch.setattr(dataset, 'capture_example_bindings', lambda *_a, **_k: pytest.fail('tools must already be saved'))
    dataset.prepare_dataset('workspace-id', 'dataset-id', DEFAULT_MODEL, data_dir,
                            fetch=False, check_render=False, test_fraction=1, validation_fraction=0)
    row, = _load_jsonl(data_dir / 'prepared/test.jsonl')
    assert [len(c['tools']) for c in build_replay_cases([row])] == [1, 2, 0]
    assert row['_source']['metadata']['smithtune_source'] == uploaded['metadata']['smithtune_source']


@pytest.mark.parametrize('change', ['definition', 'missing', 'run'])
def test_existing_dataset_requires_unchanged_tool_history(change):
    from smithtune.dataset_import import _action
    value, _ = changing_example()
    old = copy.deepcopy(value)
    old['inputs']['messages'] = old['inputs']['messages'][:2]
    old['metadata']['smithtune_source']['assistant_runs'] = old['metadata']['smithtune_source']['assistant_runs'][:1]
    assert _action(value, old, triaged=False)[0] == 'updated'
    if change == 'definition':
        old['metadata']['smithtune_source']['assistant_runs'][0]['tools'][0]['function']['description'] = 'changed'
    elif change == 'run':
        old['metadata']['smithtune_source']['assistant_runs'][0]['run_id'] = 'another-run'
    else:
        del old['metadata']['smithtune_source']
    with pytest.raises(PipelineError, match='tool availability conflicts|tool provenance differs'):
        _action(value, old, triaged=False)


@pytest.mark.parametrize('change', ['workspace', 'tools', 'source_workspace'])
def test_completed_binding_cache_rejects_changed_identity_or_content(tmp_path, monkeypatch, change):
    import json
    from smithtune.inference_contract import json_sha256
    from test_pipeline import write_raw
    unbound = example(1)
    del unbound['metadata']['smithtune_source']
    write_raw(tmp_path, [unbound])
    monkeypatch.setattr(dataset, 'download_dataset', lambda *_: None)
    monkeypatch.setattr(dataset, 'capture_example_bindings', lambda examples, *_a, **_k: ([bound_example(copy.deepcopy(e)) for e in examples], []))
    options = dict(fetch=True, check_render=False, validation_fraction=0, test_fraction=1)
    dataset.prepare_dataset('workspace-id', 'dataset-id', DEFAULT_MODEL, tmp_path, **options)
    options['fetch'] = False
    assert dataset.prepare_dataset('workspace-id', 'dataset-id', DEFAULT_MODEL, tmp_path, **options)['prepared']['accepted'] == 1
    workspace = 'workspace-id'
    if change == 'workspace':
        workspace = 'different-workspace'
    elif change == 'source_workspace':
        options['source_workspace_id'] = 'different-source'
    else:
        path = tmp_path / 'raw/bound_examples.json'
        saved = json.loads(path.read_text())
        saved['examples'][0]['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = [{'changed': True}]
        assert saved['sha256'] != json_sha256({'examples': saved['examples'], 'exclusions': saved['exclusions']})
        path.write_text(json.dumps(saved))
    with pytest.raises(PipelineError, match='different workspace|different source workspace|hash mismatch'):
        dataset.prepare_dataset(workspace, 'dataset-id', DEFAULT_MODEL, tmp_path, **options)


def test_binding_capture_retries_transient_source_reads(monkeypatch):
    import subprocess
    from curation_fakes import API, SOURCE
    from smithtune import triage_source
    api = API()
    value, _ = changing_example()
    del value['metadata']['smithtune_source']
    value['metadata'].update(source_workspace_id=SOURCE['workspace_id'], source_project_id=SOURCE['project_id'])
    failures, delays = [], []
    def transient(method, path, body):
        if path == '/v1/trajectory' and not failures:
            failures.append(path)
            raise subprocess.CalledProcessError(1, 'langsmith', stderr='HTTP 504')
    api.failure = transient
    monkeypatch.setattr(triage_source.time, 'sleep', delays.append)
    enriched, errors = dataset.capture_example_bindings([value], SOURCE['workspace_id'], runner=api)
    assert len(enriched) == 1 and not errors
    assert len(failures) == len(delays) == 1

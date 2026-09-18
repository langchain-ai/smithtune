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
        'source_sha256': hashlib.sha256((root / 'raw/examples.json').read_bytes()).hexdigest(),
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

import copy
import hashlib

import pytest

from smithtune import dataset
from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _load_jsonl, output_lock
from smithtune.inference_contract import json_sha256
from smithtune.providers import fireworks
from smithtune.providers.base import PipelineError
from test_pipeline import example, loaded_contract, write_raw


def prepare(path, examples, monkeypatch, **kwargs):
    # Avoid live downloads/tokenizers; exercise the real conversion, split and writes.
    monkeypatch.setattr(dataset, '_load_dataset_source', lambda *a, **kw: (
        {'id': 'dataset', 'name': 'test'}, examples, json_sha256(examples),
    ))
    return dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, path,
                                   inference_contract=loaded_contract(path), check_render=False, **kwargs)


def membership(path):
    return {row['_source']['source_scope_id']: partition
            for partition in dataset.SPLIT_NAMES
            for row in _load_jsonl(path / 'prepared' / f'{partition}.jsonl')}


def ledger(path):
    return _load_json(path / 'prepared' / 'split_assignments.json')


def test_prepare_retains_assignments_across_growth_removal_and_changed_example_ids(tmp_path, monkeypatch):
    examples = [example(i) for i in range(10)]
    first = prepare(tmp_path, examples, monkeypatch)
    original = membership(tmp_path)
    grown = [example(i) for i in range(15)]
    grown[0]['id'] = 'new-example-id'
    grown[0]['inputs']['messages'] += example(100)['inputs']['messages']
    prepare(tmp_path, list(reversed(grown)), monkeypatch)
    assert all(membership(tmp_path)[key] == part for key, part in original.items())
    prepare(tmp_path, grown[5:], monkeypatch)
    assert len(ledger(tmp_path)['assignments']) == 15
    prepare(tmp_path, grown, monkeypatch)
    assert all(membership(tmp_path)[key] == part for key, part in original.items())
    assert first['split']['assignments_sha256'] != ledger(tmp_path)['assignments_sha256']


def test_fixed_hash_assignment_is_independent_of_dataset_size_and_full_source_identity():
    base = example(0)
    examples = [base]
    for field, value in [('source_workspace_id', 'other'), ('source_project_id', 'other'), ('source_scope', 'trace')]:
        item = copy.deepcopy(base)
        item['id'] = field
        item['metadata'][field] = value
        examples.append(item)
    rows = dataset.prepare_sft_rows(examples, workspace_id='workspace-id')
    assignments = {}
    for row in rows:
        dataset.split_rows([row], assignments=assignments)
    assert len(assignments) == 4
    together = {}
    dataset.split_rows(rows, assignments=together)
    assert assignments == together


def legacy_preparation(path, *, captured_workspace=False):
    examples = [example(i) for i in range(10)]
    if not captured_workspace:
        for item in examples:
            item['metadata']['source_workspace_id'] = 'workspace-id'
    write_raw(path, examples)
    rows = dataset.prepare_sft_rows(examples)
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f"42:{row['_source']['source_scope_id']}".encode()).hexdigest())
    for name, values in zip(dataset.SPLIT_NAMES, (ranked[2:], ranked[:1], ranked[1:2]), strict=True):
        _jsonl_dump(path / 'prepared' / f'{name}.jsonl', values)
    _json_dump(path / 'prepared' / 'manifest.json', {
        'langsmith': {'workspace_id': 'workspace-id', 'dataset_id': 'dataset'},
        'split': {'method': 'sha256(seed:source_scope_id)', 'seed': 42,
                  'validation_fraction': .1, 'test_fraction': .1, 'train': 8, 'validation': 1, 'test': 1},
    })
    if captured_workspace:
        contracts = _load_json(path / 'raw' / 'example_contracts.json')['contracts']
        for contract in contracts.values():
            contract['provenance']['source_workspace_id'] = 'source-workspace'
        _json_dump(path / 'prepared' / 'example_contracts.json', contracts)
        manifest = _load_json(path / 'prepared' / 'manifest.json')
        manifest['example_contracts'] = {'count': len(contracts), 'sha256': json_sha256(contracts)}
        _json_dump(path / 'prepared' / 'manifest.json', manifest)
    return examples


@pytest.mark.parametrize('captured_workspace', [False, True])
def test_existing_position_splits_are_preserved_before_downloading_new_version(tmp_path, monkeypatch, captured_workspace):
    examples = legacy_preparation(tmp_path, captured_workspace=captured_workspace)
    original = membership(tmp_path)
    new = [*examples, *[example(i) for i in range(10, 15)]]
    prepare(tmp_path, new, monkeypatch, source_workspace_id='source-workspace' if captured_workspace else None)
    assert all(membership(tmp_path)[key] == part for key, part in original.items())
    # Prove migration mattered: the new algorithm would change old memberships.
    fresh = tmp_path / 'fresh'
    fresh.mkdir()
    prepare(fresh, new, monkeypatch, source_workspace_id='source-workspace' if captured_workspace else None)
    assert any(membership(fresh)[key] != part for key, part in original.items())


def test_split_from_carries_legacy_assignments_and_retains_removed_sources(tmp_path, monkeypatch):
    previous, current = tmp_path / 'old', tmp_path / 'new'
    previous.mkdir()
    current.mkdir()
    examples = legacy_preparation(previous)
    original = membership(previous)
    prepare(current, [*examples[5:], example(11)], monkeypatch, split_from=previous)
    assert len(ledger(current)['assignments']) == 11
    prepare(current, examples, monkeypatch)
    assert membership(current) == original
    assert not (previous / 'prepared' / 'split_assignments.json').exists()


def test_conflicting_split_history_fails_before_fetch(tmp_path, monkeypatch):
    previous, current = tmp_path / 'old', tmp_path / 'new'
    previous.mkdir()
    current.mkdir()
    examples = legacy_preparation(previous)
    prepare(current, examples, monkeypatch)
    before = ledger(current)
    monkeypatch.setattr(dataset, '_load_dataset_source', lambda *a, **k: pytest.fail('must fail before fetch'))
    with pytest.raises(PipelineError, match='conflicts'):
        dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, current, split_from=previous)
    assert ledger(current) == before


@pytest.mark.parametrize('damage', ['fractions', 'checksum', 'missing_ledger', 'invalid_identity'])
def test_invalid_saved_assignments_cannot_silently_resplit(tmp_path, monkeypatch, damage):
    prepare(tmp_path, [example(i) for i in range(10)], monkeypatch)
    path = tmp_path / 'prepared' / 'split_assignments.json'
    saved = ledger(tmp_path)
    kwargs = {}
    if damage == 'fractions':
        kwargs['test_fraction'] = .2
    elif damage == 'missing_ledger':
        path.unlink()
    elif damage == 'checksum':
        saved['assignments_sha256'] = 'bad'
        _json_dump(path, saved)
    else:
        saved['assignments'] = {'["incomplete"]': 'train'}
        saved['assignments_sha256'] = json_sha256(saved['assignments'])
        _json_dump(path, saved)
    monkeypatch.setattr(dataset, '_load_dataset_source', lambda *a, **k: pytest.fail('must fail before fetch'))
    with pytest.raises(PipelineError):
        dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, tmp_path, **kwargs)


def test_interruption_after_ledger_write_keeps_migrated_assignments(tmp_path, monkeypatch):
    examples = legacy_preparation(tmp_path)
    original = membership(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(dataset, '_jsonl_dump', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError, match='disk full'):
            prepare(tmp_path, [*examples, example(11)], patch)
    assert len(ledger(tmp_path)['assignments']) == 11
    prepare(tmp_path, [*examples, example(12)], monkeypatch)
    assert all(membership(tmp_path)[key] == part for key, part in original.items())
    assert len(ledger(tmp_path)['assignments']) == 12


def test_empty_partitions_are_reported_without_reassigning_conversations(tmp_path, monkeypatch, capsys):
    result = prepare(tmp_path, [example(1)], monkeypatch)
    assert result['prepared']['accepted'] == 1
    assert sum(result['split'][name] == 0 for name in dataset.SPLIT_NAMES) == 2
    assert 'Existing assignments remain fixed' in capsys.readouterr().err


def test_prepare_cannot_race_another_writer(tmp_path):
    with output_lock(tmp_path), pytest.raises(PipelineError, match='another smithtune operation'):
        dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, tmp_path)


def test_missing_explicit_split_source_does_not_start_fresh(tmp_path):
    with pytest.raises(PipelineError, match='no complete previous preparation'):
        dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, tmp_path, split_from=tmp_path/'missing')


@pytest.mark.parametrize('referenced', [False, True])
def test_legacy_recovery_does_not_trust_stale_or_corrupted_contracts(tmp_path, monkeypatch, referenced):
    legacy_preparation(tmp_path, captured_workspace=True)
    if not referenced:
        path = tmp_path / 'prepared' / 'manifest.json'
        manifest = _load_json(path)
        del manifest['example_contracts']
        _json_dump(path, manifest)
    else:
        path = tmp_path / 'prepared' / 'example_contracts.json'
        contracts = _load_json(path)
        contracts['example-0']['provenance']['source_workspace_id'] = 'wrong-workspace'
        _json_dump(path, contracts)
    monkeypatch.setattr(dataset, '_load_dataset_source', lambda *a, **k: pytest.fail('must fail before fetch'))
    with pytest.raises(PipelineError, match='source workspace|hash differs'):
        dataset.prepare_dataset('workspace-id', 'dataset', fireworks.DEFAULT_MODEL, tmp_path)
    assert not (tmp_path / 'prepared' / 'split_assignments.json').exists()

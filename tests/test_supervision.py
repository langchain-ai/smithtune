"""Explicit teaching targets survive the entire offline dataset workflow."""
import copy
import json

import pytest

from smithtune.bindings import evidence_hash, training_targets
from smithtune.dataset import prepare_sft_rows, validate_trajectories, require_current_preparation
from smithtune.dataset_import import _action
from smithtune.evaluation.replay import build_replay_cases
from smithtune.providers.base import PipelineError
from smithtune.supervision import make_supervision, target_indices, trajectory_supervision
from tests.test_assistant_bindings import example
from tests.trajectory_fixtures import items


def garden_example():
    value, _ = example()
    messages = value['inputs']['messages']
    messages[:0] = [{'role': 'human', 'content': 'old question'}, {'role': 'ai', 'content': 'HISTORICAL ANSWER'}]
    for binding in value['metadata']['smithtune_source']['assistant_runs']:
        binding['message_index'] += 2
    value['metadata']['smithtune_source']['assistant_runs'].insert(0,
        {'message_index': 1, 'run_id': 'run-0', 'trace_id': 'trace-0', 'tools': []})
    wire = items(messages)
    for i, item in enumerate(wire):
        item['metadata']['run_id'] = 'initial' if i <= 3 else f'later-{i}'
    value['metadata']['supervision'] = trajectory_supervision(wire, messages, 'skip-initial-history')
    return value


def test_boundary_preserves_history_and_all_later_assistant_actions():
    value = garden_example()
    messages = value['inputs']['messages']
    before = copy.deepcopy(value)
    assert target_indices(value['metadata'], messages) == [3, 5, 7]
    row, = prepare_sft_rows([value])
    assert row['_source']['target_message_indices'] == [3, 5, 7]
    targets = list(training_targets(row))
    assert [t['message_index'] for t in targets] == [3, 5, 7]
    assert targets[0]['messages'][1]['content'] == 'HISTORICAL ANSWER'
    assert [case['message_index'] for case in build_replay_cases([row])] == [3, 5, 7]
    assert [case['message_index'] for case in build_replay_cases([row], 1)] == [3]
    assert value == before


def test_normal_multiturn_and_default_need_no_extra_metadata():
    messages = [{'role': role, 'content': role} for role in ('human', 'ai', 'human', 'ai')]
    wire = items(messages)
    for i, entry in enumerate(wire):
        entry['metadata']['run_id'] = 'a' if i < 2 else 'b'
    assert trajectory_supervision(wire, messages, 'skip-initial-history')['target_message_indices'] == [1, 3]
    for entry in wire:
        entry.pop('metadata')
    assert trajectory_supervision(wire, messages, 'all-assistant')['target_message_indices'] == [1, 3]
    assert target_indices({}, messages) == [1, 3]


@pytest.mark.parametrize('run_ids', [[None, 'a', 'b', 'b'], ['a', 'a', 'b', 'a'], ['a', 'b', 'b', 'b']])
def test_ambiguous_boundary_fails_without_fallback(run_ids):
    messages = [{'role': role, 'content': role} for role in ('human', 'ai', 'human', 'ai')]
    wire = items(messages)
    for item, run_id in zip(wire, run_ids, strict=True):
        item['metadata']['run_id'] = run_id
    with pytest.raises(PipelineError, match='skip-initial-history'):
        trajectory_supervision(wire, messages, 'skip-initial-history')


@pytest.mark.parametrize('indices', [[], [True], [1, 1], [3, 1], [0], [-1], [200], ['1']])
def test_invalid_explicit_selection_is_rejected(indices):
    value, _ = example()
    with pytest.raises(PipelineError, match='target_message_indices'):
        make_supervision(value['inputs']['messages'], indices)


def test_metadata_integrity_and_approval_include_targets():
    value = garden_example()
    original_hash = evidence_hash(value)
    selected = value['metadata']['supervision']
    selected['target_message_indices'] = [7]
    selected['scope'] = 'explicit'
    assert evidence_hash(value) != original_hash
    value['metadata']['smithtune_triage'] = {'messages_sha256': selected['messages_sha256'], 'evidence_sha256': original_hash}
    with pytest.raises(PipelineError, match='evidence changed'):
        validate_trajectories([value], 1)
    del value['metadata']['smithtune_triage']
    value['inputs']['messages'][1]['content'] = 'changed history'
    with pytest.raises(PipelineError, match='messages_sha256'):
        validate_trajectories([value], 1)
    value['metadata']['supervision'] = None
    with pytest.raises(PipelineError, match='invalid supervision'):
        prepare_sft_rows([value])


def test_missing_prepared_selection_cannot_fall_back_to_all():
    row, = prepare_sft_rows([garden_example()])
    row['_source'].pop('target_message_indices')
    with pytest.raises(PipelineError, match='prepare again'):
        list(training_targets(row))
    row['_source']['target_message_indices'] = [1, 3, 5, 7]
    with pytest.raises(PipelineError, match='conflict'):
        build_replay_cases([row])
    require_current_preparation({'schema_version': 2, 'target_policy': 'selected_assistant_once'})


def test_import_selection_conflicts_idempotency_and_extension():
    value = garden_example()
    assert _action(value, copy.deepcopy(value), triaged=False) == ('skipped', None)
    different = copy.deepcopy(value)
    different['metadata']['supervision'] = make_supervision(different['inputs']['messages'], [7])
    with pytest.raises(PipelineError, match='supervision targets conflict'):
        _action(different, value, triaged=False)
    extended = copy.deepcopy(value)
    extended['inputs']['messages'] += [{'role': 'human', 'content': 'next'}, {'role': 'ai', 'content': 'next answer'}]
    extended['metadata']['smithtune_source']['assistant_runs'].append(
        {'message_index': 9, 'run_id': 'new', 'trace_id': 'new', 'tools': []})
    extended['metadata']['supervision'] = make_supervision(extended['inputs']['messages'], [3, 5, 7, 9])
    assert _action(extended, value, triaged=False)[0] == 'updated'
    # Missing metadata and an explicit all-assistant policy have identical targets.
    legacy, _ = example()
    explicit = copy.deepcopy(legacy)
    explicit['metadata']['supervision'] = make_supervision(explicit['inputs']['messages'], [1, 3, 5], scope='all-assistant')
    assert _action(explicit, legacy, triaged=False)[0] == 'skipped'


@pytest.mark.parametrize('provider', ['fireworks', 'baseten'])
def test_actual_provider_masks_skip_history_and_keep_selected_outputs(provider, monkeypatch):
    from smithtune import rendering
    from smithtune.providers.fireworks import provider as fireworks
    from smithtune.providers.baseten import provider as baseten
    from smithtune.hf_rendering import HFRenderer
    from smithtune.providers.fireworks.training import _render_conversation
    from training.renderer import get_renderer
    from training.recipes import sft_loop
    from tests.test_hf_rendering import _tokenizer, _loss_spans
    from tests.test_training_dependency import CharacterTokenizer

    row, = prepare_sft_rows([garden_example()])
    model = fireworks.DEFAULT_MODEL if provider == 'fireworks' else baseten.DEFAULT_MODEL
    tokenizer = CharacterTokenizer() if provider == 'fireworks' else _tokenizer()
    renderer = get_renderer(model.renderer, tokenizer) if provider == 'fireworks' else HFRenderer(model, tokenizer)
    if provider == 'fireworks':
        monkeypatch.setitem(sft_loop._worker_state, 'renderer', renderer)
        monkeypatch.setitem(sft_loop._worker_state, 'max_seq_len', model.max_seq_len)
        actual = _render_conversation(row)
    else:
        actual = baseten.render_row(row, model, renderer=renderer)
    rendered = rendering.render_row_tokens(row, model, renderer=renderer)
    assert len(actual) == len(rendered) == 3
    for i, datum in enumerate(rendered):
        text = tokenizer.decode(datum.token_ids)
        trained = ''.join(_loss_spans(datum, tokenizer))
        assert 'HISTORICAL ANSWER' in text
        assert 'HISTORICAL ANSWER' not in trained
        assert f'answer-{i}' in trained
        assert all(f'answer-{j}' not in trained for j in range(i))


def test_paginated_pull_frozen_selection_judges_and_upload(tmp_path):
    from smithtune import triage_source
    from tests.test_dataset_workflow import API, run

    api = API()
    # The initial run-ID group crosses the transport page boundary.
    api.trajectory_pages[None]['messages'] = [
        {'role': 'system', 'content': 'system'},
        {'role': 'human', 'content': 'old'},
        {'role': 'ai', 'content': 'HISTORICAL ANSWER'}]
    api.trajectory_pages['next']['messages'] = [
        {'role': 'human', 'content': 'question', 'id': 'user-1'},
        {'role': 'ai', 'content': 'new answer', 'id': 'ai-1'},
        {'role': 'human', 'content': 'second question', 'id': 'user-2'},
        {'role': 'ai', 'content': 'second answer', 'id': 'ai-2'}]
    run(tmp_path, api, supervision_scope='skip-initial-history')
    frozen = triage_source.load_snapshot(tmp_path)
    value = frozen['units'][0]['example']
    assert value['metadata']['supervision']['target_message_indices'] == [4, 6]
    seen = []
    def judge(slot, prompt, tokens):
        evidence = json.loads(prompt[1]['content'])
        assert evidence['target_message_indices'] == [4, 6]
        assert evidence['untrusted_trajectory'] == value['inputs']['messages']
        seen.append(slot['name'])
        return {'keep': 1, 'reason': 'Selected outputs are useful.'}
    api.calls.clear()
    run(tmp_path, api, 'triage', rules=['Require supported outputs'], confirm=True, judge=judge)
    assert seen and not api.calls
    with pytest.raises(PipelineError, match='frozen selection'):
        run(tmp_path, api, supervision_scope='all-assistant')
    run(tmp_path, api, 'push', name='selected', confirm=True)
    uploaded, = api.imported
    assert uploaded['metadata']['supervision'] == value['metadata']['supervision']
    row, = prepare_sft_rows([uploaded])
    assert [t['message_index'] for t in training_targets(row)] == [4, 6]


def test_selected_tool_calls_are_targets_and_tool_results_are_context():
    from smithtune.bindings import trajectory_bindings
    from tests.test_assistant_bindings import tool
    messages = [
        {'role': 'human', 'content': 'old question'},
        {'role': 'ai', 'content': 'old answer'},
        {'role': 'human', 'content': 'search please'},
        {'role': 'ai', 'content': [{'type': 'tool_call', 'id': 'call', 'name': 'search', 'args': {'query': 'x'}}]},
        {'role': 'tool', 'content': 'found x', 'tool_call_id': 'call'},
        {'role': 'ai', 'content': 'new final'}]
    wire = items(messages, tools=[tool()])
    for i, entry in enumerate(wire):
        entry['metadata']['run_id'] = 'initial' if i <= 3 else 'next'
    value = garden_example()
    value['inputs']['messages'] = messages
    value['metadata']['smithtune_source'] = trajectory_bindings(wire)
    value['metadata']['supervision'] = trajectory_supervision(wire, messages, 'skip-initial-history')
    row, = prepare_sft_rows([value])
    targets = list(training_targets(row))
    assert [t['message_index'] for t in targets] == [3, 5]
    assert targets[0]['messages'][-1]['tool_calls'][0]['function']['name'] == 'search'
    assert targets[1]['messages'][-2]['role'] == 'tool'
    assert [case['case_type'] for case in build_replay_cases([row])] == ['tool_call', 'text']


def test_reasoning_conversion_preserves_original_target_indices_and_rejects_lost_targets():
    value = garden_example()
    messages = value['inputs']['messages']
    messages[1]['content'] = [{'type': 'reasoning', 'reasoning': 'old reasoning'}]
    value['metadata']['supervision'] = make_supervision(messages, [3, 5, 7])
    row, = prepare_sft_rows([value])
    assert row['_source']['message_positions'] == [0, 2, 3, 4, 5, 6, 7]
    assert [t['message_index'] for t in training_targets(row)] == [3, 5, 7]
    messages[3]['content'] = [{'type': 'reasoning', 'reasoning': 'new reasoning'}]
    value['metadata']['supervision'] = make_supervision(messages, [3, 5, 7])
    with pytest.raises(PipelineError, match='selected assistant target is empty'):
        prepare_sft_rows([value])


def test_mixed_examples_keep_independent_policies():
    historical = garden_example()
    all_assistant = copy.deepcopy(historical)
    all_assistant['id'] = 'all'
    all_assistant['metadata']['supervision'] = make_supervision(all_assistant['inputs']['messages'], [1, 3, 5, 7], scope='all-assistant')
    explicit = copy.deepcopy(historical)
    explicit['id'] = 'explicit'
    explicit['metadata']['supervision'] = make_supervision(explicit['inputs']['messages'], [5])
    rows = prepare_sft_rows([historical, all_assistant, explicit])
    assert [[t['message_index'] for t in training_targets(row)] for row in rows] == [[3, 5, 7], [1, 3, 5, 7], [5]]


def test_real_fireworks_loader_and_validation_respect_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from training.renderer import get_renderer
    from training.recipes import sft_loop
    from training.utils import resource_autosizing
    from smithtune import rendering
    from smithtune.providers.fireworks.provider import DEFAULT_MODEL
    from smithtune.providers.fireworks.training import ServerlessTraining
    from tests.test_training_dependency import CharacterTokenizer

    row, = prepare_sft_rows([garden_example()])
    tokenizer = CharacterTokenizer()
    renderer = get_renderer(DEFAULT_MODEL.renderer, tokenizer)
    path = tmp_path / 'conversations.jsonl'
    path.write_text(json.dumps(row) + '\n')
    monkeypatch.setattr(sft_loop, '_init_render_worker', lambda *_, **kwargs: None)
    monkeypatch.setitem(sft_loop._worker_state, 'renderer', renderer)
    monkeypatch.setitem(sft_loop._worker_state, 'max_seq_len', DEFAULT_MODEL.max_seq_len)
    monkeypatch.setattr(resource_autosizing, 'select_render_worker_count', lambda **_: 0)
    config = SimpleNamespace(tokenizer_model='local', renderer_name=DEFAULT_MODEL.renderer,
        train_on_what=rendering.SFT_TARGET_POLICY, max_seq_len=DEFAULT_MODEL.max_seq_len,
        tokenizer_revision='local', thinking_trace_history_mode='', tokenizer_trust_remote_code=False,
        dataset=str(path), evaluation_dataset=str(path), max_examples=None,
        batch_size=1, group_by_length=False, length_group_factor=50, epochs=1)
    session = ServerlessTraining(config, tmp_path)
    training = sft_loop._flatten_rendered_batch(next(iter(session.loader)))
    for datums in (training, session.validation):
        assert len(datums) == 3
        for i, datum in enumerate(datums):
            target = tokenizer.decode([t for t, w in zip(datum.loss_fn_inputs['target_tokens'].data,
                                                        datum.loss_fn_inputs['weights'].data, strict=True) if w])
            assert f'answer-{i}' in target
            assert 'HISTORICAL ANSWER' not in target
            assert all(f'answer-{j}' not in target for j in range(i))


def test_default_extension_does_not_inherit_an_obsolete_destination_mask():
    from tests.test_dataset_import import example as import_example
    previous = import_example(1)
    previous['metadata']['supervision'] = make_supervision(previous['inputs']['messages'], [1], scope='all-assistant')
    incoming = import_example(1, turns=2)
    action, payload = _action(incoming, previous, triaged=False)
    assert action == 'updated'
    assert target_indices(payload['metadata'], payload['inputs']['messages']) == [1, 3]
    assert 'supervision' not in payload['metadata']


def test_cli_scope_and_resume_default_compatibility(tmp_path):
    from smithtune import checkpoint, cli
    from tests.test_dataset_workflow import API, run
    parsed = cli._parser().parse_args(['dataset', 'pull', 'data', '--supervision-scope', 'skip-initial-history'])
    assert parsed.supervision_scope == 'skip-initial-history'
    api = API()
    run(tmp_path, api)
    # Pre-feature checkpoints omitted the all-assistant source option.
    saved = checkpoint.load(tmp_path)
    saved['source'].pop('supervision_scope')
    # Check source matching directly without rewriting checkpoint identity hashes.
    from smithtune import dataset_workflow
    from unittest.mock import patch
    with patch.object(checkpoint, 'load', return_value=saved):
        assert dataset_workflow._open(tmp_path, 'pull', {'supervision_scope': 'all-assistant'}) == saved
        with pytest.raises(PipelineError, match='frozen selection'):
            dataset_workflow._open(tmp_path, 'pull', {'supervision_scope': 'skip-initial-history'})


def test_legacy_malformed_messages_keep_the_existing_whole_example_quarantine():
    from smithtune.dataset import _exclude_malformed_trajectories
    value, _ = example()
    value['inputs']['messages'][0] = None
    accepted, warnings = _exclude_malformed_trajectories([value])
    assert accepted == []
    assert warnings[0]['reason'] == 'invalid_message'


def test_adding_selection_to_legacy_contract_approval_requires_rejudging():
    value = garden_example()
    value['metadata'].pop('smithtune_source')
    value['metadata']['smithtune_triage'] = {
        'messages_sha256': value['metadata']['supervision']['messages_sha256'],
        'contract': {'legacy': 'approval only bound messages and tools'}}
    with pytest.raises(PipelineError, match='evidence changed'):
        validate_trajectories([value], 1)
    value['metadata']['smithtune_triage']['evidence_sha256'] = evidence_hash(value)
    validate_trajectories([value], 1)
    value['metadata']['supervision'] = make_supervision(value['inputs']['messages'], [7])
    with pytest.raises(PipelineError, match='evidence changed'):
        validate_trajectories([value], 1)


def test_removing_selection_cannot_broaden_a_legacy_approved_example():
    value = garden_example()
    value['metadata'].pop('smithtune_source')
    value['metadata']['smithtune_triage'] = {
        'messages_sha256': value['metadata']['supervision']['messages_sha256'],
        'evidence_sha256': evidence_hash(value)}
    validate_trajectories([value], 1)
    value['metadata'].pop('supervision')
    with pytest.raises(PipelineError, match='evidence changed'):
        validate_trajectories([value], 1)

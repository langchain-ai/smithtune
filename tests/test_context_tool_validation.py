"""Context-only assistant turns need no inference contract of their own."""
import copy
import json
from types import SimpleNamespace

import pytest

from smithtune import curation, dataset, dataset_import, dataset_workflow, triage_source
from smithtune.bindings import read_bindings, training_targets, validate_bound_messages
from smithtune.evaluation.replay import build_replay_cases
from smithtune.providers.base import PipelineError
from smithtune.supervision import make_supervision, trajectory_supervision
from tests.curation_fakes import API, SOURCE, uid
from tests.test_assistant_bindings import tool
from tests.trajectory_fixtures import items


def conversation():
    messages = [
        {'role': 'human', 'content': 'Earlier question'},
        {'role': 'ai', 'content': [{'type': 'tool_call', 'id': 'old', 'name': 'old_search', 'args': {'q': 1}}]},
        {'role': 'tool', 'tool_call_id': 'old', 'content': 'Earlier result'},
        {'role': 'ai', 'content': 'HISTORICAL ANSWER'},
        {'role': 'human', 'content': 'Current question'},
        {'role': 'ai', 'content': [{'type': 'tool_call', 'id': 'new', 'name': 'search', 'args': {'query': 'x'}}]},
        {'role': 'tool', 'tool_call_id': 'new', 'content': 'Current result'},
        {'role': 'ai', 'content': 'NEW ANSWER'},
    ]
    wire = items(messages, trace_id=uid(1), tools=[tool()])
    for i, entry in enumerate(wire):
        entry['metadata']['run_id'] = 'initial' if i <= 5 else 'later'
    for index in (1, 3):
        wire[index]['message'].pop('available_tools')
    value = {'id': 'example', 'inputs': {'messages': messages}, 'outputs': None,
             'metadata': {'source_workspace_id': SOURCE['workspace_id'], 'source_project_id': SOURCE['project_id'],
                          'source_scope': 'thread', 'source_scope_id': 'thread',
                          'trajectory_format': 'messages', 'conversation_scope': 'root',
                          'supervision': trajectory_supervision(wire, messages, 'skip-initial-history'),
                          'smithtune_source': {'schema_version': 1, 'assistant_runs': [
                              {'message_index': i, **wire[i]['metadata'], 'tools': None if i < 5 else [tool()]}
                              for i in (1, 3, 5, 7)]}}}
    return value, wire


def wire_api(wire, fallback=None):
    def api(command, *, input=None, **kwargs):
        if command[2] != '/v1/trajectory':
            return fallback(command, input=input, **kwargs)
        body = json.loads(input)
        offset = int(body.get('cursor', '0'))
        return SimpleNamespace(stdout=json.dumps({'items': wire[offset:offset + 2],
            'next_cursor': str(offset + 2) if offset + 2 < len(wire) else None}))
    return api


def fetch(wire, **kwargs):
    return curation._fetch_trajectory(SOURCE['workspace_id'], SOURCE['project_id'],
        {'key': 'trace_id', 'id': uid(1)}, runner=wire_api(wire), **kwargs)


def test_pull_selects_targets_before_requiring_tool_metadata():
    value, wire = conversation()
    before = copy.deepcopy(wire)
    result = fetch(wire, supervision_scope='skip-initial-history')
    assert result['training_error'] is None
    assert result['messages'] == value['inputs']['messages']
    assert result['supervision']['target_message_indices'] == [5, 7]
    assert result['source']['assistant_runs'][0]['tools'] is None
    assert result['source']['assistant_runs'][2]['tools'] == [tool()]
    assert wire == before


@pytest.mark.parametrize('scope', [None, 'all-assistant'])
def test_all_assistant_still_requires_historical_availability(scope):
    _, wire = conversation()
    result = fetch(wire, supervision_scope=scope)
    assert 'assistant message 1' in result['training_error']
    assert 'unknown tool availability' in result['training_error']


@pytest.mark.parametrize('change', ['missing', 'built_in', 'schema'])
def test_selected_target_tool_evidence_remains_strict(change):
    value, wire = conversation()
    if change == 'missing':
        wire[5]['message'].pop('available_tools')
        value['metadata']['smithtune_source']['assistant_runs'][2]['tools'] = None
    elif change == 'built_in':
        wire[5]['message']['available_tools'] = [{'type': 'web_search'}]
        value['metadata']['smithtune_source']['assistant_runs'][2]['tools'] = [{'type': 'web_search'}]
    else:
        wire[5]['message']['available_tools'] = [tool('integer')]
        value['metadata']['smithtune_source']['assistant_runs'][2]['tools'] = [tool('integer')]
    if change != 'schema':
        assert 'assistant message 5' in fetch(wire, supervision_scope='skip-initial-history')['training_error']
    with pytest.raises(PipelineError):
        validate_bound_messages(value)
    with pytest.raises(PipelineError):
        dataset.prepare_sft_rows([value])


@pytest.mark.parametrize('context_bindings', ['unknown', 'absent', 'known'])
def test_only_targets_need_tool_contracts_through_import_prepare_and_replay(context_bindings):
    value, _ = conversation()
    bindings = value['metadata']['smithtune_source']['assistant_runs']
    if context_bindings == 'absent':
        del bindings[:2]
    elif context_bindings == 'known':
        for binding in bindings[:2]:
            binding['tools'] = [tool()]  # The old call is outside this contract, but is context only.
    before = copy.deepcopy(value)
    validate_bound_messages(value)
    assert dataset_import.import_rejection(value, SOURCE['workspace_id']) is None
    row, = dataset.prepare_sft_rows([value])
    targets = list(training_targets(row))
    assert [t['message_index'] for t in targets] == [5, 7]
    assert [c['message_index'] for c in build_replay_cases([row])] == [5, 7]
    assert targets[0]['messages'][3]['content'] == 'HISTORICAL ANSWER'
    assert value == before


def test_missing_selected_binding_and_changed_selection_do_not_become_context():
    value, _ = conversation()
    value['metadata']['smithtune_source']['assistant_runs'].pop()
    with pytest.raises(PipelineError, match='cover'):
        read_bindings(value['metadata'], value['inputs']['messages'])
    value, _ = conversation()
    value['metadata']['supervision'] = make_supervision(value['inputs']['messages'], [1, 5, 7])
    with pytest.raises(PipelineError):
        validate_bound_messages(value)


def test_context_tool_result_pairing_is_still_required():
    value, _ = conversation()
    value['inputs']['messages'][2]['tool_call_id'] = 'unmatched'
    value['metadata']['supervision'] = make_supervision(value['inputs']['messages'], [5, 7])
    with pytest.raises(PipelineError, match='tool'):
        validate_bound_messages(value)
    with pytest.raises(PipelineError, match='tool'):
        dataset.prepare_sft_rows([value])


@pytest.mark.parametrize('provider', ['fireworks', 'baseten'])
def test_actual_provider_rendering_masks_context_with_unknown_tools(provider):
    from smithtune import rendering
    from smithtune.hf_rendering import HFRenderer
    from smithtune.providers.baseten.provider import DEFAULT_MODEL as BASETEN_MODEL
    from smithtune.providers.fireworks.provider import DEFAULT_MODEL as FIREWORKS_MODEL
    from tests.test_hf_rendering import _loss_spans, _tokenizer
    from tests.test_training_dependency import CharacterTokenizer
    from training.renderer import get_renderer

    value, _ = conversation()
    row, = dataset.prepare_sft_rows([value])
    model = FIREWORKS_MODEL if provider == 'fireworks' else BASETEN_MODEL
    tokenizer = CharacterTokenizer() if provider == 'fireworks' else _tokenizer()
    renderer = get_renderer(model.renderer, tokenizer) if provider == 'fireworks' else HFRenderer(model, tokenizer)
    rendered = rendering.render_row_tokens(row, model, renderer=renderer)
    assert len(rendered) == 2
    for datum in rendered:
        text = tokenizer.decode(datum.token_ids)
        trained = ''.join(_loss_spans(datum, tokenizer))
        assert 'HISTORICAL ANSWER' in text
        assert 'HISTORICAL ANSWER' not in trained
        assert 'old_search' not in trained
    assert 'search' in ''.join(_loss_spans(rendered[0], tokenizer))
    assert 'NEW ANSWER' in ''.join(_loss_spans(rendered[1], tokenizer))


def test_explicit_tool_override_applies_to_targets_in_replay_too(tmp_path, monkeypatch):
    from smithtune.artifacts import _load_jsonl
    from smithtune.bindings import tool_contract
    from smithtune.evaluation import replay
    from smithtune.providers.fireworks.provider import DEFAULT_MODEL
    from tests.test_pipeline import write_raw

    value, _ = conversation()
    write_raw(tmp_path, [value])
    contract = tool_contract([tool()])
    dataset.prepare_dataset('workspace-id', 'dataset-id', DEFAULT_MODEL, tmp_path,
        fetch=False, check_render=False, test_fraction=1, validation_fraction=0, inference_contract=contract)
    monkeypatch.setattr(replay, 'validate_replay_context', lambda cases, *_a, **_k:
                        ([{**case, 'prompt_tokens': 10} for case in cases], []))
    out = tmp_path / 'replay'
    replay.prepare_replay_evaluation(tmp_path, out)
    cases = _load_jsonl(out / 'cases.jsonl')
    assert [case['message_index'] for case in cases] == [5, 7]
    for case in cases:
        resolved = replay._case_contract(case, contract, None)
        request = resolved.build_chat_request(model='test-model', messages=case['messages'], max_tokens=32)
        assert request['messages'] == case['messages']
        assert request['tools'] == [tool()]
        assert resolved.contract_sha256 == contract.contract_sha256


def test_unbound_export_capture_respects_saved_explicit_selection():
    value, wire = conversation()
    del value['metadata']['smithtune_source']
    value['metadata']['supervision'] = make_supervision(value['inputs']['messages'], [5, 7])
    enriched, errors = dataset.capture_example_bindings([value], SOURCE['workspace_id'], runner=wire_api(wire))
    assert not errors
    validate_bound_messages(enriched[0])
    assert enriched[0]['metadata']['supervision'] == value['metadata']['supervision']


def test_pull_judge_upload_prepare_roundtrip_with_unknown_historical_tools(tmp_path):
    value, wire = conversation()
    api = API()
    runner = wire_api(wire, api)
    directory = tmp_path / 'curation'
    dataset_workflow.run('pull', directory, runner=runner, supervision_scope='skip-initial-history', **SOURCE)
    frozen = triage_source.load_snapshot(directory)
    assert frozen['units'][0]['training_error'] is None
    seen = []
    def judge(_judge, messages, _tokens):
        evidence = json.loads(messages[1]['content'])
        assert evidence['target_message_indices'] == [5, 7]
        assert evidence['untrusted_trajectory'] == value['inputs']['messages']
        assert evidence['untrusted_assistant_tool_bindings'][0]['tools'] is None
        seen.append(True)
        return {'keep': 1, 'reason': 'Selected teacher actions are supported.'}
    dataset_workflow.run('triage', directory, runner=runner, judge_call=judge, confirm=True,
                         rules=['Keep supported teacher actions.'])
    assert seen
    result = dataset_workflow.run('push', directory, runner=runner, confirm=True, name='selected-tools')
    assert result['created'] == 1
    uploaded, = api.examples.values()
    row, = dataset.prepare_sft_rows([uploaded])
    assert [case['message_index'] for case in build_replay_cases([row])] == [5, 7]

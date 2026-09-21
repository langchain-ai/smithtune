"""Per-output provenance and actual provider target rendering regressions."""
import copy
import json

import pytest

from smithtune.bindings import trajectory_bindings, read_bindings, tool_contract, training_targets
from smithtune.dataset import prepare_sft_rows
from smithtune.evaluation.replay import build_replay_cases, score_replay_candidate, _judge_input
from smithtune.providers.base import PipelineError
from smithtune import rendering
from test_hf_rendering import _tokenizer, _loss_spans
from test_training_dependency import CharacterTokenizer
from trajectory_fixtures import run_items, items


def tool(kind='string', name='search'):
    return {'type': 'function', 'function': {'name': name, 'description': kind,
            'parameters': {'type': 'object', 'properties': {'query': {'type': kind}}, 'required': ['query'], 'additionalProperties': False}}}


def example():
    messages = [{'role': 'human', 'content': 'question', 'id': 'question'}]
    runs = []
    for i, tools in enumerate(([tool()], [tool('integer'), tool(name='unused')], [])):
        m = {'role': 'ai', 'id': f'output-{i}', 'content': f'answer-{i}'}
        messages.append(m)
        runs.append({'id': f'run-{i}', 'trace_id': f'trace-{i}', 'run_type': 'llm',
                     'inputs': {'messages': copy.deepcopy(messages[:-1])}, 'outputs': {'messages': [m]},
                     'extra': {'invocation_params': {'tools': tools}}})
        if i < 2:
            messages.append({'role': 'human', 'content': f'next-{i}'})
    return {'id': 'example', 'inputs': {'messages': messages}, 'outputs': None,
            'metadata': {'trajectory_format': 'messages', 'conversation_scope': 'root',
                         'source_scope': 'thread', 'source_scope_id': 'thread', 'source_project_id': 'project',
                         'source_workspace_id': 'workspace', 'smithtune_source': trajectory_bindings(run_items(messages, runs))}}, runs


def test_roundtrip_bindings_and_replay_original_positions():
    value, _ = example()
    value = json.loads(json.dumps(value))
    row, = prepare_sft_rows([value])
    cases = build_replay_cases([row])
    assert [c['message_index'] for c in cases] == [1, 3, 5]
    assert [len(c['tools']) for c in cases] == [1, 2, 0]
    assert [c['source_run_id'] for c in cases] == ['run-0', 'run-1', 'run-2']
    candidate = {'role': 'assistant', 'tool_calls': [{'function': {'name': 'search', 'arguments': '{"query":"x"}'}}]}
    assert score_replay_candidate(cases[0], candidate, tool_contract(cases[0]['tools']))['arguments_schema_valid']
    assert not score_replay_candidate(cases[1], candidate, tool_contract(cases[1]['tools']))['arguments_schema_valid']
    assert not score_replay_candidate(cases[2], candidate, tool_contract([]))['arguments_schema_valid']
    assert json.loads(_judge_input(cases[1], candidate)[1]['content'])['untrusted_trajectory']['available_tools'] == cases[1]['tools']
    with pytest.raises(PipelineError, match='cover exactly'):
        value['metadata']['smithtune_source']['assistant_runs'].pop()
        read_bindings(value['metadata'], value['inputs']['messages'])


@pytest.mark.parametrize('provider', ['fireworks', 'baseten'])
def test_actual_provider_training_and_validation_targets(provider, monkeypatch):
    from smithtune.providers import fireworks, baseten
    from smithtune.hf_rendering import HFRenderer
    from training.renderer import get_renderer
    from training.recipes import sft_loop
    from smithtune.providers.fireworks_training import _render_conversation

    row, = prepare_sft_rows([example()[0]])
    model = fireworks.DEFAULT_MODEL if provider == 'fireworks' else baseten.DEFAULT_MODEL
    tokenizer = CharacterTokenizer() if provider == 'fireworks' else _tokenizer()
    renderer = get_renderer(model.renderer, tokenizer) if provider == 'fireworks' else HFRenderer(model, tokenizer)
    rendered = rendering.render_row_tokens(row, model, renderer=renderer)
    assert len(rendered) == 3
    for i, datum in enumerate(rendered):
        trained = ''.join(_loss_spans(datum, tokenizer))
        assert f'answer-{i}' in trained
        assert all(f'answer-{j}' not in trained for j in range(i))
        text = tokenizer.decode(datum.token_ids)
        assert ('"name": "unused"' in text or 'unused' in text) == (i == 1)
        if i == 2:
            assert '# Tools' not in text
    if provider == 'fireworks':
        monkeypatch.setitem(sft_loop._worker_state, 'renderer', renderer)
        monkeypatch.setitem(sft_loop._worker_state, 'max_seq_len', model.max_seq_len)
        actual = _render_conversation(row)
    else:
        actual = baseten.render_row(row, model, renderer=renderer)
    assert len(actual) == 3
    assert len(list(training_targets(row))) == 3


@pytest.mark.parametrize('provider', ['fireworks', 'baseten'])
def test_historical_calls_use_their_own_schemas_in_http_replay(provider, monkeypatch):
    from smithtune import inference
    from smithtune.evaluation.replay import _case_contract
    value, runs = example()
    messages = value['inputs']['messages']
    for i, argument in reversed(list(enumerate(['old-schema', 42]))):
        index = i * 2 + 1
        messages[index]['content'] = [{'type': 'tool_call', 'name': 'search', 'id': f'call-{i}', 'args': {'query': argument}}]
        messages.insert(index + 1, {'role': 'tool', 'content': 'found', 'tool_call_id': f'call-{i}'})
    value['metadata']['smithtune_source'] = trajectory_bindings(run_items(messages, runs))
    row, = prepare_sft_rows([value])
    cases = build_replay_cases([row])
    assert [case['message_index'] for case in cases] == [1, 4, 7]
    calls = []
    def post(request, *_, **__):
        calls.append(json.loads(request.data))
        return {'choices': [{'message': {'role': 'assistant', 'content': 'answer'}}]}
    monkeypatch.setattr(inference, '_post_json', post)
    for name in ('FIREWORKS_API_KEY', 'FIREWORKS_SESSION_ID', 'BASETEN_API_KEY'):
        monkeypatch.setenv(name, 'synthetic-test-value')
    for case in cases:
        contract = _case_contract(case, None, None)
        for model in ('base', 'tuned'):
            if provider == 'fireworks':
                inference._fireworks_chat_completion(model, case['messages'], 64, request_contract=contract)
            else:
                inference._baseten_chat_completion(model, case['messages'], 64, request_contract=contract,
                    endpoint=inference.BasetenEndpoint('model', 'deployment', 10000))
    assert [call['tools'] for call in calls] == [case['tools'] for case in cases for _ in range(2)]
    assert calls[-1]['tools'] == []
    assert len([m for m in calls[-1]['messages'] if m.get('tool_calls')]) == 2
    # Changing a prior binding cannot be hidden by the last target's empty tools.
    row['_source']['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = []
    with pytest.raises(PipelineError, match='unknown tool'):
        build_replay_cases([row])


@pytest.mark.parametrize('bad_calls', [{}, '', False, [None], [{'function': {'name': 'search', 'arguments': '[]'}}]])
def test_malformed_candidate_calls_fail_schema_scoring(bad_calls):
    row, = prepare_sft_rows([example()[0]])
    case = build_replay_cases([row])[0]
    metrics = score_replay_candidate(case, {'role': 'assistant', 'tool_calls': bad_calls}, tool_contract(case['tools']))
    assert metrics['arguments_schema_valid'] is False


def test_fireworks_real_loader_and_eager_validation_share_target_masks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from training.renderer import get_renderer
    from training.recipes import sft_loop
    from training.utils import resource_autosizing
    from smithtune.providers.fireworks import DEFAULT_MODEL
    from smithtune.providers.fireworks_training import ServerlessTraining
    row, = prepare_sft_rows([example()[0]])
    tokenizer = CharacterTokenizer()
    renderer = get_renderer(DEFAULT_MODEL.renderer, tokenizer)
    path = tmp_path / 'conversations.jsonl'
    path.write_text(json.dumps(row) + '\n')
    monkeypatch.setattr(sft_loop, '_init_render_worker', lambda *_: None)
    monkeypatch.setitem(sft_loop._worker_state, 'renderer', renderer)
    monkeypatch.setitem(sft_loop._worker_state, 'max_seq_len', DEFAULT_MODEL.max_seq_len)
    monkeypatch.setattr(resource_autosizing, 'select_render_worker_count', lambda **_: 0)
    config = SimpleNamespace(tokenizer_model='local', renderer_name=DEFAULT_MODEL.renderer,
        train_on_what=rendering.SFT_TARGET_POLICY, max_seq_len=DEFAULT_MODEL.max_seq_len,
        tokenizer_revision='local', thinking_trace_history_mode='', tokenizer_trust_remote_code=False,
        dataset=str(path), evaluation_dataset=str(path), max_examples=None,
        batch_size=1, group_by_length=False, length_group_factor=50, epochs=1)
    session = ServerlessTraining(config, tmp_path)  # No provider session is entered.
    training = sft_loop._flatten_rendered_batch(next(iter(session.loader)))
    for datums in (training, session.validation):
        assert len(datums) == 3
        for i, datum in enumerate(datums):
            target = tokenizer.decode([t for t, w in zip(datum.loss_fn_inputs['target_tokens'].data,
                                                        datum.loss_fn_inputs['weights'].data, strict=True) if w])
            assert f'answer-{i}' in target
            assert all(f'answer-{j}' not in target for j in range(i))
            assert ('unused' in tokenizer.decode(datum.model_input.to_ints())) == (i == 1)


def test_unused_builtin_is_not_silently_dropped_from_available_tools():
    value, runs = example()
    runs[0]['extra']['invocation_params']['tools'].append({'type': 'web_search_20250305', 'name': 'web_search'})
    with pytest.raises(PipelineError, match='provider-native|built-in|unsupported'):
        trajectory_bindings(run_items(value['inputs']['messages'], runs))


def test_native_availability_keeps_changing_schemas_and_final_text_tools():
    value, runs = example()
    wire = run_items(value["inputs"]["messages"], runs)
    for item in wire:
        item["message"].pop("id", None)  # Output IDs are no longer needed.
    original = copy.deepcopy(wire)
    bindings = trajectory_bindings(wire)["assistant_runs"]
    assert [b["message_index"] for b in bindings] == [1, 3, 5]
    assert [len(b["tools"]) for b in bindings] == [1, 2, 0]
    assert bindings[0]["tools"][0] != bindings[1]["tools"][0]
    assert wire == original


@pytest.mark.parametrize("missing", ["absent", None, {}, ""])
def test_unknown_availability_is_not_an_empty_tool_list(missing):
    wire = items([{"role": "ai", "content": "answer"}])
    if missing == "absent":
        del wire[0]["message"]["available_tools"]
    else:
        wire[0]["message"]["available_tools"] = missing
    with pytest.raises(PipelineError, match="unknown tool availability"):
        trajectory_bindings(wire)


@pytest.mark.parametrize("missing", ["run_id", "trace_id"])
def test_native_assistant_requires_provenance(missing):
    wire = items([{"role": "ai", "content": "answer"}])
    del wire[0]["metadata"][missing]
    with pytest.raises(PipelineError, match="run/trace provenance"):
        trajectory_bindings(wire)


def test_flat_function_definitions_from_trajectory_are_normalized():
    nested = tool()
    wire = items([{"role": "ai", "content": "answer"}], tools=[{"type": "function", **nested["function"]}])
    assert trajectory_bindings(wire)["assistant_runs"][0]["tools"] == [nested]

"""Upload -> fresh export -> prepare -> pinned splits -> actual sampler adapters -> reporting."""
import copy
import json
from types import SimpleNamespace

import pytest

from smithtune import dataset, dataset_workflow, rendering
from smithtune.artifacts import _load_jsonl
from smithtune.evaluation import langsmith as reporting, replay
from smithtune.providers import baseten, baseten_sampling, fireworks, fireworks_sampling
from smithtune.hf_rendering import HFRenderer
from smithtune.providers.base import PipelineError
from curation_fakes import API, SOURCE, DATASET
from test_hf_rendering import _tokenizer
from test_langsmith_evaluation import MemoryClient
from test_baseten_sampling import FakeService as BasetenService
from test_fireworks_sampling import Future


pytestmark = pytest.mark.sdk_integration


@pytest.mark.parametrize('provider', ['fireworks', 'baseten'])
def test_exported_bindings_reach_requests_scores_and_pinned_langsmith_resume(tmp_path, monkeypatch, provider):
    api = API()
    api.datasets[DATASET] = {'id': DATASET, 'name': 'roundtrip', 'data_type': 'kv'}
    curation = tmp_path / 'curation'
    result = dataset_workflow.run('create', curation, no_triage=True, dataset_id=DATASET,
                                  runner=api, confirm=True, **SOURCE)
    assert result['created'] == 1
    uploaded, = api.examples.values()
    parent_id = uploaded['id']
    value = copy.deepcopy(uploaded)
    value.pop('dataset_id')
    client = MemoryClient([value])
    monkeypatch.setattr(reporting, 'make_client', lambda _: client)
    monkeypatch.setattr(reporting.time, 'sleep', lambda _: None)
    api.calls.clear()
    # The fresh preparation reads only the remote export. It cannot see the
    # original curation path, and the source API refuses any subsequent reads.
    def no_source(method, path, body):
        assert not path.startswith(('/api/v2/', '/v1/trajectory', '/api/v1/sessions/'))
    api.failure = no_source
    original_download = dataset.download_dataset
    monkeypatch.setattr(dataset, 'download_dataset', lambda ws, ds, raw: original_download(ws, ds, raw, runner=api))
    if provider == 'fireworks':
        from training.renderer import get_renderer
        model = fireworks.DEFAULT_MODEL
        tokenizer = _tokenizer()
        renderer = get_renderer(model.renderer, tokenizer)
    else:
        model = baseten.DEFAULT_MODEL
        tokenizer = _tokenizer()
        renderer = HFRenderer(model, tokenizer)
    monkeypatch.setattr(rendering, 'load_training_renderer', lambda _: renderer)
    data, output = tmp_path / 'fresh-export', tmp_path / 'replay'
    manifest = dataset.prepare_dataset(SOURCE['workspace_id'], DATASET, model, data,
                                       validation_fraction=0, test_fraction=1)
    assert manifest['audit']['rendered_datums'] == 3
    assert manifest['split']['test'] == 1
    assert not (data / 'prepared/example_contracts.json').exists()
    reporting.verify_test_split(data, manifest, client=client)
    row, = _load_jsonl(data / 'prepared/test.jsonl')
    assert row['_source']['example_id'] == parent_id
    assert row['_source']['metadata']['smithtune_source'] == uploaded['metadata']['smithtune_source']
    requests, supplied = [], []

    def completion(prompt):
        text = tokenizer.decode(prompt.to_ints())
        idx = 2 if 'next-1' in text else 1 if 'next-0' in text else 0
        requests.append((idx, text))
        answer = 'answer-0' if idx == 0 else '<tool_call>\n<function=search>\n<parameter=query>\nhallucination\n</parameter>\n</function>\n</tool_call>'
        tokens = tokenizer.encode(answer + '<|im_end|>', add_special_tokens=False)
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, stop_reason='stop')])

    if provider == 'fireworks':
        monkeypatch.setattr(fireworks_sampling, 'load_training_renderer', lambda _: renderer)
        original_prompt = fireworks_sampling.replay_prompt
        def prompt(messages, tools, *args):
            supplied.append(copy.deepcopy((messages, tools)))
            return original_prompt(messages, tools, *args)
        monkeypatch.setattr(fireworks_sampling, 'replay_prompt', prompt)
        class Service:
            training_session_id = 'fake-session'
            def create_sampling_client(self, **_):
                return SimpleNamespace(sample=lambda **kw: Future(completion(kw['prompt'])), close=lambda: None)
        checkpoint_id = 'account/run-' + 'a' * 32 + '/epoch-1'
        sampler = fireworks_sampling.FireworksReplaySampler(model, checkpoint_id, output, service=Service(), snapshot='snapshot')
    else:
        monkeypatch.setattr(baseten_sampling, 'load_training_renderer', lambda _: renderer)
        original_prompt = renderer.prompt_tokens
        def prompt(messages, tools=None):
            supplied.append(copy.deepcopy((messages, tools)))
            return original_prompt(messages, tools)
        renderer.prompt_tokens = prompt
        class Service(BasetenService):
            def connect(self, *args):
                client = super().connect(*args)
                client.sample = lambda **kw: completion(kw['prompt'])
                return client
        checkpoint_id = 'bt://loops:run-1/sampler_weights/best-1'
        sampler = baseten_sampling.BasetenReplaySampler(model, checkpoint_id, output, service=Service(output))
    judge_evidence = []
    def judge(_model, messages, *_):
        evidence = json.loads(messages[-1]['content'])
        judge_evidence.append(evidence)
        candidate = evidence['candidate_next_action']
        reference = evidence['untrusted_trajectory']['reference_next_action']
        # A plausible but invalid candidate must fail despite an approving judge.
        passed = candidate == reference or 'hallucination' in json.dumps(candidate)
        return {'role': 'assistant', 'content': json.dumps({'pass': passed, 'reason': 'fake judge'})}
    client.drop_feedback = 'teacher_agreement'
    options = dict(data_dir=data, output_dir=output, tuned_model=checkpoint_id, base_model=model.base_model,
                   judge_model='judge', chat=judge, replay_sampler=sampler, concurrency=1, max_output_tokens=128, confirm=True)
    with pytest.raises(PipelineError, match='publish|upload|publication|LangSmith'):
        replay.run_replay_evaluation(**options)
    assert len(requests) == 6
    cases = _load_jsonl(output / 'cases.jsonl')
    assert [case['message_index'] for case in cases] == [1, 3, 5]
    assert [len(case['tools']) for case in cases] == [1, 2, 0]
    for case in cases:
        matched = [(m, tools) for m, tools in supplied if m == case['messages']]
        assert len(matched) >= 2 and all(tools == case['tools'] for _, tools in matched)
        assert any(e['untrusted_trajectory']['available_tools'] == case['tools'] for e in judge_evidence)
    for idx, text in requests:
        assert ('unused' in text) == (idx == 1)
        if idx == 2:
            assert '# Tools' not in text
    results = _load_jsonl(output / 'results.jsonl')
    assert len(results) == 3
    assert all(not r[label]['judgment']['pass'] and not r[label]['deterministic_metrics']['arguments_schema_valid']
               for r in results[1:] for label in ('base', 'tuned'))
    # Publication-only resume retains generations/judgments and the pinned
    # dataset version, even when the latest remote version changes.
    client.drop_feedback = None
    client.examples[parent_id]['metadata']['later_edit'] = True
    client._snapshot()
    before = len(judge_evidence)
    summary = replay.run_replay_evaluation(**options)
    assert summary['langsmith']['comparison_url']
    assert len(requests) == 6 and len(judge_evidence) == before
    parents = [r for r in client.runs_store.values() if r.reference_example_id]
    assert {str(r.reference_example_id) for r in parents} == {parent_id}
    children = [r for r in client.runs_store.values() if r.parent_run_id and r.inputs.get('tools') is not None]
    assert len(children) == 6
    for child in children:
        metadata = child.extra['metadata']
        case = next(c for c in cases if c['message_index'] == metadata['message_index'])
        assert child.inputs['tools'] == case['tools']
        assert metadata['source_run_id'] == case['source_run_id']
    # Saved cases/predictions cannot be reused with modified bound tools.
    row['_source']['metadata']['smithtune_source']['assistant_runs'][0]['tools'] = []
    from smithtune.artifacts import _jsonl_dump
    _jsonl_dump(data / 'prepared/test.jsonl', [row])
    with pytest.raises(PipelineError):
        replay.run_replay_evaluation(**options)
    assert len(requests) == 6

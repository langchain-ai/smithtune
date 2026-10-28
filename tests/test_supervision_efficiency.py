"""Resolve supervision once per operation without caching it across stages."""
import pytest

from smithtune import curation, dataset, supervision
from smithtune.bindings import validate_bound_messages
from smithtune.providers.base import PipelineError
from tests.curation_fakes import SOURCE, uid
from tests.test_context_tool_validation import conversation, wire_api


@pytest.mark.parametrize('operation', ['prepare', 'validate', 'pull', 'capture'])
def test_supervision_hashes_messages_once_per_operation(monkeypatch, operation):
    value, wire = conversation()
    original = supervision.json_sha256
    hashed = []
    def counted(payload):
        hashed.append(payload)
        return original(payload)
    monkeypatch.setattr(supervision, 'json_sha256', counted)
    if operation == 'prepare':
        dataset.prepare_sft_rows([value])
    elif operation == 'validate':
        validate_bound_messages(value)
    elif operation == 'pull':
        result = curation._fetch_trajectory(SOURCE['workspace_id'], SOURCE['project_id'],
            {'key': 'trace_id', 'id': uid(1)}, runner=wire_api(wire), supervision_scope='skip-initial-history')
        assert result['training_error'] is None
    else:
        del value['metadata']['smithtune_source']
        enriched, errors = dataset.capture_example_bindings([value], SOURCE['workspace_id'], runner=wire_api(wire))
        assert enriched and not errors
    # Capture verifies both the saved export and the independently fetched source.
    expected = 2 if operation == 'capture' else 1
    assert len(hashed) == len({id(payload) for payload in hashed}) == expected
    assert all(payload == value['inputs']['messages'] for payload in hashed)


@pytest.mark.parametrize('operation', [validate_bound_messages, lambda value: dataset.prepare_sft_rows([value])])
def test_later_operations_recheck_changed_messages(operation):
    value, _ = conversation()
    operation(value)
    value['inputs']['messages'][3]['content'] = 'Changed historical context'
    with pytest.raises(PipelineError, match='messages_sha256'):
        operation(value)

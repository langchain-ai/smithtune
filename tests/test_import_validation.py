"""Whole conversations are rejected before upload, with no source recapture."""
import copy

import pytest

from smithtune import dataset_workflow as workflow
from curation_fakes import API, SOURCE


@pytest.mark.parametrize('problem', ['unknown_tool', 'arguments', 'unmatched', 'media', 'system', 'builtin'])
def test_pull_excludes_invalid_whole_conversation_without_remote_writes(tmp_path, problem):
    api = API()
    if problem in {'unknown_tool', 'arguments', 'unmatched'}:
        api.messages[1]['content'] = [{'type': 'tool_call', 'id': 'call', 'name': 'missing' if problem == 'unknown_tool' else 'search',
                                       'args': {'query': 1 if problem == 'arguments' else 'x'}}]
        if problem != 'unmatched':
            api.messages.insert(2, {'role': 'tool', 'tool_call_id': 'call', 'content': 'result'})
    elif problem == 'media':
        api.messages[0]['content'] = [{'type': 'image', 'url': 'image'}]
    elif problem == 'system':
        api.messages.insert(2, {'role': 'system', 'content': 'late system'})
    else:
        api.runs[1]['extra']['invocation_params']['tools'].append({'type': 'web_search'})
    # Update the producing output, without giving supplied history provenance.
    api.runs[1]['outputs']['messages'] = [copy.deepcopy(api.messages[1])]
    result = workflow.run('create', tmp_path, no_triage=True, name='destination', confirm=True, runner=api, **SOURCE)
    assert len(result['rejections']) == 1
    assert not api.writes

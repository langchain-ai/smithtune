"""In-memory LangSmith source/destination with real CLI wire envelopes."""
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from test_assistant_bindings import example
from trajectory_fixtures import run_items


def uid(n):
    return str(UUID(int=n))


WORKSPACE, PROJECT, DATASET = uid(100), uid(101), uid(901)
SOURCE = dict(workspace_id=WORKSPACE, project_id=PROJECT, start_time='2026-09-02T00:00:00Z', end_time='2026-09-03T00:00:00Z')


class API:
    def __init__(self):
        self.calls, self.writes = [], []
        self.failure = None
        self.datasets, self.examples = {}, {}
        self.version = '2026-09-17T00:00:00+00:00'
        value, runs = example()
        self.messages = value['inputs']['messages']
        self.runs = [{**r, 'id': uid(1000 + i), 'trace_id': uid(1), 'project_id': PROJECT,
                      'parent_run_ids': [uid(1)], 'is_root': False,
                      'inputs': {**r['inputs'], 'run_only_marker': 'NEVER_PERSIST_RUN_TREE'}} for i, r in enumerate(runs)]
        self.runs.insert(0, {'id': uid(1), 'trace_id': uid(1), 'project_id': PROJECT, 'parent_run_ids': [], 'is_root': True, 'run_type': 'chain'})
        self.roots = [{'id': uid(1), 'trace_id': uid(1), 'thread_id': 'thread', 'project_id': PROJECT, 'start_time': '2026-09-02T01:00:00Z'}]
        self.membership = copy.deepcopy(self.roots)
        self.trajectory_requests = 0

    def __call__(self, command, *, capture=False, input=None):
        if command[1:3] == ['dataset', 'get']:
            return SimpleNamespace(stdout=json.dumps({'id': command[3], 'example_count': len(self.examples)}))
        if command[1:3] == ['dataset', 'export']:
            Path(command[4]).write_text(json.dumps(list(self.examples.values())))
            return SimpleNamespace(stdout='')
        path = command[2]
        method = command[command.index('--method') + 1]
        body = json.loads(input) if input else json.loads(command[command.index('--body') + 1]) if '--body' in command else None
        self.calls.append((method, path, copy.deepcopy(body)))
        if self.failure:
            self.failure(method, path, body)
        route = urlsplit(path).path
        query = parse_qs(urlsplit(path).query)
        if route == '/api/v2/runs/query':
            roots = self.membership if body.get('filter', '').startswith('eq(thread_id,') else self.roots
            offset = int(body.get('cursor', '0'))
            value = {'items': roots[offset:offset + 1], 'next_cursor': str(offset + 1) if offset + 1 < len(roots) else None}
        elif route.startswith('/api/v2/traces/'):
            raise AssertionError('raw trace trees must not be fetched')
        elif route.startswith('/api/v1/sessions/'):
            value = {'id': PROJECT, 'start_time': '2026-08-01T00:00:00+00:00'}
        elif route == '/v1/trajectory':
            assert body['include'] == {'system_messages': True, 'tool_definitions': True}
            self.trajectory_requests += 1
            offset = int(body.get('cursor', '0'))
            assert body['format'] == 'ui'
            value = {'items': run_items(self.messages[offset:offset + 2], self.runs[1:]), 'next_cursor': str(offset + 2) if offset + 2 < len(self.messages) else None}
        elif route == '/api/v1/datasets' and method == 'GET':
            value = [d for d in self.datasets.values() if not query.get('name') or d['name'] == query['name'][0]]
        elif route == '/api/v1/datasets' and method == 'POST':
            self.writes.append((method, path, copy.deepcopy(body)))
            self.datasets[body['id']] = copy.deepcopy(body)
            value = body
        elif route.endswith('/versions'):
            value = [{'as_of': self.version}] if self.examples else []
        elif route.startswith('/api/v1/datasets/'):
            value = self.datasets[route.rsplit('/', 1)[1]]
        elif route == '/api/v1/examples' and method == 'GET':
            offset, limit = int(query.get('offset', ['0'])[0]), int(query.get('limit', ['100'])[0])
            value = list(self.examples.values())[offset:offset + limit]
        elif route == '/api/v1/examples' and method == 'POST':
            self.writes.append((method, path, copy.deepcopy(body)))
            if body['id'] in self.examples:
                raise subprocess.CalledProcessError(1, command, stderr='HTTP 409')
            self.examples[body['id']] = copy.deepcopy(body)
            value = {'id': body['id']}
        elif route.startswith('/api/v1/examples/'):
            eid = route.rsplit('/', 1)[1]
            if method == 'GET':
                value = self.examples[eid]
            else:
                self.writes.append((method, path, copy.deepcopy(body)))
                self.examples[eid].update(copy.deepcopy(body))
                value = {'message': 'Example updated'}
        else:
            raise AssertionError((method, path, body))
        return SimpleNamespace(stdout=json.dumps(value))

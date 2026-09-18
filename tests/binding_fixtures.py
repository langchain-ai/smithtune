"""Explicit recorded bindings for synthetic source and canonical-row fixtures."""
import copy


LOOKUP = {'type': 'function', 'function': {'name': 'lookup', 'parameters': {
    'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}


def bound_example(value, tools=None):
    if tools is None:
        names = {part['name'] for m in value['inputs']['messages'] for part in (m.get('content') if isinstance(m.get('content'), list) else []) if isinstance(part, dict) and part.get('type') == 'tool_call'}
        tools = [LOOKUP] if 'lookup' in names else []
        tools += [{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}} for name in sorted(names - {'lookup'})]
    value['metadata']['smithtune_source'] = {'schema_version': 1, 'assistant_runs': [
        {'message_index': i, 'run_id': f"run-{value['id']}-{i}", 'trace_id': f"trace-{value['id']}", 'tools': copy.deepcopy(tools)}
        for i, message in enumerate(value['inputs']['messages']) if message.get('role') == 'ai']}
    return value


def bound_row(row):
    source = row['_source']
    source['message_positions'] = list(range(len(row['messages'])))
    if source.get('contract_sha256'):
        row['tool_policy'] = 'global_override'
        return row
    row.setdefault('tool_policy', 'per_assistant')
    source.setdefault('metadata', {})['smithtune_source'] = {'schema_version': 1, 'assistant_runs': [
        {'message_index': i, 'run_id': f"run-{source.get('example_id')}-{i}", 'trace_id': 'trace', 'tools': copy.deepcopy(row.get('tools', []))}
        for i, message in enumerate(row['messages']) if message.get('role') == 'assistant']}
    return row

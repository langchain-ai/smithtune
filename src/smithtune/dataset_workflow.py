"""Thin command composition: one lock and ordinary ordered stage functions."""

import shlex
import sys
from pathlib import Path

from smithtune import checkpoint as storage, dataset_import, triage, triage_source
from smithtune.artifacts import _run, output_lock
from smithtune.curation import _time, _uuid
from smithtune.dataset_artifacts import new_run_directory
from smithtune.providers.base import PipelineError


STAGES = ('pull', 'triage', 'push')


def _settings(directory, checkpoint, command, options):
    requested = ['pull', 'push'] if options.get('no_triage') else list(STAGES)
    if command == 'create':
        if options.get('no_triage') and 'triage' in checkpoint['stages']:
            raise storage.CheckpointError('--no-triage conflicts with this checkpoint; use a new directory')
        # A flagless create rerun uses saved workflow intent.
        if checkpoint['stages'] and options.get('no_triage') is None:
            requested = checkpoint['stages'] if 'push' in checkpoint['stages'] else requested
        checkpoint['stages'] = [s for s in STAGES if s in set(checkpoint['stages']) | set(requested)]
    elif command != 'resume':
        checkpoint['stages'] = [s for s in STAGES if s in set(checkpoint['stages']) | {'pull', command}]
    for key in ('workspace_id', 'project_id', 'start_time', 'end_time', 'filter', 'limit', 'seed'):
        value = options.get(key)
        if value is not None:
            value = _time(value) if key in {'start_time', 'end_time'} else value
            if value != checkpoint['source'].get(key):
                raise storage.CheckpointError(f'--{key.replace("_", "-")} conflicts with frozen source selection; use a new checkpoint')
    name, dataset_id = options.get('name'), options.get('dataset_id')
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise storage.CheckpointError('dataset name must be nonempty')
    if name and dataset_id:
        raise storage.CheckpointError('use --name or --dataset-id')
    if dataset_id:
        dataset_id = _uuid(dataset_id, 'dataset ID')
    destination = checkpoint.get('destination') or {}
    if destination.get('dataset_id') and ((dataset_id and dataset_id != destination['dataset_id']) or (name and name != destination.get('name'))):
        raise storage.CheckpointError('destination is already bound; use the saved name/ID or no destination flags')
    if name and destination.get('name') not in (None, name):
        raise storage.CheckpointError('destination name conflicts with checkpoint')
    if dataset_id and checkpoint.get('pending_write') and dataset_id != destination.get('dataset_id'):
        raise storage.CheckpointError('reconcile the pending destination write before changing destination')
    if name:
        destination['name'] = name
    if dataset_id:
        destination['dataset_id'] = dataset_id
    checkpoint['destination'] = destination
    if 'triage' in checkpoint['stages']:
        checkpoint['council'] = triage.council_settings(
            directory, judges=options.get('judges'), rules=options.get('rules'), config_path=options.get('config_path'),
            runner_mode=options.get('runner_mode'), concurrency=options.get('concurrency'),
            attempts=options.get('attempts'), max_output_tokens=options.get('max_output_tokens'))
    storage.save(directory, checkpoint)


def pending_stages(directory, checkpoint):
    result = []
    if not storage.pull_complete(checkpoint):
        result.append('pull')
    if 'triage' in checkpoint['stages'] and ('pull' in result or triage.triage_summary(directory, checkpoint)['status'] != 'complete'):
        result.append('triage')
    if 'push' in checkpoint['stages'] and (result or not dataset_import.push_complete(directory, checkpoint)):
        result.append('push')
    return result


def run(command, directory=None, *, confirm=False, runner=_run, judge_call=None, **options):
    if directory is None:
        if command not in {'pull', 'create'}:
            raise storage.CheckpointError(f'dataset {command} requires a checkpoint directory')
        directory = new_run_directory()
    directory = Path(directory)
    print(f'Curation checkpoint: {directory}', file=sys.stderr)
    concurrency = options.get('concurrency', 4) or 4
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise storage.CheckpointError('concurrency must be between 1 and 16 (pull caps at 4)')
    with output_lock(directory):
        if any((directory / name).exists() for name in ('snapshot.json', 'selection.json')):
            storage.load(directory)  # emits the migration hint
        if (directory / 'checkpoint.json').exists():
            checkpoint = storage.load(directory)
        else:
            if command not in {'pull', 'create'} or not options.get('workspace_id') or not options.get('project_id'):
                raise storage.CheckpointError('start with dataset pull DIR --workspace-id WORKSPACE --project-id PROJECT')
            if any(p.name != '.smithtune.lock' for p in directory.iterdir()):
                raise storage.CheckpointError('new curation checkpoint requires an empty directory')
            try:
                source = triage_source.source_options(options['workspace_id'], options['project_id'], options.get('start_time'), options.get('end_time'),
                                                     filter=options.get('filter'), limit=options.get('limit') if options.get('limit') is not None else 100,
                                                     seed=options.get('seed') if options.get('seed') is not None else 42)
            except PipelineError as exc:
                raise storage.CheckpointError(str(exc)) from exc
            checkpoint = {'schema_version': 1, 'source': source, 'stages': [], 'selection': None,
                          'council': None, 'destination': {}, 'pending_write': None}
        try:
            _settings(directory, checkpoint, command, options)
        except PipelineError as exc:
            raise storage.CheckpointError(str(exc)) from exc
        # This validates local integrity even on a completed, network-free resume.
        examples = storage.conversations(directory, checkpoint)
        storage.votes(directory, checkpoint, examples)
        pending = pending_stages(directory, checkpoint)
        result = {'checkpoint': str(directory), 'source': checkpoint['source'], 'destination': checkpoint['destination'],
                  'council': checkpoint['council'], 'pending_stages': pending, 'status': 'complete'}
        if command == 'resume' and not confirm:
            return {**result, 'status': 'incomplete' if pending else 'complete',
                    **({'next_command': f'smithtune dataset resume {shlex.quote(str(directory))} --confirm'} if pending else {})}
        stages = checkpoint['stages'] if command in {'create', 'resume'} else [command]
        for stage in stages:
            if command in {'create', 'resume'} and stage not in pending:
                continue
            if stage == 'pull':
                triage_source.pull(directory, checkpoint, runner=runner, concurrency=min(concurrency, 4))
                result.update(selected=len(checkpoint['selection']), downloaded=sum(s['status'] == 'complete' for s in checkpoint['selection']),
                              rejections=[{'scope_id': s['scope_id'], 'reason': s['reason']} for s in checkpoint['selection'] if s['status'] == 'excluded'])
                if not storage.pull_complete(checkpoint):
                    result.update(status='incomplete', errors=[{'scope_id': s['scope_id'], 'error': s.get('error')} for s in checkpoint['selection'] if s['status'] == 'pending'])
                    break
            elif stage == 'triage':
                value = triage.run_triage(directory, checkpoint, confirm=confirm, judge_call=judge_call)
                result['triage'] = value
                result['status'] = value['status']
                if value['status'] != 'complete':
                    break
            else:
                if not storage.pull_complete(checkpoint) or ('triage' in checkpoint['stages'] and triage.triage_summary(directory, checkpoint)['status'] != 'complete'):
                    result.update(status='incomplete' if confirm else 'preview', accepted=None,
                                  next_command=f'smithtune dataset resume {shlex.quote(str(directory))} --confirm')
                    break
                result.update(dataset_import.push(directory, checkpoint, confirm=confirm, runner=runner))
                if result['status'] != 'complete':
                    break
        result['pending_stages'] = pending_stages(directory, checkpoint)
        if result['status'] == 'incomplete':
            result.setdefault('next_command', f'smithtune dataset resume {shlex.quote(str(directory))} --confirm')
        return result

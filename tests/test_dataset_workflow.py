"""Command confirmations, frozen settings, stage intent, locks and exit codes."""
import json
from contextlib import contextmanager

import pytest

from smithtune import cli, checkpoint, dataset_workflow as workflow
from curation_fakes import API, SOURCE


@pytest.fixture
def command(monkeypatch, capsys):
    api = API()
    original = workflow.run
    judged = []
    def judge(*args):
        judged.append(args)
        return {'keep': 1, 'reason': 'Complete.'}
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: original(*a, **kw, runner=api, judge_call=judge))
    def invoke(*args):
        code = 0
        try:
            cli.main(['dataset', *map(str, args)])
        except SystemExit as exc:
            code = exc.code
        captured = capsys.readouterr()
        return code, json.loads(captured.out) if captured.out else None, captured.err
    invoke.api, invoke.judged = api, judged
    return invoke


def source_flags():
    return [argument for key, value in SOURCE.items() for argument in ('--' + key.replace('_', '-'), value)]


def test_create_preview_confirm_and_resume_one_lock(command, tmp_path, monkeypatch):
    acquisitions = []
    original = workflow.output_lock
    @contextmanager
    def lock(directory):
        acquisitions.append(directory)
        with original(directory):
            yield
    monkeypatch.setattr(workflow, 'output_lock', lock)
    code, preview, _ = command('create', tmp_path, *source_flags(), '--name', 'selected')
    assert code == 0 and preview['status'] == 'preview'
    assert preview['triage']['accepted'] is None and preview['triage']['pending_votes'] == 3
    assert not command.api.writes and not command.judged
    assert len(acquisitions) == 1
    assert command('resume', tmp_path)[0] == 1
    code, result, _ = command('create', tmp_path, '--confirm')
    assert code == 0 and result['created'] == 1 and len(command.judged) == 3
    assert len(acquisitions) == 3  # One acquisition per command, no nested stage lock.
    command.api.calls.clear()
    assert command('resume', tmp_path, '--confirm')[0] == 0
    assert not command.api.calls and len(command.judged) == 3


def test_no_triage_intent_survives_flagless_create(command, tmp_path):
    assert command('create', tmp_path, *source_flags(), '--name', 'direct', '--no-triage')[0] == 0
    code, result, _ = command('create', tmp_path, '--confirm')
    assert code == 0 and result['created'] == 1 and not command.judged
    assert checkpoint.load(tmp_path)['stages'] == ['pull', 'push']
    assert not (tmp_path / 'triage.jsonl').exists()


def test_missing_destination_and_incomplete_judges_do_not_write(command, tmp_path):
    code, result, _ = command('create', tmp_path, *source_flags(), '--no-triage', '--confirm')
    assert code == 1 and 'dataset push' in result['next_command'] and '--name' in result['next_command']
    assert not command.api.writes
    assert command('triage', tmp_path)[0] == 0
    code, result, _ = command('push', tmp_path, '--name', 'selected', '--confirm')
    assert code == 1 and result['accepted'] is None and not command.api.writes


@pytest.mark.parametrize('args,hint', [
    (['create', '--triage-dir', 'old'], 'dataset push DIR'),
    (['create', '--run-dir', 'old'], 'dataset create DIR'),
    (['create', '--output', 'old'], 'dataset pull DIR'),
    (['triage', '--workspace-id', SOURCE['workspace_id']], 'dataset pull DIR'),
    (['resume'], 'requires a checkpoint directory'),
])
def test_obsolete_and_missing_arguments_exit_two(command, args, hint):
    code, _, error = command(*args)
    assert code == 2 and hint in error and not command.api.calls


def test_hidden_triage_aliases_and_legacy_layout(command, tmp_path):
    command('pull', tmp_path, *source_flags())
    code, result, _ = command('triage', '--output-dir', tmp_path, '--dry-run')
    assert code == 0 and result['status'] == 'preview' and not command.judged
    assert command('triage', tmp_path, '--dry-run', '--confirm')[0] == 2
    assert command('triage', tmp_path, '--output-dir', tmp_path)[0] == 2
    (tmp_path / 'selection.json').write_text('{}')
    code, _, error = command('resume', tmp_path)
    assert code == 2 and 'NEW_DIR' in error and '--dataset-id' in error


def test_generated_directory_and_pull_only_resume(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, result, _ = command('pull', *source_flags(), '--concurrency', '16')
    assert code == 0 and result['checkpoint'].startswith('data/datasets/')
    assert result['pending_stages'] == []
    command.api.calls.clear()
    assert command('resume', result['checkpoint'], '--confirm')[0] == 0
    assert not command.api.calls

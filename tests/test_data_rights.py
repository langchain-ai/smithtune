"""First-use acknowledgment must precede workflow work, including --confirm."""

import io
import json
from datetime import datetime

import pytest

from smithtune import cli, data_rights
from smithtune.providers.base import PipelineError


@pytest.fixture(autouse=True)
def unacknowledged_user(local_data_rights_acknowledgment):
    data_rights._receipt_path().unlink()


def terminal(monkeypatch, text, *, interactive=True):
    stream = io.StringIO(text)
    monkeypatch.setattr(stream, "isatty", lambda: interactive)
    monkeypatch.setattr(data_rights.sys, "stdin", stream)
    return stream


@pytest.mark.parametrize("answer", ["y\n", "yes\n", " YES \n"])
def test_explicit_read_acknowledgment_is_saved_and_reused(monkeypatch, capsys, answer):
    terminal(monkeypatch, answer)
    receipt = data_rights.require_acknowledgment()
    assert json.loads(data_rights._receipt_path().read_text()) == receipt
    assert receipt["document_version"] == data_rights.DOCUMENT_VERSION
    assert datetime.fromisoformat(receipt["read_at"]).tzinfo is not None
    output = capsys.readouterr()
    assert output.out == ""
    assert "Have you read" in output.err
    assert "[y/N]" in output.err
    assert data_rights.DOCUMENT_URL in output.err
    assert data_rights._receipt_path().stat().st_mode & 0o077 == 0
    terminal(monkeypatch, "", interactive=False)
    assert data_rights.require_acknowledgment() == receipt
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("answer", ["", "\n", "n\n", "no\n", "true\n"])
def test_decline_default_and_eof_do_not_save(monkeypatch, answer):
    terminal(monkeypatch, answer)
    with pytest.raises(PipelineError, match="not been acknowledged"):
        data_rights.require_acknowledgment()
    assert not data_rights._receipt_path().exists()


def test_interrupt_does_not_save(monkeypatch):
    stream = terminal(monkeypatch, "")
    def interrupt():
        raise KeyboardInterrupt
    monkeypatch.setattr(stream, "readline", interrupt)
    with pytest.raises(PipelineError, match="not been acknowledged"):
        data_rights.require_acknowledgment()
    assert not data_rights._receipt_path().exists()


@pytest.mark.parametrize("record", ["{", "null", "[]", '{}',
    '{"document_version":"old"}',
])
def test_invalid_receipt_and_piped_yes_cannot_bypass(monkeypatch, record):
    data_rights._receipt_path().write_text(record)
    stream = terminal(monkeypatch, "yes\n", interactive=False)
    with pytest.raises(PipelineError, match="smithtune acknowledge-data-rights"):
        data_rights.require_acknowledgment()
    assert stream.tell() == 0
    assert data_rights._receipt_path().read_text() == record


@pytest.mark.parametrize("field,value", [
    ("document_version", "old"), ("document_url", "https://example.com/other"),
    ("read_at", None), ("read_at", "not-a-timestamp"),
])
def test_changed_version_or_malformed_receipt_requires_acknowledgment(monkeypatch, field, value):
    terminal(monkeypatch, "yes\n")
    receipt = data_rights.require_acknowledgment()
    receipt[field] = value
    data_rights._receipt_path().write_text(json.dumps(receipt))
    terminal(monkeypatch, "", interactive=False)
    with pytest.raises(PipelineError, match="interactive terminal"):
        data_rights.require_acknowledgment()


def test_storage_failure_stops_workflow(monkeypatch):
    terminal(monkeypatch, "yes\n")
    def fail(*args):
        raise PermissionError("read-only config")
    monkeypatch.setattr(data_rights, "_json_dump", fail)
    with pytest.raises(PipelineError, match="Cannot save"):
        data_rights.require_acknowledgment()


@pytest.mark.parametrize("argv", [
    ["dataset", "push", "example", "--confirm"],
    ["dataset", "publish-splits", "--data-dir", "example"],
    ["prepare", "--workspace-id", "w", "--dataset-id", "d", "--model", "m"],
    ["capture-contract", "--workspace-id", "w", "--run-id", "r", "--output", "example"],
    ["plan"], ["train", "--confirm"], ["evaluate", "--confirm"],
    ["eval-plan"], ["deploy", "--run-dir", "example", "--confirm"],
    ["promote", "--run-dir", "example", "--output-model-id", "m", "--confirm"],
    ["undeploy", "--account-id", "a", "--deployment-id", "d", "--confirm"],
])
def test_gate_runs_before_any_workflow_activity(monkeypatch, capsys, argv):
    terminal(monkeypatch, "yes\n", interactive=False)
    def unexpected(*args):
        raise AssertionError("workflow started before acknowledgment")
    monkeypatch.setattr(cli, "command_status", unexpected)
    with pytest.raises(SystemExit) as failure:
        cli.main(argv)
    assert failure.value.code == 2
    assert "smithtune acknowledge-data-rights" in capsys.readouterr().err


def test_setup_command_records_only_read_acknowledgment(monkeypatch, capsys):
    terminal(monkeypatch, "yes\n")
    cli.main(["acknowledge-data-rights"])
    assert json.loads(capsys.readouterr().out) == json.loads(data_rights._receipt_path().read_text())


@pytest.mark.parametrize("argv", [["--help"], ["--version"], ["train", "--help"],
    ["acknowledge-data-rights", "--help"], ["doctor"], ["models", "list"],
])
def test_discovery_does_not_require_acknowledgment(monkeypatch, argv):
    def unexpected():
        raise AssertionError("discovery requested acknowledgment")
    monkeypatch.setattr(data_rights, "require_acknowledgment", unexpected)
    try:
        cli.main(argv)
    except SystemExit as exc:
        assert exc.code == 0
    assert not data_rights._receipt_path().exists()


def test_relative_config_directory_uses_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
    monkeypatch.setattr(data_rights.Path, "home", lambda: tmp_path)
    assert data_rights._receipt_path() == tmp_path / ".config/smithtune/data-rights.json"

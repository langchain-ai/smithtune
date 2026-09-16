"""Exercise terminal rendering and CLI cleanup without provider calls."""

import io
import json
import logging
import sys
import warnings
from contextlib import contextmanager
from threading import Thread

import pytest

from smithtune import cli, progress
from smithtune.providers.base import PipelineError


class Terminal(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


@pytest.fixture
def terminal(monkeypatch):
    @contextmanager
    def open_terminal():
        stdout, stderr = Terminal(), Terminal()
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", stdout)
            patch.setattr(sys, "stderr", stderr)
            patch.setenv("TERM", "xterm-256color")
            yield stdout, stderr
    return open_terminal


def test_animation_preserves_json_and_logs(terminal, monkeypatch):
    with terminal() as streams:
        stdout, stderr = streams

        def diagnose():
            print("Checking setup", file=sys.stderr)
            sys.stderr.write("Last log without newline")
            return {"status": "ok"}

        monkeypatch.setattr(cli, "diagnose", diagnose)
        cli.main(["doctor"])
        assert json.loads(stdout.getvalue()) == {"status": "ok"}
        assert "♪ Running doctor (0s)" in stderr.getvalue()
        assert "Checking setup" in stderr.getvalue()
        assert "Last log without newline" in stderr.getvalue()
        assert "\x1b[?25h" in stderr.getvalue()  # cursor restored
        assert sys.stderr is stderr


@pytest.mark.parametrize("error", [PipelineError("failed"), KeyboardInterrupt()])
def test_cli_cleans_up_on_failure(terminal, monkeypatch, error):
    with terminal() as streams:
        stdout, stderr = streams

        def diagnose():
            raise error

        monkeypatch.setattr(cli, "diagnose", diagnose)
        with pytest.raises(SystemExit if isinstance(error, PipelineError) else KeyboardInterrupt):
            cli.main(["doctor"])
        output = stderr.getvalue()
        assert "\x1b[?25h" in output
        assert sys.stderr is stderr
        assert not stdout.getvalue()
        if isinstance(error, PipelineError):
            assert output.index("\x1b[?25h") < output.index("smithtune: error: failed")


@pytest.mark.parametrize("plain", ["stdout", "stderr", "dumb"])
def test_noninteractive_output_has_no_animation(terminal, monkeypatch, plain):
    with terminal() as streams:
        stdout, stderr = streams
        if plain == "dumb":
            monkeypatch.setenv("TERM", "dumb")
        else:
            monkeypatch.setattr(stdout if plain == "stdout" else stderr, "isatty", lambda: False)
        with progress.command_status("Running prepare"):
            print('{"ok": true}')
        assert stderr.getvalue() == "Running prepare\n"
        assert json.loads(stdout.getvalue()) == {"ok": True}


def test_frames_are_interchangeable_and_elapsed_updates(terminal, monkeypatch):
    with terminal() as streams:
        now = [0.0]
        monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
        with progress.command_status("Running prepare", frames=("a", "b")):
            now[0] = 1.1
        output = streams[1].getvalue()
        assert "a Running prepare (0s)" in output
        assert "b Running prepare (1s)" in output


def test_ascii_fallback(terminal, monkeypatch):
    with terminal() as streams:
        monkeypatch.setattr(Terminal, "encoding", "ascii")
        with progress.command_status("Running prepare"):
            pass
        output = streams[1].getvalue()
        assert "| Running prepare" in output
        assert "♪" not in output


@pytest.mark.parametrize("args", [["--help"], ["--version"], ["prepare", "--help"]])
def test_help_and_version_do_not_start_status(terminal, args):
    with terminal() as streams:
        with pytest.raises(SystemExit) as exc:
            cli.main(args)
        assert exc.value.code == 0
        assert not streams[1].getvalue()


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("fails", [False, True])
def test_command_hides_warnings_but_preserves_errors_and_restores_logging(terminal, monkeypatch, interactive, fails):
    with terminal() as (stdout, stderr):
        if not interactive:
            monkeypatch.setattr(stdout, "isatty", lambda: False)
        logger = logging.getLogger("smithtune.progress-test")
        monkeypatch.setattr(logger, "handlers", [logging.StreamHandler(stderr)])
        monkeypatch.setattr(logger, "level", logging.DEBUG)
        monkeypatch.setattr(logger, "propagate", False)
        previous_level = logging.root.manager.disable
        previous_filters = list(warnings.filters)

        def worker():
            logger.info("library chatter")
            logger.warning("library warning")
            warnings.warn("Python warning", UserWarning, stacklevel=2)
            logger.error("provider failed")

        try:
            with progress.command_status("Running evaluate"):
                thread = Thread(target=worker)
                thread.start()
                thread.join()
                print("LangSmith comparison: https://smith.langchain.com/comparison", file=sys.stderr)
                if fails:
                    raise PipelineError("command failed")
        except PipelineError:
            assert fails
        assert "Running evaluate" in stderr.getvalue()
        assert "library chatter" not in stderr.getvalue()
        assert "library warning" not in stderr.getvalue()
        assert "Python warning" not in stderr.getvalue()
        assert "provider failed" in stderr.getvalue()
        assert "LangSmith comparison:" in stderr.getvalue()
        assert logging.root.manager.disable == previous_level
        assert warnings.filters == previous_filters
        logger.warning("warnings restored")
        assert "warnings restored" in stderr.getvalue()

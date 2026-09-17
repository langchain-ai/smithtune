"""Shared artifact serialization and local command execution."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

from smithtune.providers.base import PipelineError
from smithtune.doctor import INSTALL_HELP


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def output_lock(directory: Path):
    """Prevent concurrent CLI runs from replacing each other's receipts and votes."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking this file can let a third process bypass the lock.
    with (directory / ".smithtune.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PipelineError(f"another smithtune operation is using {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def exclusive_output(argument: str):
    def decorate(function):
        parameters = signature(function)

        @wraps(function)
        def locked(*args, **kwargs):
            directory = parameters.bind(*args, **kwargs).arguments[argument]
            with output_lock(directory):
                return function(*args, **kwargs)

        return locked
    return decorate


def _json_dump(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def _atomic_text(path: Path, text: str) -> None:
    """An interrupted write must leave the previous complete artifact readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read valid JSON from {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        # JSON permits Unicode line and paragraph separators inside strings.
        # str.splitlines() treats those characters as record boundaries, while
        # JSONL written by _jsonl_dump uses only an ASCII newline delimiter.
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read valid JSONL from {path}: {exc}") from exc


def _run(command: list[str], *, capture: bool = False, input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture,
            input=input,
        )
    except FileNotFoundError as exc:
        help_text = INSTALL_HELP.get(command[0], f"Install {command[0]} and ensure it is on PATH")
        raise PipelineError(f"cannot run {command[0]}. {help_text}; run smithtune doctor to check setup") from exc

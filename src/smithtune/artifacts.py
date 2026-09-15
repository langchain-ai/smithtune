"""Shared artifact serialization and local command execution."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from http import HTTPStatus
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
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read valid JSONL from {path}: {exc}") from exc


def _langsmith_error(command: list[str], exc: subprocess.CalledProcessError) -> PipelineError:
    """Summarize diagnostics without exposing echoed requests or credentials."""
    operation = "LangSmith"
    if len(command) > 2 and command[1] == "api":
        path = command[2].split("?", 1)[0]
        if re.fullmatch(r"/?[a-zA-Z0-9_/-]{1,160}", path):
            operation += f" {path}"
    elif command[1:3] in (["dataset", "get"], ["dataset", "export"]):
        operation += " " + " ".join(command[1:3])

    diagnostic = exc.stderr or exc.stdout or ""
    if isinstance(diagnostic, bytes):
        diagnostic = diagnostic.decode("utf-8", errors="replace")
    diagnostic = diagnostic[:65536]
    # Dataset commands use the Go SDK's method/URL/status envelope.
    status = re.search(
        r'\b(?:(?:HTTP(?:/\d(?:\.\d)?)?|status(?: code)?)\s*[:=]?'
        r'|(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) "[^"\r\n]+":)\s*([45]\d\d)\b',
        diagnostic, re.I,
    )
    code = int(status[1]) if status else None
    if code is None:
        try:
            payload = json.loads(diagnostic)
        except (ValueError, RecursionError):
            payload = None
        if isinstance(payload, dict):
            value = payload.get("status_code", payload.get("status"))
            if isinstance(value, int) and 400 <= value <= 599:
                code = value

    if code is not None:
        try:
            reason = HTTPStatus(code).phrase
        except ValueError:
            reason = "request failed"
        detail = f"HTTP {code} {reason}"
        hints = {
            401: "check your LangSmith credentials",
            403: "check the API key's access to the selected workspace and resource",
            404: "check the resource ID and workspace",
            429: "rate limit reached; wait before retrying",
        }
        if code in hints:
            detail += "; " + hints[code]
        elif code >= 500:
            detail += "; LangSmith or its gateway failed to process the request"
        if code == 409 and command[1:3] == ["api", "/api/v1/datasets"] and "POST" in command:
            detail += "; dataset name already exists"
    elif re.search(r"timed? out|timeout|deadline exceeded", diagnostic, re.I):
        detail = "request timed out; check connectivity and LangSmith service status"
    elif re.search(r"connection refused|connection reset|no such host|name resolution|network is unreachable|could not resolve host", diagnostic, re.I):
        detail = "connection failed; check network connectivity and the LangSmith endpoint"
    elif re.search(r"certificate verify failed|certificate signed by unknown authority|TLS handshake", diagnostic, re.I):
        detail = "TLS connection failed; check certificates and proxy configuration"
    elif re.search(r"no API key|API key (?:is )?(?:missing|not (?:set|configured))|not authenticated", diagnostic, re.I):
        detail = "authentication is not configured; set LANGSMITH_API_KEY or log in with langsmith"
    else:
        detail = "request failed; run the operation directly with langsmith to inspect the full diagnostic"
    if "--method" in command:
        index = command.index("--method") + 1
        if index < len(command) and command[index].upper() not in {"GET", "HEAD", "OPTIONS"}:
            detail += "; outcome may be unknown; inspect existing resources before retrying"
    return PipelineError(f"{operation} failed (exit {exc.returncode}): {detail}")


def _run(command: list[str], *, capture: bool = False, input: str | None = None) -> subprocess.CompletedProcess[str]:
    langsmith = Path(command[0]).name == "langsmith"
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture,
            **({"stderr": subprocess.PIPE} if langsmith and not capture else {}),
            input=input,
        )
        if langsmith and not capture and getattr(result, "stderr", None):
            sys.stderr.write(result.stderr)
        return result
    except subprocess.CalledProcessError as exc:
        if langsmith:
            raise _langsmith_error(command, exc) from None
        raise
    except FileNotFoundError as exc:
        help_text = INSTALL_HELP.get(command[0], f"Install {command[0]} and ensure it is on PATH")
        raise PipelineError(f"cannot run {command[0]}. {help_text}; run smithtune doctor to check setup") from exc

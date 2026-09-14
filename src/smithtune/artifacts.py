"""Shared artifact serialization and local command execution."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smithtune.providers.base import PipelineError
from smithtune.doctor import INSTALL_HELP


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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

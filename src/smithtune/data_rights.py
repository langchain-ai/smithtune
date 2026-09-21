"""Local, versioned acknowledgment of the data-rights document."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from smithtune.artifacts import _json_dump
from smithtune.providers.base import PipelineError

# Bump together with the document's version when its terms change.
DOCUMENT_VERSION = "1"
DOCUMENT_URL = "https://github.com/langchain-ai/smithtune/blob/main/docs/data-rights-and-permitted-use.md"


def _receipt_path() -> Path:
    configured = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    root = configured if configured.is_absolute() else Path.home() / ".config"
    return root / "smithtune" / "data-rights.json"


def require_acknowledgment() -> dict:
    """Fail before workflow dispatch unless the user explicitly acknowledges reading."""
    path = _receipt_path()
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(receipt, dict)
                and receipt.get("document_version") == DOCUMENT_VERSION
                and receipt.get("document_url") == DOCUMENT_URL
                and isinstance(receipt.get("read_at"), str)):
            if datetime.fromisoformat(receipt["read_at"]).tzinfo is not None:
                return receipt
    except (OSError, ValueError):
        pass

    if not sys.stdin.isatty():
        raise PipelineError(
            f"Read Data Rights and Permitted Use: {DOCUMENT_URL}. "
            "Then run `smithtune acknowledge-data-rights` in an interactive terminal "
            "as the same user before running non-interactive commands."
        )

    print(
        "Before using smithtune, please read Data Rights and Permitted Use:\n\n"
        f"{DOCUMENT_URL}\n\n"
        "You are responsible for having permission to use your data for\n"
        "training and evaluation and to share it with selected providers.\n"
        "Internal-only use and open-weight models do not automatically\n"
        "make a use permitted.\n\n"
        "Have you read the Data Rights and Permitted Use document? [y/N] ",
        file=sys.stderr, end="", flush=True,
    )
    try:
        response = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = ""
    if response not in {"y", "yes"}:
        raise PipelineError("Data Rights and Permitted Use has not been acknowledged; no workflow was started.")

    receipt = {
        "document_version": DOCUMENT_VERSION,
        "document_url": DOCUMENT_URL,
        "read_at": datetime.now(UTC).isoformat(),
    }
    try:
        _json_dump(path, receipt)
    except OSError as exc:
        raise PipelineError(f"Cannot save the data-rights acknowledgment to {path}: {exc}") from exc
    return receipt

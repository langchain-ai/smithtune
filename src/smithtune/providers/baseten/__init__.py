"""Baseten training, sampling, and deployment integration."""

import os

from smithtune.providers.base import PipelineError


def require_api_key(*, message: str = "BASETEN_API_KEY is not set") -> str:
    """Require a nonblank credential without loading an optional provider SDK."""
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key.strip():
        raise PipelineError(message)
    return key

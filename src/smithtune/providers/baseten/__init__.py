"""Baseten training, sampling, and deployment integration."""

import os

from smithtune.providers.base import PipelineError


# Smithtune requests Loops capacity without a scope. Baseten then uses serverless
# capacity when the workspace and request are eligible and falls back to the
# workspace's own capacity otherwise; an explicit serverless scope would fail instead.
CAPACITY = "serverless_if_eligible"


def require_api_key(*, message: str = "BASETEN_API_KEY is not set") -> str:
    """Require a nonblank credential without loading an optional provider SDK."""
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key.strip():
        raise PipelineError(message)
    return key

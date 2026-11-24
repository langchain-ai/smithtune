"""Tinker training and sampling integration."""

import os

from smithtune.providers.base import PipelineError


# Smithtune sends Tinker credentials only to the standard Tinker API.
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod"
# Tinker samples from its shared fleet; smithtune owns no serving capacity.
CAPACITY = "tinker_shared"


def require_api_key(*, message: str = "TINKER_API_KEY is not set") -> str:
    """Require a nonblank credential without loading the Tinker SDK."""
    key = os.environ.get("TINKER_API_KEY", "")
    if not key.strip():
        raise PipelineError(message)
    base_url = os.environ.get("TINKER_BASE_URL", "")
    if base_url and base_url.rstrip("/") != TINKER_BASE_URL:
        raise PipelineError("smithtune uses the standard Tinker API; unset TINKER_BASE_URL")
    return key


def _sdk():
    try:
        import tinker
    except ImportError as exc:
        raise PipelineError(
            "the Tinker SDK is unavailable; reinstall using the installation command in the README, then run smithtune doctor"
        ) from exc
    return tinker


def service_client(**kwargs):
    """Open a Tinker session at the standard API with the checked credential."""
    key = require_api_key()
    return _sdk().ServiceClient(api_key=key, base_url=TINKER_BASE_URL, **kwargs)


def rest_client():
    """Read account-level metadata without a session; its shared connections are never closed."""
    key = require_api_key()
    return _sdk().RestClient(api_key=key, base_url=TINKER_BASE_URL)


def resolved(value):
    """Return a Tinker future's result; values that are not futures pass through.

    Tinker's client factories and REST reads return futures; using a client
    future as the client directly is deprecated.
    """
    result = getattr(value, "result", None)
    return result() if callable(result) else value


def close_service(service) -> None:
    """Stop a session's heartbeat; Tinker holds no other per-session capacity."""
    holder = getattr(service, "holder", None)
    if holder is not None:
        holder.close()

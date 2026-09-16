"""Surface synchronous SDK upload failures without replacing native retries."""

import logging
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError, filter_logs

from smithtune.providers.base import PipelineError


class PublicationRequestError(PipelineError):
    """A request failure containing only safe reporting diagnostics."""


def _request_error(error, method, pathname):
    response, seen = None, set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        if response is not None:
            break
        error = error.__cause__ or error.__context__
    status = response.status_code if response is not None else "unavailable"
    message = f"LangSmith {method} {urlsplit(pathname).path}: HTTP {status}"
    if response is None:
        return message
    # Only repeat known server diagnostics, never arbitrary response bodies.
    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = payload.get("detail", payload.get("error")) if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail.lower().rstrip(".") in {
        "rate limit exceeded", "monthly trace usage limit exceeded",
        "too many requests: tenant exceeded usage limits", "tenant exceeded usage limits",
    }:
        message += "; " + detail.rstrip(".")
    else:
        message += "; server detail omitted"
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isascii() and retry_after.isdecimal() and len(retry_after) <= 10:
        message += f"; Retry-After: {retry_after}s"
    elif retry_after:
        try:
            message += f"; Retry-After: {parsedate_to_datetime(retry_after).isoformat()}"
        except (ValueError, TypeError, OverflowError):
            pass
    return message


def _safe_retry_warning(record):
    # The SDK logs the raw malformed header while falling back to its default.
    if record.msg == "Invalid retry-after header: %s":
        record.msg = "Invalid Retry-After header; using the SDK default delay"
        record.args = ()
    return True


class PublishingClient(Client):
    """Use native SDK retries; turn logged batch failures into CLI failures."""

    def __init__(self, **kwargs):
        self._publication_error = None
        super().__init__(auto_batch_tracing=False, tracing_error_callback=self._record_error, **kwargs)

    def _record_error(self, error):
        self._publication_error = error

    def batch_ingest_runs(self, create=None, update=None):
        self._publication_error = None
        super().batch_ingest_runs(create=create, update=update)
        if self._publication_error is not None:
            raise self._publication_error

    def request_with_retries(self, method, pathname, **kwargs):
        try:
            with filter_logs(logging.getLogger("langsmith.client"), [_safe_retry_warning]):
                return super().request_with_retries(method, pathname, **kwargs)
        except Exception as error:
            # Format only after the SDK finishes its normal retry behavior.
            # Do this before SDK batch/feedback handlers can log raw exceptions.
            message = _request_error(error, method, pathname)
            if isinstance(error, LangSmithNotFoundError) and method == "GET" and urlsplit(pathname).path.rstrip("/") == "/sessions":
                raise LangSmithNotFoundError(message) from None
            raise PublicationRequestError(message) from None

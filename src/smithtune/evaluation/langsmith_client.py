"""Synchronous LangSmith publication with bounded retries and safe diagnostics."""

import json
import math
import random
import re
import sys
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from urllib3.util.retry import Retry

from smithtune.providers.base import PipelineError


class PublicationRequestError(PipelineError):
    """A request failure containing only sanitized reporting diagnostics."""


def _response_from_exception(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        if response is not None:
            return response
        error = error.__cause__ or error.__context__
    return None


def _retry_after(response):
    value = response.headers.get("Retry-After") if response is not None else None
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0, delay) if math.isfinite(delay) else None


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _server_detail(response, request, api_key):
    if response is None:
        return "server detail unavailable"
    try:
        payload = response.json()
    except ValueError:
        return "server returned a non-JSON error body"
    detail = payload
    for _ in range(3):
        if not isinstance(detail, dict):
            break
        detail = next((detail[key] for key in ("detail", "message", "error", "title") if key in detail), None)
    if not isinstance(detail, str):
        return "server detail unavailable"
    # Never emit full exceptions, headers, request bodies, or echoed inputs.
    private = [api_key or "", *list(_strings(request.get("headers", {})))]
    body = request.get("json")
    if body is None and isinstance(request.get("data"), (str, bytes)):
        try:
            body = json.loads(request["data"])
        except (ValueError, UnicodeError):
            pass
    body_strings = list(_strings(body))
    private.extend(body_strings)
    for value in sorted(set(private), key=len, reverse=True):
        if value and (len(value) >= 4 or detail == value):
            detail = detail.replace(value, "[redacted]")
    detail = re.sub(r"(?i)(?:bearer\s+\S+|(?:lsv2_|sk-|ghp_|xox[bp]-)[\w-]+)", "[redacted]", detail)
    # A server may echo only a fragment of a long message. Omit the detail
    # conservatively when a surviving phrase appears in the request payload.
    detail = " ".join(detail.split())[:500]
    request_text = " ".join(body_strings)
    # Short excerpts (email addresses, identifiers, numbers) need token checks
    # too. Common quota vocabulary is safe to retain in server explanations.
    diagnostic_words = {
        "rate", "limit", "limits", "exceeded", "too", "many", "requests", "request", "monthly", "daily", "hourly",
        "minute", "second", "seconds", "bytes", "size", "ingestion", "trace", "traces", "tenant", "organization",
        "workspace", "usage", "quota", "billing", "credits", "exhausted", "maximum", "allowed", "per", "retry",
        "after", "please", "try", "again", "later",
    }

    def tokens(text):
        return {token.strip(".") for token in re.findall(r"[\w@.+-]+", text.lower())}

    echoed = (tokens(detail) & tokens(request_text)) - diagnostic_words
    if echoed or any(detail[index:index + 12] in request_text for index in range(max(0, len(detail) - 11))):
        return "server detail omitted because it echoes request data"
    return detail


class PublishingClient(Client):
    """Keep SDK batches synchronous and surface failures the SDK otherwise logs."""

    def __init__(self, **kwargs):
        self._publication_error = None
        # Disable urllib3 retries as well as the SDK retry loop. Otherwise a
        # Retry-After can sleep inside HTTPAdapter before our budget sees it.
        super().__init__(auto_batch_tracing=False, tracing_error_callback=self._record_error,
                         retry_config=Retry(total=0, respect_retry_after_header=False, raise_on_status=False), **kwargs)

    def _record_error(self, error):
        self._publication_error = error

    def batch_ingest_runs(self, create=None, update=None):
        self._publication_error = None
        super().batch_ingest_runs(create=create, update=update)
        if self._publication_error is not None:
            raise self._publication_error

    def request_with_retries(self, method, pathname, **kwargs):
        # One retry owner for reads, batch writes, and feedback. The SDK's
        # different per-method attempt counts must not multiply these retries.
        kwargs["stop_after_attempt"] = 1
        for attempt in range(4):
            try:
                return super().request_with_retries(method, pathname, **kwargs)
            except Exception as error:
                response = _response_from_exception(error)
                status = response.status_code if response is not None else None
                delay = _retry_after(response)
                request = {**kwargs, **(kwargs.get("request_kwargs") or {})}
                detail = _server_detail(response, request, self.api_key)
                message = f"LangSmith {method} {urlsplit(pathname).path}: HTTP {status or 'unavailable'}; {detail}"
                if delay is not None:
                    message += f"; Retry-After: {delay:g}s"
                if isinstance(error, LangSmithNotFoundError) and method == "GET" and urlsplit(pathname).path.rstrip("/") == "/sessions":
                    # Keep the type for experiment lookup without the SDK
                    # exception text or its request/response traceback.
                    raise LangSmithNotFoundError(message) from None
                # A long cooldown is reported, never shortened into an early retry.
                if status != 429 or attempt == 3 or (delay is not None and delay > 60):
                    raise PublicationRequestError(message) from None
                wait = delay if delay is not None else min(5 * 2**attempt + random.uniform(0, 1), 60)
                print(f"{message}; retrying in {wait:g}s ({attempt + 1}/3)", file=sys.stderr)
                time.sleep(wait)

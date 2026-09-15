"""Shared Python sandbox with explicit host functions and bounded output."""

from __future__ import annotations

import json


def execute_code(code: str, functions: dict) -> dict:
    from pydantic_monty import CollectString, Monty, MontyError

    if not isinstance(code, str) or len(code) > 16_000:
        return {"error": "code must be a string of at most 16000 characters"}
    output = CollectString(max_bytes=16_000)
    try:
        with Monty(max_processes=1, request_timeout=30) as pool, pool.checkout(
            limits={"max_duration_secs": 5, "max_memory": 32 * 1024 * 1024, "max_suspensions": 256},
        ) as session:
            result = session.feed_run(code, external_lookup=functions, print_callback=output)
        if len(json.dumps(result, ensure_ascii=False)) > 32_000:
            return {"error": "code result too large; select fields or page strings/lists explicitly and inspect remaining pages"}
        return {"result": result, "stdout": output.output}
    except (MontyError, ValueError, TypeError):
        return {"error": "code failed; use supported Python and only these functions: " + ", ".join(functions)}

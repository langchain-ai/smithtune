"""Native trajectory UI responses, with per-message availability and provenance."""
import copy


def items(messages, *, trace_id="trace", run_id="run", tools=()):
    result = []
    for message in messages:
        message = copy.deepcopy(message)
        if message.get("role") in ("ai", "assistant"):
            message["available_tools"] = copy.deepcopy(list(tools))
        result.append({"type": "message", "message": message,
                       "metadata": {"trace_id": trace_id, "run_id": run_id}})
    return result


def run_items(messages, runs):
    result = []
    for message in messages:
        run = next((r for r in runs if any(m.get("id") == message.get("id")
                   for m in r.get("outputs", {}).get("messages", []))), None)
        entry = items([message], trace_id=run["trace_id"] if run else runs[0]["trace_id"],
                      run_id=run["id"] if run else runs[0]["id"])[0]
        if run and message.get("role") in ("ai", "assistant"):
            params = run.get("extra", {}).get("invocation_params", {})
            if "tools" in params:
                entry["message"]["available_tools"] = copy.deepcopy(params["tools"])
            else:
                entry["message"].pop("available_tools")
        result.append(entry)
    return result

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from binding_fixtures import bound_example
from smithtune import dataset
from smithtune.artifacts import _json_dump
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError


def example(index, *, thread=None, project="project-1"):
    return bound_example({
        "id": f"example-{index}",
        "inputs": {"messages": [{"role": "system", "content": f"policy {index}"},
                                {"role": "human", "content": f"question {index}"},
                                {"role": "ai", "content": f"answer {index}"}]},
        "outputs": None, "metadata": {"source_scope": "thread", "source_scope_id": thread or f"thread-{index}",
                                      "source_project_id": project,
                                      "trajectory_format": "messages", "conversation_scope": "root"},
    })


def write_empty_tool_snapshot(root, examples, dataset_id="dataset-id", workspace_id="workspace-id"):
    payload = {"examples": [bound_example(copy.deepcopy(e), tools=[]) for e in examples], "exclusions": []}
    _json_dump(root / "raw/bound_examples.json", {
        "source_sha256": hashlib.sha256((root / "raw/examples.json").read_bytes()).hexdigest(),
        "workspace_id": workspace_id, "source_workspace_id": None, **payload, "sha256": json_sha256(payload)})


@pytest.mark.parametrize("temperature", [None, 0.7])
def test_replay_temperature_default_survives_automatic_contracts(monkeypatch, temperature):
    import io
    from smithtune import inference
    from smithtune.inference_contract import parse_inference_contract
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "test-session")
    payload = {"schema_version": 1, "format": "main_model_inference_contract", "tools": [],
               "tools_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
               "inference_settings": {},
               "contract_sha256": "0c2f5299aeece0558ff0c797e3e0d0b2ae3c6b4f54c9273f9a8912f158833067",
               "provenance": {"source_workspace_id": "workspace-id", "source_thread_id": None,
                              "source_run_ids": ["run-1"], "source_trace_ids": ["trace-1"]}}
    if temperature is not None:
        payload["inference_settings"] = {"temperature": temperature}
        del payload["contract_sha256"]
    contract = parse_inference_contract(payload)
    requests = []

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())

    monkeypatch.setattr(inference, "open_without_redirects", urlopen)
    inference._fireworks_chat_completion("model", [{"role": "user", "content": "hi"}], 128, request_contract=contract)
    assert requests[0]["temperature"] == (0 if temperature is None else temperature)


@pytest.mark.parametrize("value", [None, "", " ", " padded ", 123, [], {}])
def test_invalid_source_workspace_metadata_does_not_fall_back(value):
    ex = example(1)
    ex["metadata"]["source_workspace_id"] = value

    with pytest.raises(PipelineError, match="source workspace ID"):
        dataset._source_key(ex, "workspace-id", None)


@pytest.mark.parametrize("scope", ["thread", "trace"])
def test_missing_source_project_fails_without_discovery(scope):
    ex = example(1)
    del ex["metadata"]["source_project_id"]
    ex["metadata"].update(source_scope=scope, source_scope_id=f"{scope}-1")

    with pytest.raises(PipelineError, match=r"missing source project ID; add metadata.source_project_id"):
        dataset._source_key(ex, "workspace-id", None)

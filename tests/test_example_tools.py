from __future__ import annotations

import copy
import hashlib
import json

import pytest

from binding_fixtures import bound_example
from smithtune import dataset
from smithtune.artifacts import _json_dump
from smithtune.inference_contract import ContractError, contract_from_runs, json_sha256
from smithtune.providers.base import PipelineError
from test_tool_capture import llm, tool


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
    payload = contract_from_runs([llm("run-1", [])], workspace_id="workspace-id")
    if temperature is not None:
        payload["inference_settings"] = {"temperature": temperature}
        del payload["contract_sha256"]
    contract = parse_inference_contract(payload)
    requests = []

    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())

    monkeypatch.setattr(inference.urllib.request, "urlopen", urlopen)
    inference._fireworks_chat_completion("model", [{"role": "user", "content": "hi"}], 128, request_contract=contract)
    assert requests[0]["temperature"] == (0 if temperature is None else temperature)


def test_optional_arguments_are_combined_across_multiple_snapshots():
    base = tool("list_threads")
    admin = copy.deepcopy(base)
    admin["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    limit = copy.deepcopy(base)
    limit["function"]["parameters"]["properties"]["limit"] = {"type": "integer"}
    merged = contract_from_runs([llm("a", [admin]), llm("b", [limit]), llm("c", [base])], workspace_id="workspace")
    assert merged["tools"][0]["function"]["parameters"]["properties"] == {
        "query": {"type": "string"}, "admin_threads": {"type": ["boolean", "null"]}, "limit": {"type": "integer"},
    }
    reversed_merge = contract_from_runs([llm("a", [limit]), llm("b", [base]), llm("c", [admin])], workspace_id="workspace")
    assert merged["contract_sha256"] == reversed_merge["contract_sha256"]


@pytest.mark.parametrize("change", ["required", "existing_type", "nested", "additional_properties"])
def test_optional_expansion_still_rejects_incompatible_changes(change):
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["filters"] = {"type": "object", "properties": {}}
    expanded = copy.deepcopy(base)
    parameters = expanded["function"]["parameters"]
    parameters["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    if change == "required":
        parameters["required"] = ["admin_threads"]
    elif change == "existing_type":
        parameters["properties"]["query"]["type"] = "integer"
    elif change == "nested":
        parameters["properties"]["filters"]["properties"]["new"] = {"type": "string"}
    else:
        parameters["additionalProperties"] = False
    with pytest.raises(ContractError, match="list_threads.*conflicting definitions.*a.*b"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


@pytest.mark.parametrize("constraint", [
    {"maxProperties": 1}, {"anyOf": [{"required": ["query"]}]},
    {"dependentRequired": {"query": ["admin_threads"]}},
    {"additionalProperties": {"type": "string"}}, {"required": ["admin_threads"]},
])
def test_optional_expansion_rejects_interacting_root_constraints(constraint):
    base = tool("list_threads")
    base["function"]["parameters"].update(constraint)
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")
    # Unchanged complex schemas remain supported.
    assert contract_from_runs([llm("a", [base]), llm("b", [base])], workspace_id="workspace")["tools"] == [base]


def test_optional_expansion_rejects_schema_references():
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["query"] = {"$ref": "#/properties/admin_threads"}
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


def test_optional_expansion_rejects_draft3_property_level_required():
    base = tool("list_threads")
    base["function"]["parameters"]["$schema"] = "http://json-schema.org/draft-03/schema#"
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {
        "type": ["boolean", "null"], "required": True,
    }
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


def test_optional_expansion_distinguishes_boolean_and_numeric_constraints():
    base = tool("list_threads")
    base["function"]["parameters"]["properties"]["flag"] = {"const": True}
    expanded = copy.deepcopy(base)
    expanded["function"]["parameters"]["properties"]["flag"] = {"const": 1}
    expanded["function"]["parameters"]["properties"]["admin_threads"] = {"type": ["boolean", "null"]}
    with pytest.raises(ContractError, match="conflicting definitions"):
        contract_from_runs([llm("a", [base]), llm("b", [expanded])], workspace_id="workspace")


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

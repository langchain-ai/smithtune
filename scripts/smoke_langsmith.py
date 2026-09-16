"""Create a tiny live LangSmith evaluation with synthetic model/judge responses.

No provider inference, training, tokenizer download, or deployment is performed.
The SDK dataset, split, experiment, trace and feedback operations are real.
Artifacts and remote resources remain available for inspection and resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid4, uuid5

from langsmith.utils import LangSmithNotFoundError

from smithtune import dataset, evaluation
from smithtune.artifacts import _json_dump, _load_json, output_lock
from smithtune.inference_contract import json_sha256, parse_inference_contract
from smithtune.langsmith_evaluation import make_client
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import DEFAULT_MODEL


def examples_for(workspace_id, contract):
    """Select two small, distinct conversations from each deterministic split."""
    examples = []
    for index in range(100):
        examples.append({
            "id": str(uuid4()),
            "inputs": {"messages": [
                {"role": "system", "content": "Answer each arithmetic question with only the number.", "id": f"s-{index}"},
                {"role": "human", "content": f"{index} + {index}", "id": f"u1-{index}"},
                {"role": "ai", "content": str(2 * index), "id": f"a1-{index}"},
                {"role": "human", "content": f"{index} + 1", "id": f"u2-{index}"},
                {"role": "ai", "content": str(index + 1), "id": f"a2-{index}"},
            ]},
            "outputs": None,
            "metadata": {"source_scope": "trace", "source_scope_id": str(uuid5(NAMESPACE_URL, f"smithtune-toy:{index}")),
                         "source_project_id": str(uuid5(NAMESPACE_URL, "smithtune-toy-project")),
                         "source_workspace_id": workspace_id, "trajectory_format": "messages",
                         "conversation_scope": "root", "synthetic": True},
        })
    rows = dataset.prepare_sft_rows(examples, contract, workspace_id=workspace_id, model=DEFAULT_MODEL)
    partitions = dataset.split_rows(rows, validation_fraction=.3, test_fraction=.3)
    if any(len(partition) < 2 for partition in partitions):
        raise PipelineError("toy sources did not produce two examples in each split")
    chosen = {row["_source"]["example_id"] for partition in partitions for row in partition[:2]}
    return [example for example in examples if example["id"] in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    receipt_path = root / "toy.json"
    if root.exists() and any(root.iterdir()) and not receipt_path.exists():
        raise PipelineError("toy output directory must be new, empty, or contain a saved toy.json")
    client = make_client(args.workspace_id)
    contract = parse_inference_contract({"schema_version": 1, "format": "main_model_inference_contract",
        "tools": [], "tools_sha256": json_sha256([]), "provenance": {"synthetic": True}, "inference_settings": {}})
    with output_lock(root):
        if receipt_path.exists():
            receipt = _load_json(receipt_path)
            if receipt["workspace_id"] != args.workspace_id:
                raise PipelineError("toy receipt belongs to another workspace")
        else:
            receipt = {"workspace_id": args.workspace_id, "dataset_name": f"smithtune-toy-{uuid4().hex[:12]}",
                       "examples": examples_for(args.workspace_id, contract)}
            _json_dump(receipt_path, receipt)
        try:
            remote = client.read_dataset(dataset_name=receipt["dataset_name"])
        except LangSmithNotFoundError:
            remote = client.create_dataset(receipt["dataset_name"], description="Smithtune SDK smoke test; synthetic responses and judge; no model inference.")
        receipt.update(dataset_id=str(remote.id), dataset_url=remote.url)
        _json_dump(receipt_path, receipt)
        existing = {str(example.id) for example in client.list_examples(dataset_id=remote.id)}
        missing = [example for example in receipt["examples"] if example["id"] not in existing]
        if missing:
            client.create_examples(dataset_id=remote.id, examples=missing)
        raw = [example.model_dump(mode="json") for example in client.list_examples(dataset_id=remote.id)]
        if {example["id"] for example in raw} != {example["id"] for example in receipt["examples"]}:
            raise PipelineError("toy dataset membership differs from its receipt")
        data = root / "data"
        _json_dump(data / "raw/examples.json", raw)
        _json_dump(data / "raw/dataset-export.json", [{"inputs": example["inputs"], "outputs": example.get("outputs")} for example in raw])
        _json_dump(data / "raw/dataset.json", {"id": str(remote.id), "name": remote.name, "example_count": len(raw)})
        manifest = dataset.prepare_dataset(args.workspace_id, str(remote.id), DEFAULT_MODEL, data,
            fetch=False, check_render=False, inference_contract=contract, validation_fraction=.3, test_fraction=.3)
        calls = []

        def synthetic_chat(model, messages, max_tokens, json_mode=False, request_contract=None):
            calls.append(model)
            if json_mode:
                evidence = json.loads(messages[-1]["content"])
                passed = evidence["candidate_next_action"] == evidence["reference_next_action"]
                return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "Synthetic exact-answer comparison."})}
            left, _, right = messages[-1]["content"].split()
            answer = str(int(left) + int(right)) if model == "toy-tuned" else "-1"
            return {"role": "assistant", "content": answer}

        # Rendering/model transport are the only fakes in this live SDK check.
        with patch.object(evaluation, "validate_replay_context", lambda cases, *_a, **_k: ([{**case, "prompt_tokens": 8} for case in cases], [])):
            summary = evaluation.run_replay_evaluation(data, root / "evaluation", "toy-tuned", "toy-judge",
                base_model="toy-base", chat=synthetic_chat, confirm=True)
            before = len(calls)
            resumed = evaluation.run_replay_evaluation(data, root / "evaluation", "toy-tuned", "toy-judge",
                base_model="toy-base", chat=synthetic_chat, confirm=True)
        if len(calls) != before or resumed != summary:
            raise PipelineError("toy evaluation resume repeated work or changed results")
        if summary["tuned_pass_rate"] != 1 or summary["base_pass_rate"] != 0:
            raise PipelineError("toy evaluation scores did not match the deterministic fixtures")
        result = {"status": "complete", "dataset_id": str(remote.id), "dataset_url": remote.url,
                  "workspace_id": args.workspace_id, "splits": {name: manifest["split"][name] for name in dataset.SPLIT_NAMES},
                  "langsmith": summary["langsmith"], "base_pass_rate": summary["base_pass_rate"],
                  "tuned_pass_rate": summary["tuned_pass_rate"], "resume_verified": True,
                  "synthetic_model_calls": len(calls), "paid_model_calls": 0,
                  "artifacts": str(root)}
        _json_dump(root / "result.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        # SDK exceptions can contain request details; keep credentials out of logs.
        print(f"Toy check interrupted ({type(exc).__name__}); inspect the saved receipt and rerun to resume.", file=sys.stderr)
        raise SystemExit(1) from None

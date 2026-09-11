from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

import artifacts
import dataset as dataset_ops
import evaluation as replay
import inference as inference_transport
import inference_contract
import pipeline
from providers import get_provider
import rendering
from providers import baseten, fireworks
from providers.base import ModelSpec, PipelineError


def message(role: str, content, message_id: str, **extra):
    return {"role": role, "content": content, "id": message_id, **extra}


def example(index: int, messages: list[dict] | None = None, thread: str | None = None) -> dict:
    return {
        "id": f"example-{index}",
        "inputs": {
            "messages": messages
            or [
                message("human", f"question {index}", f"human-{index}"),
                message("ai", f"answer {index}", f"ai-{index}"),
            ]
        },
        "outputs": None,
        "metadata": {
            "source_thread_id": thread or f"thread-{index}",
            "trajectory_format": "messages",
            "conversation_scope": "root",
        },
    }


def write_raw(root: Path, examples: list[dict], dataset_id: str = "dataset-id") -> None:
    raw = root / "raw"
    raw.mkdir()
    (raw / "examples.json").write_text(json.dumps(examples), encoding="utf-8")
    export = [{"inputs": item["inputs"], "outputs": item["outputs"]} for item in examples]
    (raw / "dataset-export.json").write_text(json.dumps(export), encoding="utf-8")
    (raw / "dataset.json").write_text(
        json.dumps({"id": dataset_id, "name": "test", "example_count": len(examples)}),
        encoding="utf-8",
    )


def write_manifest(
    root: Path,
    model: ModelSpec = fireworks.DEFAULT_MODEL,
    contract: inference_contract.InferenceContract | None = None,
) -> None:
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    manifest = {
        "langsmith": {"examples": 10},
        "split": {"train": 8, "validation": 1, "test": 1},
        "model": model.__dict__,
        "provider": {
            "name": model.provider,
            "renderer": model.renderer,
            "tokenizer_revision": model.tokenizer_revision,
        },
    }
    if contract is not None:
        manifest["inference_contract"] = contract.manifest_summary()
        (prepared / "inference_contract.json").write_text(
            json.dumps(contract.to_dict()), encoding="utf-8"
        )
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (prepared / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (prepared / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    (prepared / "test.jsonl").write_text("{}\n", encoding="utf-8")


def loaded_contract(tmp_path: Path) -> inference_contract.InferenceContract:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    payload = {
        "schema_version": 1,
        "format": "main_model_inference_contract",
        "tools": tools,
        "tools_sha256": inference_contract.json_sha256(tools),
        "system_prompt": {
            "role": "system",
            "content": "policy",
            "sha256": inference_contract.content_sha256("policy"),
        },
        "provenance": {"source_run_id": "run-id"},
        "inference_settings": {"temperature": 0},
    }
    path = tmp_path / "inference-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return inference_contract.load_inference_contract(path)


def test_download_dataset_pages_past_one_hundred(tmp_path: Path):
    rows = [example(i) for i in range(205)]
    for row in rows:
        row["source_thread_id"] = row["metadata"].pop("source_thread_id")
    commands: list[list[str]] = []

    def runner(command, capture=False):
        commands.append(command)
        if command[1:3] == ["dataset", "get"]:
            return SimpleNamespace(stdout=json.dumps({"id": "dataset-id", "example_count": 205}))
        if command[1:3] == ["dataset", "export"]:
            Path(command[4]).write_text(json.dumps([{}] * 205), encoding="utf-8")
            return SimpleNamespace(stdout="")
        query = parse_qs(urlparse(command[2]).query)
        offset = int(query["offset"][0])
        limit = int(query["limit"][0])
        return SimpleNamespace(stdout=json.dumps(rows[offset : offset + limit]))

    dataset, returned = dataset_ops.download_dataset("workspace-id", "dataset-id", tmp_path, runner)

    downloaded = json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))
    page_commands = [command for command in commands if command[1] == "api" and command[2].startswith("/api/v1/examples?")]
    assert len(downloaded) == 205
    assert dataset["id"] == "dataset-id"
    assert returned == rows
    assert [parse_qs(urlparse(command[2]).query)["offset"][0] for command in page_commands] == ["0", "100", "200"]
    assert all(command[-2:] == ["--method", "GET"] for command in page_commands)
    assert all("workspace-id" in command for command in commands)
    assert all(any("dataset-id" in argument for argument in command) for command in commands)
    dataset_ops.validate_trajectories(returned, 205)
    assert all(row["_source"]["source_thread_id"] for row in dataset_ops.prepare_sft_rows(returned))


def test_capture_contract_fetches_raw_invocation_parameters(tmp_path: Path):
    run = {
        "id": "run-id",
        "trace_id": "trace-id",
        "session_id": "project-id",
        "name": "ChatModel",
        "run_type": "llm",
        "start_time": "2026-09-03T12:00:00Z",
        "inputs": {"messages": [[{"role": "system", "content": "policy"}]]},
        "extra": {
            "metadata": {"ls_provider": "provider", "ls_model_name": "model"},
            "invocation_params": {
                "temperature": 0,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up a value.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        },
    }
    commands = []

    def runner(command, capture=False):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps([run]))

    output = tmp_path / "contract.json"
    summary = dataset_ops.capture_inference_contract(
        "workspace-id",
        "run-id",
        output,
        runner=runner,
    )
    contract = inference_contract.load_inference_contract(output)
    request_body = json.loads(commands[0][commands[0].index("--body") + 1])

    assert commands[0][:3] == ["langsmith", "api", "runs/query"]
    assert commands[0][commands[0].index("--workspace") + 1] == "workspace-id"
    assert request_body["id"] == ["run-id"]
    assert {"extra", "inputs"}.issubset(request_body["select"])
    assert summary["contract_sha256"] == contract.contract_sha256
    assert summary["source_run_id"] == "run-id"


def test_tool_call_conversion_preserves_ids_and_arguments():
    source = message(
        "ai",
        [{"type": "tool_call", "id": "call-1", "name": "lookup", "args": {"b": 2, "a": 1}}],
        "message-1",
    )
    original = copy.deepcopy(source)
    converted = dataset_ops.convert_message(source)
    assert converted["id"] == "message-1"
    assert converted["content"] == ""
    assert json.loads(converted["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert source == original


def test_baseten_rendering_constructs_native_loops_data(monkeypatch: pytest.MonkeyPatch):
    from baseten.loops import Datum, ModelInput, TensorData
    import training.utils

    monkeypatch.setattr(
        training.utils,
        "render_messages_to_datums",
        lambda *args, **kwargs: SimpleNamespace(
            token_ids=[10, 20, 30, 40, 50], token_weights=[0, 0, 1, 1, 0]
        ),
    )
    model = baseten.DEFAULT_MODEL
    row = {"messages": [{"role": "user", "content": "Hello"}], "tools": []}

    datum = baseten.render_row(row, model, renderer=object())[0]

    assert isinstance(datum, Datum)
    assert isinstance(datum.model_input, ModelInput)
    assert isinstance(datum.loss_fn_inputs["target_tokens"], TensorData)
    assert isinstance(datum.loss_fn_inputs["weights"], TensorData)


def test_text_blocks_are_not_flattened():
    content = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    converted = dataset_ops.convert_message(message("ai", content, "message-1"))
    assert converted["content"] == content
    assert converted["content"] is not content


def test_message_id_is_optional_and_preserved_when_present():
    without_id = dataset_ops.convert_message({"role": "system", "content": "policy"})
    with_id = dataset_ops.convert_message(message("human", "question", "human-1"))

    assert "id" not in without_id
    assert with_id["id"] == "human-1"


def test_exact_duplicate_is_warning_and_is_preserved():
    duplicate = message("ai", "same", "ai-duplicate")
    messages = [message("human", "go", "human-1"), duplicate, message("human", "continue", "human-2"), dict(duplicate)]
    examples = [example(0, messages)]
    audit = dataset_ops.validate_trajectories(examples, 1)
    rows = dataset_ops.prepare_sft_rows(examples)
    assert len(rows[0]["messages"]) == 4
    assert audit.duplicate_message_warnings == [
        {
            "code": "duplicate_message",
            "example_id": "example-0",
            "first_position": 1,
            "repeat_position": 3,
            "message_id": "ai-duplicate",
        }
    ]


def test_validation_accepts_system_prefix_and_any_valid_trailing_role():
    messages = [
        message("system", "policy", "system-1"),
        message("human", "question", "human-1"),
        message("ai", "answer", "ai-1"),
        message("human", "follow-up", "human-2"),
    ]

    audit = dataset_ops.validate_trajectories([example(0, messages)], 1)

    assert audit.roles == {"ai": 1, "human": 2, "system": 1}


def test_validation_requires_an_assistant_training_target():
    messages = [
        message("system", "policy", "system-1"),
        message("human", "question", "human-1"),
    ]

    with pytest.raises(PipelineError, match="no assistant training target"):
        dataset_ops.validate_trajectories([example(0, messages)], 1)


def test_split_groups_source_threads_without_leakage():
    examples = [example(i, thread=f"thread-{i // 2}") for i in range(100)]
    dataset_ops.validate_trajectories(examples, 100)
    rows = dataset_ops.prepare_sft_rows(examples)
    first = dataset_ops.split_rows(rows)
    second = dataset_ops.split_rows(list(reversed(rows)))
    train_threads = {row["_source"]["source_thread_id"] for row in first[0]}
    validation_threads = {row["_source"]["source_thread_id"] for row in first[1]}
    test_threads = {row["_source"]["source_thread_id"] for row in first[2]}
    assert len(first[0]) == 80
    assert len(first[1]) == 10
    assert len(first[2]) == 10
    assert validation_threads == {row["_source"]["source_thread_id"] for row in second[1]}
    assert test_threads == {row["_source"]["source_thread_id"] for row in second[2]}
    assert train_threads.isdisjoint(validation_threads | test_threads)
    assert validation_threads.isdisjoint(test_threads)


def test_source_identity_supports_native_legacy_and_standalone_trace():
    native = example(1)
    native["source_thread_id"] = "native-thread"
    native["metadata"].pop("source_thread_id")
    legacy = example(2, thread="legacy-thread")
    trace = example(3)
    trace["metadata"].pop("source_thread_id")
    trace["metadata"]["source_trace_id"] = "trace-3"

    dataset_ops.validate_trajectories([native, legacy, trace], 3)
    rows = dataset_ops.prepare_sft_rows([native, legacy, trace])

    assert rows[0]["_source"]["source_thread_id"] == "native-thread"
    assert rows[0]["_source"]["source_trace_id"] is None
    assert rows[1]["_source"]["source_thread_id"] == "legacy-thread"
    assert rows[2]["_source"]["source_thread_id"] is None
    assert rows[2]["_source"]["source_trace_id"] == "trace-3"


def test_source_identity_rejects_conflicting_thread_ids():
    conflicting = example(1, thread="legacy-thread")
    conflicting["source_thread_id"] = "native-thread"

    with pytest.raises(PipelineError, match="conflicting source_thread_id"):
        dataset_ops.validate_trajectories([conflicting], 1)


def test_split_uses_threads_before_traces_and_preserves_legacy_ranking():
    examples = [example(index, thread=f"thread-{index}") for index in range(10)]
    for index, item in enumerate(examples):
        item["metadata"]["source_trace_id"] = f"trace-{index}"
    rows = dataset_ops.prepare_sft_rows(examples)
    first = dataset_ops.split_rows(rows)
    second = dataset_ops.split_rows(list(reversed(rows)))
    legacy_ranked = sorted(
        {f"thread-{index}" for index in range(10)},
        key=lambda value: hashlib.sha256(f"{dataset_ops.SPLIT_SEED}:{value}".encode()).hexdigest(),
    )

    assert {row["_source"]["source_thread_id"] for row in first[1]} == {legacy_ranked[0]}
    assert {row["_source"]["source_thread_id"] for row in first[2]} == {legacy_ranked[1]}
    assert [[row["_source"]["example_id"] for row in part] for part in first] != [
        [row["_source"]["example_id"] for row in part] for part in second
    ]
    assert [{dataset_ops._source_group(row) for row in part} for part in first] == [
        {dataset_ops._source_group(row) for row in part} for part in second
    ]


def test_standalone_traces_are_independent_split_groups():
    examples = [example(index) for index in range(10)]
    for index, item in enumerate(examples):
        item["metadata"].pop("source_thread_id")
        item["metadata"]["source_trace_id"] = f"trace-{index}"
    rows = dataset_ops.prepare_sft_rows(examples)
    splits = dataset_ops.split_rows(rows)

    assert [len(part) for part in splits] == [8, 1, 1]
    assert all(row["_source"]["source_thread_id"] is None for part in splits for row in part)


def test_split_supports_test_only_and_validation_only_holdouts():
    rows = dataset_ops.prepare_sft_rows([example(index) for index in range(10)])

    test_only = dataset_ops.split_rows(rows, validation_fraction=0.0, test_fraction=1.0)
    validation_only = dataset_ops.split_rows(rows, validation_fraction=0.1, test_fraction=0.0)

    assert [len(partition) for partition in test_only] == [0, 0, 10]
    assert [len(partition) for partition in validation_only] == [9, 1, 0]


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction"),
    [(-0.1, 0.1), (0.1, -0.1), (0.6, 0.5)],
)
def test_split_rejects_invalid_fraction_ranges(validation_fraction: float, test_fraction: float):
    rows = dataset_ops.prepare_sft_rows([example(index) for index in range(10)])

    with pytest.raises(PipelineError, match="fractions"):
        dataset_ops.split_rows(rows, validation_fraction, test_fraction)


def test_split_isolation_rejects_duplicate_content_across_threads():
    train_row = dataset_ops.prepare_sft_rows([example(1, thread="train-thread")])[0]
    test_row = copy.deepcopy(train_row)
    test_row["_source"]["source_thread_id"] = "test-thread"

    with pytest.raises(PipelineError, match="content hash overlap"):
        dataset_ops._validate_split_isolation([train_row], [], [test_row])


def test_prepare_keeps_all_rows_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    examples = [example(i) for i in range(10)]
    write_raw(tmp_path, examples)
    monkeypatch.setattr(
        dataset_ops,
        "validate_model_context",
        lambda rows, model: (
            rows,
            [],
            {"rendered_datums": 10, "context_tokens": 100, "target_tokens": 20, "max_context_tokens": 15, "rejected_examples": 0},
        ),
    )

    manifest = dataset_ops.prepare_dataset("workspace-id", "dataset-id", fireworks.DEFAULT_MODEL, tmp_path, fetch=False)

    train = (tmp_path / "prepared" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    validation = (tmp_path / "prepared" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    test = (tmp_path / "prepared" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == 8
    assert len(validation) == 1
    assert len(test) == 1
    assert manifest["model"]["name"] == "qwen3p8-27b"
    assert manifest["conversion"]["messages_filtered"] is False
    assert manifest["prepared"] == {"accepted": 10, "rejected": 0}
    assert all("_source" in json.loads(line) for line in train + validation + test)


def test_prepare_requires_contract_for_tool_trajectories_and_writes_tools(tmp_path: Path):
    messages = [
        message("system", "policy", "system-1"),
        message("human", "find x", "human-1"),
        message(
            "ai",
            [{"type": "tool_call", "id": "call-1", "name": "lookup", "args": {"query": "x"}}],
            "ai-1",
        ),
        message("tool", "result", "tool-1", tool_call_id="call-1"),
        message("ai", "answer", "ai-2"),
    ]
    examples = []
    for index in range(10):
        trajectory = copy.deepcopy(messages)
        trajectory[1]["content"] = f"find x {index}"
        trajectory[2]["content"][0]["args"]["query"] = f"x-{index}"
        examples.append(example(index, trajectory, thread=f"thread-{index}"))
    write_raw(tmp_path, examples)

    with pytest.raises(PipelineError, match="inference contract"):
        dataset_ops.prepare_dataset(
            "workspace-id",
            "dataset-id",
            fireworks.DEFAULT_MODEL,
            tmp_path,
            fetch=False,
            check_render=False,
        )

    contract = loaded_contract(tmp_path)
    manifest = dataset_ops.prepare_dataset(
        "workspace-id",
        "dataset-id",
        fireworks.DEFAULT_MODEL,
        tmp_path,
        inference_contract=contract,
        fetch=False,
        check_render=False,
    )
    rows = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (tmp_path / "prepared" / f"{split}.jsonl").read_text().splitlines()
    ]

    assert all(row["tools"] == list(contract.tools) for row in rows)
    assert all(row["_source"]["contract_sha256"] == contract.contract_sha256 for row in rows)
    assert manifest["inference_contract"]["contract_sha256"] == contract.contract_sha256
    assert manifest["conversion"]["tool_schemas_added"] is True
    assert json.loads((tmp_path / "prepared" / "inference_contract.json").read_text())[
        "contract_sha256"
    ] == contract.contract_sha256


def test_prepare_reports_rejected_rows_without_changing_source_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    examples = [example(i) for i in range(10)]
    write_raw(tmp_path, examples)
    monkeypatch.setattr(
        dataset_ops,
        "validate_model_context",
        lambda rows, model: (
            rows[:-1],
            [{"example_id": rows[-1]["_source"]["example_id"]}],
            {"rejected_examples": 1},
        ),
    )

    manifest = dataset_ops.prepare_dataset(
        "workspace-id",
        "dataset-id",
        fireworks.DEFAULT_MODEL,
        tmp_path,
        fetch=False,
    )

    assert manifest["langsmith"]["examples"] == 10
    assert manifest["prepared"] == {"accepted": 9, "rejected": 1}
    assert sum(manifest["split"][name] for name in ("train", "validation", "test")) == 9
    assert json.loads((tmp_path / "prepared" / "rejected.json").read_text()) == [
        {"example_id": "example-9"}
    ]


def test_prepare_accepts_a_test_only_dataset(tmp_path: Path):
    examples = [example(index) for index in range(3)]
    write_raw(tmp_path, examples, "test-dataset")
    manifest = dataset_ops.prepare_dataset(
        "workspace-id",
        "test-dataset",
        fireworks.DEFAULT_MODEL,
        tmp_path,
        validation_fraction=0.0,
        test_fraction=1.0,
        fetch=False,
        check_render=False,
    )

    assert manifest["split"] == {
        "method": "sha256(seed:source_thread_id)",
        "seed": dataset_ops.SPLIT_SEED,
        "validation_fraction": 0.0,
        "test_fraction": 1.0,
        "train": 0,
        "validation": 0,
        "test": 3,
    }
    assert [
        len((tmp_path / "prepared" / f"{partition}.jsonl").read_text().splitlines())
        for partition in ("train", "validation", "test")
    ] == [0, 0, 3]
    test_rows = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "test.jsonl").read_text().splitlines()
    ]
    assert len(replay.build_replay_cases(test_rows)) == 3
    assert manifest["langsmith"]["dataset_id"] == "test-dataset"


def test_prepare_cli_accepts_one_dataset_and_fraction_controls():
    parser = pipeline._parser()
    args = parser.parse_args(
        [
            "prepare",
            "--workspace-id",
            "workspace-id",
            "--dataset-id",
            "source-dataset",
            "--validation-fraction",
            "0",
            "--test-fraction",
            "1",
            "--inference-contract",
            "contract.json",
        ]
    )

    assert args.dataset_id == "source-dataset"
    assert args.validation_fraction == 0.0
    assert args.test_fraction == 1.0

    capture = parser.parse_args(
        [
            "capture-contract",
            "--workspace-id",
            "workspace-id",
            "--run-id",
            "run-id",
            "--output",
            "contract.json",
        ]
    )
    assert capture.output == Path("contract.json")


def test_model_context_rejects_complete_long_example(monkeypatch: pytest.MonkeyPatch):
    import training.renderer
    import training.utils
    import training.utils.tokenizers

    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training.renderer, "get_renderer", lambda *args: object())
    monkeypatch.setattr(training.utils, "parse_train_on_what", lambda value: value)
    monkeypatch.setattr(
        training.utils,
        "render_messages_to_datums",
        lambda *args, **kwargs: SimpleNamespace(token_ids=[1, 2, 3], token_weights=[0, 1, 1]),
    )
    model = ModelSpec("tiny", "accounts/fireworks/models/tiny", "tokenizer", "revision", "renderer", 2)
    row = {
        "messages": [],
        "_source": {
            "example_id": "one",
            "source_thread_id": "thread-one",
            "source_trace_id": "trace-one",
        },
    }
    accepted, rejected, audit = rendering.validate_model_context([row], model)
    assert accepted == []
    assert rejected == [
        {
            "example_id": "one",
            "source_thread_id": "thread-one",
            "source_trace_id": "trace-one",
            "rendered_tokens": 3,
            "context_limit": 2,
            "reason": "rendered example exceeds model context limit",
        }
    ]
    assert audit["rejected_examples"] == 1


def test_model_context_passes_tools_to_renderer_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    import training.renderer
    import training.utils
    import training.utils.supervised
    import training.utils.tokenizers

    rendered_tools = []
    renderer = object()
    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training.renderer, "get_renderer", lambda *args: renderer)
    monkeypatch.setattr(training.utils, "parse_train_on_what", lambda value: value)
    monkeypatch.setattr(training.utils.supervised, "renderer_declares_tools", lambda value: True)

    def render(*args, **kwargs):
        rendered_tools.append(kwargs["tools"])
        return SimpleNamespace(token_ids=[1, 2], token_weights=[0, 1])

    monkeypatch.setattr(training.utils, "render_messages_to_datums", render)
    model = ModelSpec(
        "tiny",
        "accounts/fireworks/models/tiny",
        "tokenizer",
        "revision",
        "renderer",
        10,
        requires_tool_declarations=True,
    )
    row = {
        "messages": [{"role": "user", "content": "find x"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        "_source": {"example_id": "one", "source_thread_id": "thread-one"},
    }

    accepted, rejected, _ = rendering.validate_model_context([row], model)

    assert accepted == [row]
    assert rejected == []
    assert rendered_tools == [row["tools"]]

    monkeypatch.setattr(training.utils.supervised, "renderer_declares_tools", lambda value: False)
    with pytest.raises(PipelineError, match="cannot declare tools"):
        rendering.validate_model_context([row], model)


def tool_call(name: str, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "id": f"message-{call_id}",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{\"query\":\"x\"}"},
            }
        ],
    }


def test_replay_cases_slice_before_even_tool_boundaries():
    messages = [{"role": "user", "content": "task", "id": "user-1"}]
    for index in range(5):
        call_id = f"call-{index}"
        messages.extend(
            [
                tool_call(f"tool-{index}", call_id),
                {"role": "tool", "content": f"result-{index}", "tool_call_id": call_id, "id": f"tool-{index}"},
            ]
        )
    rows = [
        {
            "messages": messages,
            "_source": {"example_id": "example-1", "source_thread_id": "thread-1"},
        }
    ]

    cases = replay.build_replay_cases(rows, max_points_per_trajectory=3)

    assert [case["message_index"] for case in cases] == [1, 5, 9]
    assert [case["reference"]["tool_calls"][0]["function"]["name"] for case in cases] == [
        "tool-0", "tool-2", "tool-4"
    ]
    assert cases[1]["messages"] == messages[:5]
    assert cases[1]["tool_results"][0]["content"] == "result-2"


def test_replay_cases_include_every_assistant_message_by_default():
    messages = [
        {"role": "user", "content": "task", "id": "user-1"},
        {"role": "assistant", "content": "I will look that up.", "id": "assistant-1"},
        tool_call("lookup"),
        {"role": "tool", "content": "result", "tool_call_id": "call-1", "id": "tool-1"},
        {"role": "assistant", "content": "The answer is result.", "id": "assistant-2"},
    ]
    rows = [
        {
            "messages": messages,
            "_source": {"example_id": "example-1", "source_thread_id": "thread-1"},
        }
    ]

    cases = replay.build_replay_cases(rows)

    assert [case["message_index"] for case in cases] == [1, 2, 4]
    assert [case["case_type"] for case in cases] == ["text", "tool_call", "text"]


def test_replay_evaluation_requires_test_rows(tmp_path: Path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"]["test"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "prepared" / "test.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(PipelineError, match="no test rows"):
        replay.prepare_replay_evaluation(tmp_path, tmp_path / "evaluation")


def test_replay_cases_preserve_tools_and_contract_hash():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    rows = [
        {
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "find x"},
                tool_call("lookup"),
            ],
            "tools": tools,
            "_source": {
                "example_id": "example-1",
                "source_thread_id": "thread-1",
                "contract_sha256": "contract-hash",
            },
        }
    ]

    cases = replay.build_replay_cases(rows)

    assert cases[0]["tools"] == tools
    assert cases[0]["contract_sha256"] == "contract-hash"


def test_replay_cases_include_trace_provenance():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "_source": {
                "example_id": "example-1",
                "source_thread_id": None,
                "source_trace_id": "trace-1",
            },
        }
    ]

    cases = replay.build_replay_cases(rows)

    assert cases[0]["source_thread_id"] is None
    assert cases[0]["source_trace_id"] == "trace-1"


def test_replay_context_rejects_without_truncation(monkeypatch: pytest.MonkeyPatch):
    import training.renderer
    import training.utils.tokenizers

    class Prompt:
        def to_ints(self):
            return [1, 2, 3]

    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training.renderer, "get_renderer", lambda *args: SimpleNamespace(build_generation_prompt=lambda messages: Prompt()))
    model = ModelSpec("tiny", "accounts/fireworks/models/tiny", "tok", "rev", "renderer", 4)
    case = {
        "id": "case",
        "example_id": "example",
        "source_thread_id": None,
        "source_trace_id": "trace-1",
        "messages": [],
    }

    accepted, rejected = rendering.validate_replay_context([case], model, max_output_tokens=2)

    assert accepted == []
    assert rejected[0]["prompt_tokens"] == 3
    assert rejected[0]["context_limit"] == 4
    assert rejected[0]["source_thread_id"] is None
    assert rejected[0]["source_trace_id"] == "trace-1"


def test_replay_context_counts_tool_declarations(monkeypatch: pytest.MonkeyPatch):
    import training.renderer
    import training.utils.supervised
    import training.utils.tokenizers

    class Prompt:
        def to_ints(self):
            return [1, 2, 3]

    captured = []
    renderer = SimpleNamespace(build_generation_prompt=lambda messages: Prompt())
    monkeypatch.setattr(training.utils.tokenizers, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(training.renderer, "get_renderer", lambda *args: renderer)
    monkeypatch.setattr(training.utils.supervised, "renderer_declares_tools", lambda value: True)

    def prefix(messages, *, renderer, tools):
        captured.append((messages, tools))
        return [{"role": "system", "content": "rendered tool prefix"}, *messages]

    monkeypatch.setattr(training.utils.supervised, "build_tool_prefixed_messages", prefix)
    model = ModelSpec(
        "tiny",
        "accounts/fireworks/models/tiny",
        "tok",
        "rev",
        "renderer",
        10,
        requires_tool_declarations=True,
    )
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    case = {
        "id": "case",
        "example_id": "example",
        "messages": [{"role": "user", "content": "find x"}],
        "tools": tools,
    }

    accepted, rejected = rendering.validate_replay_context([case], model, max_output_tokens=2)

    assert rejected == []
    assert accepted[0]["prompt_tokens"] == 3
    assert captured == [(case["messages"], tools)]

    monkeypatch.setattr(training.utils.supervised, "renderer_declares_tools", lambda value: False)
    with pytest.raises(PipelineError, match="cannot declare tools"):
        rendering.validate_replay_context([case], model, max_output_tokens=2)


def test_replay_evaluation_calibrates_and_compares_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    contract = loaded_contract(tmp_path)
    write_manifest(data_dir, contract=contract)
    row = {
        "messages": [
            {"role": "system", "content": "policy", "id": "system-1"},
            {"role": "user", "content": "look up x", "id": "user-1"},
            tool_call("lookup"),
            {"role": "tool", "content": "x is 1", "tool_call_id": "call-1", "id": "tool-1"},
        ],
        "tools": list(contract.tools),
        "_source": {
            "example_id": "example-1",
            "source_thread_id": "thread-1",
            "contract_sha256": contract.contract_sha256,
        },
    }
    (data_dir / "prepared" / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        replay,
        "validate_replay_context",
        lambda cases, model, max_output_tokens: ([{**case, "prompt_tokens": 10} for case in cases], []),
    )

    candidate_contracts = []

    def fake_chat(model, messages, max_tokens, json_mode, request_contract=None):
        if json_mode:
            assert request_contract is None
            evidence = json.loads(messages[1]["content"])
            candidate = evidence["candidate_next_action"]["tool_calls"][0]["function"]
            reference = evidence["reference_next_action"]["tool_calls"][0]["function"]
            passed = candidate["name"] == reference["name"] and candidate["arguments"] == reference["arguments"]
            return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "tool check"})}
        candidate_contracts.append(request_contract)
        return tool_call("lookup" if model == "tuned-model" else "wrong")

    summary = replay.run_replay_evaluation(
        data_dir,
        tmp_path / "evaluation",
        "tuned-model",
        "judge-model",
        base_model="base-deployment",
        confirm=True,
        chat=fake_chat,
    )

    assert summary["calibration_passed"] is True
    assert summary["calibration_checks"] == 3
    assert summary["base_model"] == "base-deployment"
    assert summary["base_pass_rate"] == 0
    assert summary["tuned_pass_rate"] == 1
    assert summary["paired_wins"] == 1
    assert summary["contract_sha256"] == contract.contract_sha256
    assert summary["deterministic_metrics"]["base"]["tool_name_match"]["rate"] == 0
    assert summary["deterministic_metrics"]["tuned"]["tool_name_match"]["rate"] == 1
    assert summary["deterministic_metrics"]["tuned_minus_base"][
        "tool_name_match_rate"
    ] == 1
    assert candidate_contracts == [contract, contract]
    assert (tmp_path / "evaluation" / "results.jsonl").is_file()


def test_replay_evaluation_reports_text_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    write_manifest(data_dir)
    row = {
        "messages": [
            {"role": "user", "content": "What is x?", "id": "user-1"},
            {"role": "assistant", "content": "x is 1", "id": "assistant-1"},
        ],
        "_source": {"example_id": "example-1", "source_thread_id": "thread-1"},
    }
    (data_dir / "prepared" / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        replay,
        "validate_replay_context",
        lambda cases, model, max_output_tokens: ([{**case, "prompt_tokens": 10} for case in cases], []),
    )

    def fake_chat(model, messages, max_tokens, json_mode, request_contract=None):
        if json_mode:
            evidence = json.loads(messages[1]["content"])
            passed = evidence["candidate_next_action"]["content"] == evidence["reference_next_action"]["content"]
            return {"role": "assistant", "content": json.dumps({"pass": passed, "reason": "text check"})}
        return {"role": "assistant", "content": "x is 1" if model == "tuned-model" else "x is 2"}

    summary = replay.run_replay_evaluation(
        data_dir,
        tmp_path / "evaluation",
        "tuned-model",
        "judge-model",
        confirm=True,
        chat=fake_chat,
    )

    assert summary["case_types"] == {"text": 1}
    assert summary["calibration_checks"] == 3
    assert summary["by_case_type"]["text"]["tuned_pass_rate"] == 1
    assert "base_pass_rate" not in summary["by_case_type"]["text"]


def test_deterministic_metrics_score_tool_decisions_arguments_and_schema(tmp_path: Path):
    contract = loaded_contract(tmp_path)
    reference = tool_call("lookup")
    case = {
        "case_type": "tool_call",
        "reference": reference,
    }

    exact = replay.score_replay_candidate(case, copy.deepcopy(reference), contract)
    wrong_name = replay.score_replay_candidate(case, tool_call("missing"), contract)
    malformed = tool_call("lookup")
    malformed["tool_calls"][0]["function"]["arguments"] = "not-json"
    malformed_score = replay.score_replay_candidate(case, malformed, contract)
    wrong_type = tool_call("lookup")
    wrong_type["tool_calls"][0]["function"]["arguments"] = '{"query":1}'
    wrong_type_score = replay.score_replay_candidate(case, wrong_type, contract)
    text_score = replay.score_replay_candidate(
        case,
        {"role": "assistant", "content": "I cannot look that up."},
        contract,
    )

    assert exact == {
        "tool_decision_match": True,
        "tool_name_match": True,
        "arguments_json_valid": True,
        "arguments_schema_valid": True,
        "reference_arguments_match": True,
        "parallel_call_set_match": None,
    }
    assert wrong_name["tool_decision_match"] is True
    assert wrong_name["tool_name_match"] is False
    assert wrong_name["arguments_schema_valid"] is False
    assert wrong_name["reference_arguments_match"] is False
    assert malformed_score["arguments_json_valid"] is False
    assert malformed_score["arguments_schema_valid"] is False
    assert wrong_type_score["arguments_json_valid"] is True
    assert wrong_type_score["arguments_schema_valid"] is False
    assert text_score["tool_decision_match"] is False
    assert text_score["tool_name_match"] is False


def test_deterministic_metrics_match_parallel_call_sets_without_order(tmp_path: Path):
    contract = loaded_contract(tmp_path)
    second_tool = {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch an item.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    }
    payload = contract.to_dict()
    payload["tools"].append(second_tool)
    payload["tools_sha256"] = inference_contract.json_sha256(payload["tools"])
    payload.pop("contract_sha256")
    path = tmp_path / "parallel-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    contract = inference_contract.load_inference_contract(path)
    lookup = tool_call("lookup", "call-1")
    fetch = tool_call("fetch", "call-2")
    fetch["tool_calls"][0]["function"]["arguments"] = '{"id":2}'
    reference = copy.deepcopy(lookup)
    reference["tool_calls"].extend(fetch["tool_calls"])
    reversed_candidate = copy.deepcopy(reference)
    reversed_candidate["tool_calls"].reverse()
    changed_candidate = copy.deepcopy(reversed_candidate)
    changed_candidate["tool_calls"][0]["function"]["arguments"] = '{"id":3}'
    case = {"case_type": "tool_call", "reference": reference}

    exact = replay.score_replay_candidate(case, reversed_candidate, contract)
    changed = replay.score_replay_candidate(case, changed_candidate, contract)

    assert exact["tool_name_match"] is True
    assert exact["reference_arguments_match"] is True
    assert exact["parallel_call_set_match"] is True
    assert changed["tool_name_match"] is True
    assert changed["arguments_schema_valid"] is True
    assert changed["reference_arguments_match"] is False
    assert changed["parallel_call_set_match"] is False


def test_deterministic_metric_summary_reports_base_tuned_rates_and_deltas():
    results = [
        {
            "case": {"case_type": "tool_call"},
            "base": {
                "deterministic_metrics": {
                    "tool_decision_match": False,
                    "tool_name_match": False,
                    "arguments_json_valid": False,
                    "arguments_schema_valid": False,
                    "reference_arguments_match": False,
                    "parallel_call_set_match": None,
                }
            },
            "tuned": {
                "deterministic_metrics": {
                    "tool_decision_match": True,
                    "tool_name_match": True,
                    "arguments_json_valid": True,
                    "arguments_schema_valid": True,
                    "reference_arguments_match": True,
                    "parallel_call_set_match": None,
                }
            },
        },
        {
            "case": {"case_type": "text"},
            "base": {
                "deterministic_metrics": {
                    "tool_decision_match": True,
                    "tool_name_match": None,
                    "arguments_json_valid": None,
                    "arguments_schema_valid": None,
                    "reference_arguments_match": None,
                    "parallel_call_set_match": None,
                }
            },
            "tuned": {
                "deterministic_metrics": {
                    "tool_decision_match": True,
                    "tool_name_match": None,
                    "arguments_json_valid": None,
                    "arguments_schema_valid": None,
                    "reference_arguments_match": None,
                    "parallel_call_set_match": None,
                }
            },
        },
    ]

    summary = replay.summarize_deterministic_metrics(results, ["base", "tuned"])

    assert summary["base"]["tool_decision_match"] == {
        "matches": 1,
        "cases": 2,
        "rate": 0.5,
    }
    assert summary["tuned"]["arguments_schema_valid"]["rate"] == 1
    assert summary["tuned_minus_base"]["tool_decision_match_rate"] == 0.5
    assert summary["tuned_minus_base"]["arguments_schema_valid_rate"] == 1
    assert summary["base"]["parallel_call_set_match"]["rate"] is None


def test_fireworks_inference_uses_the_shared_contract_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}
    contract = loaded_contract(tmp_path)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return Response(
            json.dumps({"choices": [{"message": {"role": "assistant", "content": "done"}}]}).encode()
        )

    monkeypatch.setattr(inference_transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "session-id")
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "find x"},
    ]

    inference_transport._chat_completion(
        "accounts/fireworks/models/model",
        messages,
        256,
        False,
        contract,
    )

    assert captured["body"]["messages"] == messages
    assert captured["body"]["tools"] == list(contract.tools)
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 256


def test_anthropic_judge_uses_langsmith_gateway(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return Response(json.dumps({"content": [{"type": "text", "text": '{"pass":true,"reason":"match"}'}]}).encode())

    monkeypatch.setattr(inference_transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value")

    result = inference_transport._chat_completion(
        "anthropic/claude-sonnet-5-model-id",
        [{"role": "system", "content": "judge"}, {"role": "user", "content": "case"}],
        256,
        True,
    )

    assert captured["url"] == "https://gateway.smith.langchain.com/anthropic/v1/messages"
    assert captured["body"]["model"] == "claude-sonnet-5-model-id"
    assert captured["body"]["system"] == "judge"
    assert "output_config" not in captured["body"]
    assert result["content"].startswith("{")


def test_anthropic_judge_prefers_gateway_custom_headers(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["key"] = request.get_header("X-api-key")
        response = {"content": [{"type": "text", "text": '{"pass":true,"reason":"match"}'}]}
        return Response(json.dumps(response).encode())

    monkeypatch.setattr(inference_transport.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", '{"X-Api-Key":"gateway-key"}')

    inference_transport._chat_completion(
        "anthropic/claude-sonnet-5",
        [{"role": "user", "content": "case"}],
        256,
        True,
    )

    assert captured["key"] == "gateway-key"


def test_claude_sonnet_5_is_the_default_judge():
    args = pipeline._parser().parse_args(
        ["evaluate", "--output-dir", "evaluation", "--tuned-model", "tuned"]
    )

    assert args.judge_model == "anthropic/claude-sonnet-5"


def test_judge_marks_reference_tool_results_as_future_evidence():
    captured = {}

    def chat(model, messages, max_tokens, json_mode, request_contract=None):
        captured["messages"] = messages
        return {"role": "assistant", "content": '{"pass":true,"reason":"match"}'}

    reference = tool_call("lookup")
    case = {
        "messages": [{"role": "user", "content": "question"}],
        "reference": reference,
        "tool_results": [
            {
                "role": "tool",
                "tool_call_id": reference["tool_calls"][0]["id"],
                "content": "reference result",
            }
        ],
    }

    replay.judge_replay_candidate(case, reference, "judge", chat)

    instructions = captured["messages"][0]["content"]
    evidence = json.loads(captured["messages"][1]["content"])
    assert "not visible to the candidate" in instructions
    assert evidence["trajectory_prefix_visible_to_candidate"] == case["messages"]
    assert evidence["reference_action_tool_results_not_visible_to_candidate"] == case["tool_results"]
    assert "trajectory_prefix" not in evidence
    assert "recorded_tool_results" not in evidence


def test_judge_retries_an_invalid_response():
    responses = iter(
        [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": '{"pass":true,"reason":"match"}'},
        ]
    )
    calls = []

    def chat(model, messages, max_tokens, json_mode, request_contract=None):
        calls.append((model, max_tokens, json_mode))
        return next(responses)

    case = {
        "messages": [{"role": "user", "content": "question"}],
        "reference": tool_call("lookup"),
        "tool_results": [],
    }

    result = replay.judge_replay_candidate(case, case["reference"], "judge", chat)

    assert result["pass"] is True
    assert len(calls) == 2


def test_builtin_and_custom_model_resolution():
    parser = pipeline._parser()
    source = ["--workspace-id", "workspace-id", "--dataset-id", "dataset-id"]
    built_in = parser.parse_args(["prepare", *source, "--model-profile", "kimi-k3"])
    custom = parser.parse_args(
        [
            "prepare", *source, "--model-profile", "custom",
            "--base-model", "accounts/fireworks/models/model-x",
            "--tokenizer-model", "org/model-x", "--tokenizer-revision", "abc123",
            "--renderer", "model_x", "--max-seq-len", "4096",
            "--requires-tool-declarations",
        ]
    )
    assert get_provider(built_in.provider).model_from_options(pipeline._model_options(built_in)).max_seq_len == 196_608
    assert get_provider(custom.provider).model_from_options(pipeline._model_options(custom)).base_model.endswith("model-x")
    assert get_provider(custom.provider).model_from_options(pipeline._model_options(custom)).requires_tool_declarations is True


def test_tool_declaration_flag_is_only_valid_for_custom_profiles():
    args = pipeline._parser().parse_args(
        [
            "prepare",
            "--workspace-id",
            "workspace-id",
            "--dataset-id",
            "dataset-id",
            "--model-profile",
            "qwen3p8-27b",
            "--requires-tool-declarations",
        ]
    )

    with pytest.raises(PipelineError, match="custom model fields"):
        get_provider(args.provider).model_from_options(pipeline._model_options(args))


def test_prepare_rows_rejects_recorded_calls_outside_the_contract(tmp_path: Path):
    contract = loaded_contract(tmp_path)
    messages = [
        message("system", "policy", "system-1"),
        message("human", "look up x", "human-1"),
        message(
            "ai",
            [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"query": 1},
                }
            ],
            "ai-1",
        ),
        message("tool", "not found", "tool-1", tool_call_id="call-1"),
    ]

    with pytest.raises(PipelineError, match="do not match its JSON Schema"):
        dataset_ops.prepare_sft_rows([example(1, messages)], contract)


def test_plan_uses_manifest_model_and_sft_defaults(tmp_path: Path):
    write_manifest(tmp_path, fireworks.MODEL_SPECS["kimi-k3"])
    plan = fireworks.FireworksProvider().plan(
        tmp_path,
        "run-id",
        fireworks.SFTSettings(),
    )
    assert plan["training_api"] == "https://api.fireworks.ai/training/v1/serverless"
    assert plan["base_model"] == "accounts/fireworks/models/kimi-k3"
    assert plan["config"]["renderer"] == "kimi_k3"
    assert plan["config"]["max_epochs"] == 5
    assert plan["config"]["early_stopping_patience"] == 1
    assert plan["config"]["loss_target"] == "all assistant text and tool calls"
    assert "train_on_what" not in plan["config"]
    assert plan["dataset"]["validation_rows"] == 1
    assert plan["dataset"]["test_rows"] == 1


def test_plan_accepts_legacy_fireworks_manifest_without_provider_identity(tmp_path: Path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["provider"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = fireworks.FireworksProvider().plan(
        tmp_path,
        "run-id",
        fireworks.SFTSettings(),
    )

    assert plan["base_model"] == fireworks.DEFAULT_MODEL.base_model


def test_plan_rejects_explicitly_null_provider_identity(tmp_path: Path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PipelineError, match="prepared provider mismatch"):
        fireworks.FireworksProvider().plan(
            tmp_path,
            "run-id",
            fireworks.SFTSettings(),
        )


def test_training_plan_requires_train_and_validation_rows(tmp_path: Path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "prepared" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for missing_partition in ("train", "validation"):
        manifest["split"][missing_partition] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(PipelineError, match=f"no {missing_partition} rows"):
            fireworks.FireworksProvider().plan(
                tmp_path,
                "run-id",
                fireworks.SFTSettings(),
            )
        manifest["split"][missing_partition] = 1


def test_qwen_renderer_resolves_for_preserved_trajectory_history():
    from training.recipes import sft_loop

    model = fireworks.DEFAULT_MODEL
    resolved = sft_loop._resolved_renderer_name(
        tokenizer_model=model.tokenizer_model,
        renderer_name=model.renderer,
        thinking_trace_history_mode=model.thinking_trace_history_mode,
        renderer_name_is_resolved=False,
    )
    assert resolved == "qwen3_8_preserved"


def test_early_stopping_selects_best_checkpoint():
    losses = iter([1.0, 0.8, 0.9])

    def run_epoch(epoch, checkpoint):
        return {
            "job_id": f"job-{epoch}",
            "steps": epoch,
            "eval_loss": next(losses),
            "resume_checkpoint": f"run-{epoch}/step-{epoch}",
        }

    result = fireworks.run_early_stopping(
        fireworks.SFTSettings(max_epochs=5, early_stopping_patience=1),
        run_epoch,
    )
    assert len(result["epochs"]) == 3
    assert result["best"]["job_id"] == "job-2"
    assert result["stopped_early"] is True


def test_serverless_checkpoint_refs_use_training_session_api(monkeypatch: pytest.MonkeyPatch):
    calls = []
    run_id = "run-0123456789abcdef0123456789abcdef"

    class Client:
        account_id = "account-id"

        def __init__(self, **kwargs):
            pass

        def list_training_session_checkpoints(self, name):
            calls.append(name)
            return [
                {
                    "name": f"accounts/account-id/trainingSessions/session-1/checkpoints/{run_id}-step-4",
                    "checkpointType": "CHECKPOINT_TYPE_TRAINING_LORA",
                    "createTime": "2026-01-01T00:00:00Z",
                    "promotable": False,
                },
                {
                    "name": f"accounts/account-id/trainingSessions/session-1/checkpoints/{run_id}-step-4-abcd1234",
                    "checkpointType": "CHECKPOINT_TYPE_INFERENCE_LORA",
                    "createTime": "2026-01-01T00:00:01Z",
                    "promotable": True,
                },
            ]

        def close(self):
            pass

    import fireworks.training.sdk as fireworks_sdk

    monkeypatch.setattr(fireworks_sdk, "FireworksClient", Client)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "00000000-0000-0000-0000-000000000000")

    result = fireworks._epoch_checkpoints("session-1")

    assert calls == ["accounts/account-id/trainingSessions/session-1"]
    assert result["resume_checkpoint"] == f"account-id/{run_id}/step-4"
    assert result["promotable_checkpoint"].endswith(f"{run_id}-step-4-abcd1234")


def test_mocked_training_runs_recipe_by_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from training.recipes import sft_loop

    data_dir = tmp_path / "data"
    write_manifest(data_dir)
    losses = iter([1.0, 0.7, 0.8])
    configs = []

    def fake_main(config):
        configs.append(config)
        loss = next(losses)
        Path(config.runner.metrics_file).write_text(json.dumps({"step": 1, "eval/loss": loss}) + "\n", encoding="utf-8")
        return {"job_id": f"job-{len(configs)}", "steps": 1}

    monkeypatch.setattr(sft_loop, "main", fake_main)
    monkeypatch.setattr(
        fireworks,
        "_epoch_checkpoints",
        lambda job_id: {
            "resume_checkpoint": f"account/run-{job_id}/step-1",
            "promotable_checkpoint": f"accounts/a/trainingSessions/{job_id}/checkpoints/step-1",
        },
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")
    result = fireworks.FireworksProvider().train(
        data_dir,
        tmp_path / "run",
        "run-id",
        fireworks.SFTSettings(max_epochs=5, early_stopping_patience=1),
        confirm=True,
        init_from_checkpoint=None,
    )
    assert len(configs) == 3
    assert configs[1].init_from_checkpoint == "account/run-job-1/step-1"
    assert configs[0].base_model == fireworks.DEFAULT_MODEL.base_model
    assert result["best"]["job_id"] == "job-2"


def test_promotion_uses_best_checkpoint_and_planned_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "best": {
                    "job_id": "job-2",
                    "resume_checkpoint": "account/run-2/step-4",
                    "promotable_checkpoint": "accounts/a/trainingSessions/job-2/checkpoints/run-2-step-4",
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "plan.json").write_text(json.dumps({"base_model": "accounts/fireworks/models/model-x"}), encoding="utf-8")
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def promote_session_checkpoint(self, *args):
            calls.append(args)

        def close(self):
            pass

    import fireworks.training.sdk as fireworks_sdk

    monkeypatch.setattr(fireworks_sdk, "FireworksClient", Client)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")

    fireworks.FireworksProvider().promote(run, "output-model", confirm=True)

    assert calls == [
        (
            "accounts/a/trainingSessions/job-2/checkpoints/run-2-step-4",
            "output-model",
            "accounts/fireworks/models/model-x",
        )
    ]


def test_mocked_deployment_returns_official_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commands = []
    monkeypatch.setattr(fireworks, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(
        fireworks,
        "_inference_smoke_test",
        lambda route: {"http_status": 200, "finish_reason": "stop"},
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-value")

    endpoint = fireworks.FireworksProvider().deploy(
        tmp_path,
        "account-id",
        "output-model",
        "deployment-id",
        "accounts/fireworks/deploymentShapes/shape-id",
        confirm=True,
    )

    assert commands[0][:3] == ["firectl", "deployment", "create"]
    assert endpoint["inference_url"].startswith("https://api.fireworks.ai/inference/v1/")
    assert endpoint["model"] == (
        "accounts/account-id/models/output-model#"
        "accounts/account-id/deployments/deployment-id"
    )


def test_mutating_steps_require_confirmation(tmp_path: Path):
    settings = fireworks.SFTSettings()
    with pytest.raises(PipelineError, match="--confirm"):
        fireworks.FireworksProvider().train(
            tmp_path,
            tmp_path / "run",
            "run-id",
            settings,
            confirm=False,
            init_from_checkpoint=None,
        )
    with pytest.raises(PipelineError, match="--confirm"):
        fireworks.FireworksProvider().promote(tmp_path / "run", "model-id", confirm=False)
    with pytest.raises(PipelineError, match="--confirm"):
        fireworks.FireworksProvider().deploy(tmp_path, "account", "model-id", "deployment-id", "shape", confirm=False)
    with pytest.raises(PipelineError, match="--confirm"):
        replay.run_replay_evaluation(tmp_path, tmp_path / "eval", "tuned", "judge", confirm=False)


def test_source_uses_official_provider_urls_and_no_embedded_secret():
    source = "\n".join(
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (
            artifacts,
            dataset_ops,
            replay,
            inference_transport,
            rendering,
            fireworks,
        )
    )
    assert "gateway.smith.langchain.com/fireworks" not in source
    assert "https://api.fireworks.ai/training/v1/serverless" in source
    assert "https://api.fireworks.ai/inference/v1" in source
    assert "https://gateway.smith.langchain.com/anthropic/v1/messages" in source
    assert "fw_" not in source
    assert "lsv2_pt_" not in source


def test_fireworks_settings_errors_use_the_cli_error_path():
    result = subprocess.run(
        [sys.executable, pipeline.__file__, "plan", "--max-epochs", "0"],
        cwd=Path(pipeline.__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "max_epochs must be positive" in result.stderr
    assert "Traceback" not in result.stderr

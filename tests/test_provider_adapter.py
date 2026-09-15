from __future__ import annotations

import builtins
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from smithtune import capabilities, dataset
from smithtune import cli as pipeline
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from smithtune.providers import get_provider


def _write_raw_dataset(root, count=100):
    examples = [
        {
            "id": f"example-{index}",
            "inputs": {
                "messages": [
                    {"role": "human", "content": f"question {index}"},
                    {"role": "ai", "content": f"answer {index}"},
                ]
            },
            "outputs": None,
            "metadata": {
                "source_project_id": "project-1",
                "source_scope": "thread", "source_scope_id": f"thread-{index}",
                "trajectory_format": "messages",
                "conversation_scope": "root",
            },
        }
        for index in range(count)
    ]
    raw = root / "raw"
    raw.mkdir()
    (raw / "examples.json").write_text(json.dumps(examples), encoding="utf-8")
    (raw / "dataset-export.json").write_text(
        json.dumps(
            [
                {"inputs": item["inputs"], "outputs": item["outputs"]}
                for item in examples
            ]
        ),
        encoding="utf-8",
    )
    (raw / "dataset.json").write_text(
        json.dumps({"id": "dataset-id", "name": "test", "example_count": count}),
        encoding="utf-8",
    )

    from test_example_tools import write_empty_tool_snapshot
    write_empty_tool_snapshot(root, examples)


def test_provider_aware_commands_default_to_fireworks_and_accept_baseten(monkeypatch):
    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    parser = pipeline._parser()

    default_prepare = parser.parse_args(
        ["prepare", "--model", "qwen3p8-27b", "--workspace-id", "workspace-id", "--dataset-id", "dataset-id"]
    )
    default_plan = parser.parse_args(["plan"])
    default_train = parser.parse_args(["train"])
    baseten_plan = parser.parse_args(["plan", "--provider", "baseten"])

    assert default_prepare.provider == "fireworks"
    assert default_plan.provider == "fireworks"
    assert default_train.provider == "fireworks"
    assert baseten_plan.provider == "baseten"


def test_baseten_resolves_qwen_profile(monkeypatch):
    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    args = pipeline._parser().parse_args(
        [
            "prepare",
            "--model",
            "qwen3p8-27b",
            "--provider",
            "baseten",
            "--workspace-id",
            "workspace-id",
            "--dataset-id",
            "dataset-id",
        ]
    )

    model = get_provider(args.provider).model_from_options(pipeline._model_options(args))

    assert model.base_model == "Qwen/Qwen3.8-27B"
    assert model.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert model.tokenizer_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert model.renderer == "hf_assistant"
    assert model.max_seq_len == 262_144


def test_provider_specific_training_options_are_rejected(monkeypatch):
    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    parser = pipeline._parser()
    baseten = parser.parse_args(
        ["plan", "--provider", "baseten", "--lora-alpha", "32"]
    )
    fireworks = parser.parse_args(
        ["plan", "--provider", "fireworks", "--max-spend-usd", "75"]
    )

    with pytest.raises(PipelineError, match="Fireworks-only"):
        pipeline._settings_from_args(baseten)
    with pytest.raises(PipelineError, match="Baseten-only"):
        pipeline._settings_from_args(fireworks)


@pytest.mark.parametrize("provider", ["fireworks", "baseten"])
@pytest.mark.parametrize("test_fraction,train_rows,test_rows", [(None, 80, 10), (0.0, 90, 0)])
def test_cli_preparation_can_be_planned_by_standalone_provider(
    tmp_path, monkeypatch, capsys, provider, test_fraction, train_rows, test_rows
):
    _write_raw_dataset(tmp_path)
    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    capability_calls = []

    def fireworks_capability(model, length):
        capability_calls.append((model, length))
        return capabilities.FireworksModelCapability(model, "Qwen/Qwen3.8-27B", 131_072, True)

    def baseten_capability(model, length):
        capability_calls.append((model, length))
        return baseten.BasetenModelCapability(model, 262_144)

    monkeypatch.setattr(capabilities, "fetch_fireworks_model_capability", fireworks_capability)
    monkeypatch.setattr(baseten, "fetch_model_capability", baseten_capability)
    adapter_module = fireworks if provider == "fireworks" else baseten
    monkeypatch.setattr(adapter_module, "resolve_rendering_model", lambda model: model)
    monkeypatch.setattr(
        dataset,
        "validate_model_context",
        lambda rows, model: (
            rows,
            [],
            {
                "rendered_datums": 100,
                "context_tokens": 1000,
                "target_tokens": 500,
                "max_context_tokens": 10,
                "rejected_examples": 0,
            },
        ),
    )

    argv = [
        "pipeline.py", "prepare", "--provider", provider, "--model", "qwen3p8-27b",
        "--workspace-id", "workspace-id", "--dataset-id", "dataset-id",
        "--data-dir", str(tmp_path), "--no-fetch",
    ]
    if test_fraction is not None:
        argv.extend(["--test-fraction", str(test_fraction)])
    monkeypatch.setattr(sys, "argv", argv)
    pipeline.main()
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["provider"] == {
        "name": provider,
        "renderer": adapter_module.DEFAULT_MODEL.renderer,
        "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    assert capability_calls == [(adapter_module.DEFAULT_MODEL.base_model, adapter_module.DEFAULT_MODEL.training_context_limit)]
    assert manifest["split"]["train"] == train_rows
    assert manifest["split"]["validation"] == 10
    assert manifest["split"]["test"] == test_rows
    stored = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    assert stored == manifest

    # A fresh interpreter must plan the prepared data without the CLI or SDKs.
    result = subprocess.run(
        [sys.executable, "-c", """
import builtins
import json
import sys
from pathlib import Path

original_import = builtins.__import__
def without_cli_or_sdk(name, *args, **kwargs):
    if name.split('.')[0] in {'pipeline', 'baseten', 'fireworks', 'training'}:
        raise AssertionError(f'planning must not import {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_cli_or_sdk

from smithtune.providers.base import TrainingOptions
from smithtune.providers import get_provider
adapter = get_provider(sys.argv[1])
plan = adapter.plan(Path(sys.argv[2]), 'standalone', adapter.settings_from_options(TrainingOptions()))
assert 'smithtune.cli' not in sys.modules
print(json.dumps(plan))
""", provider, str(tmp_path)],
        cwd=Path(pipeline.__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["base_model"] == manifest["model"]["base_model"]
    assert plan["dataset"]["train_rows"] == train_rows


def test_training_provider_registry_returns_adapters():
    assert get_provider("fireworks").name == "fireworks"
    assert get_provider("baseten").name == "baseten"
    with pytest.raises(PipelineError, match="unknown training provider"):
        get_provider("missing")


@pytest.mark.parametrize("provider", ["fireworks", "baseten"])
@pytest.mark.parametrize("explicit_paths", [False, True])
def test_cli_training_requires_confirmation_without_traceback(tmp_path, provider, explicit_paths):
    run_dir = tmp_path / "run"
    extras = ["--run-dir", str(run_dir), "--run-id", "no-provision"] if explicit_paths else []
    result = subprocess.run(
        [
            sys.executable, "-m", "smithtune", "train", "--provider", provider,
            "--data-dir", str(tmp_path / "missing"), *extras,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--confirm" in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("provider_name", ["fireworks", "baseten"])
@pytest.mark.parametrize("run_id,run_dir", [(None, None), ("my-sft", None), (None, "output"), ("my-sft", "output")])
def test_cli_training_generates_identity_and_preserves_overrides(
    tmp_path, monkeypatch, capsys, provider_name, run_id, run_dir
):
    provider = get_provider(provider_name)
    calls = []

    def train(data_dir, output_dir, identity, settings, **kwargs):
        calls.append((data_dir, output_dir, identity, kwargs))
        return {"checkpoint": "saved-checkpoint"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provider, "train", train)
    monkeypatch.setattr(pipeline, "get_provider", lambda name: provider)
    argv = ["train", "--provider", provider_name, "--confirm"]
    if run_id is not None:
        argv.extend(["--run-id", run_id])
    if run_dir is not None:
        argv.extend(["--run-dir", run_dir])

    identities = []
    for _ in range(2):
        pipeline.main(argv)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        data_dir, output_dir, identity, kwargs = calls[-1]
        assert data_dir == tmp_path / "data"
        assert kwargs == {"confirm": True, "init_from_checkpoint": None}
        if run_id is None:
            assert re.fullmatch(r"sft-\d{8}-\d{6}-[0-9a-f]{12}", identity)
        else:
            assert identity == run_id
        assert output_dir == (Path(run_dir) if run_dir else tmp_path / "runs" / identity)
        assert result == {"checkpoint": "saved-checkpoint", "run_id": identity, "run_dir": str(output_dir.resolve())}
        assert f"Run ID: {identity}\nRun directory: {output_dir.resolve()}\n" == captured.err
        identities.append(identity)
    if run_id is None:
        assert identities[0] != identities[1]


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "/absolute", "nested/name", "nested\\name"])
def test_cli_training_rejects_path_ids_for_default_directory(tmp_path, monkeypatch, capsys, run_id):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        pipeline.main(["train", "--run-id", run_id, "--confirm"])
    assert error.value.code == 2
    assert "single directory name" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_baseten_runtime_errors_use_the_cli_error_path(tmp_path, monkeypatch, capsys):
    """Credential-free Baseten service failures print like other CLI errors."""

    class Provider:
        name = "baseten"

        def settings_from_options(self, options):
            return baseten.BasetenProvider().settings_from_options(options)

        def train(self, *args, **kwargs):
            raise baseten.BasetenRuntimeError("BASETEN_API_KEY is required for preflight")

    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    monkeypatch.setattr(pipeline, "get_provider", lambda name: Provider())
    with pytest.raises(SystemExit) as error:
        pipeline.main([
            "train", "--provider", "baseten", "--data-dir", str(tmp_path),
            "--run-dir", str(tmp_path / "run"), "--run-id", "test", "--confirm",
        ])

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "BASETEN_API_KEY is required for preflight" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_fireworks_training_configuration_is_owned_by_provider(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    model = fireworks.DEFAULT_MODEL
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "langsmith": {"examples": 100},
                "split": {"train": 80, "validation": 10, "test": 10},
                "model": model.__dict__,
                "provider": {
                    "name": "fireworks",
                    "renderer": model.renderer,
                    "tokenizer_revision": model.tokenizer_revision,
                },
            }
        ),
        encoding="utf-8",
    )

    original_import = builtins.__import__

    def import_without_cli(name, *args, **kwargs):
        if name == "pipeline":
            raise AssertionError("provider planning must work without the CLI")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_cli)
    plan = fireworks.FireworksProvider().plan(
        tmp_path,
        "run-id",
        fireworks.SFTSettings(),
    )

    assert plan["training_api"] == fireworks.TRAINING_BASE_URL
    assert plan["config"]["max_epochs"] == 5


def test_fireworks_provider_rejects_a_baseten_prepared_manifest(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "langsmith": {"examples": 100},
                "split": {"train": 90, "validation": 10, "test": 0},
                "model": baseten.MODEL_SPECS["qwen3p8-27b"].__dict__,
                "provider": {
                    "name": "baseten",
                    "renderer": "qwen3_8_preserved",
                    "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="prepared provider mismatch"):
        get_provider("fireworks").plan(
            tmp_path,
            "run-id",
            fireworks.SFTSettings(),
        )


def test_fireworks_provider_train_rejects_a_baseten_prepared_manifest(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "langsmith": {"examples": 100},
                "split": {"train": 90, "validation": 10, "test": 0},
                "model": baseten.MODEL_SPECS["qwen3p8-27b"].__dict__,
                "provider": {
                    "name": "baseten",
                    "renderer": "qwen3_8_preserved",
                    "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PipelineError, match="prepared provider mismatch"):
        get_provider("fireworks").train(
            tmp_path,
            tmp_path / "run",
            "run-id",
            fireworks.SFTSettings(),
            confirm=True,
            init_from_checkpoint=None,
        )


@pytest.mark.parametrize("provider", ["baseten", "fireworks"])
def test_cli_cross_workspace_preparation(tmp_path, monkeypatch, capsys, provider):
    from test_example_tools import example
    from test_tool_capture import llm
    from smithtune.inference_contract import contract_from_runs, parse_inference_contract

    monkeypatch.setattr(pipeline, "get_version", lambda: "0.1.0")
    module = baseten if provider == "baseten" else fireworks
    monkeypatch.setattr(module, "preflight_model", lambda model, **kwargs: model)
    monkeypatch.setattr(module, "resolve_rendering_model", lambda model: model)
    calls = []

    def download(workspace, dataset_id, raw):
        calls.append(("download", workspace))
        examples = [example(1)]
        raw.mkdir(parents=True)
        (raw / "examples.json").write_text(json.dumps(examples))
        (raw / "dataset-export.json").write_text(json.dumps([{"inputs": examples[0]["inputs"]}]))
        (raw / "dataset.json").write_text(json.dumps({"id": dataset_id, "example_count": 1}))

    def capture(workspace, examples, *, source_workspace_id, checkpoint_path):
        calls.append(("capture", workspace, source_workspace_id))
        payload = contract_from_runs([llm("run-1", [])], workspace_id=source_workspace_id)
        payload["provenance"]["source_example_id"] = examples[0]["id"]
        return {examples[0]["id"]: parse_inference_contract(payload)}

    monkeypatch.setattr(dataset, "download_dataset", download)
    monkeypatch.setattr(dataset, "capture_example_contracts", capture)
    monkeypatch.setattr(sys, "argv", [
        "smithtune", "prepare", "--provider", provider, "--model", "qwen3p8-27b",
        "--workspace-id", "dataset-workspace", "--source-workspace-id", "trace-workspace",
        "--dataset-id", "dataset-id", "--data-dir", str(tmp_path), "--skip-render-check",
        "--validation-fraction", "0", "--test-fraction", "0",
    ])
    assert pipeline.main() is None
    assert calls == [("download", "dataset-workspace"), ("capture", "dataset-workspace", "trace-workspace")]
    manifest = json.loads((tmp_path / "prepared" / "manifest.json").read_text())
    assert manifest["langsmith"]["workspace_id"] == "dataset-workspace"

    calls.clear()
    sys.argv.append("--no-fetch")
    assert pipeline.main() is None
    assert calls == []
    sys.argv[sys.argv.index("--source-workspace-id") + 1] = "other-workspace"
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2
    assert "different source workspace" in capsys.readouterr().err

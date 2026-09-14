from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import shlex

import pytest

from smithtune import cli, curation
from smithtune.providers import get_provider
from smithtune.providers.base import ModelOptions, PipelineError, TrainingOptions
from smithtune.training_plan import load_training_plan, save_training_plan


@pytest.fixture(autouse=True)
def cli_version(monkeypatch):
    # These are command behavior tests; distribution metadata is tested separately.
    monkeypatch.setattr(cli, "get_version", lambda: "test")


def prepared_data(root, provider_name):
    provider = get_provider(provider_name)
    model = provider.model_from_options(ModelOptions())
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    (prepared / "manifest.json").write_text(json.dumps({
        "langsmith": {"examples": 3, "dataset_id": "source-dataset"},
        "split": {"train": 1, "validation": 1, "test": 1},
        "model": asdict(model),
        "provider": {"name": provider_name, "renderer": model.renderer, "tokenizer_revision": model.tokenizer_revision},
        "audit": {"max_context_tokens": 32},
    }))
    for partition in ("train", "validation", "test"):
        (prepared / f"{partition}.jsonl").write_text(json.dumps({
            "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": partition}],
            "tools": [],
        }) + "\n")
    return provider


def saved_plan(tmp_path, provider_name="fireworks"):
    provider = prepared_data(tmp_path / "data", provider_name)
    path = tmp_path / "plan.json"
    value = save_training_plan(
        provider, tmp_path / "data", "reviewed-run",
        provider.settings_from_options(TrainingOptions()), path,
    )
    return path, value


@pytest.mark.parametrize("provider_name,extra", [
    ("fireworks", ["--lora-alpha", "16", "--pipeline-depth", "2"]),
    ("baseten", ["--replicas", "2", "--max-spend-usd", "75", "--hourly-rate-usd", "30",
                 "--spend-reserve-fraction", "0.2", "--max-dropped-training-rows", "0"]),
])
def test_cli_train_uses_saved_plan_from_another_directory(tmp_path, monkeypatch, capsys, provider_name, extra):
    provider = prepared_data(tmp_path / "data", provider_name)
    monkeypatch.chdir(tmp_path)
    cli.main([
        "plan", "--provider", provider_name, "--data-dir", "data", "--run-id", "reviewed-run",
        "--output", "plans/reviewed.json", "--learning-rate", "0.0003", "--max-epochs", "2",
        "--early-stopping-patience", "2", "--early-stopping-min-delta", "0.01",
        "--batch-size", "4", "--seed", "7", "--init-from-checkpoint", "saved-checkpoint", *extra,
    ])
    output = capsys.readouterr()
    value = json.loads(output.out)
    path = tmp_path / "plans/reviewed.json"
    assert json.loads(path.read_text()) == value
    assert "Training plan saved to" in output.err
    assert value["training"]["settings"]["lora_rank"] == 8
    if provider_name == "baseten":
        assert value["training"]["settings"]["microbatch_token_budget"] == 262_144
    calls = []

    def train(self, data_dir, run_dir, run_id, settings, **kwargs):
        calls.append((self.name, data_dir, run_dir, run_id, asdict(settings), kwargs))
        return {"status": "trained"}

    monkeypatch.setattr(type(provider), "train", train)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cli.main(["train", "--plan", str(path), "--run-dir", "run", "--confirm"])
    assert calls == [(
        provider_name, tmp_path / "data", Path("run"), "reviewed-run", value["training"]["settings"],
        {"confirm": True, "init_from_checkpoint": "saved-checkpoint"},
    )]
    assert json.loads(capsys.readouterr().out) == {"status": "trained"}


def test_plan_defaults_to_file_in_working_directory(tmp_path, monkeypatch, capsys):
    prepared_data(tmp_path / "data", "fireworks")
    monkeypatch.chdir(tmp_path)
    cli.main(["plan"])
    value = json.loads(capsys.readouterr().out)
    assert json.loads((tmp_path / "plan.json").read_text()) == value
    assert load_training_plan(tmp_path / "plan.json").run_id == "langsmith-sft"


@pytest.mark.parametrize("option", [
    ["--provider", "fireworks"], ["--data-dir", "data"], ["--run-id", "other"],
    ["--init-from-checkpoint", "other"], ["--learning-rate=0.0001"], ["--learning-r", "0.0001"],
    ["--max-epochs", "5"], ["--early-stopping-patience", "1"], ["--early-stopping-min-delta", "0"],
    ["--batch-size", "32"], ["--seed", "42"], ["--lora-rank", "8"],
    ["--lora-alpha", "32"], ["--pipeline-depth", "4"], ["--microbatch-token-budget", "100"],
    ["--max-spend-usd", "75"], ["--hourly-rate-usd", "30"], ["--replicas", "1"],
    ["--spend-reserve-fraction", "0.1"], ["--max-dropped-training-rows", "1"],
])
def test_saved_plan_rejects_explicit_overrides_before_loading_or_training(option, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--plan", "missing.json", "--run-dir", str(tmp_path / "run"), "--confirm", *option])
    assert exc.value.code == 2
    assert "--plan cannot be combined with" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_saved_plan_still_requires_confirmation(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--plan", "missing.json", "--run-dir", str(tmp_path / "run")])
    assert exc.value.code == 2
    assert "rerun with --confirm" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("provider_name", ["fireworks", "baseten"])
def test_direct_training_remains_supported(provider_name, tmp_path, monkeypatch):
    provider = get_provider(provider_name)
    calls = []

    def train(self, data_dir, run_dir, run_id, settings, **kwargs):
        calls.append((self.name, data_dir, run_id, settings.learning_rate))
        return {}

    monkeypatch.setattr(type(provider), "train", train)
    monkeypatch.chdir(tmp_path)
    cli.main(["train", "--provider", provider_name, "--run-dir", "run", "--run-id", "direct", "--confirm"])
    assert calls == [(provider_name, tmp_path / "data", "direct", 1e-4)]


@pytest.mark.parametrize("name", ["manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"])
def test_saved_plan_rejects_changes_to_prepared_files_before_training(tmp_path, name, monkeypatch, capsys):
    path, _ = saved_plan(tmp_path)
    prepared = tmp_path / "data/prepared" / name
    prepared.write_text(prepared.read_text() + "\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("must reject changed data before training")

    monkeypatch.setattr(type(get_provider("fireworks")), "train", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--plan", str(path), "--run-dir", "run", "--confirm"])
    assert exc.value.code == 2
    assert "prepared data changed since planning" in capsys.readouterr().err


@pytest.mark.parametrize("change", [
    lambda v: v.update(schema_version=99),
    lambda v: v.update(training=[]),
    lambda v: v["training"].update(data_dir="relative"),
    lambda v: v["training"].update(provider="unknown"),
    lambda v: v["training"]["settings"].pop("learning_rate"),
    lambda v: v["training"]["settings"].update(learning_rate="oops"),
    lambda v: v["training"]["settings"].update(learning_rate=float("nan")),
    lambda v: v["training"]["settings"].update(learning_rate=True),
    lambda v: v["training"]["settings"].update(learning_rate=0),
    lambda v: v["training"]["settings"].update(max_epochs=2.5),
    lambda v: v["training"]["settings"].update(max_epochs=2.0),
    lambda v: v["training"]["settings"].update(extra_setting=1),
    lambda v: v["training"]["settings"].update(learning_rate=0.0009),
    lambda v: v["plan"]["config"].update(learning_rate=0.0009),
])
def test_invalid_or_inconsistent_plan_fails_locally(tmp_path, change):
    path, value = saved_plan(tmp_path)
    change(value)
    path.write_text(json.dumps(value))
    with pytest.raises(PipelineError):
        load_training_plan(path)


@pytest.mark.parametrize("symlinked", [False, True])
def test_plan_cannot_overwrite_prepared_data(tmp_path, symlinked):
    path, _ = saved_plan(tmp_path)
    request = load_training_plan(path)
    prepared = request.data_dir / "prepared"
    if symlinked:
        external = tmp_path / "external-prepared"
        prepared.rename(external)
        prepared.symlink_to(external, target_is_directory=True)
    manifest = prepared / "manifest.json"
    original = manifest.read_bytes()
    with pytest.raises(PipelineError, match="outside the prepared"):
        save_training_plan(request.provider, request.data_dir, request.run_id, request.settings, manifest)
    assert manifest.read_bytes() == original


def test_readme_dataset_command_submits_the_supported_feedback_expression(tmp_path, monkeypatch, capsys):
    from test_curation import API, root, uid

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    command = next(block for block in readme.split("```bash\n")[1:] if block.startswith("smithtune dataset create"))
    command = command.split("```", 1)[0].replace("\\\n", "")
    argv = shlex.split(command)[1:]
    argv[argv.index("--workspace-id") + 1] = uid(100)
    argv[argv.index("--project-id") + 1] = uid(101)
    api = API([[root(1, "selected-thread")]])
    monkeypatch.setattr(curation, "create_dataset", partial(curation.create_dataset, runner=api))
    monkeypatch.chdir(tmp_path)
    cli.main(argv)
    query = api.calls[0][1]
    assert query["filter"] == 'and(and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9)), lt(start_time, "2026-09-08T00:00:00+00:00"))'
    assert query["is_root"] is True
    assert len(api.examples) == 1
    assert json.loads(capsys.readouterr().out)["dataset_id"] == uid(200)


@pytest.mark.parametrize("option", [
    ["--learning-rate", "nan"], ["--learning-rate", "inf"],
    ["--run-id", ""], ["--init-from-checkpoint", ""],
])
def test_plan_rejects_invalid_inputs_before_writing(tmp_path, monkeypatch, capsys, option):
    prepared_data(tmp_path / "data", "fireworks")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["plan", *option])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "plan.json").exists()

"""Offline SageMaker request/lifecycle tests. No AWS resource is created."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from smithtune import cli, rendering
from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json
from smithtune.providers import get_provider, sagemaker as aws
from smithtune.providers.sagemaker import provider as aws_provider
from smithtune.providers.base import ModelOptions, PipelineError, TrainingOptions
from smithtune.providers.baseten.provider import MODEL_SPECS as BASETEN_MODELS
from smithtune.training.hf_sft import as_example, read_bundle, render_conversations
from tests.binding_fixtures import bound_row


def config(**kwargs):
    return aws.AWSJobConfig(**({
        "region": "us-east-1", "role_arn": "arn:aws:iam::123456789012:role/SmithTune",
        "training_image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/smithtune@sha256:" + "a" * 64,
        "s3_prefix": "s3://smithtune-test/jobs", "instance_type": "ml.g6e.2xlarge",
        "max_runtime_seconds": 600,
    } | kwargs))


def fallback_config(**kwargs):
    return config(**({"instance_type": None, "max_pending_seconds": 1800,
                     "instance_preferences": [
                         {"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": 2.8},
                         {"instance_type": "ml.g6e.4xlarge", "hourly_rate_usd": 3.76},
                     ]} | kwargs))


def prepared(root, profile="qwen3p5-9b"):
    model = replace(aws.MODEL_SPECS[profile], max_seq_len=4096, trainer_max_seq_len=4096,
                    template_sha256="b" * 64, rendering_version="synthetic-test-renderer")
    manifest = {
        "model": asdict(model), "provider": {"name": "sagemaker", "renderer": model.renderer,
                                             "tokenizer_revision": model.tokenizer_revision},
        "split": {"train": 1, "validation": 1, "test": 1}, "audit": {"max_context_tokens": 12},
    }
    _json_dump(root / "prepared/manifest.json", manifest)
    for split in ("train", "validation", "test"):
        _jsonl_dump(root / f"prepared/{split}.jsonl", [bound_row({
            "messages": [{"role": "user", "content": "Calculate"},
                         {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "42"},
                         {"role": "assistant", "content": "42"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}],
            "_source": {"example_id": split, "source_key": ["workspace", "project", "thread", split],
                        "source_scope": "thread", "source_scope_id": split},
        })])
    return model


class MissingJob(Exception):
    response: ClassVar[dict] = {"Error": {"Code": "ResourceNotFound"}}


class FakeAWS:
    def __init__(self, *, timeout=False, occupied=False):
        self.calls = []
        self.timeout = timeout
        self.occupied = occupied
        self.request = None

    def client(self, name):
        return self

    def get_caller_identity(self):
        return {"Account": "123456789012"}

    def describe_training_job(self, **kwargs):
        self.calls.append(("describe", kwargs))
        if self.request is None:
            raise MissingJob()
        return self.request | {
            "TrainingJobStatus": "InProgress", "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/sft-test",
            "SecondaryStatus": "Training",
        }

    def list_objects_v2(self, **kwargs):
        return {"KeyCount": 1 if self.occupied else 0}

    def upload_file(self, *args, **kwargs):
        self.calls.append(("upload", args))

    def create_training_job(self, **kwargs):
        self.calls.append(("create", kwargs))
        self.request = kwargs
        if self.timeout:
            raise TimeoutError("simulated lost submission response")
        return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/sft-test"}

    def stop_training_job(self, **kwargs):
        self.calls.append(("stop", kwargs))


def test_plan_is_offline_and_contains_a_valid_aws_request(tmp_path):
    prepared(tmp_path)
    provider = aws.SageMakerProvider(session_factory=lambda _: pytest.fail("planning contacted AWS"))
    plan = provider.plan(tmp_path, "sft-test", aws.SageMakerSFTSettings(aws=config(hourly_rate_usd=6)))
    assert plan["cost"]["runtime_compute_estimate_usd"] == 1
    assert plan["worker_settings"]["lora_rank"] == 8
    assert plan["create_training_job"]["StoppingCondition"] == {"MaxRuntimeInSeconds": 600}
    assert len(plan["input_sha256"]) == 4
    botocore = pytest.importorskip("botocore.session")
    from botocore.validate import validate_parameters
    shape = botocore.Session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    validate_parameters(plan["create_training_job"], shape)


@pytest.mark.parametrize("changes", [
    {"instance_count": 2}, {"max_runtime_seconds": 0}, {"max_runtime_seconds": True},
    {"hourly_rate_usd": float("nan")}, {"s3_prefix": "s3://bucket"},
    {"s3_prefix": "s3://bucket/path/../other"}, {"training_image": "image:latest"},
    {"role_arn": "user"}, {"security_group_ids": ["sg-abcd"]},
])
def test_invalid_configs_rejected(changes):
    with pytest.raises(PipelineError):
        config(**changes).validate()


def test_fallback_plan_is_one_native_job_with_conservative_cost(tmp_path):
    prepared(tmp_path)
    provider = aws.SageMakerProvider(session_factory=lambda _: pytest.fail("planning contacted AWS"))
    plan = provider.plan(tmp_path, "fallback", aws.SageMakerSFTSettings(aws=fallback_config()))
    request = plan["create_training_job"]
    assert request["ResourceConfig"] == {
        "InstanceCount": 1, "VolumeSizeInGB": 100,
        "InstancePreferences": [{"InstanceType": "ml.g6e.2xlarge"}, {"InstanceType": "ml.g6e.4xlarge"}],
    }
    assert "EnableManagedSpotTraining" not in request
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 600, "MaxPendingTimeInSeconds": 1800}
    assert plan["cost"]["hourly_rate_usd"] == 3.76
    assert plan["cost"]["runtime_compute_estimate_usd"] == pytest.approx(3.76 / 6)
    assert [v["hourly_rate_usd"] for v in plan["cost"]["instance_options"]] == [2.8, 3.76]
    botocore = pytest.importorskip("botocore.session")
    from botocore.validate import validate_parameters
    shape = botocore.Session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    validate_parameters(request, shape)
    path = tmp_path / "aws.json"
    _json_dump(path, asdict(fallback_config()))
    assert aws.AWSJobConfig.read(path) == fallback_config()


@pytest.mark.parametrize("changes", [
    {"instance_type": "ml.g6e.2xlarge"}, {"hourly_rate_usd": 2.8},
    {"instance_preferences": []}, {"instance_preferences": None},
    {"instance_preferences": ["ml.g6e.2xlarge"]},
    {"instance_preferences": [{"instance_type": "ml.g6e.2xlarge"}]},
    {"instance_preferences": [{"instance_type": "ml.t3.medium", "hourly_rate_usd": 1}]},
    {"instance_preferences": [{"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": float("nan")}]},
    {"instance_preferences": [{"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": True}]},
    {"instance_preferences": [{"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": 2.8}] * 2},
    {"instance_preferences": [{"instance_type": f"ml.g6e.{n}xlarge", "hourly_rate_usd": n} for n in range(1, 7)]},
    {"max_pending_seconds": None}, {"max_pending_seconds": 600},
    {"max_pending_seconds": True}, {"max_pending_seconds": 86401},
    {"instance_count": 2},
])
def test_fallback_rejects_ambiguous_unpriced_or_unbounded_options(changes):
    with pytest.raises(PipelineError):
        fallback_config(**changes).validate()


def test_fallback_status_reports_selected_instance_without_resubmitting(tmp_path):
    prepared(tmp_path)
    remote = FakeAWS()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=fallback_config()), confirm=True)
    assert provider.status(run)["selected_instance_type"] is None
    remote.request["ResourceConfig"].update(SelectedInstanceType="ml.g6e.4xlarge", SelectedInstanceCount=1)
    status = provider.status(run)
    assert status["selected_instance_type"] == "ml.g6e.4xlarge"
    assert status["selected_hourly_rate_usd"] == 3.76
    assert status["selected_instance_count"] == 1
    assert provider.status(run, stop=True, confirm=True)["status"] == "Stopping"
    assert sum(call[0] == "create" for call in remote.calls) == 1


@pytest.mark.parametrize("changes", [
    {"SelectedInstanceType": "ml.p5.48xlarge", "SelectedInstanceCount": 1},
    {"SelectedInstanceType": "ml.g6e.4xlarge", "SelectedInstanceCount": 2},
    {"InstancePreferences": [{"InstanceType": "ml.g6e.8xlarge"}]},
])
def test_fallback_rejects_changed_remote_allocation_before_stopping(tmp_path, changes):
    prepared(tmp_path)
    remote = FakeAWS()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=fallback_config()), confirm=True)
    remote.request["ResourceConfig"].update(changes)
    with pytest.raises(PipelineError, match="resources|allocation"):
        provider.status(run, stop=True, confirm=True)
    assert not any(call[0] == "stop" for call in remote.calls)


def test_submit_requires_confirmation_before_any_remote_action(tmp_path):
    provider = aws.SageMakerProvider(session_factory=lambda _: pytest.fail("contacted AWS"))
    with pytest.raises(PipelineError, match="paid resources"):
        provider.train(tmp_path, tmp_path / "run", "sft-test", None, confirm=False)
    assert not (tmp_path / "run").exists()


def test_submit_freezes_input_and_status_can_reconnect_and_stop(tmp_path):
    prepared(tmp_path)
    remote = FakeAWS()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    result = provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=config()), confirm=True)
    assert result["status"] == "Submitted"
    assert len([c for c in remote.calls if c[0] == "upload"]) == 5
    assert read_bundle(run / "input")[1].provider == "sagemaker"
    training = _load_json(run / "input/bundle.json")["training"]
    assert training == _load_json(run / "plan.json")["training"]
    assert training["model_revision"] == _load_json(tmp_path / "prepared/manifest.json")["model"]["tokenizer_revision"]
    assert training["prepared_manifest_uri"] == "s3://smithtune-test/jobs/sft-test/input/prepared/manifest.json"
    assert training["lora_rank"] == 8 and training["lora_alpha"] == 16
    assert training["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert "aws" not in training["settings"] and "trainer_backend" not in training["settings"]
    assert provider.status(run)["status"] == "InProgress"
    assert provider.status(run, stop=True, confirm=True)["status"] == "Stopping"
    assert sum(c[0] == "create" for c in remote.calls) == 1
    assert sum(c[0] == "stop" for c in remote.calls) == 1
    with pytest.raises(PipelineError, match="new or empty"):
        provider.train(tmp_path, run, "another", aws.SageMakerSFTSettings(aws=config()), confirm=True)


@pytest.mark.parametrize("changed", [
    {"lora_alpha": 99}, {"target_modules": ["q_proj"]}, {"model_revision": "different"},
    {"settings": {"batch_size": 999}}, {"schema_version": 2},
])
def test_worker_rejects_drift_between_portable_training_description_and_recipe(tmp_path, changed):
    prepared(tmp_path)
    run = tmp_path / "run"
    provider = aws.SageMakerProvider(session_factory=lambda _: FakeAWS())
    provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=config()), confirm=True)
    bundle_path = run / "input/bundle.json"
    bundle = _load_json(bundle_path)
    bundle["training"].update(changed)
    _json_dump(bundle_path, bundle)
    with pytest.raises(PipelineError, match="training specification"):
        read_bundle(run / "input")
    # Old pilot bundles have no shared description and remain readable.
    del bundle["training"]
    _json_dump(bundle_path, bundle)
    assert read_bundle(run / "input")[1].provider == "sagemaker"


def test_submission_timeout_keeps_receipt_and_does_not_retry(tmp_path):
    prepared(tmp_path)
    remote = FakeAWS(timeout=True)
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    with pytest.raises(PipelineError, match="may still be running"):
        provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=config()), confirm=True)
    assert _load_json(run / "job.json")["status"] == "submission_pending"
    assert provider.status(run)["status"] == "InProgress"
    assert sum(c[0] == "create" for c in remote.calls) == 1


def test_refuses_to_overwrite_existing_s3_run(tmp_path):
    prepared(tmp_path)
    remote = FakeAWS(occupied=True)
    with pytest.raises(PipelineError, match="prefix already contains"):
        aws.SageMakerProvider(session_factory=lambda _: remote).train(
            tmp_path, tmp_path / "run", "sft-test", aws.SageMakerSFTSettings(aws=config()), confirm=True,
        )
    assert not any(call[0] in {"upload", "create"} for call in remote.calls)


def test_altered_bundle_and_remote_identity_are_rejected(tmp_path):
    prepared(tmp_path)
    remote = FakeAWS()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=config()), confirm=True)
    with (run / "input/prepared/train.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(PipelineError, match="checksum"):
        read_bundle(run / "input")
    remote.request["RoleArn"] = "arn:aws:iam::123456789012:role/Other"
    with pytest.raises(PipelineError, match="does not match"):
        provider.status(run, stop=True, confirm=True)
    assert not any(call[0] == "stop" for call in remote.calls)


def test_rejects_source_leakage_and_provider_mismatch(tmp_path):
    prepared(tmp_path)
    source = tmp_path / "prepared/train.jsonl"
    (tmp_path / "prepared/validation.jsonl").write_bytes(source.read_bytes())
    with pytest.raises(PipelineError, match="multiple splits"):
        aws.load_prepared(tmp_path)
    prepared(tmp_path)
    manifest = _load_json(tmp_path / "prepared/manifest.json")
    manifest["provider"]["name"] = "baseten"
    _json_dump(tmp_path / "prepared/manifest.json", manifest)
    with pytest.raises(PipelineError, match="provider mismatch"):
        aws.load_prepared(tmp_path)


def test_cli_wires_aws_settings_and_rejects_unsupported_operations(tmp_path, capsys):
    path = tmp_path / "aws.json"
    _json_dump(path, asdict(config()))
    parser = cli._parser()
    args = parser.parse_args(["plan", "--provider", "sagemaker", "--aws-config", str(path)])
    assert cli._settings_from_args(args).aws == config()
    for provider in ("fireworks", "baseten"):
        args = parser.parse_args(["plan", "--provider", provider, "--aws-config", str(path)])
        with pytest.raises(PipelineError, match="requires --provider sagemaker"):
            cli._settings_from_args(args)
    cli.main(["models", "list", "--provider", "sagemaker"])
    listed = json.loads(capsys.readouterr().out)["models"]
    assert {model["alias"] for model in listed} == {"qwen3p5-9b", "qwen3p5-0p8b-test"}
    assert all(model["experimental"] and not model["hardware_validated"] for model in listed)
    with pytest.raises(SystemExit):
        cli.main(["undeploy", "--provider", "sagemaker", "--confirm"])
    assert "not implemented" in capsys.readouterr().err


def test_model_context_requires_explicit_selection_and_reuses_baseten_formatter():
    provider = get_provider("sagemaker")
    with pytest.raises(PipelineError, match="explicit --max-seq-len"):
        provider.model_from_options(ModelOptions(model="qwen3p5-9b"))
    spec = provider.model_from_options(ModelOptions(model="qwen3p5-9b", max_seq_len=4096))
    assert spec.renderer == BASETEN_MODELS[spec.name].renderer
    assert rendering.rendering_version(spec) == rendering.rendering_version(BASETEN_MODELS[spec.name])


def test_prepare_reuses_dataset_pipeline_without_provider_service_calls(tmp_path, monkeypatch):
    from tests.test_provider_adapter import _write_raw_dataset

    _write_raw_dataset(tmp_path)
    monkeypatch.setattr(aws_provider, "resolve_rendering_model", lambda spec: replace(
        spec, template_sha256="b" * 64, rendering_version="synthetic-test-renderer",
    ))
    renderer = SimpleNamespace(render=lambda messages, tools=None, final_target=False: [SimpleNamespace(
        token_ids=[1, 2, 3, 4], token_weights=[0, 0, 1, 1],
    )])
    monkeypatch.setattr(rendering, "load_training_renderer", lambda _: renderer)
    provider = aws.SageMakerProvider(session_factory=lambda _: pytest.fail("prepare contacted AWS"))
    provider.prepare("workspace-id", "dataset-id", tmp_path,
                     model_options=ModelOptions(model="qwen3p5-9b", max_seq_len=4096),
                     fetch=False, sync_splits=False)
    manifest, rows = aws.load_prepared(tmp_path)
    assert manifest["provider"]["name"] == "sagemaker"
    assert manifest["audit"]["max_context_tokens"] == 4
    assert all(rows[split] for split in ("train", "validation", "test"))


def test_tool_context_is_preserved_and_masks_shift_only_once(tmp_path):
    model = prepared(tmp_path)
    rows = aws.load_prepared(tmp_path)[1]["train"]
    datum = SimpleNamespace(token_ids=[10, 11, 12, 13, 14, 15], token_weights=[0, 1, 1, 0, 0, 1])
    calls = []
    renderer = SimpleNamespace(render=lambda messages, tools=None, final_target=False: calls.append((messages, tools)) or [datum])
    examples = render_conversations(rows, model, renderer)[0]
    # Each assistant message renders once with its history and its own tools.
    assert calls == [(rows[0]["messages"][:2], rows[0]["tools"]), (rows[0]["messages"], rows[0]["tools"])]
    assert len(examples) == 2
    item = examples[0]
    assert item.input_ids == datum.token_ids
    # HF receives unshifted labels. Its loss shifts to [11, 12, -100, -100, 15].
    assert item.labels == [-100, 11, 12, -100, -100, 15]
    assert item.active_tokens == 3


@pytest.mark.parametrize("tokens,weights,limit", [
    ([1, 2, 3], [0, 0, 1], 2), ([1, 2], [1, 0], 2),
    ([1, 2], [0, float("nan")], 2), ([1, True], [0, 1], 2),
])
def test_worker_rejects_truncation_empty_targets_and_invalid_masks(tokens, weights, limit):
    with pytest.raises(PipelineError):
        as_example(SimpleNamespace(token_ids=tokens, token_weights=weights), limit)


def test_cpu_instances_are_allowed_only_for_the_pipeline_test_profile(tmp_path):
    cpu = config(instance_type="ml.m5.xlarge", hourly_rate_usd=0.23)
    provider = aws.SageMakerProvider(session_factory=lambda _: pytest.fail("planning contacted AWS"))
    settings = provider.settings_from_options(TrainingOptions(aws_config=_write_config(tmp_path, cpu)))
    assert settings.device == "cpu"
    prepared(tmp_path)
    with pytest.raises(PipelineError, match="CPU instance types are allowed only"):
        provider.plan(tmp_path, "cpu-test", settings)
    prepared(tmp_path, profile="qwen3p5-0p8b-test")
    plan = provider.plan(tmp_path, "cpu-test", settings)
    assert plan["worker_settings"]["device"] == "cpu"
    assert plan["training"]["model_revision"] == aws.MODEL_SPECS["qwen3p5-9b"].tokenizer_revision
    assert any("CPU training" in warning for warning in plan["warnings"])
    gpu = provider.settings_from_options(TrainingOptions(aws_config=_write_config(tmp_path, config())))
    assert gpu.device == "cuda:0"
    with pytest.raises(PipelineError, match="all GPU types or all CPU types"):
        fallback_config(instance_preferences=[{"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": 2.8},
                                              {"instance_type": "ml.m5.xlarge", "hourly_rate_usd": 0.23}]).validate()
    with pytest.raises(PipelineError, match="instance_type"):
        config(instance_type="ml.t3.medium").validate()


def _write_config(root, value):
    path = root / f"aws-{id(value)}.json"
    _json_dump(path, asdict(value))
    return path


def test_test_profile_shares_the_pinned_tokenizer_but_pins_its_own_weights():
    from smithtune.training.contracts import model_revision

    provider = get_provider("sagemaker")
    nine = provider.model_from_options(ModelOptions(model="Qwen/Qwen3.5-9B", max_seq_len=4096))
    small = provider.model_from_options(ModelOptions(model="Qwen/Qwen3.5-0.8B", max_seq_len=4096))
    assert nine.name == "qwen3p5-9b" and small.name == "qwen3p5-0p8b-test"
    assert small.tokenizer_model == nine.tokenizer_model and small.tokenizer_revision == nine.tokenizer_revision
    assert small.renderer == nine.renderer and small.base_model == "Qwen/Qwen3.5-0.8B"
    assert model_revision(small) != model_revision(nine) and len(model_revision(small)) == 40
    assert provider.model_from_options(ModelOptions(model="qwen3p5-0p8b-test", max_seq_len=64)).max_seq_len == 64
    with pytest.raises(PipelineError, match="no supported SageMaker model profile"):
        provider.model_from_options(ModelOptions(model="qwen3p5-99b", max_seq_len=64))

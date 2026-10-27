"""Real tiny CPU training/export/reload with AWS boundaries replaced in memory."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace
from typing import ClassVar

import pytest

from smithtune import cli
from smithtune.artifacts import _json_dump, _load_json
from smithtune.providers import sagemaker as aws
from smithtune.providers.sagemaker import deployment as deploy
from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker.artifacts import collect, extract_archive, package_artifact
from smithtune.providers.sagemaker.monitoring import wait_for_training
from smithtune.training import hf_sft as worker, serving
from smithtune.training.contracts import verify_artifact, write_artifact_manifest
from tests.test_sagemaker import FakeAWS, MissingJob, config, prepared
from tests.test_sagemaker_worker import tiny_qwen


class MissingObject(Exception):
    response: ClassVar[dict] = {"Error": {"Code": "404"}}


class Cloud(FakeAWS):
    def __init__(self):
        super().__init__()
        self.resources = {"model": {}, "config": {}, "endpoint": {}}
        self.objects, self.metadata = {}, {}
        self.completed = False
        self.lose_create_response = False
        self.image_media_type = "application/vnd.docker.distribution.manifest.v2+json"

    def describe_images(self, **kwargs):
        return {"imageDetails": [{"imageDigest": kwargs["imageIds"][0]["imageDigest"], "imageManifestMediaType": self.image_media_type}]}

    def describe_training_job(self, **kwargs):
        value = super().describe_training_job(**kwargs)
        if self.completed:
            value.update(TrainingJobStatus="Completed", SecondaryStatus="Completed",
                         ModelArtifacts={"S3ModelArtifacts": "s3://smithtune-test/jobs/sft-test/output/sft-test/output/model.tar.gz"})
        return value

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.metadata[(bucket, key)] = (ExtraArgs or {}).get("Metadata", {})

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise MissingObject()
        return {"ContentLength": len(self.objects[(Bucket, Key)]), "Metadata": self.metadata.get((Bucket, Key), {})}

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def _create(self, kind, operation, request):
        from botocore.session import Session
        from botocore.validate import validate_parameters
        shape = Session().get_service_model("sagemaker").operation_model(operation).input_shape
        validate_parameters(request, shape)
        name_field = {"model": "ModelName", "config": "EndpointConfigName", "endpoint": "EndpointName"}[kind]
        arn_field = {"model": "ModelArn", "config": "EndpointConfigArn", "endpoint": "EndpointArn"}[kind]
        name = request[name_field]
        remote = deepcopy(request) | {arn_field: f"arn:aws:sagemaker:us-east-1:123456789012:{kind}/{name}"}
        if kind == "endpoint":
            remote["EndpointStatus"] = "InService"
        self.resources[kind][name] = remote
        self.calls.append((f"create_{kind}", name))
        if kind == "endpoint" and self.lose_create_response:
            raise TimeoutError("lost response after successful creation")
        return {arn_field: remote[arn_field]}

    def create_model(self, **kwargs):
        return self._create("model", "CreateModel", kwargs)

    def create_endpoint_config(self, **kwargs):
        return self._create("config", "CreateEndpointConfig", kwargs)

    def create_endpoint(self, **kwargs):
        return self._create("endpoint", "CreateEndpoint", kwargs)

    def _describe(self, kind, name):
        if name not in self.resources[kind]:
            raise MissingJob()
        return self.resources[kind][name]

    def describe_model(self, ModelName):
        return self._describe("model", ModelName)

    def describe_endpoint_config(self, EndpointConfigName):
        return self._describe("config", EndpointConfigName)

    def describe_endpoint(self, EndpointName):
        return self._describe("endpoint", EndpointName)

    def list_tags(self, ResourceArn):
        kind, name = ResourceArn.split(":")[-1].split("/")
        return {"Tags": self.resources[kind][name]["Tags"]}

    def delete_endpoint(self, EndpointName):
        self.calls.append(("delete_endpoint", EndpointName))
        del self.resources["endpoint"][EndpointName]

    def delete_endpoint_config(self, EndpointConfigName):
        self.calls.append(("delete_config", EndpointConfigName))
        del self.resources["config"][EndpointConfigName]

    def delete_model(self, ModelName):
        self.calls.append(("delete_model", ModelName))
        del self.resources["model"][ModelName]

    def invoke_endpoint(self, **kwargs):
        payload = json.loads(kwargs["Body"])
        return {"Body": io.BytesIO(json.dumps(self.engine.predict(payload)).encode())}


@pytest.fixture(params=["torch", "trl"])
def trained(tmp_path, monkeypatch, request):
    """Run the actual local optimizer and safetensors export without Hub access."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast

    prepared(tmp_path)
    remote = Cloud()
    provider = aws.SageMakerProvider(session_factory=lambda _: remote)
    run = tmp_path / "run"
    provider.train(tmp_path, run, "sft-test", aws.SageMakerSFTSettings(aws=config(), max_epochs=1, batch_size=2, trainer_backend=request.param), confirm=True)

    def model_factory(name, **_kwargs):
        model = tiny_qwen()
        model.config._name_or_path = name
        model.name_or_path = name
        return model

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<unk>": 0, "hello": 1, "<|im_end|>": 2, **{f"token{i}": i for i in range(3, 32)}}, unk_token="<unk>")),
        unk_token="<unk>", eos_token="<|im_end|>", pad_token="<|im_end|>")
    renderer = SimpleNamespace(
        tokenizer=tokenizer, prompt_tokens=lambda messages, tools: [1, 3, 4],
        render=lambda messages, tools=None, final_target=False: [SimpleNamespace(token_ids=[1, 3, 4, 2], token_weights=[0, 1, 0, 1])],
    )
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", model_factory)
    monkeypatch.setattr(worker, "load_training_renderer", lambda _: renderer)
    monkeypatch.setattr(serving, "load_training_renderer", lambda _: renderer)
    output = tmp_path / "model"
    worker.run(run / "input", output, tmp_path / "checkpoints", device="cpu")
    verify_artifact(output)
    archive = tmp_path / "model.tar.gz"
    package_artifact(output, archive)
    remote.objects[("smithtune-test", "jobs/sft-test/output/sft-test/output/model.tar.gz")] = archive.read_bytes()
    remote.completed = True
    collect(run, provider=provider)
    return run, remote, provider


def settings():
    return deploy.EndpointSettings(config().training_image, "ml.g6.4xlarge", 1.654, 64, max_new_tokens=4)


def test_cpu_train_collect_deploy_reload_invoke_cleanup(trained):
    run, cloud, provider = trained
    collected = collect(run, provider=provider)
    assert collected["status"] == "verified"
    assert collected["artifacts"]["selected_model_uri"] == (run / "artifact/adapter").resolve().as_uri()
    assert collected["artifacts"]["metrics_uri"] == (run / "artifact/result.json").resolve().as_uri()
    assert collected["artifacts"]["result_manifest_uri"] == (run / "artifact/artifact.json").resolve().as_uri()
    assert collected["artifacts"]["resume_checkpoint_uri"] is None
    proposal = deploy.plan(run, settings(), "tiny-pilot")
    assert proposal["cost"]["continuous_billing"]
    assert "InferenceAmiVersion" not in proposal["requests"]["config"]["ProductionVariants"][0]
    assert "VariantInstanceProvisionTimeoutInSeconds" not in proposal["requests"]["config"]["ProductionVariants"][0]
    assert "InstancePools" not in proposal["requests"]["config"]["ProductionVariants"][0]
    assert "instance_preferences" not in proposal["settings"]
    deploy.deploy(run, settings(), "tiny-pilot", confirm=True, session_factory=lambda _: cloud)
    assert deploy.status(run, session_factory=lambda _: cloud)["status"] == "InService"
    cloud.engine = serving.InferenceEngine.load(run / "artifact", device="cpu", max_context_tokens=64, max_new_tokens=4)
    for variant in ("base", "tuned"):
        value = deploy.invoke(run, {"messages": [{"role": "user", "content": "hello"}], "variant": variant}, session_factory=lambda _: cloud)
        assert value["source_run_id"] == "sft-test"
        assert value["variant"] == variant
        assert value["usage"]["completion_tokens"] > 0
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleting"
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleted"
    assert not any(cloud.resources.values())
    assert (run / "artifact/adapter/adapter_model.safetensors").exists()


def test_explicit_gpu_host_runtime_and_provision_wait_reach_aws(trained):
    run, cloud, _ = trained
    selected = replace(settings(), inference_ami_version="al2023-ami-sagemaker-inference-gpu-4-1",
                       provision_timeout_seconds=300)
    deploy.deploy(run, selected, "explicit-runtime", confirm=True, session_factory=lambda _: cloud)
    variant = cloud.resources["config"]["explicit-runtime"]["ProductionVariants"][0]
    assert variant["InferenceAmiVersion"] == "al2023-ami-sagemaker-inference-gpu-4-1"
    assert variant["VariantInstanceProvisionTimeoutInSeconds"] == 300
    assert deploy.status(run, session_factory=lambda _: cloud)["status"] == "InService"
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleted"


@pytest.mark.parametrize("changes", [
    {"inference_ami_version": "al2-ami-sagemaker-inference-neuron-2"},
    {"inference_ami_version": "unknown"},
    {"inference_ami_version": []},
    {"provision_timeout_seconds": True}, {"provision_timeout_seconds": 299},
    {"provision_timeout_seconds": 3601}, {"provision_timeout_seconds": 300.5},
])
def test_invalid_host_runtime_or_provision_limit_rejected_before_aws(changes):
    with pytest.raises(PipelineError, match="inference_ami_version|provision_timeout_seconds"):
        replace(settings(), **changes).validate(config())


def test_failed_endpoint_preserves_aws_reason_without_requiring_logs(trained):
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "capacity-failure", confirm=True, session_factory=lambda _: cloud)
    cloud.resources["endpoint"]["capacity-failure"].update(
        EndpointStatus="Failed", FailureReason="InsufficientInstanceCapacity: requested capacity unavailable")
    receipt = deploy.wait(run, timeout=1, session_factory=lambda _: cloud)
    assert receipt["status"] == "Failed"
    assert receipt["failure_reason"].startswith("InsufficientInstanceCapacity")
    assert _load_json(run / "endpoint/receipt.json")["failure_reason"] == receipt["failure_reason"]


def test_endpoint_pools_preserve_priority_rates_recovery_and_cleanup(trained):
    run, cloud, _ = trained
    config_file = run / "hosting.json"
    _json_dump(config_file, {
        "serving_image": settings().serving_image, "max_context_tokens": 64, "max_new_tokens": 4,
        "provision_timeout_seconds": 300,
        "instance_preferences": [
            {"instance_type": "ml.g6.4xlarge", "hourly_rate_usd": 1.654},
            {"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": 2.03},
        ],
    })
    selected = deploy.EndpointSettings.read(config_file)
    proposal = deploy.plan(run, selected, "pooled")
    variant = proposal["requests"]["config"]["ProductionVariants"][0]
    assert "InstanceType" not in variant
    assert variant["InitialInstanceCount"] == 1
    assert variant["InstancePools"] == [
        {"InstanceType": "ml.g6.4xlarge", "Priority": 1},
        {"InstanceType": "ml.g5.4xlarge", "Priority": 2},
    ]
    assert proposal["cost"]["hourly_compute_usd"] == 2.03
    assert proposal["cost"]["instance_hourly_rates_usd"] == {"ml.g6.4xlarge": 1.654, "ml.g5.4xlarge": 2.03}
    deploy.deploy(run, selected, "pooled", confirm=True, session_factory=lambda _: cloud)
    cloud.resources["endpoint"]["pooled"]["ProductionVariants"] = [{"VariantName": "AllTraffic", "InstancePools": [
        {"InstanceType": "ml.g6.4xlarge", "CurrentInstanceCount": 0},
        {"InstanceType": "ml.g5.4xlarge", "CurrentInstanceCount": 1},
    ]}]
    assert deploy.status(run, session_factory=lambda _: cloud)["provisioned_instances"] == {
        "ml.g6.4xlarge": 0, "ml.g5.4xlarge": 1,
    }
    # JSON receipts must compare identically after normalization; no duplicate endpoint.
    deploy.deploy(run, selected, "pooled", confirm=True, session_factory=lambda _: cloud)
    assert sum(call[0] == "create_endpoint" for call in cloud.calls) == 1
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleted"
    assert not any(cloud.resources.values())


@pytest.mark.parametrize("changes", [
    {"instance_type": "ml.g6.4xlarge"}, {"hourly_rate_usd": 1.654},
    {"provision_timeout_seconds": None}, {"instance_preferences": "ml.g5.4xlarge"},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge"}]},
    {"instance_preferences": [None]},
    {"instance_preferences": [{"instance_type": "ml.c5.4xlarge", "hourly_rate_usd": 1}]},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": None}]},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": True}]},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": float("nan")}]},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": 1}] * 2},
    {"instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": 1}] * 6},
])
def test_invalid_endpoint_pools_rejected_before_aws(changes):
    values = {"instance_type": None, "hourly_rate_usd": None, "provision_timeout_seconds": 300,
              "instance_preferences": [{"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": 2.03}]} | changes
    with pytest.raises(PipelineError):
        replace(settings(), **values).validate(config())


def test_uncertain_create_recovers_same_resources_without_duplicate(trained):
    run, cloud, _ = trained
    cloud.lose_create_response = True
    with pytest.raises(PipelineError, match="submission incomplete"):
        deploy.deploy(run, settings(), "uncertain", confirm=True, session_factory=lambda _: cloud)
    cloud.lose_create_response = False
    deploy.deploy(run, settings(), "uncertain", confirm=True, session_factory=lambda _: cloud)
    assert sum(call[0] == "create_endpoint" for call in cloud.calls) == 1
    assert deploy.status(run, session_factory=lambda _: cloud)["status"] == "InService"


def test_ownership_checks_block_deleting_replaced_resource(trained):
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "owned", confirm=True, session_factory=lambda _: cloud)
    cloud.resources["endpoint"]["owned"]["Tags"] = [{"Key": "smithtune:deployment", "Value": "somebody-else"}]
    with pytest.raises(PipelineError, match="not owned"):
        deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert "owned" in cloud.resources["endpoint"]
    assert not any(call[0].startswith("delete_") for call in cloud.calls)


def test_deploy_requires_confirmation_and_rejects_tampered_export(trained):
    run, _cloud, _ = trained
    with pytest.raises(PipelineError, match="billed"):
        deploy.deploy(run, settings(), "blocked", session_factory=lambda _: pytest.fail("AWS contacted"))
    path = run / "artifact/adapter/adapter_model.safetensors"
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(PipelineError, match="checksums"):
        deploy.plan(run, settings(), "blocked")


@pytest.mark.parametrize("name,kind", [("../outside", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)])
def test_untrusted_archive_cannot_escape_or_create_links(tmp_path, name, kind):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.linkname = "../../outside"
        bundle.addfile(item, io.BytesIO())
    with pytest.raises(PipelineError, match="unsafe"):
        extract_archive(path, tmp_path / "out")
    assert not (tmp_path / "outside").exists()


def test_legacy_artifact_upgrade_requires_original_provenance(trained, tmp_path):
    run, cloud, provider = trained
    local = run / "artifact"
    (local / "artifact.json").unlink()
    archive = tmp_path / "legacy.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for file in local.rglob("*"):
            if file.is_file():
                bundle.add(file, arcname=file.relative_to(local))
    cloud.objects[("smithtune-test", "jobs/sft-test/output/sft-test/output/model.tar.gz")] = archive.read_bytes()
    shutil.rmtree(local)
    collect(run, provider=provider)
    assert verify_artifact(local)[0]["run_id"] == "sft-test"
    result = _load_json(local / "result.json")
    result["run_id"] = "wrong-run"
    _json_dump(local / "result.json", result)
    write_artifact_manifest(local)
    with pytest.raises(PipelineError, match="training inputs"):
        collect(run, provider=provider)


def test_queue_cutoff_stops_only_pending_jobs_and_reports_cleanup_pending(tmp_path):
    calls, now = [], [0]
    def status(_run, **kwargs):
        calls.append(kwargs)
        return {"status": "Stopping" if kwargs.get("stop") else "InProgress", "secondary_status": "Pending"}
    def sleep(seconds):
        now[0] += seconds
    value = wait_for_training(tmp_path, timeout=2, stop_if_queued=True, confirm=True,
                              provider=SimpleNamespace(status=status), sleep=sleep, clock=lambda: now[0])
    assert value["status"] == "Stopping" and not value["cleanup_complete"]
    assert calls[-1] == {"stop": True, "confirm": True}


def test_cli_deploy_plan_is_offline_and_mutations_are_guarded(trained, tmp_path, capsys):
    run, _, _ = trained
    path = tmp_path / "endpoint.json"
    _json_dump(path, asdict(settings()))
    args = ["--run-dir", str(run), "--endpoint-config", str(path), "--endpoint-name", "cli-pilot"]
    cli.main(["aws", "deploy-plan", *args])
    assert json.loads(capsys.readouterr().out)["endpoint_name"] == "cli-pilot"
    with pytest.raises(SystemExit):
        cli.main(["aws", "deploy", *args])


@pytest.mark.parametrize("failure", [None, "invoke", "readiness"])
def test_temporary_endpoint_smoke_always_cleans_up(trained, monkeypatch, failure):
    run, cloud, _ = trained
    cloud.engine = SimpleNamespace(predict=lambda payload: {
        "source_run_id": "sft-test", "variant": payload["variant"], "message": {"role": "assistant", "content": "hello"}})
    if failure == "invoke":
        def fail(**_kwargs):
            raise TimeoutError("lost invocation response")
        monkeypatch.setattr(cloud, "invoke_endpoint", fail)
    real_wait = deploy.wait
    def immediate_wait(*args, deleting=False, **kwargs):
        if failure == "readiness" and not deleting:
            return {"status": "Failed"}
        return real_wait(*args, deleting=deleting, sleep=lambda _: None, **kwargs)
    monkeypatch.setattr(deploy, "wait", immediate_wait)
    kwargs = {"confirm": True, "timeout": 1, "session_factory": lambda _: cloud}
    if failure:
        with pytest.raises(PipelineError, match="resources cleaned up"):
            deploy.smoke(run, settings(), "smoke-pilot", {"messages": [{"role": "user", "content": "hello"}]}, **kwargs)
    else:
        result = deploy.smoke(run, settings(), "smoke-pilot", {"messages": [{"role": "user", "content": "hello"}]}, **kwargs)
        assert result["cleanup_complete"] and len(result["generations"]) == 2
        assert not result["all_formats_valid"]  # A response alone does not prove native format validity.
    assert not any(cloud.resources.values())
    assert _load_json(run / "endpoint/smoke-result.json")["cleanup_complete"]


def test_smoke_reports_pending_deletion_without_claiming_cleanup(trained, monkeypatch):
    run, cloud, _ = trained
    monkeypatch.setattr(deploy, "wait", lambda *args, **kwargs: {"status": "Deleting" if kwargs.get("deleting") else "Failed"})
    with pytest.raises(PipelineError, match="cleanup is incomplete"):
        deploy.smoke(run, settings(), "pending", {"messages": [{"role": "user", "content": "hello"}]},
                     confirm=True, timeout=1, session_factory=lambda _: cloud)
    assert not _load_json(run / "endpoint/smoke-result.json")["cleanup_complete"]
    assert "pending" in cloud.resources["endpoint"]


def test_new_endpoint_after_deletion_preserves_previous_receipt(trained):
    run, cloud, _ = trained
    old = deploy.deploy(run, settings(), "first", confirm=True, session_factory=lambda _: cloud)
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    with pytest.raises(PipelineError, match="new endpoint name"):
        deploy.deploy(run, settings(), "first", confirm=True, session_factory=lambda _: cloud)
    new = deploy.deploy(run, settings(), "second", confirm=True, session_factory=lambda _: cloud)
    assert old["owner"] != new["owner"]
    archived = _load_json(run / "endpoint/history" / old["owner"] / "receipt.json")
    assert archived["status"] == "Deleted" and archived["plan"]["endpoint_name"] == "first"
    assert set(cloud.resources["endpoint"]) == {"second"}


def test_cleanup_does_not_assume_absence_after_unacknowledged_create(trained, monkeypatch):
    run, cloud, _ = trained
    create = cloud.create_endpoint
    def timeout(**_kwargs):
        raise TimeoutError("AWS may still be processing create")
    monkeypatch.setattr(cloud, "create_endpoint", timeout)
    with pytest.raises(PipelineError, match="submission incomplete"):
        deploy.deploy(run, settings(), "ambiguous", confirm=True, session_factory=lambda _: cloud)
    receipt = deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert receipt["status"] == "CleanupUncertain"
    assert deploy.status(run, session_factory=lambda _: cloud)["status"] == "CleanupUncertain"
    # If the delayed resource becomes visible, it is found by its original
    # name/ownership tag and can be cleaned; no replacement gets submitted.
    create(**(receipt["plan"]["requests"]["endpoint"] | {"Tags": [{"Key": "smithtune:deployment", "Value": receipt["owner"]}]}))
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleted"


def test_owned_but_modified_endpoint_configuration_blocks_inference(trained):
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "modified", confirm=True, session_factory=lambda _: cloud)
    cloud.resources["config"]["modified"]["ProductionVariants"][0]["InstanceType"] = "ml.p5.48xlarge"
    with pytest.raises(PipelineError, match="differs"):
        deploy.invoke(run, {"messages": []}, session_factory=lambda _: cloud)


def test_acknowledged_but_temporarily_invisible_endpoint_is_not_declared_deleted(trained, monkeypatch):
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "invisible", confirm=True, session_factory=lambda _: cloud)
    original = cloud.describe_endpoint
    monkeypatch.setattr(cloud, "describe_endpoint", lambda **_kwargs: (_ for _ in ()).throw(MissingJob()))
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "CleanupUncertain"
    assert "invisible" in cloud.resources["endpoint"]
    monkeypatch.setattr(cloud, "describe_endpoint", original)
    deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == "Deleted"


def test_serving_never_truncates_and_rejects_invalid_generation_settings(trained):
    run, _cloud, _ = trained
    engine = serving.InferenceEngine.load(run / "artifact", device="cpu", max_context_tokens=6, max_new_tokens=4)
    request = {"messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(PipelineError, match="no truncation"):
        engine.predict(request)  # 3 prompt tokens + 4 requested outputs > 6
    for invalid in (True, -1, 5, 1.5):
        with pytest.raises(PipelineError, match="max_new_tokens"):
            engine.predict(request | {"max_new_tokens": invalid})
    for bad in ({"messages": []}, request | {"variant": "wrong"}, request | {"temperature": 0.7}):
        with pytest.raises(PipelineError):
            engine.predict(bad)
    assert engine.predict(request | {"max_new_tokens": 2})["usage"]["completion_tokens"] <= 2


@pytest.mark.parametrize("retry_attempts,expected", [(0, "Deleted"), (1, "CleanupUncertain")])
def test_definitive_service_rejection_is_distinct_from_retried_create(trained, monkeypatch, retry_attempts, expected):
    from botocore.exceptions import ClientError
    run, cloud, _ = trained
    def reject(**_kwargs):
        raise ClientError({"Error": {"Code": "ValidationException", "Message": "Unsupported manifest media type"},
                           "ResponseMetadata": {"RetryAttempts": retry_attempts, "RequestId": "known-rejection"}}, "CreateModel")
    monkeypatch.setattr(cloud, "create_model", reject)
    with pytest.raises(PipelineError, match="submission incomplete"):
        deploy.deploy(run, settings(), "rejected", confirm=True, session_factory=lambda _: cloud)
    assert deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)["status"] == expected
    assert not any(cloud.resources.values())


def test_oci_image_rejected_before_model_creation(trained):
    run, cloud, _ = trained
    cloud.image_media_type = "application/vnd.oci.image.manifest.v1+json"
    with pytest.raises(PipelineError, match="Docker V2"):
        deploy.deploy(run, settings(), "bad-format", confirm=True, session_factory=lambda _: cloud)
    assert not any(cloud.resources.values())
    assert not _load_json(run / "endpoint/receipt.json")["attempted"]


def test_cleanup_wait_retries_aws_in_progress_rejection_and_preserves_dependencies(trained, monkeypatch):
    from botocore.exceptions import ClientError
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "still-creating", confirm=True, session_factory=lambda _: cloud)
    cloud.resources["endpoint"]["still-creating"]["EndpointStatus"] = "Creating"
    real_delete, attempts, now = cloud.delete_endpoint, [], [0]

    def reject_until_ready(**kwargs):
        attempts.append(kwargs)
        if cloud.resources["endpoint"]["still-creating"]["EndpointStatus"] == "Creating":
            raise ClientError({"Error": {"Code": "ValidationException", "Message":
                               'Cannot update in-progress endpoint "example".'}}, "DeleteEndpoint")
        return real_delete(**kwargs)

    monkeypatch.setattr(cloud, "delete_endpoint", reject_until_ready)
    receipt = deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert receipt["status"] == "CleanupPending"
    assert "endpoint" not in receipt["delete_requested"]
    assert all(cloud.resources.values())
    assert not any(call[0].startswith("delete_") for call in cloud.calls)

    # The rejected delete must not turn temporary invisibility into proven cleanup.
    describe = cloud.describe_endpoint
    monkeypatch.setattr(cloud, "describe_endpoint", lambda **_kwargs: (_ for _ in ()).throw(MissingJob()))
    assert deploy.status(run, session_factory=lambda _: cloud)["status"] == "CleanupUncertain"
    monkeypatch.setattr(cloud, "describe_endpoint", describe)

    def sleep(seconds):
        now[0] += seconds
    value = deploy.wait(run, timeout=1, deleting=True, session_factory=lambda _: cloud,
                        sleep=sleep, clock=lambda: now[0])
    assert value["status"] == "CleanupPending" and value["wait_timed_out"]
    assert all(cloud.resources.values())

    def become_ready(seconds):
        now[0] += seconds
        if "still-creating" in cloud.resources["endpoint"]:
            cloud.resources["endpoint"]["still-creating"]["EndpointStatus"] = "InService"
    value = deploy.wait(run, timeout=30, deleting=True, session_factory=lambda _: cloud,
                        sleep=become_ready, clock=lambda: now[0])
    assert value["status"] == "Deleted"
    assert "cleanup_pending_reason" not in value
    assert not any(cloud.resources.values())
    assert len(attempts) >= 3


@pytest.mark.parametrize("code,message", [("AccessDeniedException", "Not authorized"),
                                         ("ValidationException", "Unrelated invalid request")])
def test_cleanup_does_not_swallow_other_service_errors(trained, monkeypatch, code, message):
    from botocore.exceptions import ClientError
    run, cloud, _ = trained
    deploy.deploy(run, settings(), "delete-error", confirm=True, session_factory=lambda _: cloud)
    def reject(**_kwargs):
        raise ClientError({"Error": {"Code": code, "Message": message}}, "DeleteEndpoint")
    monkeypatch.setattr(cloud, "delete_endpoint", reject)
    with pytest.raises(PipelineError, match="cleanup incomplete"):
        deploy.undeploy(run, confirm=True, session_factory=lambda _: cloud)
    assert all(cloud.resources.values())

"""Recover completed job artifacts and package verified local exports for serving."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from smithtune.artifacts import _json_dump, _load_json, output_lock
from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker import AWSJobConfig, SageMakerProvider, split_s3_uri
from smithtune.training.contracts import artifact_files, read_result, sha256_file, verify_artifact, write_artifact_manifest
from smithtune.training.spec import TrainingArtifacts

MAX_ARTIFACT_BYTES = 2 * 1024**3


def _collected_artifacts(target: Path, result: dict) -> dict:
    """Portable handles to verified local outputs; restart loading is not implemented."""
    resolved = target.resolve()
    artifacts = TrainingArtifacts(
        selected_model_uri=(resolved / "adapter").as_uri(), resume_checkpoint_uri=None,
        metrics_uri=(resolved / "result.json").as_uri(), result_manifest_uri=(resolved / "artifact.json").as_uri(),
    )
    return {"status": "verified", "artifact_dir": str(target), "source_run_id": result["run_id"],
            "artifacts": asdict(artifacts)}


def extract_archive(archive: Path, destination: Path):
    """Bounded regular-file extraction; no links, devices or path traversal."""
    total, count, seen = 0, 0, set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = Path(member.name)
            count += 1
            if (count > 10000 or path.is_absolute() or ".." in path.parts or member.issym()
                    or member.islnk() or not (member.isdir() or member.isfile()) or member.name in seen):
                raise PipelineError("unsafe or oversized model archive")
            seen.add(member.name)
            total += member.size
            if total > MAX_ARTIFACT_BYTES:
                raise PipelineError("model archive exceeds the adapter artifact size limit")
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise PipelineError("unreadable model artifact member")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def collect(run_dir: Path, *, provider=None) -> dict:
    provider = provider or SageMakerProvider()
    receipt = provider.status(run_dir)
    if receipt["status"] != "Completed" or not receipt.get("model_artifacts"):
        raise PipelineError("artifact collection requires a completed training job")
    config = AWSJobConfig(**receipt["aws"])
    expected_bucket, expected_prefix = split_s3_uri(config.s3_prefix)
    bucket, key = split_s3_uri(receipt["model_artifacts"])
    if bucket != expected_bucket or not key.startswith(f"{expected_prefix}/{receipt['job_name']}/output/"):
        raise PipelineError("AWS returned an artifact outside this job's output prefix")
    with output_lock(run_dir):
        target = run_dir / "artifact"
        if target.exists():
            result, _spec = verify_artifact(target)
            _verify_origin(result, run_dir, receipt)
            return _collected_artifacts(target, result)
        with tempfile.TemporaryDirectory(prefix=".collect-", dir=run_dir) as scratch:
            scratch = Path(scratch)
            archive, extracted = scratch / "model.tar.gz", scratch / "artifact"
            extracted.mkdir()
            client = provider._session(config).client("s3")
            # Bound the compressed download too; don't download arbitrary model weights.
            head = client.head_object(Bucket=bucket, Key=key)
            if not 0 < head["ContentLength"] <= MAX_ARTIFACT_BYTES:
                raise PipelineError("unexpected model archive size")
            client.download_file(bucket, key, str(archive))
            if archive.stat().st_size != head["ContentLength"]:
                raise PipelineError("model archive changed during download")
            extract_archive(archive, extracted)
            result, _spec = read_result(extracted)
            _verify_origin(result, run_dir, receipt)
            if (extracted / "artifact.json").exists():
                verify_artifact(extracted)
            else:
                # Earlier smoke exports predate per-file manifests. Validate their
                # exact training provenance before upgrading the local export.
                write_artifact_manifest(extracted)
            _json_dump(scratch / "collection.json", {
                "schema_version": 1, "source_uri": receipt["model_artifacts"],
                "archive_sha256": sha256_file(archive), "artifact_sha256": sha256_file(extracted / "artifact.json"),
            })
            os.replace(extracted, target)
            os.replace(scratch / "collection.json", run_dir / "collection.json")
        return _collected_artifacts(target, result)


def _verify_origin(result, run_dir, receipt):
    bundle = _load_json(run_dir / "input/bundle.json")
    if (result.get("run_id") != receipt["job_name"] or result.get("bundle") != bundle
            or result.get("model") != _load_json(run_dir / "input/prepared/manifest.json")["model"]):
        raise PipelineError("model export does not match the saved training inputs")
    for name, digest in bundle["files"].items():
        if sha256_file(run_dir / "input/prepared" / name) != digest:
            raise PipelineError("saved training input was changed")


def package_artifact(directory: Path, target: Path):
    verify_artifact(directory)
    with tarfile.open(target, "w:gz") as archive:
        for name in [*artifact_files(directory), "artifact.json"]:
            archive.add(directory / name, arcname=name, recursive=False)

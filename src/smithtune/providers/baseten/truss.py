"""Pinned adapter for Baseten's checkpoint inference-template implementation.

Truss exposes this through internal CLI modules, so its version is checked before
use. The raw mutation result must reach the lifecycle owner before parsing the
generated config: Truss's higher-level helper parses config after provisioning
but before returning resource IDs. The provider chooses the serving template;
this adapter does not configure context limits or tool parsers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import logging
import os
import threading
from typing import Any

from smithtune.providers.base import PipelineError


TRUSS_VERSION = "0.18.30"


class _OtherThreadsOnly(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread_id


@contextmanager
def _suppress_provider_response_logs() -> Iterator[None]:
    # Truss logs complete failed HTTP response bodies before raising. The caller
    # receives sanitized errors; unrelated threads keep their existing logging.
    logger = logging.getLogger("truss.remote.baseten.api")
    log_filter = _OtherThreadsOnly()
    logger.addFilter(log_filter)
    try:
        yield
    finally:
        logger.removeFilter(log_filter)


def prepare_deployment(
    *,
    checkpoint_id: str,
    model_name: str,
    accelerator: str,
) -> Callable[[], dict[str, Any]]:
    """Read instance availability now; return a single-use create operation.

The caller persists deployment intent before invoking the returned callable and
persists its raw model/deployment IDs before inspecting ``truss_config``. A
creation failure can have an unknown server-side outcome and must not be retried
automatically. Credentials are read from BASETEN_API_KEY without using .trussrc.
"""
    try:
        installed = version("truss")
    except PackageNotFoundError:
        raise PipelineError(
            "Baseten deployment requires the baseten-deploy extra "
            f"(truss=={TRUSS_VERSION})."
        ) from None
    if installed != TRUSS_VERSION:
        raise PipelineError(
            f"Baseten deployment requires truss=={TRUSS_VERSION}; "
            "install the baseten-deploy extra."
        )
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise PipelineError("BASETEN_API_KEY is not set")
    if not all(value.strip() for value in (checkpoint_id, model_name, accelerator)):
        raise PipelineError("Checkpoint, model name, and accelerator are required")
    parts = accelerator.split(":")
    if len(parts) > 2 or (len(parts) == 2 and (not parts[1].isdigit() or int(parts[1]) < 1)):
        raise PipelineError("Accelerator must be a GPU type with an optional positive count, e.g. H200:1")
    try:
        from truss.base.truss_config import Accelerator, AcceleratorSpec
        from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
            _build_inference_template_request,
        )
        from truss.cli.train.types import DeployCheckpointsConfigComplete
        from truss.remote.baseten.remote import BasetenRemote
        from truss_train.definitions import (
            CheckpointList,
            Compute,
            DeployCheckpointsRuntime,
        )

        remote = BasetenRemote("https://app.baseten.co", api_key=api_key)
        config = DeployCheckpointsConfigComplete(
            checkpoint_details=CheckpointList(loops_checkpoint_ids=[checkpoint_id]),
            model_name=model_name,
            compute=Compute(
                accelerator=AcceleratorSpec(
                    accelerator=Accelerator(parts[0]),
                    count=int(parts[1]) if len(parts) == 2 else 1,
                )
            ),
            runtime=DeployCheckpointsRuntime(environment_variables={}),
        )
        with _suppress_provider_response_logs():
            request = _build_inference_template_request(config, remote, dry_run=False)
            selected = next(
                (instance for instance in remote.api.get_instance_types() if instance.id == request["instance_type_id"]),
                None,
            )
    except Exception:
        raise PipelineError(
            "Could not prepare the Baseten deployment request. Check the pinned "
            "baseten-deploy dependencies, credentials, and accelerator availability."
        ) from None

    # Truss may select a larger instance when the requested count is unavailable.
    # A paid create must stay within the exact GPU allocation the user requested.
    if (
        selected is None
        or selected.node_count != 1
        or selected.gpu_count != (int(parts[1]) if len(parts) == 2 else 1)
        or selected.gpu_type != parts[0]
    ):
        raise PipelineError("The requested exact GPU allocation is unavailable; refusing a different Baseten instance")

    used = False

    def create() -> dict[str, Any]:
        nonlocal used
        if used:
            raise PipelineError("This Baseten deployment create operation has already been attempted")
        used = True
        try:
            with _suppress_provider_response_logs():
                result = remote.api.create_model_version_from_inference_template(request)
        except Exception:
            raise PipelineError(
                "Baseten deployment creation failed; its server-side outcome is unknown. "
                "Reconcile the saved deployment intent before attempting another deployment."
            ) from None
        if not isinstance(result, dict):
            raise PipelineError(
                "Baseten returned an invalid deployment result; its server-side outcome is unknown. "
                "Reconcile the saved deployment intent before attempting another deployment."
            )
        return result

    return create

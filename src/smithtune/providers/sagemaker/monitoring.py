"""Bounded waiting that never submits a replacement training job."""
from __future__ import annotations

import math
import time

from smithtune.providers.base import PipelineError
from smithtune.providers.sagemaker import SageMakerProvider, TERMINAL_STATUSES


def wait_for_training(run_dir, *, timeout=600, stop_if_queued=False, confirm=False,
                      provider=None, sleep=time.sleep, clock=time.monotonic):
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise PipelineError("training wait timeout must be in (0, 3600]")
    if stop_if_queued and not confirm:
        raise PipelineError("stopping a queued job requires --confirm")
    provider = provider or SageMakerProvider()
    deadline = clock() + timeout
    while True:
        receipt = provider.status(run_dir)
        if receipt["status"] in TERMINAL_STATUSES:
            return receipt
        remaining = deadline - clock()
        if remaining <= 0:
            if stop_if_queued and receipt.get("secondary_status") == "Pending" and not receipt.get("training_started_at"):
                receipt = provider.status(run_dir, stop=True, confirm=True)
            return receipt | {"wait_timed_out": True, "cleanup_complete": receipt["status"] in TERMINAL_STATUSES}
        sleep(min(10, remaining))

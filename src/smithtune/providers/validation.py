"""Optimizer-step validation schedules shared by training adapters."""

import math


def training_schedule(
    training_units: int,
    batch_size: int,
    max_epochs: int,
    validation_every_steps: int | None,
) -> dict:
    """Count interval and epoch-end checks without double-counting boundaries.

    The count assumes all requested epochs complete. Early stopping or a spend
    limit can shorten the run. Each adapter supplies its own batching unit.
    """
    steps = math.ceil(training_units / batch_size)
    total = steps * max_epochs
    passes = max_epochs if steps else 0
    if steps and validation_every_steps is not None:
        passes += total // validation_every_steps - total // math.lcm(steps, validation_every_steps)
    return {
        "steps_per_epoch": steps,
        "max_optimizer_steps": total,
        "expected_validation_passes": passes,
        "step_unit": "optimizer_updates",
        "early_stopping_patience_unit": "epochs",
    }

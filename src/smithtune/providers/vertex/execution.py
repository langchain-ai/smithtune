"""Execution inputs for a single-machine Vertex CustomJob.

This module describes compute independently of the HF SFT recipe. Live job
submission, identity reconciliation, monitoring, and cancellation are not
implemented. There is no Google SDK import or resource allocation here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VertexExecutionSpec:
    """Caller-supplied deployment settings; not a validated capacity claim."""

    project: str
    location: str
    image_uri: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    service_account: str
    output_uri: str
    max_runtime_seconds: int

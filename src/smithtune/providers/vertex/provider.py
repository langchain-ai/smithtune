"""Compose an HF training description with Vertex execution settings.

This provider scaffold is deliberately absent from the supported-provider
registry. A qualified model profile and the prepare/plan/train lifecycle are
required before exposing Vertex through the CLI. Fireworks and Baseten retain
their existing TrainingProvider implementations.
"""

from dataclasses import dataclass, field

from smithtune.providers.vertex.execution import VertexExecutionSpec
from smithtune.training.spec import SFTJobSpec


@dataclass(frozen=True)
class VertexTrainingPlan:
    """Composition of training intent and compute configuration, without I/O."""

    training: SFTJobSpec
    execution: VertexExecutionSpec

    provider: str = field(default="vertex", init=False)
    implementation: str = field(default="hf_trl", init=False)

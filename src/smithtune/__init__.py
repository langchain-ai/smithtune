"""Fine-tune models on LangSmith trajectories."""

from importlib.metadata import version


def get_version() -> str:
    """Read the version from the installed distribution."""
    return version("smithtune")

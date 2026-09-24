"""Offline installation diagnostics. Never read or display credential values."""

import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


INSTALL_HELP = {
    "langsmith": "Install the LangSmith CLI with `curl -fsSL https://cli.langsmith.com/install.sh | sh`, then ensure langsmith is on PATH",
    "firectl": "Install firectl: https://docs.fireworks.ai/tools-sdks/firectl/firectl and ensure firectl is on PATH",
}


CREDENTIALS = {
    "LANGSMITH_API_KEY": "every workflow: dataset access, split publication, experiments",
    "BASETEN_API_KEY": "Baseten training and evaluation; the default triage council and default evaluation judge",
    "FIREWORKS_API_KEY": "Fireworks training and evaluation; Fireworks judges",
    "ANTHROPIC_API_KEY": "anthropic/<model> evaluation judges",
    "OPENAI_API_KEY": "the optional gpt-5.6-terra triage judge",
}


# `firectl deployment-shape-version match` first shipped in firectl 1.8.5.
MIN_FIRECTL_SHAPE_MATCH = (1, 8, 5)


def firectl_version() -> tuple[int, int, int] | None:
    """Installed firectl version, read offline from `firectl version`."""
    try:
        result = subprocess.run(["firectl", "version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    versions = re.findall(r"^\s*(\d+)\.(\d+)\.(\d+)\s*$", f"{result.stdout}\n{result.stderr}", re.M)
    return tuple(int(part) for part in versions[-1]) if versions else None


def diagnose() -> dict:
    # artifacts imports this module, and data_rights imports artifacts.
    from smithtune import data_rights

    packages = {}
    for name in ("smithtune", "fireworks-training-cookbook", "tinker-cookbook", "fireworks-ai", "baseten-loops", "jsonschema", "langsmith", "transformers", "trl", "torch", "deepagents", "langchain-openai", "pydantic-monty"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    tools = {}
    for name, operations in (
        ("langsmith", "dataset pull, dataset push, prepare (unless --no-fetch)"),
        ("firectl", "Fireworks deploy, undeploy"),
    ):
        available = shutil.which(name) is not None
        tools[name] = {"available": available, "required_for": operations}
        if not available:
            tools[name]["help"] = INSTALL_HELP[name]
    if tools["firectl"]["available"]:
        installed = firectl_version()
        tools["firectl"]["version"] = ".".join(map(str, installed)) if installed else None
        tools["firectl"]["automatic_deployment_shapes"] = bool(installed and installed >= MIN_FIRECTL_SHAPE_MATCH)
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "tools": tools,
        "credentials": {name: "set" if os.environ.get(name) else "unset" for name in CREDENTIALS},
        "credentials_required_for": CREDENTIALS,
        "data_rights_acknowledged": data_rights.saved_acknowledgment() is not None,
        "note": "Offline checks only; credential validity and service access have not been tested. Missing prerequisites are required only for the operations that use them.",
    }

"""Offline installation diagnostics. Never read or display credential values."""

import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version


INSTALL_HELP = {
    "langsmith": "Install the LangSmith CLI: https://github.com/langchain-ai/langsmith-cli and ensure langsmith is on PATH",
    "firectl": "Install firectl: https://docs.fireworks.ai/tools-sdks/firectl/firectl and ensure firectl is on PATH",
}


def diagnose() -> dict:
    packages = {}
    for name in ("smithtune", "fireworks-training-cookbook", "tinker-cookbook", "fireworks-ai", "baseten-loops", "jsonschema", "transformers", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    tools = {}
    for name, operations in (
        ("langsmith", "dataset create, capture-contract, prepare (unless --no-fetch)"),
        ("firectl", "deploy, undeploy"),
    ):
        available = shutil.which(name) is not None
        tools[name] = {"available": available, "required_for": operations}
        if not available:
            tools[name]["help"] = INSTALL_HELP[name]
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "tools": tools,
        "credentials": {
            name: "set" if os.environ.get(name) else "unset"
            for name in ("LANGSMITH_API_KEY", "FIREWORKS_API_KEY", "BASETEN_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_CUSTOM_HEADERS")
        },
        "note": "Offline checks only; credential validity and service access have not been tested. Missing prerequisites are required only for the operations that use them.",
    }

"""Build and test customer distributions in clean environments, outside the repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT, env: dict | None = None) -> None:
    # Build backends emit hundreds of file-copy lines; keep successful runs concise.
    quiet = args[:3] == ("sfw", "uv", "build")
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=quiet, text=True)
    if result.returncode and quiet:
        print(result.stdout)
        print(result.stderr)
    result.check_returncode()
    if quiet:
        print("Built distributions:", " ".join(args[3:]), flush=True)


def check(dist: Path, scratch: Path, *, full_tests: bool) -> None:
    wheels = [next(dist.glob(f"{name}-*.whl")) for name in ("smithtune", "smithtune_training_runtime")]
    venv = scratch / "venv"
    run("uv", "venv", "--python", "3.12", str(venv))
    python = str(venv / "bin/python")
    run("sfw", "uv", "pip", "install", "--python", python, *(str(p) for p in wheels), "pytest==9.1.1")
    run("uv", "pip", "check", "--python", python)
    work = scratch / "customer"
    work.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    run(str(venv / "bin/smithtune"), "--version", cwd=work, env=env)
    run(python, "-I", "-m", "smithtune", "--help", cwd=work, env=env)
    run(python, "-I", "-c", "import sys, smithtune, training; from pathlib import Path; "
        "assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) for m in (smithtune, training)); "
        "from training.recipes import sft_loop; import baseten.loops; "
        "from fireworks.training.sdk import FireworksClient", cwd=work, env=env)
    # Copy only test inputs and project metadata; never application source.
    shutil.copytree(ROOT / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, work / name)
    provenance = work / "packages/training-runtime"
    provenance.mkdir(parents=True)
    shutil.copy2(ROOT / "packages/training-runtime/upstream.json", provenance / "upstream.json")
    targets = ["tests"] if full_tests else ["tests/test_cli_installation.py", "tests/test_runtime_packaging.py"]
    run(python, "-I", "-m", "pytest", *targets, cwd=work, env=env)
    if full_tests:
        tool_env = {**env, "UV_TOOL_DIR": str(scratch / "tools"), "UV_TOOL_BIN_DIR": str(scratch / "bin")}
        run("sfw", "uv", "tool", "install", "--python", "3.12", "--with", str(wheels[1]), str(wheels[0]), cwd=work, env=tool_env)
        run(str(scratch / "bin/smithtune"), "--version", cwd=work, env=tool_env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    dist = args.dist_dir.resolve()
    if not args.skip_build:
        run("sfw", "uv", "build", "--all-packages", "--out-dir", str(dist))
    with tempfile.TemporaryDirectory(prefix="smithtune-dist-") as temporary:
        scratch = Path(temporary)
        check(dist, scratch / "wheel", full_tests=True)
        rebuilt = scratch / "rebuilt"
        for project in (ROOT, ROOT / "packages/training-runtime"):
            metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
            archive = dist / f"{metadata['name'].replace('-', '_')}-{metadata['version']}.tar.gz"
            run("sfw", "uv", "build", "--wheel", str(archive), "--out-dir", str(rebuilt))
        check(rebuilt, scratch / "sdist", full_tests=False)


if __name__ == "__main__":
    main()

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
OVERRIDES = ROOT / "overrides.txt"


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
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    wheel = dist / f"smithtune-{version}-py3-none-any.whl"
    venv = scratch / "venv"
    run("uv", "venv", "--python", "3.12", str(venv))
    python = str(venv / "bin/python")
    run("sfw", "uv", "pip", "install", "--python", python, "--overrides", str(OVERRIDES), str(wheel), "pytest==9.1.1")
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
    for name in ("README.md", "pyproject.toml", "overrides.txt"):
        shutil.copy2(ROOT / name, work / name)
    targets = ["tests"] if full_tests else ["tests/test_cli_installation.py", "tests/test_training_dependency.py"]
    run(python, "-I", "-m", "pytest", *targets, cwd=work, env=env)
    run(python, "-I", "-c", "from importlib.resources import files; root = files('smithtune'); "
        "assert all(root.joinpath(p).is_file() for p in ('skills/smithtune/SKILL.md', 'triage_prompts/coordinator.md', "
        "'triage_prompts/judge.md', 'triage_prompts/default_council.json'))", cwd=work, env=env)
    if full_tests:
        # Verify the opt-in runtime from the installed wheel, with real graphs
        # and deterministic local models. The default install is tested first.
        run("sfw", "uv", "pip", "install", "--python", python, "--overrides", str(OVERRIDES), str(wheel) + "[deepagents]")
        run(python, "-I", "-m", "pytest", "tests/test_triage_agent.py", "tests/test_triage_coordinator.py", "tests/test_triage.py", cwd=work, env=env)
        run("sfw", "uv", "pip", "install", "--python", python, "--overrides", str(OVERRIDES), str(wheel) + "[baseten-deploy]")
        run(python, "-I", "-m", "pytest", "tests/test_baseten_truss.py", "tests/test_baseten_deployment.py", "tests/test_baseten_deploy_cli.py", cwd=work, env=env)
        # Commit a clean source snapshot in a disposable repository. This tests
        # Git installation of the current working tree without committing it to
        # the developer's repository or depending on a published branch.
        source = scratch / "source"
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "CONTRIBUTING.md", "MANIFEST.in", ".gitignore", "overrides.txt"):
            shutil.copy2(ROOT / name, source / name)
        shutil.copytree(ROOT / "src/smithtune", source / "src/smithtune", ignore=shutil.ignore_patterns("__pycache__"))
        run("git", "init", "--quiet", str(source))
        run("git", "add", "pyproject.toml", "README.md", "CONTRIBUTING.md", "MANIFEST.in", ".gitignore", "overrides.txt", "src/smithtune", cwd=source)
        run("git", "diff", "--cached", "--name-only", cwd=source)
        run("git", "-c", "user.name=Distribution test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Distribution test snapshot", cwd=source)
        tool_env = {**env, "UV_TOOL_DIR": str(scratch / "tools"), "UV_TOOL_BIN_DIR": str(scratch / "bin")}
        run("sfw", "uv", "tool", "install", "--no-config", "--python", "3.12", "--overrides", str(source / "overrides.txt"), "git+" + source.as_uri(), cwd=work, env=tool_env)
        run(str(scratch / "bin/smithtune"), "--version", cwd=work, env=tool_env)
        run(str(scratch / "bin/smithtune"), "doctor", cwd=work, env=tool_env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    dist = args.dist_dir.resolve()
    if not args.skip_build:
        run("sfw", "uv", "build", "--out-dir", str(dist))
    with tempfile.TemporaryDirectory(prefix="smithtune-dist-") as temporary:
        scratch = Path(temporary)
        check(dist, scratch / "wheel", full_tests=True)
        rebuilt = scratch / "rebuilt"
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        archive = dist / f"smithtune-{metadata['version']}.tar.gz"
        run("sfw", "uv", "build", "--wheel", str(archive), "--out-dir", str(rebuilt))
        check(rebuilt, scratch / "sdist", full_tests=False)


if __name__ == "__main__":
    main()

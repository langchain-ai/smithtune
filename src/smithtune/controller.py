"""Detached, single-attempt controllers on an existing machine."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

from smithtune.artifacts import _json_dump, _load_json, _utc_now, output_lock
from smithtune.providers.base import PipelineError


RECOVERY_GUIDE = "https://github.com/langchain-ai/smithtune/blob/main/docs/reference.md#remote-controller-recovery"
TERMINAL = {"completed", "failed", "interrupted", "launch_failed"}


def directory(run_dir: Path) -> Path:
    """Keep launch artifacts outside the provider's new/empty run directory."""
    run_dir = run_dir.resolve()
    return run_dir.with_name(f"{run_dir.name}.controller")


def launch(arguments: list[str], run_dir: Path, run_id: str, provider: str) -> dict:
    if os.name != "posix":
        raise PipelineError("detached controllers require Linux or macOS")
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PipelineError(f"run directory must be new or empty: {run_dir}")
    control_dir = directory(run_dir)
    control_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # One reservation per launch, including failed or ambiguous launches.
        control_dir.mkdir(mode=0o700)
    except FileExistsError:
        raise PipelineError(f"controller already recorded for {run_dir}; inspect smithtune status --run-dir {run_dir}") from None
    receipt = {
        "schema_version": 1, "status": "starting", "run_id": run_id,
        "run_dir": str(run_dir), "provider": provider, "host": os.uname().nodename,
        "created_at_utc": _utc_now(), "automatic_restart": False,
        "log_path": str(control_dir / "controller.log"), "recovery_guide": RECOVERY_GUIDE,
    }
    try:
        _json_dump(control_dir / "request.json", {"arguments": arguments, "cwd": str(Path.cwd())})
        _json_dump(control_dir / "controller.json", receipt)
        with (control_dir / "controller.log").open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "smithtune.controller", str(control_dir)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
    except OSError as exc:
        receipt.update(status="launch_failed", ended_at_utc=_utc_now())
        _json_dump(control_dir / "controller.json", receipt)
        raise PipelineError(f"could not start controller; inspect {control_dir}") from exc
    # Only the worker writes controller.json after spawning. An uncertain
    # acknowledgment must never cause a second launch.
    return {
        **receipt, "pid": process.pid, "controller_dir": str(control_dir),
        "message": "Controller launched on this machine. Use status to check startup and progress.",
    }


def status(run_dir: Path) -> dict:
    control_dir = directory(run_dir)
    receipt_path = control_dir / "controller.json"
    if not receipt_path.is_file():
        raise PipelineError(f"no detached controller recorded for {run_dir}")
    with ExitStack() as observation:
        try:
            observation.enter_context(output_lock(control_dir))
            active = False
        except PipelineError:
            active = True
        # If the worker has exited, read its final receipt while holding the
        # lock. Reading before the lock can mistake completed work for a crash.
        receipt = _load_json(receipt_path)
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1
            or receipt.get("run_dir") != str(run_dir.resolve())
            or receipt.get("status") not in TERMINAL | {"starting", "running"}):
        raise PipelineError(f"invalid controller receipt in {control_dir}")
    result = {**receipt, "controller_active": active, "controller_dir": str(control_dir)}
    if not active and receipt["status"] == "running":
        result["status"] = "interrupted"
    elif not active and receipt["status"] == "starting":
        # Startup and death before the first worker receipt are indistinguishable.
        result["status"] = "unknown"
    result["provider_resources"] = "not_checked"
    result["artifacts"] = {
        name: str(run_dir.resolve() / name) for name in (
            "plan.json", "epochs.json", "run-state.json", "session.json", "result.json", "replay",
        ) if (run_dir / name).exists()
    }
    if result["status"] in {"failed", "interrupted", "unknown", "launch_failed"}:
        result["recovery"] = [
            "Inspect the controller log and saved run artifacts; status does not check provider resources.",
            "Reconcile and release remaining paid provider resources before starting another run.",
            "For unfinished training, initialize a NEW run from a saved resumable checkpoint and choose remaining epochs/budget explicitly.",
            "If training completed, rerun evaluate with the original provider, directories, and settings.",
            f"Manual recovery: {RECOVERY_GUIDE}",
        ]
    return result


def logs(run_dir: Path, *, tail: int = 100, follow: bool = False) -> None:
    if tail < 0:
        raise PipelineError("--tail must be non-negative")
    control_dir = directory(run_dir)
    startup_deadline = time.monotonic() + 30
    try:
        with (control_dir / "controller.log").open(encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=tail)
            sys.stdout.writelines(lines)
            sys.stdout.flush()
            while follow:
                text = handle.read()
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    continue
                observed = status(run_dir)
                if not observed["controller_active"]:
                    if observed["status"] == "unknown" and time.monotonic() < startup_deadline:
                        time.sleep(0.2)
                        continue
                    # A final write may have raced with the status read.
                    sys.stdout.write(handle.read())
                    sys.stdout.flush()
                    if observed["status"] == "unknown":
                        print("Controller startup is still unknown; inspect status and the saved log.", file=sys.stderr)
                    return
                time.sleep(0.2)
    except OSError as exc:
        raise PipelineError(f"cannot read controller log in {control_dir}") from exc


def run(control_dir: Path) -> int:
    from smithtune.cli import main
    from smithtune.providers.baseten import TerminationRequested

    # An observer may briefly hold the lock while the worker is starting.
    with output_lock(control_dir, wait=True):
        receipt = _load_json(control_dir / "controller.json")
        if receipt["status"] != "starting":
            raise PipelineError("controller already attempted; use manual recovery")
        request = _load_json(control_dir / "request.json")
        receipt.update(status="running", pid=os.getpid(), started_at_utc=_utc_now())
        _json_dump(control_dir / "controller.json", receipt)
        requested_signal = None

        def terminate(signum, _frame):
            nonlocal requested_signal
            requested_signal = signum
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, terminate)
        exit_code = 1
        try:
            os.chdir(request["cwd"])
            main(request["arguments"])
            exit_code = 0
        except SystemExit as exc:
            exit_code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
        except TerminationRequested:
            exit_code = 128 + signal.SIGTERM
        except KeyboardInterrupt:
            exit_code = 128 + (requested_signal or signal.SIGINT)
        except BaseException:
            # Keep the traceback in the controller log, never in the receipt.
            import traceback

            traceback.print_exc()
        finally:
            signal.signal(signal.SIGTERM, previous)
            receipt.update(
                status="completed" if exit_code == 0 else ("interrupted" if exit_code in (130, 143) else "failed"),
                exit_code=exit_code, ended_at_utc=_utc_now(),
            )
            _json_dump(control_dir / "controller.json", receipt)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))

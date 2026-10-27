"""SageMaker filesystem adapter for the shared training/serving image."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from smithtune.artifacts import _json_dump
from smithtune.training import hf_sft as worker


# Leave generation time for the final upload before SageMaker enforces its limit.
EVALUATION_MARGIN_SECONDS = 180


def generation_deadline(started: float, environment=os.environ) -> float | None:
    """Stop generating before the job's runtime limit so saved answers upload."""
    raw = environment.get("SMITHTUNE_MAX_RUNTIME_SECONDS")
    if raw is None:
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    return started + max(60, limit - EVALUATION_MARGIN_SECONDS)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "evaluate", "serve"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/opt/ml/checkpoints"))
    args = parser.parse_args(argv)
    if args.command == "train" and os.environ.get("SMITHTUNE_WORKER_COMMAND") == "evaluate":
        args.command = "evaluate"
    try:
        if args.command == "serve":
            from smithtune.training.serving import serve
            serve(args.model_dir)
            return
        from smithtune.training.evaluation import request_stop
        # A stop request (smithtune aws stop, or the runtime limit) finishes the
        # current answer and records the evaluation as incomplete.
        signal.signal(signal.SIGTERM, request_stop)
        deadline = generation_deadline(started)
        if args.command == "evaluate":
            worker.run_evaluation_only(args.input_dir, args.model_dir, args.checkpoint_dir, deadline=deadline)
        else:
            worker.run(args.input_dir, args.model_dir, args.checkpoint_dir, deadline=deadline)
    except Exception as exc:
        diagnostic = worker.failure_diagnostics(exc)
        print(json.dumps(diagnostic), flush=True)
        output = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
        output.mkdir(parents=True, exist_ok=True)
        _json_dump(output / "worker-failure.json", diagnostic)
        (output.parent / "failure").write_text(f"Worker failed at {diagnostic['stage']} ({diagnostic['error_kind']}); inspect worker diagnostics.\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

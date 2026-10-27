"""SageMaker filesystem adapter for the shared training/serving image."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from smithtune.artifacts import _json_dump
from smithtune.training import hf_sft as worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "serve"], default="train")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model")))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/opt/ml/checkpoints"))
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from smithtune.training.serving import serve
            serve(args.model_dir)
        else:
            worker.run(args.input_dir, args.model_dir, args.checkpoint_dir)
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

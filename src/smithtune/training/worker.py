"""Portable local-file entrypoint for the shared Hugging Face SFT recipe.

Training implementation lives in hf_sft.py. Reexports preserve earlier pilot
imports; new training code should import the recipe directly.
"""
from __future__ import annotations

from pathlib import Path

from smithtune.training.hf_sft import (
    Example as Example,
    IGNORE_INDEX as IGNORE_INDEX,
    as_example as as_example,
    configure_lora as configure_lora,
    failure_diagnostics as failure_diagnostics,
    gpu_diagnostics as gpu_diagnostics,
    read_bundle as read_bundle,
    render_conversations as render_conversations,
    report_stage as report_stage,
    run as run,
    train_loop as train_loop,
    _loss as _loss,
)

def main(argv=None):
    """Cloud-neutral entrypoint; wrappers stage these local directories."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    run(args.input_dir, args.model_dir, args.checkpoint_dir, device=args.device)


if __name__ == "__main__":
    main()

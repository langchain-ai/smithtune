"""Hugging Face SFT input preparation using smithtune's existing renderer.

Model loading and optimization belong in this module; Vertex submission does
not. Input conversion requires no torch, PEFT, TRL, or cloud SDK imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smithtune.providers.base import PipelineError

if TYPE_CHECKING:
    from smithtune.hf_rendering import TokenDatum


def to_trainer_example(datum: TokenDatum, *, max_seq_len: int) -> dict[str, list[int]]:
    """Preserve rendered tokens and target masks; the causal model shifts labels.

    Padding and sequence packing are not performed here. A collator must mask
    any padding it adds, and must preserve these explicit labels.
    """
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise PipelineError("max_seq_len must be a positive integer")
    tokens = datum.token_ids
    weights = datum.token_weights
    if not tokens or len(tokens) != len(weights):
        raise PipelineError("rendered tokens and loss weights must have equal nonzero length")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise PipelineError("rendered token IDs must be nonnegative integers")
    if any(type(weight) not in (int, float) or weight not in (0, 1) for weight in weights):
        raise PipelineError("HF SFT requires binary rendered loss weights")
    if len(tokens) > max_seq_len:
        raise PipelineError("rendered target exceeds max_seq_len; reject its whole trajectory")
    if not any(weights[1:]):
        raise PipelineError("rendered target has no supervised tokens after causal shifting")
    return {
        "input_ids": list(tokens),
        "attention_mask": [1] * len(tokens),
        "labels": [token if weight else -100 for token, weight in zip(tokens, weights, strict=True)],
    }

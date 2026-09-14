from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from smithtune.models import resolve_prepared_model
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import PipelineError
from smithtune.rendering import validate_model_context


@pytest.mark.parametrize("limit", [0, -1, 8193, True, 1.5])
def test_model_rejects_invalid_trainer_limits(limit):
    model = replace(baseten.DEFAULT_MODEL, max_seq_len=8192, trainer_max_seq_len=limit)
    with pytest.raises(PipelineError, match="trainer context limit"):
        model.validate()


@pytest.mark.parametrize("mode", [None, 5, ["preserved"]])
def test_model_rejects_non_string_thinking_history_mode(mode):
    model = replace(fireworks.DEFAULT_MODEL, thinking_trace_history_mode=mode)
    with pytest.raises(PipelineError, match="thinking_trace_history_mode must be a string"):
        model.validate()


def test_legacy_baseten_profile_restores_its_configured_trainer_limit():
    model = asdict(baseten.DEFAULT_MODEL)
    del model["trainer_max_seq_len"]
    manifest = {
        "model": model,
        "provider": {
            "name": "baseten",
            "renderer": model["renderer"],
            "tokenizer_revision": model["tokenizer_revision"],
        },
    }
    resolved = resolve_prepared_model(manifest, baseten.MODEL_SPECS, provider="baseten")
    assert resolved == baseten.DEFAULT_MODEL
    assert resolved.training_context_limit == 131_072


@pytest.mark.parametrize("stage", ["preparation", "baseten_training"])
def test_conflicting_thinking_history_mode_is_rejected_before_tokenizer_loading(
    monkeypatch, stage
):
    from smithtune import hf_rendering

    def unexpected_load(*args, **kwargs):
        raise AssertionError("conflicting model options must fail before loading the tokenizer")

    monkeypatch.setattr(hf_rendering, "load_tokenizer", unexpected_load)
    model = replace(baseten.DEFAULT_MODEL, thinking_trace_history_mode="interleaved")
    with pytest.raises(PipelineError, match="conflicts with.*thinking_trace_history_mode"):
        if stage == "preparation":
            validate_model_context([], model)
        else:
            baseten.render_row({"messages": []}, model)

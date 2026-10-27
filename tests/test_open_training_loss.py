"""Chunked next-token loss keeps the objective and adapter recipe; only memory changes."""
from copy import deepcopy

import pytest

from smithtune.providers.base import PipelineError
from smithtune.training.contracts import LORA_TARGET_SETS, LOSS_IMPLEMENTATION, WorkerSettings, describe_sft_job
from smithtune.training.hf_sft import Example, _loss, configure_lora, enable_chunked_loss
from tests.test_sagemaker_worker import tiny_qwen

torch = pytest.importorskip("torch")


def test_chunked_loss_matches_full_logits_loss_and_gradients():
    base = tiny_qwen(layer_types=["linear_attention", "full_attention"])
    full = configure_lora(base, 2)
    chunked = deepcopy(full)
    enable_chunked_loss(chunked)
    enable_chunked_loss(chunked)  # idempotent
    example = Example([1, 2, 3, 4, 5, 6, 7], [-100, -100, 3, 4, -100, 6, 7])
    expected = _loss(full, example, "cpu")
    actual = _loss(chunked, example, "cpu")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    expected.backward()
    actual.backward()
    for (name, value), (_, wanted) in zip(chunked.named_parameters(), full.named_parameters(), strict=True):
        if value.requires_grad:
            torch.testing.assert_close(value.grad, wanted.grad, atol=1e-5, rtol=1e-4, msg=name)
    ids = torch.tensor([example.input_ids])
    labelled = chunked(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([example.labels]), use_cache=False)
    assert labelled.logits is None, "scored positions must never materialize full logits"
    assert int(labelled.num_valid_tokens) == example.active_tokens
    unlabelled = chunked(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert unlabelled.logits.shape[-1] == 32, "generation keeps the original forward"


def test_lora_target_sets_cover_decoder_projections_only():
    adapted = {}
    for targets in LORA_TARGET_SETS:
        model = configure_lora(tiny_qwen(layer_types=["linear_attention", "full_attention"]), 2, targets)
        adapted[targets] = {name.split(".lora_")[0].rsplit(".", 1)[-1] for name, value in model.named_parameters() if value.requires_grad}
        names = [name for name, value in model.named_parameters() if value.requires_grad]
        assert not any(any(part in name for part in ("lm_head", "embed_tokens", "visual")) for name in names), targets
        assert all(".language_model.layers." in name for name in names), targets
    assert adapted["attention"] == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert adapted["all-linear"] == set(LORA_TARGET_SETS["all-linear"])
    with pytest.raises(KeyError):
        configure_lora(tiny_qwen(), 2, "lm_head")


def test_job_description_records_targets_loss_and_default_backend():
    from smithtune.providers.sagemaker import MODEL_SPECS

    assert WorkerSettings().trainer_backend == "trl"
    spec = MODEL_SPECS["qwen3p5-9b"]
    wide = describe_sft_job(spec, WorkerSettings(lora_targets="all-linear"), lora_rank=4, prepared_manifest_uri="s3://b/p/manifest.json")
    assert wide["target_modules"] == list(LORA_TARGET_SETS["all-linear"]) and wide["lora_alpha"] == 8
    assert wide["loss"] == LOSS_IMPLEMENTATION
    narrow = describe_sft_job(spec, WorkerSettings(), lora_rank=4, prepared_manifest_uri="s3://b/p/manifest.json")
    assert narrow["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    with pytest.raises(PipelineError, match="lora_targets"):
        WorkerSettings(lora_targets="everything").validate()

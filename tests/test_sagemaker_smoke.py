"""Real tiny CPU adapter reload and generation, with no downloads or AWS calls."""

from copy import deepcopy

import pytest

from smithtune.providers.base import CommonSFTSettings, PipelineError
from smithtune.providers.sagemaker.smoke import generate_tokens, last_logits, verify_loaded_adapter
from smithtune.training.hf_sft import Example, configure_lora, train_loop

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from tests.test_sagemaker_worker import tiny_qwen  # noqa: E402


def test_reload_preserves_adapter_logits_and_greedy_cached_generation(tmp_path):
    original = tiny_qwen()
    fresh = deepcopy(original)
    trained = configure_lora(original, 2)
    example = Example([1, 2, 3, 4, 5], [-100, -100, 3, 4, 5])
    train_loop(trained, [[example]], [[example]], CommonSFTSettings(max_epochs=1, batch_size=1), device="cpu")
    trained.eval()
    trained.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = peft.PeftModel.from_pretrained(fresh, tmp_path, is_trainable=False).eval()
    assert not any(parameter.requires_grad for parameter in reloaded.parameters())
    assert verify_loaded_adapter(reloaded, tmp_path)["nonzero_lora_b_matrices"] > 0
    tokens = [1, 2, 3]
    torch.testing.assert_close(last_logits(trained, tokens, "cpu"), last_logits(reloaded, tokens, "cpu"))
    with reloaded.disable_adapter():
        base = last_logits(reloaded, tokens, "cpu")
    assert torch.max(torch.abs(last_logits(reloaded, tokens, "cpu") - base)) > 0
    expected = generate_tokens(trained, tokens, device="cpu", eos_token_id=31, max_new_tokens=3)
    actual = generate_tokens(reloaded, tokens, device="cpu", eos_token_id=31, max_new_tokens=3)
    assert actual["tokens"] == expected["tokens"]
    assert 0 < actual["output_tokens"] <= 3
    assert actual["stop_reason"] in {"stop", "length"}
    with torch.no_grad():
        next(value for name, value in reloaded.named_parameters() if "lora_B" in name).add_(1)
    with pytest.raises(PipelineError, match="weights differ"):
        verify_loaded_adapter(reloaded, tmp_path)


def test_probe_rejects_nonfinite_logits():
    class Broken(torch.nn.Module):
        def forward(self, **kwargs):
            return type("Output", (), {"logits": torch.full((1, 2, 3), float("nan"))})()

    with pytest.raises(PipelineError, match="non-finite"):
        last_logits(Broken(), [1, 2], "cpu")

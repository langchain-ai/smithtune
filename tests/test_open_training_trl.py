"""Real SFTTrainer updates compared against the original explicit SFT objective."""
from copy import deepcopy

import pytest

from smithtune.training.contracts import WorkerSettings
from smithtune.training.trl_trainer import train
from smithtune.training.hf_sft import Example, configure_lora, train_loop
from tests.test_sagemaker_worker import tiny_qwen

torch = pytest.importorskip("torch")


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"<unk>": 0, "<pad>": 1, "<eos>": 2, **{f"token{i}": i for i in range(3, 32)}}, unk_token="<unk>")),
        unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")


@pytest.mark.parametrize("batch_size,epochs", [(8, 1), (2, 2)])
def test_sft_trainer_preserves_token_normalization_partial_groups_and_updates(tmp_path, batch_size, epochs, monkeypatch):
    model = configure_lora(tiny_qwen(), 2)
    reference = deepcopy(model)
    rows = [[Example([3, 4, 5, 6], [-100, -100, -100, 6]),
             Example([3, 4, 5, 6, 7], [-100, 4, 5, -100, 7])],
            [Example([6, 5, 4], [-100, 5, -100])]]
    settings = WorkerSettings(max_epochs=epochs, batch_size=batch_size, learning_rate=1e-3,
                              early_stopping_patience=epochs + 1, trainer_backend="trl")
    gradients = []
    original = torch.optim.AdamW.step
    def record_step(optimizer, *args, **kwargs):
        gradients.append(torch.cat([parameter.grad.detach().reshape(-1).clone() for group in optimizer.param_groups
                                    for parameter in group["params"] if parameter.grad is not None]))
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", record_step)
    expected = train_loop(reference, rows, rows, settings, device="cpu")
    records = []
    actual = train(model, rows, rows, settings, tokenizer=tokenizer(), output_dir=tmp_path,
                   device="cpu", on_epoch=lambda row, state: records.append((row, state)))
    assert actual["optimizer_steps"] == expected["optimizer_steps"]
    assert actual["best_validation_loss"] == pytest.approx(expected["best_validation_loss"], abs=2e-6)
    for (name, value), (_, wanted) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(value, wanted, atol=2e-6, rtol=1e-4, msg=name)
    steps = expected["optimizer_steps"]
    for expected_gradient, actual_gradient in zip(gradients[:steps], gradients[steps:], strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=1e-4)
    assert records[-1][1]["optimizer"]["state"]
    assert actual["epochs"][-1]["train_target_tokens"] == 5

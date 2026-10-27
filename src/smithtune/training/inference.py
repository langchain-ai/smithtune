"""Shared adapter integrity and bounded greedy generation helpers."""
import time
from pathlib import Path
from smithtune.providers.base import PipelineError


def verify_loaded_adapter(model, adapter_dir: Path) -> dict:
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(adapter_dir / "adapter_model.safetensors"), device="cpu")
    loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(saved) != set(loaded):
        raise PipelineError("reloaded adapter tensor names differ from the export")
    nonzero_b = 0
    for name, expected in saved.items():
        actual = loaded[name].detach().cpu()
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise PipelineError("reloaded adapter weights differ from the exported finite tensors")
        if "lora_B" in name and torch.count_nonzero(actual).item():
            nonzero_b += 1
    if not nonzero_b:
        raise PipelineError("export contains no changed LoRA B matrices")
    return {"tensor_count": len(saved), "nonzero_lora_b_matrices": nonzero_b, "exact_export_match": True}


def last_logits(model, tokens: list[int], device):
    import torch

    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise PipelineError("reloaded model returned non-finite logits")
    return logits


def generate_tokens(model, tokens: list[int], *, device, eos_token_id: int, max_new_tokens: int) -> dict:
    import torch

    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    if device != "cpu":
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                                eos_token_id=eos_token_id, pad_token_id=eos_token_id)
    if device != "cpu":
        torch.cuda.synchronize()
    completion = output[0, len(tokens):].tolist()
    if not completion:
        raise PipelineError("reloaded model returned no completion tokens")
    return {"tokens": completion, "prompt_tokens": len(tokens), "output_tokens": len(completion),
            "elapsed_seconds": time.monotonic() - started,
            "stop_reason": "stop" if completion[-1] == eos_token_id else "length"}


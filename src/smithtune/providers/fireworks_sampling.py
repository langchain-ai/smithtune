"""Replay through the official Fireworks serverless Training API sampler."""

from __future__ import annotations

import copy
import os
import re
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from smithtune.artifacts import _json_dump, _load_json, _utc_now
from smithtune.dataset import _model_from_manifest, _require_prepared_provider
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import CLIENT_SOURCE, TRAINING_BASE_URL, _set_skill_session
from smithtune.rendering import load_training_renderer, replay_prompt


def create_service():
    from fireworks.training.sdk import FiretitanServiceClient

    if not os.environ.get("FIREWORKS_API_KEY", "").strip():
        raise PipelineError("FIREWORKS_API_KEY is not set")
    _set_skill_session()
    return FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"], base_url=TRAINING_BASE_URL,
        default_headers={"X-Fireworks-Client-Source": CLIENT_SOURCE,
                         "X-Fireworks-Session-Id": os.environ["FIREWORKS_SESSION_ID"]},
    )


def checkpoint_from_run(data_dir: Path, run_dir: Path) -> tuple[Any, dict[str, Any]]:
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    _require_prepared_provider(manifest, "fireworks")
    model = _model_from_manifest(manifest)
    plan = _load_json(run_dir / "plan.json")
    result = _load_json(run_dir / "result.json")
    best = result.get("best") if isinstance(result, dict) else None
    checkpoint = best.get("resume_checkpoint") if isinstance(best, dict) else None
    if not isinstance(checkpoint, str) or re.fullmatch(r"[a-zA-Z0-9_-]+/run-[0-9a-f]{32}/[a-zA-Z0-9_-]+", checkpoint) is None:
        raise PipelineError("Fireworks evaluation requires a saved serverless training checkpoint in --run-dir")
    if plan.get("base_model") != model.base_model:
        raise PipelineError("prepared data base model differs from the Fireworks training run")
    config = plan.get("config", {})
    return model, {**best, "lora_rank": config.get("lora_rank", model.default_lora_rank),
                   "lora_alpha": config.get("lora_alpha", 32)}


def restore_stop_suffix(tokens: list[int], renderer, stop_reason: str) -> list[int]:
    """Complete a partially returned string stop marker, never the response body.

    Fireworks excludes string stops from response text. Its raw token output
    can include only the first tokens of a multi-token stop (Kimi K3). The
    renderer needs the complete framing. Match token IDs so literal control-
    looking text cannot be mistaken for a structural marker.
    """
    if stop_reason != "stop":
        return tokens
    for stop in renderer.get_stop_sequences():
        if not isinstance(stop, str):
            continue
        encode_control = getattr(renderer.tokenizer, "_encode_text_piece", None)
        ending = list(encode_control(stop, allow_special_tokens=True) if encode_control else
                      renderer.tokenizer.encode(stop, add_special_tokens=False))
        for size in range(min(len(ending) - 1, len(tokens)), 0, -1):
            if tokens[-size:] == ending[:size]:
                return [*tokens, *ending[size:]]
    return tokens


class FireworksReplaySampler:
    """Use a live snapshot, or restore a saved run when evaluation starts later.

    The checkpoint identifies results across retries. Session-specific routes
    are recorded separately, since an expired session cannot be resumed.
    """

    def __init__(self, model, checkpoint: str, output_dir: Path, *, service=None, snapshot: str | None = None,
                 lora_rank: int | None = None, lora_alpha: int = 32):
        if (service is None) != (snapshot is None):
            raise PipelineError("a live sampler requires both its service and snapshot")
        self.model = model
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.service = service
        self.snapshot = snapshot
        self.lora_rank = model.default_lora_rank if lora_rank is None else lora_rank
        self.lora_alpha = lora_alpha
        if any(type(value) is not int or value < 1 for value in (self.lora_rank, self.lora_alpha)):
            raise PipelineError("saved LoRA rank and alpha must be positive integers")
        self.renderer = load_training_renderer(model)
        self._stack = ExitStack()
        self._samplers: dict[str, Any] = {}
        self.config = {"provider": "fireworks", "serving_mode": "serverless",
                       "checkpoint": checkpoint, "base_model": model.base_model,
                       "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha}

    def __enter__(self):
        try:
            if self.service is None:
                self.service = create_service()
                self._stack.callback(self.service.close)
                client = self.service.create_lora_training_client(
                    self.model.base_model, rank=self.lora_rank, alpha=self.lora_alpha,
                )
                client.load_state(self.checkpoint).result()
                self.snapshot = client.save_weights_for_sampler(f"replay-{uuid.uuid4().hex[:8]}").result().path
            for identity, options in (
                (self.checkpoint, {"model_path": self.snapshot}),
                (self.model.base_model, {"base_model": self.model.base_model}),
            ):
                sampler = self.service.create_sampling_client(tokenizer=self.renderer.tokenizer, **options)
                self._stack.callback(sampler.close)
                self._samplers[identity] = sampler
            self.receipt = {**self.config, "session_id": self.service.training_session_id,
                            "snapshot": self.snapshot, "started_at_utc": _utc_now(), "status": "running"}
            _json_dump(self.output_dir / "sampler.json", self.receipt)
            return self.checkpoint
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            self._stack.close()
        except BaseException:
            self.receipt["status"] = "cleanup_failed"
            _json_dump(self.output_dir / "sampler.json", self.receipt)
            raise
        self.receipt.update(status="closed", closed_at_utc=_utc_now())
        _json_dump(self.output_dir / "sampler.json", self.receipt)

    def generate(self, model, messages, max_tokens, json_mode=False, request_contract=None):
        from tinker import SamplingParams

        if model not in self._samplers or json_mode:
            raise PipelineError("serverless replay can sample only its checkpoint and matching base model")
        tools = list(request_contract.tools) if request_contract is not None else []
        prompt = replay_prompt(messages, tools, self.model, self.renderer)
        if len(prompt.to_ints()) + max_tokens > self.model.max_seq_len:
            raise PipelineError("replay prompt and output budget exceed model context limit")
        response = self._samplers[model].sample(
            prompt=prompt, num_samples=1,
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0, stop=self.renderer.get_stop_sequences()),
        ).result()
        if len(response.sequences or []) != 1:
            raise PipelineError("serverless sampler returned no unique completion")
        sequence = response.sequences[0]
        tokens = list(sequence.tokens or [])
        parse_tokens = restore_stop_suffix(tokens, self.renderer, sequence.stop_reason)
        parsed, termination = self.renderer.parse_response(parse_tokens)
        candidate = self.renderer.to_openai_message(copy.deepcopy(parsed))
        unparsed = parsed.get("unparsed_tool_calls", [])
        candidate["sampling"] = {
            "session_id": self.service.training_session_id,
            "snapshot": self.snapshot if model == self.checkpoint else None,
            "prompt_tokens": len(prompt.to_ints()), "output_tokens": len(tokens),
            "stop_reason": sequence.stop_reason, "termination": str(termination),
            "format_valid": bool(getattr(termination, "is_clean", termination)) and not unparsed,
            "restored_stop_tokens": len(parse_tokens) - len(tokens),
            "raw_text": self.renderer.tokenizer.decode(tokens),
        }
        return candidate

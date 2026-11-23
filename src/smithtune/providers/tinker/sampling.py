"""Fixed-checkpoint replay through Tinker's shared sampling service.

Tinker serves saved sampler weights and base models from its own fleet and bills
sampled tokens, so smithtune provisions no serving capacity and has nothing to
deactivate. It closes the sampling session when replay ends.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from smithtune.artifacts import _json_dump, _utc_now
from smithtune.providers.base import PipelineError
from smithtune.providers.tinker import CAPACITY, close_service, resolved, service_client
from smithtune.rendering import load_training_renderer
from smithtune.run_artifacts import load_checkpoint_context, require_matching_base_model


_CHECKPOINT = re.compile(r"tinker://([A-Za-z0-9][A-Za-z0-9:._-]*)/sampler_weights/([A-Za-z0-9][A-Za-z0-9._-]*)")


def checkpoint_from_run(data_dir: Path, run_dir: Path) -> tuple[Any, str]:
    model, plan, result = load_checkpoint_context(data_dir, run_dir, provider="tinker")
    if not isinstance(plan, dict) or not isinstance(result, dict) or any(
        item.get("provider") != "tinker" for item in (plan, result)
    ):
        raise PipelineError("Tinker evaluation requires a completed Tinker training run")
    checkpoint = result.get("best_sampler_weights_uri")
    match = _CHECKPOINT.fullmatch(checkpoint) if isinstance(checkpoint, str) else None
    if not match or match[1] != result.get("tinker_run_id"):
        raise PipelineError("Tinker evaluation requires the saved best sampler checkpoint in --run-dir")
    require_matching_base_model(model, plan, provider_label="Tinker")
    return model, checkpoint


class TinkerReplaySampler:
    """Sample a saved checkpoint and its base model; never restore trainer state."""

    def __init__(self, model, checkpoint: str, output_dir: Path, *, service_factory=None):
        if model.provider != "tinker" or not isinstance(checkpoint, str) or not _CHECKPOINT.fullmatch(checkpoint):
            raise PipelineError("Tinker replay requires a saved sampler checkpoint and its Tinker model")
        from smithtune.providers.baseten.sampling_formats import PARSING_VERSION

        self.model = model
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.service_factory = service_factory or service_client
        self.renderer = load_training_renderer(model)
        self.config = {"parsing_version": PARSING_VERSION, "provider": "tinker", "serving_mode": "sampler",
                       "capacity": CAPACITY, "checkpoint": checkpoint, "base_model": model.base_model,
                       "max_seq_len": model.max_seq_len, "tokenizer_revision": model.tokenizer_revision,
                       "template_sha256": model.template_sha256, "rendering_version": model.rendering_version,
                       "renderer": model.renderer}
        self.receipt: dict[str, Any] = {}
        self._service: Any | None = None
        self._samplers: dict[str, Any] = {}

    def _save(self):
        _json_dump(self.output_dir / "sampler.json", self.receipt)

    def __enter__(self):
        self.receipt = {**self.config, "started_at_utc": _utc_now(), "status": "starting"}
        self._save()
        try:
            self._service = self.service_factory(user_metadata={"smithtune_replay": self.checkpoint})
            self._samplers = {
                self.checkpoint: resolved(self._service.create_sampling_client(model_path=self.checkpoint)),
                self.model.base_model: resolved(
                    self._service.create_sampling_client(base_model=self.model.base_model)
                ),
            }
        except BaseException:
            self._close()
            raise
        self.receipt["status"] = "running"
        self._save()
        return self.checkpoint

    def _close(self):
        self._samplers = {}
        service, self._service = self._service, None
        failed = False
        if service is not None:
            try:
                close_service(service)
            except BaseException:
                failed = True
        self.receipt.update(status="close_failed" if failed else "closed", closed_at_utc=_utc_now())
        self._save()

    def __exit__(self, exc_type, exc, tb):
        self._close()

    def generate(self, model, messages, max_tokens, json_mode=False, request_contract=None):
        import tinker
        from smithtune.providers.baseten.sampling_formats import parse_completion, stop_sequences

        if model not in self._samplers or json_mode:
            raise PipelineError("Tinker replay can sample only its checkpoint and matching base model")
        if type(max_tokens) is not int or max_tokens < 1:
            raise PipelineError("max output tokens must be positive")
        tools = list(request_contract.tools) if request_contract is not None else []
        tokens = self.renderer.prompt_tokens(messages, tools=tools)
        if len(tokens) + max_tokens > self.model.max_seq_len:
            raise PipelineError("replay prompt and output budget exceed model context limit")
        response = self._samplers[model].sample(
            prompt=tinker.ModelInput.from_ints(tokens), num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=0, stop=stop_sequences(self.renderer, self.model),
            ),
        ).result()
        if len(response.sequences or []) != 1:
            raise PipelineError("Tinker sampler returned no unique completion")
        sequence = response.sequences[0]
        output = list(sequence.tokens or [])
        candidate = parse_completion(self.renderer, self.model, output, sequence.stop_reason, tools=tools)
        candidate.setdefault("sampling", {}).update(
            checkpoint=self.checkpoint if model == self.checkpoint else None,
            prompt_tokens=len(tokens), output_tokens=len(output),
        )
        return candidate

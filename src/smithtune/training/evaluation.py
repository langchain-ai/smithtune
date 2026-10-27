"""In-job replay generation: base and tuned answers for saved cases, written durably.

Generation runs on the training machine after the best adapter is exported. Each
answer is appended to a JSONL file under the checkpoint directory, which the
cloud adapter synchronizes while the job runs, so a hard stop keeps every answer
written so far. Judging and publication stay outside the job.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

from smithtune.artifacts import _json_dump, _load_json, _load_jsonl
from smithtune.inference_contract import json_sha256
from smithtune.providers.base import PipelineError

EVALUATION_FILES = ("evaluation/cases.jsonl", "evaluation/settings.json")
VARIANTS = ("base", "tuned")
MAX_CASES_LIMIT = 500
_stop_requested = False


def request_stop(*_args) -> None:
    """Signal handler target: finish the current answer, then stop cleanly."""
    global _stop_requested
    _stop_requested = True


def stop_requested() -> bool:
    return _stop_requested


def reset_stop() -> None:
    global _stop_requested
    _stop_requested = False


@dataclass(frozen=True)
class EvaluationSettings:
    max_new_tokens: int
    max_cases: int
    max_points_per_trajectory: int | None = None
    schema_version: int = 1

    def validate(self) -> None:
        for name, maximum in (("max_new_tokens", 32_768), ("max_cases", MAX_CASES_LIMIT)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise PipelineError(f"evaluation {name} must be an integer in [1, {maximum}]")
        if self.max_points_per_trajectory is not None and (
            isinstance(self.max_points_per_trajectory, bool) or not isinstance(self.max_points_per_trajectory, int)
            or self.max_points_per_trajectory < 1
        ):
            raise PipelineError("evaluation max_points_per_trajectory must be a positive integer")
        if self.schema_version != 1:
            raise PipelineError("unsupported evaluation settings version")


def request_sha256(messages, variant: str, max_new_tokens: int) -> str:
    """Identity of one generation request; the local sampler looks answers up by it.

    Tools are the case's own at both ends, so they are not part of the key.
    """
    return json_sha256({"messages": messages, "variant": variant, "max_new_tokens": max_new_tokens})


def read_evaluation(input_dir: Path, bundle: dict) -> tuple[EvaluationSettings, list[dict]] | None:
    """Return the bundled cases and settings, or None when no evaluation was requested."""
    listed = set(bundle.get("files", {}))
    if not listed & set(EVALUATION_FILES):
        return None
    if listed & set(EVALUATION_FILES) != set(EVALUATION_FILES):
        raise PipelineError("bundle must list both evaluation files or neither")
    raw = _load_json(Path(input_dir) / "evaluation/settings.json")
    if not isinstance(raw, dict):
        raise PipelineError("evaluation settings must be an object")
    try:
        settings = EvaluationSettings(**raw)
    except TypeError as exc:
        raise PipelineError("evaluation settings have missing or unknown fields") from exc
    settings.validate()
    cases = _load_jsonl(Path(input_dir) / "evaluation/cases.jsonl")
    if not cases or len(cases) > settings.max_cases:
        raise PipelineError("bundled evaluation cases exceed max_cases or are empty")
    seen = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]
                or not isinstance(case.get("messages"), list) or not case["messages"]):
            raise PipelineError("invalid bundled evaluation case")
        if case["id"] in seen:
            raise PipelineError("duplicate bundled evaluation case")
        seen.add(case["id"])
    return settings, cases


def load_predictions(path: Path, *, repair: bool = False) -> dict[tuple[str, str], dict]:
    """Read answers written so far. A torn final line from a hard stop is dropped.

    With repair, the torn line is also removed from the file so later appends
    leave a well-formed record stream.
    """
    if not path.exists():
        return {}
    result = {}
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            if index == len(lines) - 1 or (index == len(lines) - 2 and lines[-1] == ""):
                if repair:
                    path.write_text("".join(f"{item}\n" for item in lines[:index] if item.strip()), encoding="utf-8")
                break
            raise PipelineError("saved predictions are corrupt") from exc
        if (not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("variant") not in VARIANTS
                or not isinstance(record.get("case_id"), str) or not isinstance(record.get("request_sha256"), str)):
            raise PipelineError("saved predictions have an unexpected record shape")
        result[(record["case_id"], record["variant"])] = record
    return result


def evaluation_status(completed: int, total: int, *, reason: str | None = None, error_kind: str | None = None) -> dict:
    if error_kind:
        status = "failed"
    elif completed >= total:
        status = "complete"
    else:
        status = "incomplete"
    value = {"status": status, "completed": completed, "total": total}
    if reason:
        value["reason"] = reason
    if error_kind:
        value["error_kind"] = error_kind
    return value


def generate_predictions(model, renderer, spec, cases, settings: EvaluationSettings, *, device, output_path: Path,
                         deadline: float | None = None, clock=time.monotonic, generate=None, parse=None) -> dict:
    """Append one record per (case, variant); resume by skipping records already saved."""
    from smithtune.providers.baseten.sampling_formats import parse_completion, stop_sequences
    from smithtune.training.inference import generate_tokens

    generate = generate or generate_tokens
    parse = parse or parse_completion
    settings.validate()
    stop_ids = renderer.tokenizer.encode(stop_sequences(renderer, spec)[0], add_special_tokens=False)
    if len(stop_ids) != 1:
        raise PipelineError("in-job generation requires one native stop token")
    done = load_predictions(output_path, repair=True)
    total = len(VARIANTS) * len(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    for variant in VARIANTS:
        for case in cases:
            key = (case["id"], variant)
            if key in done:
                continue
            if stop_requested():
                return evaluation_status(len(done), total, reason="stop_requested")
            if deadline is not None and clock() >= deadline:
                return evaluation_status(len(done), total, reason="time_budget_exhausted")
            tools = case.get("tools") or []
            tokens = renderer.prompt_tokens(case["messages"], tools=tools)
            if not tokens or len(tokens) + settings.max_new_tokens > spec.max_seq_len:
                record = {"schema_version": 1, "case_id": case["id"], "variant": variant,
                          "request_sha256": request_sha256(case["messages"], variant, settings.max_new_tokens),
                          "candidate": None, "skipped": "prompt plus output budget exceeds the model context",
                          "prompt_tokens": len(tokens)}
            else:
                with model.disable_adapter() if variant == "base" else nullcontext():
                    generated = generate(model, tokens, device=device, eos_token_id=stop_ids[0],
                                         max_new_tokens=settings.max_new_tokens)
                candidate = parse(renderer, spec, generated.pop("tokens"), generated["stop_reason"], tools=tools)
                record = {"schema_version": 1, "case_id": case["id"], "variant": variant,
                          "request_sha256": request_sha256(case["messages"], variant, settings.max_new_tokens),
                          "candidate": candidate, **generated,
                          "truncated": generated["stop_reason"] == "length"}
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            done[key] = record
            print(json.dumps({"smithtune_event": "generation", "variant": variant, "case_id": case["id"],
                              "completed": len(done), "total": total,
                              "output_tokens": record.get("output_tokens"), "stop_reason": record.get("stop_reason")}), flush=True)
    return evaluation_status(len(done), total)


def write_evaluation_summary(path: Path, status: dict, settings: EvaluationSettings | None) -> None:
    _json_dump(path, {"schema_version": 1, **status,
                      "settings": None if settings is None else {"max_new_tokens": settings.max_new_tokens,
                                                                  "max_cases": settings.max_cases,
                                                                  "max_points_per_trajectory": settings.max_points_per_trajectory}})

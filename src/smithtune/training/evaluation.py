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


def request_sha256(messages, variant: str, max_new_tokens: int, tools=()) -> str:
    """Bind an answer to the exact messages, ordered tool schemas and output budget."""
    return json_sha256({"messages": messages, "tools": list(tools or ()),
                        "variant": variant, "max_new_tokens": max_new_tokens})


def validate_predictions(records: dict, cases: list[dict], settings: EvaluationSettings, *, legacy_cases=None) -> dict:
    """Check saved answers before reuse; legacy lookup needs the original exported cases.

    Version 1 did not hash tools. Reuse needs a verified export's original
    cases, and the current request must match those cases exactly.
    Return normalized copies; never rewrite the saved evidence.
    """
    by_id = {case["id"]: case for case in cases}
    legacy_by_id = {case["id"]: case for case in (legacy_cases or [])}
    if len(by_id) != len(cases):
        raise PipelineError("duplicate evaluation case IDs")
    checked = {}
    for key, record in records.items():
        case = by_id.get(record["case_id"])
        if case is None:
            raise PipelineError("saved prediction refers to a different set of evaluation cases")
        expected = request_sha256(case["messages"], record["variant"], settings.max_new_tokens, case.get("tools"))
        if record["schema_version"] == 1:
            original = legacy_by_id.get(record["case_id"])
            if original is None:
                raise PipelineError("legacy saved predictions cannot safely resume: tool schemas were not hashed; "
                                    "keep the old answers and use a fresh evaluation checkpoint directory")
            if request_sha256(original["messages"], record["variant"], settings.max_new_tokens, original.get("tools")) != expected:
                raise PipelineError("legacy saved prediction differs from the original exported evaluation case")
            recorded = json_sha256({"messages": case["messages"], "variant": record["variant"],
                                    "max_new_tokens": settings.max_new_tokens})
        else:
            recorded = expected
        if record["request_sha256"] != recorded:
            raise PipelineError("saved prediction differs from the evaluation messages, tools or output budget")
        checked[key] = {**record, "request_sha256": expected}
    return checked


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
                return result
            raise PipelineError("saved predictions are corrupt") from exc
        if (not isinstance(record, dict) or record.get("schema_version") not in (1, 2) or record.get("variant") not in VARIANTS
                or not isinstance(record.get("case_id"), str) or not isinstance(record.get("request_sha256"), str)):
            raise PipelineError("saved predictions have an unexpected record shape")
        key = (record["case_id"], record["variant"])
        if key in result:
            raise PipelineError("saved predictions contain a duplicate case and variant")
        result[key] = record
    if repair and text and not text.endswith("\n"):
        # A hard stop can leave a complete JSON object without its line separator.
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
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
                         deadline: float | None = None, clock=time.monotonic, generate=None, parse=None,
                         legacy_cases=None) -> dict:
    """Append one record per (case, variant); resume only matching saved requests."""
    from smithtune.providers.baseten.sampling_formats import parse_completion, stop_sequences
    from smithtune.training.inference import generate_tokens

    generate = generate or generate_tokens
    parse = parse or parse_completion
    settings.validate()
    stop_ids = renderer.tokenizer.encode(stop_sequences(renderer, spec)[0], add_special_tokens=False)
    if len(stop_ids) != 1:
        raise PipelineError("in-job generation requires one native stop token")
    done = validate_predictions(load_predictions(output_path), cases, settings, legacy_cases=legacy_cases)
    # Repair a torn tail only after the complete records passed identity checks.
    load_predictions(output_path, repair=True)
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
                record = {"schema_version": 2, "case_id": case["id"], "variant": variant,
                          "request_sha256": request_sha256(case["messages"], variant, settings.max_new_tokens, tools),
                          "candidate": None, "skipped": "prompt plus output budget exceeds the model context",
                          "prompt_tokens": len(tokens)}
            else:
                with model.disable_adapter() if variant == "base" else nullcontext():
                    generated = generate(model, tokens, device=device, eos_token_id=stop_ids[0],
                                         max_new_tokens=settings.max_new_tokens)
                candidate = parse(renderer, spec, generated.pop("tokens"), generated["stop_reason"], tools=tools)
                record = {"schema_version": 2, "case_id": case["id"], "variant": variant,
                          "request_sha256": request_sha256(case["messages"], variant, settings.max_new_tokens, tools),
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

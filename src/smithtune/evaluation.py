"""Plan and score held-out next-message replay evaluations."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from smithtune.artifacts import _json_dump, _jsonl_dump, _load_json, _load_jsonl
from smithtune.dataset import (
    _canonical, _model_from_manifest, _prepared_inference_contract,
    _prepared_example_contracts,
)
from smithtune.inference import ANTHROPIC_MODEL_PREFIX, _chat_completion, _inference_messages
from smithtune.inference_contract import ContractError, InferenceContract
from smithtune.providers.base import PipelineError
from smithtune.providers.fireworks import _require_confirm, _set_skill_session
from smithtune.rendering import DEFAULT_REPLAY_MAX_TOKENS, validate_reasoning_support, validate_replay_context


DEFAULT_REPLAY_POINTS: int | None = None


DEFAULT_JUDGE_MAX_TOKENS = 4_096


DEFAULT_EVALUATION_CONCURRENCY = 4


JUDGE_MAX_ATTEMPTS = 3


JUDGE_CALIBRATION_CASES = 5


DETERMINISTIC_METRIC_KEYS = (
    "tool_decision_match",
    "tool_name_match",
    "arguments_json_valid",
    "arguments_schema_valid",
    "reference_arguments_match",
    "parallel_call_set_match",
)


DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"


JUDGE_INSTRUCTIONS = """You judge the next assistant message against a recorded trajectory.
Return JSON with exactly two fields: pass (boolean) and reason (short string).
The candidate is generated immediately after trajectory_prefix_visible_to_candidate.
reference_action_tool_results_not_visible_to_candidate occurred only after reference_next_action.
Those future results were not visible to the candidate and are not part of the trajectory prefix.
Use them only as evidence of what the reference action accomplished; never treat them as prior context.
For a tool call, pass when the candidate selects an equivalent tool with correct material arguments.
For a text response, pass when its meaning, usefulness, and factual claims agree with the reference.
Exact wording and tool-call IDs do not matter.
Fail missing, wrong, malformed, contradictory, or materially different responses. Do not prefer either model.
Keep the reason under 30 words. Return only JSON. Do not use Markdown."""


def _pick_evenly(values: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return values
    if limit < 1:
        raise PipelineError("max replay points must be positive")
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    return [values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)]


def build_replay_cases(
    test_rows: list[dict[str, Any]],
    max_points_per_trajectory: int | None = DEFAULT_REPLAY_POINTS,
) -> list[dict[str, Any]]:
    """Slice test trajectories before assistant-message boundaries."""
    cases: list[dict[str, Any]] = []
    for row in test_rows:
        messages = row.get("messages")
        source = row.get("_source")
        if not isinstance(messages, list) or not isinstance(source, dict):
            raise PipelineError("test rows must contain messages and source metadata")
        positions = [
            index for index, message in enumerate(messages)
            if message.get("role") == "assistant"
            and _has_visible_action(message)
        ]
        for position in _pick_evenly(positions, max_points_per_trajectory):
            reference = messages[position]
            case_type = "tool_call" if reference.get("tool_calls") else "text"
            call_ids = {call["id"] for call in reference.get("tool_calls", [])}
            tool_results: list[dict[str, Any]] = []
            for message in messages[position + 1 :]:
                if message.get("role") != "tool":
                    break
                if message.get("tool_call_id") in call_ids:
                    tool_results.append(copy.deepcopy(message))
            example_id = source["example_id"]
            case_id = hashlib.sha256(f"{example_id}:{position}".encode()).hexdigest()[:16]
            cases.append(
                {
                    "id": case_id,
                    "example_id": example_id,
                    "source_thread_id": source["source_thread_id"],
                    "source_trace_id": source.get("source_trace_id"),
                    "message_index": position,
                    "case_type": case_type,
                    "messages": copy.deepcopy(messages[:position]),
                    "reference": copy.deepcopy(reference),
                    "tool_results": tool_results,
                    "tools": copy.deepcopy(row.get("tools", [])),
                    "contract_sha256": source.get("contract_sha256"),
                }
            )
    return cases


def _has_visible_action(message: dict[str, Any]) -> bool:
    """Reasoning-only and empty messages are context, not scored actions."""
    content = message.get("content")
    text = content if isinstance(content, str) else "".join(
        part.get("text", "") for part in (content or [])
    )
    return bool(message.get("tool_calls") or text.strip())


def _case_contract(
    case: dict[str, Any], global_contract: InferenceContract | None,
    example_contracts: dict[str, InferenceContract] | None,
) -> InferenceContract | None:
    if example_contracts is None:
        return global_contract
    example_id = case.get("example_id")
    if example_id not in example_contracts:
        raise PipelineError(f"replay example {example_id} has no captured tool schemas")
    return example_contracts[example_id]


def prepare_replay_evaluation(
    data_dir: Path,
    output_dir: Path,
    max_points_per_trajectory: int | None = DEFAULT_REPLAY_POINTS,
    max_output_tokens: int = DEFAULT_REPLAY_MAX_TOKENS,
) -> dict[str, Any]:
    """Build model-ready replay cases from the untouched test split."""
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    if manifest.get("split", {}).get("test", 0) < 1:
        raise PipelineError("prepared dataset has no test rows")
    model = _model_from_manifest(manifest)
    global_contract = _prepared_inference_contract(data_dir, manifest)
    example_contracts = _prepared_example_contracts(data_dir, manifest)
    test_rows = _load_jsonl(data_dir / "prepared" / "test.jsonl")
    reasoning_policy = manifest.get("conversion", {}).get("reasoning_policy", "omit")
    if reasoning_policy not in ("omit", "preserve"):
        raise PipelineError("prepared manifest has an invalid reasoning policy")
    has_reasoning = any(
        message.get("reasoning_content")
        for row in test_rows for message in row["messages"]
    )
    if has_reasoning:
        if reasoning_policy == "omit":
            raise PipelineError("prepared test rows contain reasoning despite reasoning_policy='omit'; prepare again")
        validate_reasoning_support(model)
    cases = build_replay_cases(test_rows, max_points_per_trajectory)
    for case in cases:
        contract = _case_contract(case, global_contract, example_contracts)
        if contract is None and (case["case_type"] == "tool_call" or case["tools"]):
            raise PipelineError("tool replay cases require a prepared inference contract")
        if contract is not None:
            expected_tools = list(contract.tools)
            if case.get("contract_sha256") != contract.contract_sha256:
                raise PipelineError(f"replay case {case['id']} has a different inference contract hash")
            if case.get("tools") != expected_tools:
                raise PipelineError(f"replay case {case['id']} has different tool schemas")
            try:
                contract.validate_messages([*case["messages"], case["reference"]])
            except ContractError as exc:
                raise PipelineError(f"replay case {case['id']} violates inference contract: {exc}") from exc
    accepted, rejected = validate_replay_context(cases, model, max_output_tokens)
    if not accepted:
        raise PipelineError("the test split contains no usable assistant replay cases")
    case_types = Counter(case["case_type"] for case in accepted)
    plan = {
        "test_trajectories": len(test_rows),
        "cases": len(accepted),
        "case_types": dict(sorted(case_types.items())),
        "rejected": len(rejected),
        "max_points_per_trajectory": max_points_per_trajectory,
        "max_output_tokens": max_output_tokens,
        "reasoning_policy": reasoning_policy,
        "training_base_model": model.base_model,
        "total_prompt_tokens": sum(case["prompt_tokens"] for case in accepted),
        "max_prompt_tokens": max(case["prompt_tokens"] for case in accepted),
        "generation_calls_per_model": len(accepted),
        "judge_calibration_calls": len(_calibration_cases(accepted)) * 3,
        "judge_scoring_calls_per_model": len(accepted),
    }
    if global_contract is not None:
        plan["contract_sha256"] = global_contract.contract_sha256
    if example_contracts is not None:
        plan["example_contracts_sha256"] = manifest["example_contracts"]["sha256"]
    _jsonl_dump(output_dir / "cases.jsonl", accepted)
    _json_dump(output_dir / "rejected.json", rejected)
    _json_dump(output_dir / "plan.json", plan)
    return plan


def _judge_input(case: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    reference_action = _inference_messages([case["reference"]])[0]
    candidate_action = _inference_messages([candidate])[0]
    # Score observable behavior. A rationale cannot substitute for a missing
    # answer or a correct tool call, even when it describes the right action.
    reference_action.pop("reasoning_content", None)
    candidate_action.pop("reasoning_content", None)
    evidence = {
        "trajectory_prefix_visible_to_candidate": _inference_messages(case["messages"]),
        "reference_next_action": reference_action,
        "reference_action_tool_results_not_visible_to_candidate": _inference_messages(case["tool_results"]),
        "candidate_next_action": candidate_action,
    }
    return [
        {"role": "system", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))},
    ]


def _parse_judgment(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str):
        raise PipelineError("judge returned no JSON content")
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise PipelineError("judge returned invalid JSON")
    try:
        value = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PipelineError("judge returned invalid JSON") from exc
    if not isinstance(value.get("pass"), bool) or not isinstance(value.get("reason"), str):
        raise PipelineError("judge must return boolean pass and string reason")
    return {"pass": value["pass"], "reason": value["reason"]}


def judge_replay_candidate(
    case: dict[str, Any],
    candidate: dict[str, Any],
    judge_model: str,
    chat: Callable[
        [str, list[dict[str, Any]], int, bool, InferenceContract | None],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    last_error: PipelineError | None = None
    for _ in range(JUDGE_MAX_ATTEMPTS):
        response = chat(
            judge_model,
            _judge_input(case, candidate),
            DEFAULT_JUDGE_MAX_TOKENS,
            True,
            None,
        )
        try:
            return _parse_judgment(response)
        except PipelineError as exc:
            last_error = exc
    raise PipelineError(f"judge returned an invalid result after {JUDGE_MAX_ATTEMPTS} attempts") from last_error


def _tool_call_details(message: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return bool(raw_calls), []
    details: list[dict[str, Any]] = []
    for call in raw_calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        raw_arguments = function.get("arguments") if isinstance(function, dict) else None
        parsed_arguments: Any = None
        arguments_json_valid = False
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
                arguments_json_valid = True
            except json.JSONDecodeError:
                pass
        details.append(
            {
                "name": name,
                "arguments": parsed_arguments,
                "arguments_json_valid": arguments_json_valid,
            }
        )
    return True, details


def _tool_call_signatures(details: list[dict[str, Any]]) -> Counter[tuple[str, str]] | None:
    if not details or any(
        not isinstance(call["name"], str) or not call["arguments_json_valid"]
        for call in details
    ):
        return None
    return Counter(
        (call["name"], _canonical(call["arguments"]))
        for call in details
    )


def score_replay_candidate(
    case: dict[str, Any],
    candidate: dict[str, Any],
    contract: InferenceContract | None,
) -> dict[str, bool | None]:
    """Compute deterministic next-action and tool-call correctness metrics."""
    reference_is_tool, reference_calls = _tool_call_details(case["reference"])
    candidate_is_tool, candidate_calls = _tool_call_details(candidate)
    metrics: dict[str, bool | None] = {
        "tool_decision_match": reference_is_tool == candidate_is_tool,
        "tool_name_match": None,
        "arguments_json_valid": None,
        "arguments_schema_valid": None,
        "reference_arguments_match": None,
        "parallel_call_set_match": None,
    }
    if not reference_is_tool:
        return metrics
    if contract is None:
        raise PipelineError("tool replay scoring requires an inference contract")

    reference_names = Counter(
        call["name"] if isinstance(call["name"], str) else None
        for call in reference_calls
    )
    candidate_names = Counter(
        call["name"] if isinstance(call["name"], str) else None
        for call in candidate_calls
    )
    metrics["tool_name_match"] = bool(candidate_calls) and candidate_names == reference_names
    metrics["arguments_json_valid"] = bool(candidate_calls) and all(
        call["arguments_json_valid"] for call in candidate_calls
    )
    schema_valid = bool(candidate_calls) and bool(metrics["arguments_json_valid"])
    if schema_valid:
        for call in candidate_calls:
            name = call["name"]
            if not isinstance(name, str):
                schema_valid = False
                break
            try:
                contract.validate_tool_arguments(name, call["arguments"])
            except ContractError:
                schema_valid = False
                break
    metrics["arguments_schema_valid"] = schema_valid

    reference_signatures = _tool_call_signatures(reference_calls)
    candidate_signatures = _tool_call_signatures(candidate_calls)
    exact_call_set = (
        reference_signatures is not None
        and candidate_signatures is not None
        and candidate_signatures == reference_signatures
    )
    metrics["reference_arguments_match"] = exact_call_set
    if len(reference_calls) > 1:
        metrics["parallel_call_set_match"] = exact_call_set
    return metrics


def summarize_deterministic_metrics(
    results: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Aggregate deterministic metric rates and paired tuned-minus-base deltas."""
    summary: dict[str, Any] = {}
    for label in labels:
        model_scores: dict[str, Any] = {}
        for metric in DETERMINISTIC_METRIC_KEYS:
            values = [
                result[label]["deterministic_metrics"].get(metric)
                for result in results
            ]
            evaluated = [value for value in values if isinstance(value, bool)]
            matches = sum(value is True for value in evaluated)
            model_scores[metric] = {
                "matches": matches,
                "cases": len(evaluated),
                "rate": matches / len(evaluated) if evaluated else None,
            }
        summary[label] = model_scores
    if "base" in labels and "tuned" in labels:
        summary["tuned_minus_base"] = {
            f"{metric}_rate": (
                summary["tuned"][metric]["rate"] - summary["base"][metric]["rate"]
                if summary["tuned"][metric]["rate"] is not None
                and summary["base"][metric]["rate"] is not None
                else None
            )
            for metric in DETERMINISTIC_METRIC_KEYS
        }
    return summary


def _wrong_tool_candidate(reference: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(reference)
    candidate["tool_calls"][0]["function"]["name"] = "intentionally_wrong_tool"
    return candidate


def _wrong_arguments_candidate(reference: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(reference)
    candidate["tool_calls"][0]["function"]["arguments"] = '{"intentionally_wrong":true}'
    return candidate


def _wrong_text_candidate(reference: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(reference)
    candidate["content"] = "This response is intentionally unrelated to the recorded answer."
    candidate.pop("tool_calls", None)
    return candidate


def _missing_text_candidate(reference: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(reference)
    candidate["content"] = ""
    candidate.pop("tool_calls", None)
    return candidate


def _calibration_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_cases = [case for case in cases if case["case_type"] == "tool_call"][:JUDGE_CALIBRATION_CASES]
    text_cases = [case for case in cases if case["case_type"] == "text"][:JUDGE_CALIBRATION_CASES]
    return tool_cases + text_cases


def calibrate_judge(
    cases: list[dict[str, Any]],
    judge_model: str,
    chat: Callable[
        [str, list[dict[str, Any]], int, bool, InferenceContract | None],
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    """Score obvious positive and negative judge controls."""
    results: list[dict[str, Any]] = []
    for case in _calibration_cases(cases):
        if case["case_type"] == "tool_call":
            negative_controls = (
                ("wrong_tool", _wrong_tool_candidate(case["reference"])),
                ("wrong_arguments", _wrong_arguments_candidate(case["reference"])),
            )
        else:
            negative_controls = (
                ("wrong_text", _wrong_text_candidate(case["reference"])),
                ("missing_text", _missing_text_candidate(case["reference"])),
            )
        for label, candidate, expected in (
            ("recorded_reference", case["reference"], True),
            (negative_controls[0][0], negative_controls[0][1], False),
            (negative_controls[1][0], negative_controls[1][1], False),
        ):
            judgment = judge_replay_candidate(case, candidate, judge_model, chat)
            results.append(
                {
                    "case_id": case["id"],
                    "case_type": case["case_type"],
                    "control": label,
                    "expected": expected,
                    "actual": judgment["pass"],
                    "reason": judgment["reason"],
                }
            )
    return results


def run_replay_evaluation(
    data_dir: Path,
    output_dir: Path,
    tuned_model: str,
    judge_model: str,
    *,
    base_model: str | None = None,
    concurrency: int = DEFAULT_EVALUATION_CONCURRENCY,
    max_points_per_trajectory: int | None = DEFAULT_REPLAY_POINTS,
    max_output_tokens: int = DEFAULT_REPLAY_MAX_TOKENS,
    confirm: bool,
    chat: Callable[
        [str, list[dict[str, Any]], int, bool, InferenceContract | None],
        dict[str, Any],
    ]
    | None = None,
) -> dict[str, Any]:
    """Score tuned next messages and optionally compare a base model."""
    _require_confirm(confirm, "model and judge inference")
    if concurrency < 1:
        raise PipelineError("evaluation concurrency must be positive")
    if chat is None and not os.environ.get("FIREWORKS_API_KEY"):
        raise PipelineError("FIREWORKS_API_KEY is not set")
    if (
        chat is None
        and judge_model.startswith(ANTHROPIC_MODEL_PREFIX)
        and not os.environ.get("ANTHROPIC_CUSTOM_HEADERS")
        and not os.environ.get("ANTHROPIC_API_KEY")
    ):
        raise PipelineError("ANTHROPIC_CUSTOM_HEADERS or ANTHROPIC_API_KEY is not set")
    _set_skill_session()
    plan = prepare_replay_evaluation(
        data_dir,
        output_dir,
        max_points_per_trajectory,
        max_output_tokens,
    )
    cases = _load_jsonl(output_dir / "cases.jsonl")
    manifest = _load_json(data_dir / "prepared" / "manifest.json")
    global_contract = _prepared_inference_contract(data_dir, manifest)
    example_contracts = _prepared_example_contracts(data_dir, manifest)
    chat_fn = chat or _chat_completion
    calibration = calibrate_judge(cases, judge_model, chat_fn)
    _jsonl_dump(output_dir / "calibration.jsonl", calibration)
    if any(result["actual"] != result["expected"] for result in calibration):
        raise PipelineError("judge failed positive or negative calibration controls")
    results_path = output_dir / "results.jsonl"
    results = _load_jsonl(results_path) if results_path.exists() else []
    models = [("tuned", tuned_model)]
    if base_model:
        models.insert(0, ("base", base_model))
    cases_by_id = {case["id"]: case for case in cases}
    for result in results:
        saved_case = result.get("case", {})
        current_case = cases_by_id.get(saved_case.get("id"))
        if current_case is None or saved_case != current_case:
            raise PipelineError("existing replay results use different cases; use a new output directory")
        request_contract = _case_contract(current_case, global_contract, example_contracts)
        contract_sha256 = request_contract.contract_sha256 if request_contract is not None else None
        if result.get("reasoning_policy", "omit") != plan["reasoning_policy"]:
            raise PipelineError("existing replay results use a different reasoning policy")
        if result.get("contract_sha256") != contract_sha256:
            raise PipelineError("existing replay results use a different inference contract")
        if result.get("judge_model") != judge_model or any(
            result.get(label, {}).get("model") != model for label, model in models
        ):
            raise PipelineError("existing replay results use different models")
    completed = {result["case"]["id"] for result in results}
    case_order = {case["id"]: index for index, case in enumerate(cases)}

    def score_case(case: dict[str, Any]) -> dict[str, Any]:
        request_contract = _case_contract(case, global_contract, example_contracts)
        contract_sha256 = request_contract.contract_sha256 if request_contract is not None else None
        scored: dict[str, Any] = {
            "case": case,
            "reasoning_policy": plan["reasoning_policy"],
            "judge_model": judge_model,
            "contract_sha256": contract_sha256,
        }
        for label, model in models:
            candidate = chat_fn(
                model,
                case["messages"],
                max_output_tokens,
                False,
                request_contract,
            )
            scored[label] = {
                "model": model,
                "candidate": candidate,
                "deterministic_metrics": score_replay_candidate(
                    case,
                    candidate,
                    request_contract,
                ),
                "judgment": judge_replay_candidate(case, candidate, judge_model, chat_fn),
            }
        return scored

    pending = [case for case in cases if case["id"] not in completed]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(score_case, case) for case in pending]
        for future in as_completed(futures):
            results.append(future.result())
            results.sort(key=lambda result: case_order[result["case"]["id"]])
            _jsonl_dump(results_path, results)
    for result in results:
        request_contract = _case_contract(result["case"], global_contract, example_contracts)
        for label, _ in models:
            result[label]["deterministic_metrics"] = score_replay_candidate(
                result["case"],
                result[label]["candidate"],
                request_contract,
            )
    _jsonl_dump(results_path, results)
    tuned_passes = sum(result["tuned"]["judgment"]["pass"] for result in results)
    count = len(results)
    by_case_type = {}
    for case_type in sorted({result["case"]["case_type"] for result in results}):
        group = [result for result in results if result["case"]["case_type"] == case_type]
        group_tuned = sum(result["tuned"]["judgment"]["pass"] for result in group)
        group_count = len(group)
        by_case_type[case_type] = {
            "cases": group_count,
            "tuned_passes": group_tuned,
            "tuned_pass_rate": group_tuned / group_count,
        }
    summary = {
        **plan,
        "tuned_model": tuned_model,
        "judge_model": judge_model,
        "evaluated_models": len(models),
        "generation_calls": len(results) * len(models),
        "judge_scoring_calls": len(results) * len(models),
        "calibration_passed": True,
        "calibration_checks": len(calibration),
        "tuned_passes": tuned_passes,
        "tuned_pass_rate": tuned_passes / count,
        "by_case_type": by_case_type,
        "deterministic_metrics": summarize_deterministic_metrics(
            results,
            [label for label, _ in models],
        ),
    }
    if base_model:
        base_passes = sum(result["base"]["judgment"]["pass"] for result in results)
        wins = sum(
            result["tuned"]["judgment"]["pass"] and not result["base"]["judgment"]["pass"]
            for result in results
        )
        regressions = sum(
            result["base"]["judgment"]["pass"] and not result["tuned"]["judgment"]["pass"]
            for result in results
        )
        summary.update(
            {
                "base_model": base_model,
                "base_passes": base_passes,
                "base_pass_rate": base_passes / count,
                "pass_rate_delta": (tuned_passes - base_passes) / count,
                "paired_wins": wins,
                "paired_regressions": regressions,
                "paired_ties": count - wins - regressions,
            }
        )
        for case_type, scores in by_case_type.items():
            group = [result for result in results if result["case"]["case_type"] == case_type]
            group_base = sum(result["base"]["judgment"]["pass"] for result in group)
            group_wins = sum(
                result["tuned"]["judgment"]["pass"] and not result["base"]["judgment"]["pass"]
                for result in group
            )
            group_regressions = sum(
                result["base"]["judgment"]["pass"] and not result["tuned"]["judgment"]["pass"]
                for result in group
            )
            scores.update(
                {
                    "base_passes": group_base,
                    "base_pass_rate": group_base / len(group),
                    "pass_rate_delta": (scores["tuned_passes"] - group_base) / len(group),
                    "paired_wins": group_wins,
                    "paired_regressions": group_regressions,
                    "paired_ties": len(group) - group_wins - group_regressions,
                }
            )
    _json_dump(output_dir / "summary.json", summary)
    return summary

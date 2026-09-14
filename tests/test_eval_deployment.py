import copy
import json
from dataclasses import replace

import pytest

from smithtune import evaluation, eval_deployment
from smithtune.eval_deployment import ControlError, EvalDeployment, TemporaryDeployment
from smithtune.providers.base import PipelineError


CONFIG = EvalDeployment("accounts/acct/models/tuned", "acct", "eval-1", "accounts/fireworks/deploymentShapes/test", timeout=5)


class API:
    def __init__(self):
        self.resource = None
        self.calls = []
        self.create_error = False
        self.preemptible = True
        self.state = "READY"
        self.delete_error = False

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        if method == "GET":
            if self.resource is None:
                raise ControlError(404)
            return copy.deepcopy(self.resource)
        if method == "POST":
            self.resource = {**body, "name": CONFIG.resource, "state": self.state, "preemptible": self.preemptible}
            if self.create_error:
                raise ControlError(None)
            return copy.deepcopy(self.resource)
        if method == "DELETE":
            if self.delete_error:
                raise ControlError(503)
            self.resource = None
            return {}
        raise AssertionError(method)


def test_temporary_eval_creates_preemptible_and_confirms_cleanup(tmp_path):
    api = API()
    with TemporaryDeployment(CONFIG, tmp_path, request=api) as route:
        assert route == CONFIG.route
        assert api.resource["preemptible"] is True
        assert api.resource["baseModel"] == CONFIG.model
        assert api.resource["minReplicaCount"] == api.resource["maxReplicaCount"] == 1
    assert api.resource is None
    receipt = json.loads((tmp_path / "deployments/eval-1.json").read_text())
    assert receipt["state"] == "deleted"
    assert api.calls[1][1].endswith("?deploymentId=eval-1")


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_cleanup_after_evaluation_failure_or_interrupt(tmp_path, error):
    api = API()
    with pytest.raises(error), TemporaryDeployment(CONFIG, tmp_path, request=api):
        raise error("interrupted")
    assert api.resource is None


def test_lost_create_response_is_reconciled_by_ownership(tmp_path):
    api = API()
    api.create_error = True
    with TemporaryDeployment(CONFIG, tmp_path, request=api):
        assert api.resource
    assert sum(method == "POST" for method, *_ in api.calls) == 1
    assert api.resource is None


def test_unowned_deployment_is_never_reused_or_deleted(tmp_path):
    api = API()
    api.resource = {"name": CONFIG.resource, "baseModel": CONFIG.model, "description": "someone else", "state": "READY"}
    with pytest.raises(PipelineError, match="not owned"), TemporaryDeployment(CONFIG, tmp_path, request=api):
        pytest.fail("must not enter")
    assert [method for method, *_ in api.calls] == ["GET"]


def test_wrong_capacity_mode_is_refused_and_cleaned(tmp_path):
    api = API()
    api.preemptible = False
    with pytest.raises(PipelineError, match="preemptible mode"), TemporaryDeployment(CONFIG, tmp_path, request=api):
        pytest.fail("must not enter")
    assert api.resource is None


def test_startup_timeout_is_bounded_and_cleans(tmp_path):
    api = API()
    api.state = "CREATING"
    now = [0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(PipelineError, match="timed out"), TemporaryDeployment(CONFIG, tmp_path, request=api, sleeper=sleep, clock=lambda: now[0]):
        pytest.fail("must not enter")
    assert now[0] == CONFIG.timeout
    assert api.resource is None


def test_cleanup_failure_leaves_actionable_receipt(tmp_path):
    api = API()
    api.delete_error = True
    with pytest.raises(PipelineError, match="cleanup needs attention"), TemporaryDeployment(CONFIG, tmp_path, request=api):
        pass
    receipt = json.loads((tmp_path / "deployments/eval-1.json").read_text())
    assert receipt["state"] == "cleanup_required"
    assert "--deployment-id eval-1" in receipt["cleanup_command"]


@pytest.mark.parametrize("change", [{"model": "accounts/fireworks/models/base"}, {"account_id": "../bad"},
                                    {"model": "accounts/other/models/tuned"}, {"deployment_id": "a/b"},
                                    {"deployment_shape": "https://attacker.invalid"}, {"timeout": float("nan")}])
def test_invalid_resource_options_fail_locally(change):
    with pytest.raises(PipelineError):
        replace(CONFIG, **change).validate()


def test_control_transport_uses_official_api_and_sanitizes_errors(monkeypatch):
    import urllib.error
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    monkeypatch.setenv("FIREWORKS_SESSION_ID", "test-session")
    captured = []

    def fail(request, **kwargs):
        captured.append(request)
        raise urllib.error.HTTPError(request.full_url, 403, "secret body", {}, None)

    monkeypatch.setattr(eval_deployment.urllib.request, "urlopen", fail)
    with pytest.raises(ControlError, match="HTTP 403") as exc:
        eval_deployment.control_request("GET", "/v1/" + CONFIG.resource)
    assert "secret body" not in str(exc.value)
    assert captured[0].full_url == "https://api.fireworks.ai/v1/" + CONFIG.resource


def replay_data(tmp_path, monkeypatch):
    from test_pipeline import write_manifest
    data = tmp_path / "data"
    write_manifest(data)
    rows = [{"messages": [{"role": "user", "content": f"question-{i}"}, {"role": "assistant", "content": f"answer-{i}"}],
             "_source": {"example_id": f"example-{i}", "source_thread_id": f"thread-{i}", "source_trace_id": None}} for i in range(2)]
    (data / "prepared/test.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(evaluation, "validate_replay_context", lambda cases, *_: ([{**case, "prompt_tokens": 4} for case in cases], []))
    return data


def judge_or_generate(model, messages, max_tokens, json_mode=False, request_contract=None):
    if model == "anthropic/judge":
        evidence = json.loads(messages[-1]["content"])
        passed = evidence["candidate_next_action"] == evidence["reference_next_action"]
        return {"content": json.dumps({"pass": passed, "reason": "same" if passed else "different"})}
    return {"role": "assistant", "content": messages[-1]["content"].replace("question", "answer")}


def test_full_replay_preemptible_mode_cleanup_and_resume(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    api = API()
    monkeypatch.setattr(evaluation, "TemporaryDeployment", lambda config, output: TemporaryDeployment(config, output, request=api))
    routes = []

    def chat(model, *args):
        if model != "anthropic/judge":
            routes.append(model)
        return judge_or_generate(model, *args)

    output = tmp_path / "eval"
    result = evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True, chat=chat)
    assert result["tuned_pass_rate"] == 1
    assert routes == [CONFIG.route, CONFIG.route]
    assert api.resource is None
    creates = len(api.calls)
    evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True, chat=chat)
    assert all(method == "GET" for method, *_ in api.calls[creates:])
    assert len(routes) == 2  # Completed cases require no new inference.
    assert json.loads((output / "plan.json").read_text())["deployment"]["preemptible"] is True
    creates = len(api.calls)
    with pytest.raises(PipelineError, match="different evaluation settings"):
        evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, max_output_tokens=1000, confirm=True, chat=chat)
    assert len(api.calls) == creates


def test_resume_retries_cleanup_after_all_cases_were_saved(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    api = API()
    api.delete_error = True
    monkeypatch.setattr(evaluation, "TemporaryDeployment", lambda config, output: TemporaryDeployment(config, output, request=api))
    output = tmp_path / "eval"
    with pytest.raises(PipelineError, match="cleanup needs attention"):
        evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True, chat=judge_or_generate)
    assert len((output / "results.jsonl").read_text().splitlines()) == 2
    assert api.resource is not None
    api.delete_error = False
    result = evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True,
                                            chat=lambda *_: pytest.fail("must not repeat paid inference"))
    assert result["tuned_passes"] == 2
    assert api.resource is None
    assert sum(method == "POST" for method, *_ in api.calls) == 1


@pytest.mark.parametrize("changed", [{"deploymentShape": "accounts/fireworks/deploymentShapes/other"}, {"maxReplicaCount": 2}])
def test_owned_capacity_changes_are_rejected_and_cleaned(tmp_path, changed):
    api = API()

    def request(method, path, body=None):
        result = api(method, path, body)
        if method == "GET":
            result.update(changed)
        return result

    with pytest.raises(PipelineError, match="different"), TemporaryDeployment(CONFIG, tmp_path, request=request):
        pytest.fail("must not enter")
    assert api.resource is None


def test_server_resolved_shape_version_is_accepted(tmp_path):
    api = API()

    def request(method, path, body=None):
        result = api(method, path, body)
        if method == "GET":
            result["deploymentShape"] += "/versions/v1"
        return result

    with TemporaryDeployment(CONFIG, tmp_path, request=request):
        pass
    assert api.resource is None


def test_unknown_create_outcome_is_not_reported_as_deleted(tmp_path):
    api = API()

    def request(method, path, body=None):
        if method == "POST":
            raise ControlError(None)
        return api(method, path, body)

    with pytest.raises(PipelineError, match="cleanup needs attention"), TemporaryDeployment(CONFIG, tmp_path, request=request):
        pytest.fail("must not enter")
    assert json.loads((tmp_path / "deployments/eval-1.json").read_text())["state"] == "cleanup_required"


def test_cli_preemptible_plan_evaluate_and_cleanup(tmp_path, monkeypatch, capsys):
    from smithtune import cli
    data = replay_data(tmp_path, monkeypatch)
    api = API()
    monkeypatch.setattr(evaluation, "TemporaryDeployment", lambda config, output: TemporaryDeployment(config, output, request=api))
    monkeypatch.setattr(evaluation, "_chat_completion", judge_or_generate)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-credential")
    flags = ["--data-dir", str(data), "--output-dir", str(tmp_path / "eval"), "--serving-mode", "preemptible",
             "--tuned-model", CONFIG.model, "--account-id", CONFIG.account_id,
             "--deployment-id", CONFIG.deployment_id, "--deployment-shape", CONFIG.deployment_shape]
    cli.main(["eval-plan", *flags])
    assert json.loads(capsys.readouterr().out)["deployment"]["preemptible"] is True
    assert api.calls == []
    cli.main(["evaluate", *flags, "--judge-model", "anthropic/judge", "--confirm"])
    assert json.loads(capsys.readouterr().out)["tuned_pass_rate"] == 1
    assert api.resource is None


def test_preempted_case_is_not_scored_as_model_failure(tmp_path, monkeypatch):
    data = replay_data(tmp_path, monkeypatch)
    api = API()
    monkeypatch.setattr(evaluation, "TemporaryDeployment", lambda config, output: TemporaryDeployment(config, output, request=api))

    def chat(model, messages, *args):
        if model != "anthropic/judge" and messages[-1]["content"] == "question-1":
            api.resource = None
            raise PipelineError("HTTP 404")
        return judge_or_generate(model, messages, *args)

    output = tmp_path / "eval"
    with pytest.raises(PipelineError, match="interrupted"):
        evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True, concurrency=1, chat=chat)
    assert len((output / "results.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "evaluation-state.json").read_text())["status"] == "interrupted"
    assert not (output / "summary.json").exists()
    result = evaluation.run_replay_evaluation(data, output, CONFIG.model, "anthropic/judge", deployment=CONFIG, confirm=True, chat=judge_or_generate)
    assert result["tuned_passes"] == 2

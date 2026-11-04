"""Bedrock council transport and recovery, with the AWS service stubbed out."""

import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("boto3")
import boto3
from botocore.exceptions import ClientError
from botocore.httpsession import URLLib3Session
from botocore.stub import Stubber

from smithtune import bedrock, dataset_workflow, doctor, triage, triage_judges
from smithtune.providers.base import PipelineError
from tests.test_triage import API, SYSTEM, messages, source


JUDGE = {"name": "judge-1", "provider": "bedrock", "model": "amazon.nova-lite-v1:0", "region": "us-east-1"}
VERDICT = {"keep": 1, "reason": "The full conversation completes the request."}


def response(text=None, *, stop="end_turn", content=None):
    return {"output": {"message": {"role": "assistant", "content": content if content is not None else [{"text": text or json.dumps(VERDICT)}]}},
            "stopReason": stop, "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
            "metrics": {"latencyMs": 1}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("Bedrock tests must not call an AWS service")
    monkeypatch.setattr(URLLib3Session, "send", unexpected)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", JUDGE["region"])
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "unused-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "unused-aws-credentials"))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    bedrock._client.cache_clear()
    yield
    bedrock._client.cache_clear()


@pytest.fixture
def runtime(monkeypatch):
    session = boto3.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name=JUDGE["region"])
    runtime = session.client("bedrock-runtime")
    control = session.client("bedrock")
    monkeypatch.setattr(bedrock, "client", lambda region, service="bedrock-runtime": runtime if service == "bedrock-runtime" else control)
    with Stubber(runtime) as stub:
        yield runtime, stub
        stub.assert_no_pending_responses()


def test_api_judge_sends_the_full_trajectory_and_no_tools(runtime):
    _, stub = runtime
    trajectory = {"messages": SYSTEM + messages(1) + messages(2) + [{"role": "tool", "content": "x" * 300_000 + "original tail"}]}
    prompt = triage_judges.judge_messages(trajectory, triage_judges.JUDGE_PROMPT, ["Keep factual answers."])
    stub.add_response("converse", response(), {
        "modelId": JUDGE["model"], "system": [{"text": prompt[0]["content"]}],
        "messages": [{"role": "user", "content": [{"text": prompt[1]["content"]}]}],
        "inferenceConfig": {"maxTokens": 1024},
    })
    assert triage_judges.api_judge(JUDGE, prompt, 1024) == VERDICT
    assert json.loads(prompt[1]["content"])["untrusted_trajectory"] == trajectory["messages"]


@pytest.mark.parametrize("stop,text", [("max_tokens", json.dumps(VERDICT)), ("guardrail_intervened", json.dumps(VERDICT)), ("end_turn", "not json")])
def test_unfinished_or_invalid_answers_are_not_votes(runtime, stop, text):
    _, stub = runtime
    stub.add_response("converse", response(text, stop=stop))
    with pytest.raises(PipelineError, match="did not finish|invalid JSON"):
        triage_judges.api_judge(JUDGE, [{"role": "user", "content": "Judge this."}], 128)


@pytest.mark.parametrize("code,message,expected", [
    ("ValidationException", "Input is too long for requested model.", True),
    ("ValidationException", "Input length 200000 exceeds maximum context length.", True),
    ("ValidationException", "The model does not support tool use.", False),
    ("ValidationException", "The requested maxTokens exceeds the output limit.", False),
    ("AccessDeniedException", "Input is too long", False),
    ("ThrottlingException", "Input is too long", False),
    ("ExpiredTokenException", "The security token has expired.", False),
])
def test_only_explicit_context_rejections_filter_the_trajectory(code, message, expected):
    error = ClientError({"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse")
    assert triage_judges.context_window_exceeded(error) is expected


@pytest.mark.parametrize("code", ["AccessDeniedException", "ThrottlingException", "ExpiredTokenException", "ValidationException"])
def test_aws_failures_stay_incomplete_and_can_resume(runtime, tmp_path, code):
    _, stub = runtime
    for _ in range(2):
        stub.add_client_error("converse", service_error_code=code, service_message="private provider detail", http_status_code=400)
    args = dict(config={"judges": [JUDGE], "rules": ["Keep completed requests."]}, confirm=True,
                runner=API(), attempts=2, sleeper=lambda _: None)
    result = triage.run_triage(source(), tmp_path, **args)
    assert result["status"] == "incomplete" and result["kept"] == 0
    record = json.loads((tmp_path / "judgments.jsonl").read_text())
    assert record["status"] == "error" and record["attempts"] == 2
    assert record["aws_error_code"] == code
    assert "private provider detail" not in (tmp_path / "judgments.jsonl").read_text()
    stub.add_response("converse", response())
    assert triage.run_triage(source(), tmp_path, **args)["kept"] == 1


def test_context_rejection_filters_once_without_truncation_or_retry(runtime, tmp_path):
    _, stub = runtime
    stub.add_client_error("converse", service_error_code="ValidationException", service_message="Input is too long", http_status_code=400)
    args = dict(config={"judges": [JUDGE], "rules": ["Keep completed requests."]}, confirm=True, runner=API())
    result = triage.run_triage(source(), tmp_path, **args)
    assert result["filtered_context"] == 1 and result["kept"] == 0
    assert json.loads((tmp_path / "judgments.jsonl").read_text())["attempts"] == 1
    assert triage.run_triage(source(), tmp_path, **args)["filtered_context"] == 1


def test_preview_pins_region_and_resume_preserves_successful_mixed_votes(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_REGION", raising=False)
    state = dataset_workflow.run("pull", tmp_path, runner=API(), **{k: v for k, v in source().items() if k != "seed"})
    assert state["collection"]["status"] == "needs_review"
    chosen = ["bedrock:amazon.nova-lite-v1:0", "openai:test"]
    preview = dataset_workflow.run("triage", tmp_path, judges=chosen, rules=["Keep completed requests."], runner_mode="api", attempts=1)
    assert preview["triage"]["config"]["judges"][0]["region"] == "us-east-1"
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    seen = []

    def first(judge, prompt, limit):
        seen.append(copy.deepcopy(judge))
        if judge["provider"] == "openai":
            raise RuntimeError("temporary failure")
        return VERDICT

    dataset_workflow.run("triage", tmp_path, confirm=True, judge_call=first)
    assert next(j for j in seen if j["provider"] == "bedrock")["region"] == "us-east-1"
    seen.clear()

    def retry(judge, prompt, limit):
        seen.append(judge["provider"])
        return VERDICT

    result = dataset_workflow.run("triage", tmp_path, confirm=True, judge_call=retry)
    assert result["collection"]["eligible"] == 1
    assert seen == ["openai"]


def test_region_resolution_reads_profile_settings_without_credentials(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(boto3, "Session", lambda: SimpleNamespace(region_name="eu-west-1"))
    assert bedrock.resolve_region() == "eu-west-1"
    original = {"judges": [{k: v for k, v in JUDGE.items() if k != "region"}]}
    assert triage.validate_config(original)["judges"][0]["region"] == "eu-west-1"
    assert "region" not in original["judges"][0]
    with pytest.raises(PipelineError, match="only Bedrock"):
        triage.validate_config({"judges": [{**JUDGE, "provider": "openai"}]})
    monkeypatch.setattr(boto3, "Session", lambda: SimpleNamespace(region_name=None))
    with pytest.raises(PipelineError, match="needs a region"):
        bedrock.resolve_region()


def test_clients_keep_sdk_credentials_and_use_selected_profile(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "council-test")
    seen = []
    sdk_client = object()

    class Session:
        def get_credentials(self):
            # Deliberately opaque: code must leave refresh/signing to the SDK.
            return object()

        def client(self, service, *, region_name, config):
            seen.append((service, region_name, config))
            return sdk_client

    monkeypatch.setattr(boto3, "Session", Session)
    assert bedrock.client("us-east-1") is sdk_client
    assert bedrock.client("us-east-1") is sdk_client
    assert len(seen) == 1
    service, region, config = seen[0]
    assert (service, region) == ("bedrock-runtime", "us-east-1")
    assert config.signature_version == "v4" and config.retries["total_max_attempts"] == 1
    monkeypatch.setenv("AWS_PROFILE", "another-profile")
    bedrock.client("us-east-1")
    assert len(seen) == 2  # Do not reuse clients across profile selections.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-only-bearer")
    with pytest.raises(PipelineError, match="IAM credentials"):
        bedrock.client("us-east-1")


def test_missing_credentials_and_profile_errors_are_actionable_and_redacted(monkeypatch):
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: SimpleNamespace(get_credentials=lambda: None))
    with pytest.raises(PipelineError, match="AWS credentials were not found"):
        bedrock.client("us-east-1")

    def failed_session(**kwargs):
        raise ValueError("private configuration")
    monkeypatch.setattr(boto3, "Session", failed_session)
    with pytest.raises(PipelineError, match="check AWS_PROFILE") as caught:
        bedrock.client("us-east-1")
    assert "private configuration" not in str(caught.value)


def test_doctor_reports_bedrock_setup_without_resolving_credentials(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "council-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: pytest.fail("doctor looked up credentials"))
    report = doctor.diagnose()["bedrock"]
    assert report["profile"] == "council-test" and report["region"] == "us-east-1"
    assert report["authentication"] == "AWS SDK credential chain (IAM)"


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens"])
def test_deepagent_judge_uses_converse_without_json_mode_or_tools(runtime, stop):
    pytest.importorskip("langchain_aws")
    pytest.importorskip("deepagents")
    runtime_client, stub = runtime
    seen = []
    runtime_client.meta.events.register("before-parameter-build.bedrock-runtime.Converse", lambda params, **kwargs: seen.append(copy.deepcopy(params)))
    stub.add_response("converse", response(stop=stop))
    prompt = triage_judges.judge_messages({"messages": SYSTEM + messages(1)}, triage_judges.JUDGE_PROMPT, [])
    diagnostics = {}
    if stop == "end_turn":
        assert triage_judges.deepagent_judge(JUDGE, prompt, 1024, diagnostics=diagnostics) == VERDICT
        assert diagnostics["finish_reason"] == "end_turn"
    else:
        with pytest.raises(PipelineError, match="did not finish"):
            triage_judges.deepagent_judge(JUDGE, prompt, 1024)
    request, = seen
    assert "toolConfig" not in request and "additionalModelRequestFields" not in request
    assert request["modelId"] == JUDGE["model"]
    assert request["inferenceConfig"]["maxTokens"] == 1024
    assert json.loads(request["messages"][-1]["content"][0]["text"])["untrusted_trajectory"] == SYSTEM + messages(1)


def test_bedrock_runs_the_real_coordinator_and_judge_then_resumes(runtime, tmp_path, monkeypatch):
    pytest.importorskip("langchain_aws")
    pytest.importorskip("deepagents")
    pytest.importorskip("pydantic_monty")
    runtime_client, stub = runtime
    seen = []
    runtime_client.meta.events.register("before-parameter-build.bedrock-runtime.Converse", lambda params, **kwargs: seen.append(copy.deepcopy(params)))
    stub.add_response("converse", response(stop="tool_use", content=[{"toolUse": {
        "toolUseId": "call_code", "name": "code_mode", "input": {"code": "judge_batch(pending_tasks())"},
    }}]))
    stub.add_response("converse", response())
    stub.add_response("converse", response("Finished."))
    args = dict(config={"judges": [JUDGE], "rules": ["Keep completed requests."]}, confirm=True, runner=API(), runner_mode="deepagent", concurrency=1)
    result = triage.run_triage(source(), tmp_path, **args)
    state = json.loads((tmp_path / "agent-state.json").read_text())
    assert state["status"] == "complete" and state["code_calls"] == 1, state
    assert result["status"] == "complete" and result["kept"] == 1
    coordinator, judge_request, final = seen
    assert {t["toolSpec"]["name"] for t in coordinator["toolConfig"]["tools"]} == {"code_mode", "task"}
    assert "toolConfig" not in judge_request
    evidence = json.loads(judge_request["messages"][-1]["content"][0]["text"])
    assert evidence["untrusted_trajectory"] == SYSTEM + messages(1) + messages(2)
    tool_result = final["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "call_code"
    assert '"status": "complete"' in json.dumps(tool_result["content"]).replace('\\"', '"')
    before = (tmp_path / "judgments.jsonl").read_bytes()
    monkeypatch.setattr(bedrock, "client", lambda *_: pytest.fail("completed council repeated AWS calls"))
    assert triage.run_triage(source(), tmp_path, **args)["kept"] == 1
    assert before == (tmp_path / "judgments.jsonl").read_bytes()


def test_coordinator_access_errors_are_saved_without_private_details(runtime, tmp_path):
    pytest.importorskip("langchain_aws")
    pytest.importorskip("deepagents")
    pytest.importorskip("pydantic_monty")
    _, stub = runtime
    stub.add_client_error("converse", service_error_code="AccessDeniedException", service_message="private provider detail", http_status_code=403)
    result = triage.run_triage(source(), tmp_path,
        config={"judges": [JUDGE], "rules": ["Keep completed requests."]}, confirm=True, runner=API(), runner_mode="deepagent")
    assert result["status"] == "incomplete" and result["kept"] == 0
    state = json.loads((tmp_path / "agent-state.json").read_text())
    assert state["status"] == "incomplete" and state["aws_error_code"] == "AccessDeniedException"
    assert state["finished"] == 0
    assert "private provider detail" not in (tmp_path / "agent-state.json").read_text()


def test_arn_judges_use_api_mode_and_keep_the_full_identifier(runtime, tmp_path):
    _, stub = runtime
    arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"
    judge, = triage.council_settings(tmp_path, judges=[f"bedrock:{arn}"])["config"]["judges"]
    assert judge["model"] == arn
    stub.add_response("converse", response(), {
        "modelId": arn, "system": [], "messages": [{"role": "user", "content": [{"text": "Judge this."}]}],
        "inferenceConfig": {"maxTokens": 128},
    })
    assert triage_judges.api_judge(judge, [{"role": "user", "content": "Judge this."}], 128) == VERDICT
    with pytest.raises(PipelineError, match="--runner api"):
        bedrock.chat_model(judge, 128)


@pytest.mark.parametrize('runner', ['api', 'deepagent'])
@pytest.mark.parametrize('wrapper', ['fence', 'inline'])
def test_nova_markdown_json_wrapper_is_accepted_without_loosening_the_schema(runtime, runner, wrapper):
    _, stub = runtime
    if runner == 'deepagent':
        pytest.importorskip('langchain_aws')
        pytest.importorskip('deepagents')
    wrapped = '```json\n' + json.dumps(VERDICT) + '\n```' if wrapper == 'fence' else '`' + json.dumps(VERDICT) + '`'
    stub.add_response('converse', response(wrapped))
    call = triage_judges.api_judge if runner == 'api' else triage_judges.deepagent_judge
    result = call(JUDGE, [{'role': 'user', 'content': 'Judge this synthetic case.'}], 512)
    assert triage_judges.validate_judgment(result) == VERDICT
    stub.add_response('converse', response('```json\n{"keep": 1, "reason": "Done.", "extra": true}\n```'))
    with pytest.raises(PipelineError, match='keep.*reason'):
        triage_judges.validate_judgment(call(JUDGE, [{'role': 'user', 'content': 'Judge this synthetic case.'}], 512))


@pytest.mark.parametrize('text', [
    'Here is my verdict:\n```json\n{"keep":1,"reason":"Done."}\n```',
    '```json\n{"keep":1,"reason":"Done."}\n```\n```json\n{"keep":0,"reason":"No."}\n```',
    '```json\n{"keep":1,"reason":"Done."}\n```\nExtra explanation.',
])
def test_markdown_normalization_does_not_extract_answers_from_extra_prose(runtime, text):
    _, stub = runtime
    stub.add_response('converse', response(text))
    with pytest.raises(PipelineError, match='invalid JSON'):
        triage_judges.api_judge(JUDGE, [{'role': 'user', 'content': 'Judge this synthetic case.'}], 512)

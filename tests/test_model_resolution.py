from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from smithtune import capabilities, cli
from smithtune.models import resolve_model_options
from smithtune.providers import baseten, fireworks
from smithtune.providers.base import ModelOptions, PipelineError


@pytest.mark.parametrize("provider,expected", [
    *((baseten, model) for model in baseten.MODEL_SPECS.values()),
    *((fireworks, model) for model in fireworks.MODEL_SPECS.values()),
])
@pytest.mark.parametrize("selector", ["name", "base_model", "tokenizer_model"])
def test_explicit_model_resolves_the_provider_rendering_configuration(provider, expected, selector):
    actual = resolve_model_options(
        ModelOptions(model=getattr(expected, selector)), provider.MODEL_SPECS,
        provider=expected.provider,
    )
    assert actual == expected


@pytest.mark.parametrize("model", [*baseten.MODEL_SPECS.values(), *fireworks.MODEL_SPECS.values()])
def test_supported_model_preflight_checks_its_selected_identity_and_context(model):
    calls = []

    def resolver(base_model, context):
        calls.append((base_model, context))
        if model.provider == "baseten":
            return baseten.BasetenModelCapability(base_model, context)
        return capabilities.FireworksModelCapability(base_model, model.tokenizer_model, context, True)

    assert capabilities.preflight_model(model, capability_resolver=resolver) == model
    assert calls == [(model.base_model, model.training_context_limit)]


def test_explicit_fireworks_model_selects_kimi_without_qwen_defaults():
    expected = fireworks.MODEL_SPECS["kimi-k3"]
    actual = resolve_model_options(
        ModelOptions(model=expected.base_model), fireworks.MODEL_SPECS, provider="fireworks",
    )
    assert actual == expected
    assert actual.tokenizer_model == "moonshotai/Kimi-K3"


@pytest.mark.parametrize("provider", [baseten, fireworks])
def test_programmatic_profile_default_stays_compatible(provider):
    assert resolve_model_options(
        ModelOptions(), provider.MODEL_SPECS, provider=provider.DEFAULT_MODEL.provider,
    ) == provider.DEFAULT_MODEL


@pytest.mark.parametrize("options,message", [
    (ModelOptions(model="qwen3p8-27b", model_profile="qwen3p8-27b"), "choose either"),
    (ModelOptions(model="unknown"), "rendering configuration"),
    (ModelOptions(model_profile=""), "unknown fireworks model profile"),
])
def test_model_selection_rejects_ambiguous_or_unsupported_options(options, message):
    with pytest.raises(PipelineError, match=message):
        resolve_model_options(options, fireworks.MODEL_SPECS, provider="fireworks")


def test_cli_requires_an_explicit_model_choice(capsys, monkeypatch):
    monkeypatch.setattr(cli, "get_version", lambda: "0.1.0")
    with pytest.raises(SystemExit) as failure:
        cli._parser().parse_args(["prepare", "--workspace-id", "workspace", "--dataset-id", "dataset"])
    assert failure.value.code == 2
    assert "--model --model-profile" in capsys.readouterr().err


@pytest.mark.parametrize("choice", ["--model", "--model-profile"])
def test_cli_passes_explicit_model_choice(choice, monkeypatch):
    monkeypatch.setattr(cli, "get_version", lambda: "0.1.0")
    args = cli._parser().parse_args([
        "prepare", "--workspace-id", "workspace", "--dataset-id", "dataset",
        choice, "qwen3p8-27b",
    ])
    assert getattr(cli._model_options(args), choice[2:].replace("-", "_")) == "qwen3p8-27b"


@pytest.mark.parametrize("provider", [baseten, fireworks])
@pytest.mark.parametrize("selector", ["model", "model_profile"])
def test_context_can_be_lowered_for_supported_models(provider, selector):
    options = ModelOptions(**{selector: "qwen3p8-27b"}, max_seq_len=4096)
    model = resolve_model_options(options, provider.MODEL_SPECS, provider=provider.DEFAULT_MODEL.provider)
    assert model.max_seq_len == model.training_context_limit == 4096
    assert model.tokenizer_model == provider.DEFAULT_MODEL.tokenizer_model


@pytest.mark.parametrize("context", [0, -1, True, 1.5, 262_145])
def test_selected_model_context_must_fit_the_supported_training_limit(context):
    with pytest.raises(PipelineError, match="supported training context"):
        resolve_model_options(
            ModelOptions(model="qwen3p8-27b", max_seq_len=context),
            baseten.MODEL_SPECS, provider="baseten",
        )


@pytest.mark.parametrize("field,value", [
    ("template_sha256", "abc"), ("template_sha256", "A" * 64),
    ("template_sha256", None), ("rendering_version", 1),
])
def test_model_rejects_invalid_rendering_identity(field, value):
    with pytest.raises(PipelineError, match=field):
        replace(fireworks.DEFAULT_MODEL, **{field: value}).validate()


def _fireworks_document(**overrides):
    return {
        "name": fireworks.DEFAULT_MODEL.base_model,
        "huggingFaceUrl": "https://huggingface.co/Qwen/Qwen3.8-27B",
        "supervisedLoraTunable": True,
        "trainingContextLength": 131_072,
        **overrides,
    }


def _fetch_document(document):
    payload = document if isinstance(document, bytes) else json.dumps(document).encode()
    return capabilities.fetch_fireworks_model_capability(
        fireworks.DEFAULT_MODEL.base_model, 131_072, api_key="test-key",
        opener=lambda *args, **kwargs: io.BytesIO(payload),
    )


def test_fireworks_preflight_uses_read_only_fixed_origin_and_sanitized_metadata():
    calls = []

    def opener(request, *, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(_fireworks_document()).encode())

    capability = capabilities.fetch_fireworks_model_capability(
        fireworks.DEFAULT_MODEL.base_model, 131_072, api_key="test-key", opener=opener,
    )
    request, timeout = calls[0]
    assert request.full_url == "https://api.fireworks.ai/v1/accounts/fireworks/models/qwen3p8-27b"
    assert request.method == "GET"
    assert request.data is None
    assert request.get_header("Authorization") == "Bearer test-key"
    assert timeout == 30
    assert capability.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert "test-key" not in repr(capability)


@pytest.mark.parametrize("inference_serverless", [False, True, None])
def test_inference_serverless_flag_does_not_determine_training_support(inference_serverless):
    capability = _fetch_document(_fireworks_document(supportsServerless=inference_serverless))
    assert capability.supervised_lora_tunable is True


@pytest.mark.parametrize("value", [None, "true", 1])
def test_fireworks_rejects_invalid_managed_lora_metadata(value):
    with pytest.raises(PipelineError, match="LoRA training|supervised LoRA"):
        _fetch_document(_fireworks_document(supervisedLoraTunable=value, supportsServerless=True))


@pytest.mark.parametrize("alias", ["deepseek-v4-flash-0731", "muse-glimmer-30b"])
@pytest.mark.parametrize("managed_flag", [False, True, "absent"])
def test_serverless_training_does_not_require_managed_sft_support(alias, managed_flag):
    model = fireworks.MODEL_SPECS[alias]
    document = _fireworks_document(
        name=model.base_model, huggingFaceUrl=f"https://huggingface.co/{model.tokenizer_model}",
        supervisedLoraTunable=managed_flag, trainingContextLength=65_536,
    )
    if managed_flag == "absent":
        del document["supervisedLoraTunable"]
    capability = capabilities.fetch_fireworks_model_capability(
        model.base_model, model.training_context_limit, api_key="test-key",
        opener=lambda *args, **kwargs: io.BytesIO(json.dumps(document).encode()),
    )
    assert capabilities.preflight_model(model, capability_resolver=lambda *args: capability) == model


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "131072"])
def test_fireworks_rejects_invalid_training_context_metadata(value):
    with pytest.raises(PipelineError, match="training context limit"):
        _fetch_document(_fireworks_document(trainingContextLength=value))


@pytest.mark.parametrize("value", [
    None, "http://huggingface.co/Qwen/Qwen3.8-27B",
    "https://huggingface.co.attacker.invalid/Qwen/Qwen3.8-27B",
    "https://secret@huggingface.co/Qwen/Qwen3.8-27B",
    "https://huggingface.co/Qwen/Qwen3.8-27B?token=secret",
    "https://huggingface.co/Qwen/Qwen3.8-27B/tree/main",
    "https://huggingface.co/Qwen/%2E%2E", "https://huggingface.co/../Qwen",
    "https://huggingface.co/Qwen/Qwen3.8-27B\n",
])
def test_fireworks_rejects_untrusted_or_ambiguous_tokenizer_urls(value):
    with pytest.raises(PipelineError, match="Hugging Face model URL"):
        _fetch_document(_fireworks_document(huggingFaceUrl=value))


@pytest.mark.parametrize("payload", [
    b"[]", b"null", b'{"name":"one","name":"two"}', b'{"value":NaN}', b"\xff",
    b'{"secret":"do not print response body"', b"x" * (1024 * 1024 + 1),
    b"[" * 2000 + b"]" * 2000,
])
def test_fireworks_rejects_invalid_metadata_without_printing_it(payload):
    with pytest.raises(PipelineError) as failure:
        _fetch_document(payload)
    assert "do not print response body" not in str(failure.value)


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 429, 500])
def test_fireworks_http_failures_expose_only_status(status):
    def opener(*args, **kwargs):
        raise urllib.error.HTTPError("https://secret-url.invalid", status, "secret-key", {}, None)

    with pytest.raises(PipelineError) as failure:
        capabilities.fetch_fireworks_model_capability(
            fireworks.DEFAULT_MODEL.base_model, 1024, api_key="secret-key", opener=opener,
        )
    assert str(failure.value) == f"Fireworks model preflight returned HTTP {status}"
    assert failure.value.__suppress_context__


def test_authenticated_redirect_is_blocked_before_opening_another_destination():
    handler = capabilities._NoRedirect()
    opener = urllib.request.build_opener(handler)
    opened = []
    opener.open = lambda *args, **kwargs: opened.append(args)
    request = urllib.request.Request(
        "https://api.fireworks.ai/v1/accounts/fireworks/models/qwen3p8-27b",
        headers={"Authorization": "Bearer secret-key"},
    )
    with pytest.raises(urllib.error.HTTPError):
        opener.error(
            "http", request, io.BytesIO(b""), 302, "Found",
            {"location": "https://attacker.invalid/steal"},
        )
    assert opened == []


@pytest.mark.parametrize("model", [
    "https://attacker.invalid/model", "accounts/fireworks/models/../../secret",
    "accounts/fireworks/models/qwen3p8-27b?secret", "accounts/fireworks/models/unknown",
])
def test_invalid_or_unknown_pool_is_rejected_before_request(model):
    def unexpected(*args, **kwargs):
        raise AssertionError("unverified model must not trigger a request")

    with pytest.raises(PipelineError, match="resource is invalid|availability is unverified"):
        capabilities.fetch_fireworks_model_capability(model, 1024, api_key="test-key", opener=unexpected)


def test_missing_api_key_fails_before_request(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(PipelineError, match="FIREWORKS_API_KEY is required"):
        capabilities.fetch_fireworks_model_capability(fireworks.DEFAULT_MODEL.base_model, 1024)


def test_kimi_uses_documented_shared_pool_context_instead_of_generic_model_context():
    model = fireworks.MODEL_SPECS["kimi-k3"]
    capability = capabilities.FireworksModelCapability(
        model.base_model, model.tokenizer_model, 65_536, True,
    )
    assert capabilities.preflight_model(model, capability_resolver=lambda *args: capability) is model


def test_shared_pool_limit_is_enforced_before_metadata_fetch():
    model = replace(fireworks.DEFAULT_MODEL, max_seq_len=262_144)
    with pytest.raises(PipelineError, match="serverless training context limit is 131,072"):
        capabilities.preflight_model(model)


def test_fireworks_metadata_tokenizer_must_match_selected_model():
    model = fireworks.DEFAULT_MODEL
    capability = capabilities.FireworksModelCapability(model.base_model, "wrong/tokenizer", 131_072, True)
    with pytest.raises(PipelineError, match="does not match the selected tokenizer"):
        capabilities.preflight_model(model, capability_resolver=lambda *args: capability)


def test_baseten_preflight_reuses_capabilities_without_importing_rendering_dependencies():
    model = baseten.DEFAULT_MODEL
    calls = []

    def resolver(base_model, context):
        calls.append((base_model, context))
        return baseten.BasetenModelCapability(base_model, 262_144, True)

    assert capabilities.preflight_model(model, capability_resolver=resolver) is model
    assert calls == [(model.base_model, model.training_context_limit)]


@pytest.mark.parametrize("capability", [
    baseten.BasetenModelCapability("wrong/model", 262_144),
    baseten.BasetenModelCapability("Qwen/Qwen3.8-27B", True),
    baseten.BasetenModelCapability("Qwen/Qwen3.8-27B", 1024),
    baseten.BasetenModelCapability("Qwen/Qwen3.8-27B", 262_144, "true"),
])
def test_baseten_rejects_invalid_or_insufficient_capability(capability):
    with pytest.raises((PipelineError, baseten.BasetenRuntimeError)):
        capabilities.preflight_model(baseten.DEFAULT_MODEL, capability_resolver=lambda *args: capability)


def test_baseten_does_not_infer_cross_entropy_support_from_catalog_presence():
    model = replace(baseten.DEFAULT_MODEL, base_model="zai-org/GLM-5.3")
    with pytest.raises(PipelineError, match="cross-entropy training compatibility is unverified"):
        capabilities.preflight_model(model)


def test_baseten_tokenizer_cannot_silently_point_to_another_model():
    model = replace(baseten.DEFAULT_MODEL, tokenizer_model="other/tokenizer")
    with pytest.raises(PipelineError, match="official base model identity"):
        capabilities.preflight_model(model)

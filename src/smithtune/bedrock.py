"""Optional Bedrock council transport using the AWS SDK credential chain."""

from __future__ import annotations

from functools import lru_cache
import os
import re

from smithtune.providers.base import PipelineError


def resolve_region(region: str | None = None) -> str:
    """Resolve non-secret configuration only; previews make no service calls."""
    if region is None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            try:
                import boto3
                region = boto3.Session().region_name
            except ImportError:
                raise PipelineError("Bedrock requires the optional [bedrock] extra") from None
            except Exception:
                raise PipelineError("could not read AWS region configuration; set AWS_DEFAULT_REGION or check AWS_PROFILE") from None
    if not isinstance(region, str) or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", region):
        raise PipelineError("Bedrock needs a region: set AWS_DEFAULT_REGION or region in the council config")
    return region


def client(region: str, service: str = "bedrock-runtime"):
    """Keep refreshable SDK credentials on clients; never copy resolved keys."""
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise PipelineError("Bedrock councils currently use IAM credentials; unset AWS_BEARER_TOKEN_BEDROCK and select an AWS profile or role")
    profiles = (os.environ.get("AWS_PROFILE"), os.environ.get("AWS_DEFAULT_PROFILE"))
    return _client(resolve_region(region), profiles, service)


@lru_cache(maxsize=16)
def _client(region: str, profiles: tuple[str | None, str | None], service: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise PipelineError("Bedrock requires the optional [bedrock] extra") from None
    try:
        # Profiles participate in the cache key. Let the SDK choose credential
        # precedence rather than forcing a profile over environment credentials.
        session = boto3.Session()
        if session.get_credentials() is None:
            raise PipelineError("AWS credentials were not found for Bedrock; configure a profile, sign in with aws sso login, or use a workload role")
        return session.client(service, region_name=region, config=Config(
            signature_version="v4", connect_timeout=10, read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 1},
        ))
    except PipelineError:
        raise
    except Exception:
        raise PipelineError("could not load AWS credentials for Bedrock; check AWS_PROFILE and refresh an expired SSO or temporary session") from None


def chat_model(judge: dict, max_tokens: int):
    """Supply the existing Deep Agents coordinator with a tool-capable model."""
    if judge["model"].startswith("arn:"):
        raise PipelineError("Bedrock agent councils need a model ID or system inference profile ID; use --runner api for ARN-based judges")
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError:
        raise PipelineError("Bedrock agent councils require the optional [bedrock] extra") from None
    region = resolve_region(judge.get("region"))
    return ChatBedrockConverse(
        model=judge["model"], region_name=region, max_tokens=max_tokens,
        client=client(region), bedrock_client=client(region, "bedrock"),
        bedrock_api_key=None, disable_streaming=True,
    )


def complete(judge: dict, messages: list[dict], max_tokens: int) -> str:
    """One judge request, without tools or assumptions about JSON-mode support."""
    system, conversation = [], []
    for message in messages:
        if not isinstance(message.get("content"), str):
            raise PipelineError("Bedrock council messages must contain text")
        if message["role"] == "system":
            system.append({"text": message["content"]})
        elif message["role"] in {"user", "assistant"}:
            conversation.append({"role": message["role"], "content": [{"text": message["content"]}]})
        else:
            raise PipelineError("unsupported Bedrock council message role")
    response = client(resolve_region(judge.get("region"))).converse(
        modelId=judge["model"], system=system, messages=conversation,
        inferenceConfig={"maxTokens": max_tokens},
    )
    if response.get("stopReason") not in {"end_turn", "stop_sequence"}:
        raise PipelineError("Bedrock judge did not finish its answer; check the model and output-token budget")
    try:
        text = "".join(block["text"] for block in response["output"]["message"]["content"] if "text" in block)
    except (KeyError, TypeError):
        raise PipelineError("Bedrock judge returned no text response") from None
    if not text:
        raise PipelineError("Bedrock judge returned no text response")
    return text

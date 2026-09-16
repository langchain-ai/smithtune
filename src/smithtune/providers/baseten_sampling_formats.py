"""Parse completions from the pinned Baseten training tokenizer formats.

Qwen3.8/Qwen3.5 and GLM use their official chat templates. Kimi K3 uses
encoding_k3.py's typed XTML format; its control tokens are structural only
when represented by their special token IDs, never by ordinary text.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any
from xml.etree import ElementTree

from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012, specification_with

from smithtune.providers.base import PipelineError


PARSING_VERSION = "smithtune-baseten-replay-v1"


_STOPS = {
    "hf_assistant": ("<|im_end|>",),
    "hf_prefix_qwen3_5": ("<|im_end|>",),
    "hf_prefix_glm53_flash": ("<|user|>", "<|observation|>"),
    "hf_prefix_kimi_k3": ("<|close|>message<|sep|>",),
}
_NAME = r"[A-Za-z0-9_.:-]+"


def stop_sequences(renderer, model) -> list[str]:
    """Native assistant termination markers, excluding history delimiters."""
    if model.renderer not in _STOPS:
        raise PipelineError("no Baseten sampling parser for the prepared renderer")
    return list(_STOPS[model.renderer])


def _control_tokens(tokenizer, text: str) -> list[int]:
    encode = getattr(tokenizer, "_encode_text_piece", None)
    return list(encode(text, allow_special_tokens=True) if encode else
                tokenizer.encode(text, add_special_tokens=False))


def _without_stop(tokenizer, tokens: list[int], stops: list[str], stopped: bool) -> tuple[list[int], str | None]:
    for stop in stops:
        ending = _control_tokens(tokenizer, stop)
        if ending and len(tokens) >= len(ending) and tokens[-len(ending):] == ending:
            return tokens[:-len(ending)], stop
        # A string stop can be excluded, or returned as a partial token suffix.
        # Only normalize a reported stop, never a length-truncated generation.
        if stopped:
            for size in range(min(len(ending) - 1, len(tokens)), 0, -1):
                if tokens[-size:] == ending[:size]:
                    return tokens[:-size], stop
    return tokens, None


def _json(text: str) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate tool argument")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def _matches_type(value: Any, kind: str) -> bool:
    return {"object": isinstance(value, dict), "array": isinstance(value, list),
            "boolean": type(value) is bool, "null": value is None,
            "number": type(value) in (int, float), "integer": type(value) is int,
            "string": isinstance(value, str)}.get(kind, False)


def _typed_value(text: str, kind: str | tuple[str, ...] | None) -> Any:
    if kind == "string":
        return text
    # Qwen3.5's official template stringifies Python boolean/null scalars.
    normalized = {"True": "true", "False": "false", "None": "null"}.get(text, text)
    try:
        value = _json(normalized)
    except ValueError:
        if kind is None:
            if text.lstrip().startswith(("{", "[", '\"')):
                raise ValueError("ambiguous tool argument without a schema type") from None
            return text
        if isinstance(kind, tuple) and "string" in kind:
            return text
        raise
    if kind is None:
        # The templates emit string arguments without quotes: JSON-looking
        # strings cannot be distinguished from typed values without a schema.
        raise ValueError("ambiguous tool argument without a schema type")
    kinds = kind if isinstance(kind, tuple) else (kind,)
    matches = any(_matches_type(value, allowed) for allowed in kinds if allowed != "string")
    if "string" in kinds:
        if matches:
            raise ValueError("ambiguous nullable string argument" if set(kinds) == {"string", "null"}
                             else "ambiguous tool argument schema types")
        return text
    if not matches:
        raise ValueError("tool argument does not match its declared type")
    return value


def _resolve_schema(schema, resolver, seen: frozenset[int]):
    """Follow saved fragment references only; the registry has no retriever."""
    while isinstance(schema, dict):
        if id(schema) in seen:
            raise ValueError("cyclic tool argument schema reference")
        seen = seen | {id(schema)}
        ref = schema.get("$ref")
        if ref is None:
            return schema, resolver, seen
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError("only saved local tool argument schema references are supported")
        if set(schema) - {"$ref", "$defs", "definitions", "$schema", "title", "description", "default"}:
            raise ValueError("ambiguous tool argument reference with sibling constraints")
        try:
            resolved = resolver.lookup(ref)
        except Unresolvable as exc:
            raise ValueError("cannot resolve saved tool argument schema reference") from exc
        schema, resolver = resolved.contents, resolved.resolver
    return {}, resolver, seen


def _schema_types(schema: dict, resolver, seen: frozenset[int]) -> str | tuple[str, ...] | None:
    schema, resolver, seen = _resolve_schema(schema, resolver, seen)
    kind = schema.get("type")
    if isinstance(kind, str):
        return kind
    if isinstance(kind, list) and all(isinstance(value, str) for value in kind):
        return tuple(sorted(set(kind)))
    alternatives = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(alternatives, list):
        types = [_schema_types(value, resolver, seen) for value in alternatives if isinstance(value, dict)]
        if len(types) != len(alternatives) or any(not isinstance(value, str) for value in types):
            raise ValueError("ambiguous tool argument schema")
        return tuple(sorted(set(types)))
    return None


def _argument_type(tools, name: str, key: str) -> str | tuple[str, ...] | None:
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name") == name:
            parameters = function.get("parameters", {})
            specification = specification_with(parameters.get("$schema", ""), default=DRAFT202012)
            resolver = Registry().resolver_with_root(specification.create_resource(parameters))
            parameters, resolver, _ = _resolve_schema(parameters, resolver, frozenset())
            schema = parameters.get("properties", {}).get(key, parameters.get("additionalProperties", {}))
            return _schema_types(schema, resolver, frozenset())
    return None


def _call(name: str, arguments: dict, index: int) -> dict:
    if not re.fullmatch(_NAME, name):
        raise ValueError("invalid tool name")
    return {"id": f"call_{index}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
    }}


def _xml_calls(body: str, *, glm: bool, tools) -> tuple[str, list[dict]]:
    start = body.find("<tool_call>")
    if start < 0:
        if any(tag in body for tag in ("<tool_call", "</tool_call", "<function=", "<arg_key>")):
            raise ValueError("incomplete tool call")
        return body.strip(), []
    content, remaining = body[:start].strip(), body[start:]
    calls = []
    while remaining.strip():
        match = re.match(r"\s*<tool_call>(.*?)</tool_call>", remaining, re.DOTALL)
        if match is None:
            raise ValueError("unclosed tool call or trailing text")
        block = match[1].strip()
        if glm:
            name_match = re.match(rf"({_NAME})(?=<arg_key>|$)", block)
            if name_match is None:
                raise ValueError("invalid GLM call")
            name, params = name_match[1], block[name_match.end():]
            pattern = r"<arg_key>([^<>]+)</arg_key><arg_value>(.*?)</arg_value>"
        else:
            function = re.fullmatch(rf"<function=({_NAME})>\s*(.*?)\s*</function>", block, re.DOTALL)
            if function is None:
                raise ValueError("invalid Qwen call")
            name, params = function[1], function[2]
            pattern = r"\s*<parameter=([^<>]+)>\n?(.*?)\n?</parameter>"
        arguments = {}
        while params.strip():
            parameter = re.match(pattern, params, re.DOTALL)
            if parameter is None:
                raise ValueError("invalid tool argument framing")
            key, value = parameter[1], parameter[2]
            if key in arguments:
                raise ValueError("duplicate tool argument")
            arguments[key] = _typed_value(value, _argument_type(tools, name, key))
            params = params[parameter.end():]
        calls.append(_call(name, arguments, len(calls)))
        remaining = remaining[match.end():]
    return content, calls


def _kimi_xml(tokenizer, tokens: list[int]) -> str:
    controls = {}
    for text, replacement in (("<|open|>", "<"), ("<|close|>", "</"), ("<|sep|>", ">")):
        encoded = _control_tokens(tokenizer, text)
        if len(encoded) != 1:
            raise PipelineError("Kimi control marker is not a single special token")
        controls[encoded[0]] = replacement
    pieces, buffer = [], []
    in_tag = False

    def flush():
        text = tokenizer.decode(buffer, skip_special_tokens=False)
        if in_tag and not re.fullmatch(r'(?:think|response|tools|call|argument|json)(?: [a-z]+="[^<>\x00-\x1f]*")*', text):
            raise ValueError("invalid Kimi tag header")
        pieces.append(text if in_tag else html.escape(text, quote=False))
        buffer.clear()

    for token in tokens:
        if token in controls:
            flush()
            marker = controls[token]
            if (marker == ">") != in_tag:
                raise ValueError("invalid Kimi control sequence")
            pieces.append(marker)
            in_tag = marker != ">"
        else:
            buffer.append(token)
    flush()
    if in_tag:
        raise ValueError("incomplete Kimi tag")
    return "".join(pieces)


def _plain(element) -> str:
    if len(element):
        raise ValueError("nested content in scalar channel")
    return element.text or ""


def _kimi_parse(tokenizer, tokens: list[int]) -> tuple[str, str, list[dict]]:
    # The inference prompt has already opened message and think. The sampler
    # removes the configured message stop; all inner channels must still close.
    root = ElementTree.fromstring("<message><think>" + _kimi_xml(tokenizer, tokens) + "</message>")
    children = list(root)
    if [node.tag for node in children] not in (["think", "response"], ["think", "response", "tools"]):
        raise ValueError("invalid Kimi response channels")
    if root.text or any(node.tail for node in children):
        raise ValueError("text outside Kimi response channels")
    reasoning, content = _plain(children[0]), _plain(children[1])
    calls = []
    if len(children) == 3:
        tools = children[2]
        if tools.text or not len(tools):
            raise ValueError("invalid Kimi tools channel")
        for index, call in enumerate(tools, 1):
            if call.tag != "call" or set(call.attrib) != {"tool", "index"} or call.attrib["index"] != str(index) or call.text or call.tail:
                raise ValueError("invalid Kimi tool call")
            arguments = {}
            for argument in call:
                if argument.tail:
                    raise ValueError("text outside Kimi argument")
                if argument.tag == "json" and len(call) == 1 and argument.attrib == {"type": "object"}:
                    arguments = _typed_value(_plain(argument), "object")
                elif argument.tag == "argument" and set(argument.attrib) == {"key", "type"}:
                    key = argument.attrib["key"]
                    if key in arguments:
                        raise ValueError("duplicate tool argument")
                    arguments[key] = _typed_value(_plain(argument), argument.attrib["type"])
                else:
                    raise ValueError("invalid Kimi argument")
            calls.append(_call(call.attrib["tool"], arguments, index - 1))
    return reasoning, content, calls


def parse_completion(renderer, model, tokens: list[int], stop_reason: str, tools=None) -> dict[str, Any]:
    """Return evaluator-compatible assistant output, retaining malformed raw text.

    The selected official generation prompts all open the reasoning channel.
    Incomplete reasoning, malformed tools, and length-limited output are invalid.
    """
    stops = stop_sequences(renderer, model)
    tokenizer = renderer.tokenizer
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    candidate: dict[str, Any] = {"role": "assistant", "content": raw}
    sampling = {"format_valid": False, "raw_text": raw, "stop_reason": stop_reason}
    candidate["sampling"] = sampling
    parse_tokens, ending = _without_stop(tokenizer, tokens, stops, stop_reason == "stop")
    try:
        if model.renderer == "hf_prefix_kimi_k3":
            reasoning, content, calls = _kimi_parse(tokenizer, parse_tokens)
        else:
            text = tokenizer.decode(parse_tokens, skip_special_tokens=False)
            if "</think>" not in text:
                raise ValueError("unclosed reasoning channel")
            reasoning, body = text.split("</think>", 1)
            if "<think>" in reasoning or any(tag in body for tag in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|user|>", "<|observation|>")):
                raise ValueError("unexpected response boundary")
            content, calls = _xml_calls(body, glm=model.renderer == "hf_prefix_glm53_flash", tools=tools)
            reasoning = reasoning.strip()
            if model.renderer == "hf_prefix_glm53_flash" and ending is not None:
                expected = "<|observation|>" if calls else "<|user|>"
                if ending != expected:
                    raise ValueError("GLM stop does not match the response")
        candidate.update(content=content, reasoning_content=reasoning)
        if calls:
            candidate["tool_calls"] = calls
        sampling["format_valid"] = stop_reason == "stop"
    except (ValueError, ElementTree.ParseError) as exc:
        sampling["parse_error"] = str(exc)
    return candidate

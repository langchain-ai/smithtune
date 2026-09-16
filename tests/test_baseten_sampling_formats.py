"""Offline completions grounded in the pinned official tokenizer templates."""

import json
import os
from types import SimpleNamespace

import pytest

from smithtune.baseten_sampling_formats import parse_completion, stop_sequences
from smithtune.providers.baseten import MODEL_SPECS


class Tokenizer:
    """Keep special-token identity distinct from identical literal characters."""

    markers = ("<|open|>", "<|close|>", "<|sep|>", "<|im_end|>", "<|user|>", "<|observation|>")

    def encode(self, text, add_special_tokens=False):
        return self._encode_text_piece(text, allow_special_tokens=True)

    def _encode_text_piece(self, text, allow_special_tokens=False):
        tokens = []
        while text:
            marker = next((m for m in self.markers if text.startswith(m)), None) if allow_special_tokens else None
            if marker:
                tokens.append(0x110000 + self.markers.index(marker))
                text = text[len(marker):]
            else:
                tokens.append(ord(text[0]))
                text = text[1:]
        return tokens

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.markers[t - 0x110000] if t >= 0x110000 else chr(t) for t in tokens)


TOKENIZER = Tokenizer()
RENDERER = SimpleNamespace(tokenizer=TOKENIZER)
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "count": {"type": "integer"},
                                      "enabled": {"type": "boolean"}, "nested": {"type": "object"}},
}}}]


def parse(model, text, reason="stop", tools=TOOLS):
    return parse_completion(RENDERER, MODEL_SPECS[model], TOKENIZER.encode(text), reason, tools)


@pytest.mark.parametrize("model", ["qwen3p8-27b", "qwen3p5-9b", "glm-5p3-flash"])
@pytest.mark.parametrize("reasoning", ["", "Consider café 🐢 東京"])
def test_reasoning_is_separate_from_visible_text(model, reasoning):
    candidate = parse(model, reasoning + "</think>\n\nThe answer.")
    assert candidate["content"] == "The answer."
    assert candidate["reasoning_content"] == reasoning
    assert candidate["sampling"]["format_valid"]


@pytest.mark.parametrize("model", ["qwen3p8-27b", "qwen3p5-9b"])
def test_qwen_tools_preserve_schema_types_and_parallel_calls(model):
    text = ('reason</think>Let me check.\n<tool_call>\n<function=weather>\n'
            '<parameter=city>\n123\n</parameter>\n<parameter=count>\n2\n</parameter>\n'
            '<parameter=enabled>\nTrue\n</parameter>\n</function>\n</tool_call>\n'
            '<tool_call>\n<function=weather>\n</function>\n</tool_call><|im_end|>')
    candidate = parse(model, text)
    assert candidate["sampling"]["format_valid"]
    assert candidate["content"] == "Let me check."
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"city": "123", "count": 2, "enabled": True}
    assert len(candidate["tool_calls"]) == 2


def test_glm_calls_and_stop_marker_agree():
    text = 'reason</think>Checking.<tool_call>weather<arg_key>city</arg_key><arg_value>123</arg_value></tool_call>'
    candidate = parse("glm-5p3-flash", text + "<|observation|>")
    assert candidate["sampling"]["format_valid"]
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"city": "123"}
    assert not parse("glm-5p3-flash", text + "<|user|>")["sampling"]["format_valid"]


@pytest.mark.parametrize("text", [
    'unclosed reasoning', 'x</think><tool_call>weather',
    'x</think><tool_call>\n<function=weather>\n<parameter=count>\nwrong\n</parameter>\n</function>\n</tool_call>',
    'x</think><tool_call>\n<function=weather>\n<parameter=count>\n1\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>',
    'x</think><tool_call><function=weather></function></tool_call>extra',
    'x</think>answer<|im_start|>user', 'x</think>answer</think>',
])
def test_invalid_output_is_retained_for_the_evaluator(text):
    candidate = parse("qwen3p8-27b", text)
    assert not candidate["sampling"]["format_valid"]
    assert candidate["sampling"]["raw_text"] == text


KIMI_RESPONSE = 'reason<|close|>think<|sep|><|open|>response<|sep|>café 🐢 東京<|close|>response<|sep|>'
KIMI_TOOLS = ('<|open|>tools<|sep|><|open|>call tool="weather" index="1"<|sep|>'
              '<|open|>argument key="city" type="string"<|sep|>123<|close|>argument<|sep|>'
              '<|open|>argument key="count" type="number"<|sep|>2<|close|>argument<|sep|>'
              '<|close|>call<|sep|><|close|>tools<|sep|>')
KIMI_STOP = '<|close|>message<|sep|>'


@pytest.mark.parametrize("suffix", ["", KIMI_STOP, "<|close|>", "<|close|>mess"])
def test_kimi_stop_normalization_preserves_closed_channels(suffix):
    candidate = parse("kimi-k3", KIMI_RESPONSE + KIMI_TOOLS + suffix)
    assert candidate["sampling"]["format_valid"]
    assert candidate["content"] == "café 🐢 東京"
    assert candidate["reasoning_content"] == "reason"
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"city": "123", "count": 2}


def test_kimi_literals_cannot_close_response_or_inject_xml():
    before = TOKENIZER.encode('reason<|close|>think<|sep|><|open|>response<|sep|>')
    literal = '<|close|>response<|sep|><!DOCTYPE x [<!ENTITY y "bad">]>&y;'
    after = TOKENIZER.encode('<|close|>response<|sep|>')
    tokens = before + TOKENIZER._encode_text_piece(literal) + after
    result = parse_completion(RENDERER, MODEL_SPECS["kimi-k3"], tokens, "stop")
    assert result["sampling"]["format_valid"]
    assert result["content"] == literal


@pytest.mark.parametrize("text", [
    'reason<|close|>think<|sep|><|open|>response<|sep|>unclosed',
    KIMI_RESPONSE + KIMI_TOOLS.replace('type="number"', 'type="object"'),
    KIMI_RESPONSE + KIMI_TOOLS.replace('index="1"', 'index="2"'),
    KIMI_RESPONSE + '<|open|>tools<|sep|><|close|>tools<|sep|>',
    KIMI_RESPONSE + '<|open|>!DOCTYPE x<|sep|>',
])
def test_kimi_rejects_truncated_or_invalid_structures(text):
    assert not parse("kimi-k3", text)["sampling"]["format_valid"]


@pytest.mark.parametrize("model", list(MODEL_SPECS))
def test_length_finish_is_invalid_even_with_complete_inner_body(model):
    text = KIMI_RESPONSE if model == "kimi-k3" else "reason</think>answer"
    assert not parse(model, text, reason="length")["sampling"]["format_valid"]


def test_model_stops_match_native_training_boundaries():
    assert stop_sequences(RENDERER, MODEL_SPECS["kimi-k3"]) == [KIMI_STOP]
    assert stop_sequences(RENDERER, MODEL_SPECS["glm-5p3-flash"]) == ["<|user|>", "<|observation|>"]


@pytest.mark.skipif(os.environ.get("SMITHTUNE_TOKENIZER_TESTS") != "1", reason="requires pinned cached tokenizer assets")
@pytest.mark.parametrize("model", list(MODEL_SPECS.values()), ids=lambda m: m.name)
@pytest.mark.parametrize("arguments", [
    {"city": "123", "count": 2, "enabled": True},
    {"city": "\n leading & trailing \n", "count": 0, "enabled": False},
    {"city": "", "nested": {"items": [1, True, None, "two"]}},
    {},
])
def test_official_tokenizer_generated_completions_round_trip(model, arguments):
    from smithtune.hf_rendering import _normalize_messages
    from smithtune.rendering import load_training_renderer

    renderer = load_training_renderer(model)
    messages = [{"role": "user", "content": "Weather?"}]
    assistant = {"role": "assistant", "content": "Checking café 🐢 東京", "reasoning_content": "Choose a tool.",
                 "tool_calls": [{"type": "function", "id": "original", "function": {
                     "name": "weather", "arguments": json.dumps(arguments),
                 }}]}
    prefix = renderer.prompt_tokens(messages, TOOLS)
    full = renderer.tokenizer.apply_chat_template(_normalize_messages([*messages, assistant]), tools=TOOLS,
                                                   tokenize=False, add_generation_prompt=False)
    prefix_text = renderer.tokenizer.decode(prefix)
    assert full.startswith(prefix_text)
    completion = full[len(prefix_text):]
    if model.renderer == "hf_prefix_kimi_k3":
        completion = completion.removesuffix("<|end_of_msg|>")
    else:
        completion = completion.removesuffix("\n")
    if model.renderer == "hf_prefix_glm53_flash":
        completion += "<|observation|>"
    encode = getattr(renderer.tokenizer, "_encode_text_piece", None)
    tokens = list(encode(completion, allow_special_tokens=True) if encode else
                  renderer.tokenizer.encode(completion, add_special_tokens=False))
    candidate = parse_completion(renderer, model, tokens, "stop", TOOLS)
    assert candidate["sampling"]["format_valid"], candidate
    assert candidate["content"] == assistant["content"]
    assert candidate["reasoning_content"] == assistant["reasoning_content"]
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == json.loads(assistant["tool_calls"][0]["function"]["arguments"])


@pytest.mark.parametrize("schema", [{"type": ["string", "null"]}, {"anyOf": [{"type": "string"}, {"type": "null"}]}])
def test_nullable_string_argument_does_not_become_a_number(schema):
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": schema},
    }}}]
    candidate = parse("qwen3p8-27b", 'r</think><tool_call><function=weather><parameter=city>\n123\n</parameter></function></tool_call>', tools=tools)
    assert candidate["sampling"]["format_valid"]
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"city": "123"}


def test_ambiguous_string_number_union_fails_without_guessing():
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }}}]
    candidate = parse("qwen3p8-27b", 'r</think><tool_call><function=weather><parameter=city>\n123\n</parameter></function></tool_call>', tools=tools)
    assert not candidate["sampling"]["format_valid"]


def test_kimi_json_object_block_and_escaped_argument_key():
    json_call = ('<|open|>tools<|sep|><|open|>call tool="weather" index="1"<|sep|>'
                 '<|open|>json type="object"<|sep|>{"city":"123"}<|close|>json<|sep|>'
                 '<|close|>call<|sep|><|close|>tools<|sep|>')
    result = parse("kimi-k3", KIMI_RESPONSE + json_call)
    assert result["sampling"]["format_valid"]
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"city": "123"}
    result = parse("kimi-k3", KIMI_RESPONSE + KIMI_TOOLS.replace('key="city"', 'key="city&amp;&quot;"'))
    assert result["sampling"]["format_valid"]
    assert json.loads(result["tool_calls"][0]["function"]["arguments"])["city&\""] == "123"


@pytest.mark.parametrize("value", ["null", "None"])
@pytest.mark.parametrize("schema", [{"type": ["string", "null"]}, {"anyOf": [{"type": "string"}, {"type": "null"}]}])
def test_nullable_string_null_spellings_fail_closed(value, schema):
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": schema},
    }}}]
    text = f'r</think><tool_call><function=weather><parameter=city>\n{value}\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert not candidate["sampling"]["format_valid"]
    assert "ambiguous nullable string" in candidate["sampling"]["parse_error"]
    assert candidate["sampling"]["raw_text"] == text


@pytest.mark.parametrize("schema,value,expected", [
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "2", 2),
    ({"type": ["boolean", "null"]}, "True", True),
    ({"type": ["object", "null"]}, '{"nested":[1,true,null]}', {"nested": [1, True, None]}),
    ({"type": ["integer", "null"]}, "null", None),
    ({"type": ["integer", "null"]}, "None", None),
    ({"type": ["integer", "number"]}, "2", 2),
    ({"type": ["integer", "number"]}, "2.5", 2.5),
])
def test_non_string_unions_parse_matching_native_values(schema, value, expected):
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"value": schema},
    }}}]
    text = f'r</think><tool_call><function=weather><parameter=value>\n{value}\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert candidate["sampling"]["format_valid"], candidate
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"value": expected}


@pytest.mark.parametrize("value", ["123", "True", "False", "None", "null", '{"x":1}', "[1]", '"123"', "{"])
@pytest.mark.parametrize("schema", [{}, {"$ref": "#/$defs/value"}])
def test_unknown_or_referenced_types_do_not_guess_json_looking_strings(value, schema):
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"value": schema},
    }}}]
    text = f'r</think><tool_call><function=weather><parameter=value>\n{value}\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert not candidate["sampling"]["format_valid"]


def test_untyped_additional_property_numeric_string_is_not_coerced():
    text = 'r</think><tool_call><function=weather><parameter=undeclared>\n123\n</parameter></function></tool_call>'
    assert not parse("qwen3p8-27b", text)["sampling"]["format_valid"]


@pytest.mark.parametrize("ref", ["#/$defs/Code", "#code"])
@pytest.mark.parametrize("root_ref", [False, True])
def test_saved_local_argument_refs_preserve_numeric_string_enum(ref, root_ref):
    parameters = {"type": "object", "$defs": {"Code": {
        "$anchor": "code", "type": "string", "enum": ["123"],
    }}, "properties": {"value": {"$ref": ref}}}
    if root_ref:
        properties = parameters.pop("properties")
        parameters.pop("type")
        parameters["$defs"]["Parameters"] = {"type": "object", "properties": properties}
        parameters["$ref"] = "#/$defs/Parameters"
    tools = [{"type": "function", "function": {"name": "weather", "parameters": parameters}}]
    text = 'r</think><tool_call><function=weather><parameter=value>\n123\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert candidate["sampling"]["format_valid"], candidate
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"value": "123"}


@pytest.mark.parametrize("definition,value,expected", [
    ({"type": "object"}, '{"nested":[1,true]}', {"nested": [1, True]}),
    ({"type": "integer"}, "2", 2),
    ({"anyOf": [{"$ref": "#/$defs/Scalar"}, {"type": "null"}]}, "2", 2),
])
def test_saved_refs_reveal_object_scalar_and_nullable_types(definition, value, expected):
    parameters = {"type": "object", "$defs": {"Value": definition, "Scalar": {"type": "integer"}},
                  "properties": {"value": {"$ref": "#/$defs/Value"}}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": parameters}}]
    text = f'r</think><tool_call><function=weather><parameter=value>\n{value}\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert candidate["sampling"]["format_valid"], candidate
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"value": expected}


@pytest.mark.parametrize("reference,definitions", [
    ("#/$defs/missing", {}), ("#missing", {}),
    ("https://example.com/schema.json", {}),
    ("#/$defs/A", {"A": {"$ref": "#/$defs/A"}}),
    ("#/$defs/A", {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}),
    ("#/$defs/A", {"A": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}}),
    ("#/$defs/A", {"A": {"$ref": "#/$defs/B", "type": "string"}, "B": {"type": "integer"}}),
])
def test_missing_cyclic_external_or_ambiguous_refs_fail_without_retrieval(reference, definitions):
    parameters = {"type": "object", "$defs": definitions, "properties": {"value": {"$ref": reference}}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": parameters}}]
    text = 'r</think><tool_call><function=weather><parameter=value>\n123\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert not candidate["sampling"]["format_valid"]



def test_recursive_object_reference_exposes_concrete_type_without_expanding_children():
    parameters = {"type": "object", "properties": {"value": {"$ref": "#"}}}
    tools = [{"type": "function", "function": {"name": "weather", "parameters": parameters}}]
    text = 'r</think><tool_call><function=weather><parameter=value>\n{"value":{}}\n</parameter></function></tool_call>'
    candidate = parse("qwen3p8-27b", text, tools=tools)
    assert candidate["sampling"]["format_valid"], candidate
    assert json.loads(candidate["tool_calls"][0]["function"]["arguments"]) == {"value": {"value": {}}}

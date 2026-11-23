from smithtune.artifacts import _jsonl_dump
from smithtune.providers import lora_sft


def test_prepared_rows_keep_unicode_line_separators_inside_strings(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "first second third\x85end"}]}, {"messages": []}]
    path = tmp_path / "train.jsonl"
    _jsonl_dump(path, rows)

    assert lora_sft.read_jsonl(path) == rows

# Runtime renderer reference

`runtime-renderer.json` was generated with `renderer_snapshot()` from
`test_runtime_packaging.py`, using the unmodified Fireworks source at commit
`09bbe1170c804eb5e5d7e7286f0feeef4e26055a` and the original
`tinker_cookbook/supervised/common.py` from the 0.4.3 wheel recorded in the
runtime's `upstream.json`. The helper was loaded directly to avoid unrelated
imports from `tinker_cookbook.supervised.__init__`.

The fixture records character-tokenized model inputs, targets, loss masks, and
replay prompts for a multi-turn conversation with tools and reasoning. It checks
that packaging and vendoring preserve those operations without downloading a
model tokenizer or calling a provider. It does not establish parity with a
particular Hugging Face tokenizer revision.

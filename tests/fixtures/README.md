# Runtime renderer reference

`runtime-renderer.json` was generated with `renderer_snapshot()` from
`test_training_dependency.py`, using the unmodified Fireworks source at commit
`09bbe1170c804eb5e5d7e7286f0feeef4e26055a` and the original
`tinker_cookbook/supervised/common.py` from the 0.4.3 wheel (SHA-256
`4a794a015a41462ff18a8085ea5a282139cca54325cbd38b4736b6916b5a40ce`).
The helper was loaded directly to avoid unrelated
imports from `tinker_cookbook.supervised.__init__`.

The fixture records character-tokenized model inputs, targets, loss masks, and
replay prompts for a multi-turn conversation with tools and reasoning. It checks
that upstream dependency updates preserve those operations without downloading a
model tokenizer or calling a provider. It does not establish parity with a
particular Hugging Face tokenizer revision.

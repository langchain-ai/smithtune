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

# Per-assistant provenance fixtures

`test_assistant_bindings.py` and `curation_fakes.py` construct synthetic wire
responses; no private captured conversation or run data is committed. The wire
shapes were checked against the LangSmith backend source available during this
change (2026-09-17):

- `smith-go/runs/v2/trajectories/types.go`, `projection.go`, and
  `smith-polly/src/trajectory.mts`: native messages and UI `metadata.run_id`.
- `smith-go/runs/v2/messages/v2/process.go`, `dedup.go`, and `types.go`, plus
  `messages/process_test.go` and `batch_test.go`: StandardMessage identities and
  LangChain `generations` / constructor `kwargs` / message `data` envelopes.
  Input history can be attributed to its consuming run; UI attribution is not
  sufficient to prove production.
- `smith-go/runs/v2/query/types.go` (`QueryTraceRequestQueryParams`,
  `QueryTraceResponseBody`) and `parse.go`: `/api/v2/traces/{id}/runs` returns all
  runs in one response and has no cursor. Omitting both time bounds includes all
  history. `/api/v2/runs/query` and `/v1/trajectory` have continuation cursors.
- `smith-backend/app/schemas.py` (`DatasetCreate`): accepts a caller-assigned UUID,
  enabling attributable reconciliation after ambiguous creation.

Capture accepts only a unique stable message ID present in an LLM output and
matching normalized content/calls. Output identity is different from run identity;
unused tools are retained, explicit empty tools are verified, and missing tool
availability fails. Fixtures cover supplied history, reversed run order, duplicate
retry identities, repeated message occurrences, schema changes and removed tools.
Backend schema inspection and these fake responses are evidence of the supported
wire formats, not a live verification of every tracing integration. A live source
with missing stable output IDs or explicit `invocation_params.tools` needs improved
instrumentation before reliable automatic provenance can be established.

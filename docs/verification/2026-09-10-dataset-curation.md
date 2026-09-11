# Dataset curation: live verification

Verified September 10, 2026 (Pacific), on `jake/dataset-curation`, implementation commit `8b7a329`.
Installed official LangSmith CLI **v0.2.54** in `~/.local/bin/langsmith` using the release installer and checksum verification. Used the existing saved authentication profile and the LangChain Inc workspace.

## Results

| Check | Result |
| --- | --- |
| Existing `chat-langchain` trace import | Passed: 18 messages preserved exactly, including nine matched tool calls/results |
| Existing `chat-langchain` server-side thread import | Passed: same complete source messages, correct dedicated thread source field |
| Synthetic standalone trace selection | Passed: root feedback `correctness >= 0.9` plus tag filter selected three traces without thread IDs |
| Synthetic thread selection | Passed: six matching roots deduplicated into three threads |
| Synthetic message preservation | Passed: recorded system prompts, absent system prompt, and full two-turn thread history matched the expected content |
| Existing `prepare`, trace scope | Passed: three accepted, zero rejected; real Qwen tokenizer/renderer; one row in each split |
| Existing `prepare`, thread scope | Passed: three accepted, zero rejected; real Qwen tokenizer/renderer; one thread in each split |

No trajectory messages were removed during preparation. The default 80/10/10 fractions produce one row per split with these small three-example datasets. No model calls, training, or deployments were started. No product code changes were needed for this verification.

## LangSmith artifacts

| Dataset | Examples |
| --- | --- |
| [Existing source — trace import](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/509eb534-6528-4dd9-8a3b-e237d8b89c72) | 1 |
| [Existing source — thread import](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/b9b2610c-7356-4e5f-a575-b46b9f018f3a) | 1 |
| [Synthetic standalone traces](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/57cf087d-413e-4754-a043-5f9140480bdb) | 3 |
| [Synthetic full threads](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/2c201d16-c0f7-40bd-9975-df1e0933b524) | 3 |

Synthetic tracing project: `smithtune-live-curation-20260911-022936`, ID `5adce596-ef0e-49d5-8767-8076b4f11998`. It contains nine small synthetic root runs and root feedback. Verification artifacts remain available for inspection; existing source traces were not modified.

Local selections, receipts, source comparison hashes, fixture expectations, downloaded preparation inputs, and manifests are under ignored `data/live-20260910/`. Summary files: `verification-summary.json` and `fixture-verification-summary.json`.

## Limits and observations

The retained `chat-langchain` source has tool calls but lacks `extra.invocation_params.tools` in the inspected representative LLM run. Its import and trajectory validation passed; `capture-contract` correctly refused to invent the missing tool schemas. Full preparation was therefore verified with the synthetic fixtures, not that older tool trace.

The existing `prepare` command emits dataset-export output before its final JSON result. The verification runner initially tried to parse all stdout as one JSON object; preparation itself had succeeded. Verification then read the saved manifest without rerunning the completed import or preparation. That existing output behavior was left unchanged.

This is a live functional smoke test, not a scale or live failure-injection test. Pagination, sampling stability, ambiguous-write handling and legacy compatibility remain covered by the previously passing 268-test suite.

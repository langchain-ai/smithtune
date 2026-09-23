# Trajectory council audit

`dataset triage` selects training examples. `evaluate` tests model responses
through replay. These are separate flows.

## Current path

1. Reuse the saved full conversations from the dataset download path.
2. Filter any conversation with media in its messages or source runs.
3. Let a Deep Agent coordinator dispatch one task per conversation and council
   member through Python code mode.
4. Send the full ordered conversation messages and selection rubric in one
   request. The judge has no tools and returns only `keep` (0 or 1) and `reason`.
5. Require a completed council and use its majority label for the conversation.
6. Import kept conversations with the exact saved messages and tool schemas.

There are no per-turn votes, message previews, paged reads, or character caps.
If a provider rejects the full request because it exceeds its context window,
the whole conversation gets a filter label of 0. Other request failures remain
incomplete. The CLI never shortens the input or invents a score after a failed
request. Recorded conversation instructions are data, not judge instructions.

## Checks

Local tests cover conversation deduplication, complete message preservation,
media in early and late turns, provider context rejection, strict majority,
resume, and unchanged messages through dataset import and preparation. The
coordinator tests run the real Deep Agents graph with a local model.

Provider transport checks cover the official Fireworks endpoint and OpenAI
Responses for Terra. The Fireworks adapter retains reasoning fields for the
coordinator's tool calls. Package checks verify the packaged skill and the optional
agent dependency outside the checkout.

The full saved batch contains 100 selected roots in 95 conversations. Three
conversations contain media. This plans 276 votes for 92 full conversations,
not one set of votes for each source trace. Private source data and live test
artifacts stay under ignored run directories. Live validation results are
recorded in the pull request.

The final council uses DeepSeek, GLM-5.3-Flash, and Terra. A live check used 15
saved full conversations and 16 concurrent requests. All 42 votes completed on
the first attempt: 8 kept, 6 dropped, 1 media filter, no context filters, and no
incomplete labels. GLM used low reasoning; DeepSeek and Terra used reasoning
off. Resume made no new calls. Four kept conversations passed the training checks.

An earlier Muse council check uploaded its 3 kept conversations and read them
back with identical messages and saved tool contracts. Re-importing into the
same dataset skipped all 3 without duplicates or updates.

Label accuracy still needs comparison with a human-reviewed sample. A valid
JSON score does not establish that the model's quality decision is correct.

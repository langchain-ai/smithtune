# Dataset curation

[Back to the workflow](../README.md)

## Create a dataset from trajectories

Download and preview a whole-conversation selection:

```bash
smithtune dataset create data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name my-sft-dataset --limit 100 \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'

smithtune dataset create data/datasets/my-sft --confirm
```

The first command downloads and reports candidates and pending council votes.
It makes no judge calls or LangSmith writes. The second uses the saved source,
destination, and council settings, judges the conversations, then uploads keeps.
`create` includes triage by default. Add `--no-triage` to the first command to
upload training-valid conversations without council review.

`pull` and `create` can generate a directory under `data/datasets/` when omitted;
the path is printed and returned as `checkpoint`. Keep it for subsequent commands.
Use the returned `dataset_id` with `prepare`. Zero eligible conversations create
no empty dataset.

Without time flags, selection covers the last 24 hours. `--end-time` defaults to
now; `--start-time` defaults to 24 hours before that end. Absolute bounds are
saved once. Explicit bounds use ISO 8601 timestamps with timezones. Filters apply
to root runs; see [LangSmith filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax).

Roots are grouped by thread ID, otherwise trace ID, before a seeded sample of
distinct conversations is selected. Default `--limit` is 100 (maximum 2000), and
`--seed` is 42. Root pagination is bounded; narrow the window/filter if exceeded.
Each selected thread includes earlier history and turns outside the window.
Rejected selections are not replaced to fill the limit. The limit caps this
checkpoint's selection, not the size of an existing destination dataset.

## Separate pull, triage, and push

```bash
smithtune dataset pull data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' --limit 100
smithtune dataset triage data/datasets/my-sft
smithtune dataset triage data/datasets/my-sft --confirm
smithtune dataset push data/datasets/my-sft --name my-sft-dataset
smithtune dataset push data/datasets/my-sft --confirm
```

Pull is read-only remotely. It fetches `/v1/trajectory` with system messages and
follows continuation cursors. Supporting V2 trace-run inputs, outputs, and
attachments establish provenance and detect media. Thread membership is checked
before/after download; a change excludes that conversation. This is not an atomic
snapshot guarantee for all changing run content.

Pull saves selected identities before fetching their conversations. Completed
conversations and terminal exclusions are skipped on resume; interrupted
conversations restart from the first page. No raw responses or run trees are
retained. Transient source failures remain incomplete and retryable. Pull uses
at most four workers and one bounded retry layer for idempotent reads.

## Label full trajectories with an agent council

Triage reads only completed local conversation files. Each council member sees
the same full unchanged messages and per-assistant tool bindings as untrusted
evidence. Recorded tools are never executable judge tools. A keep requires all
council slots to succeed and a strict majority; ties drop. Media, malformed
messages, unsupported tools, and missing provenance are excluded before judging.
A provider context-window rejection excludes the whole conversation without
truncation. Other judge errors remain incomplete.

The default council is DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks, plus
GPT-5.6 Terra on OpenAI. The first model also coordinates Deep Agent Python code
mode. Install the optional `deepagents` extra using the README's installation
command and override file. Set `FIREWORKS_API_KEY` and `OPENAI_API_KEY` for judging;
`LANGSMITH_API_KEY` is needed for source reads and uploads. Judging incurs charges.

Choose a council and project rules on preview:

```bash
smithtune dataset triage data/datasets/my-sft \
  --judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra \
  --rule 'Drop answers that claim an action succeeded without evidence.'
```

Aliases can be mixed with `provider:model`. Direct Anthropic uses
`anthropic:claude-sonnet-5` and `ANTHROPIC_API_KEY`; `anthropic-gateway:<model>`
uses `LANGSMITH_GATEWAY_API_KEY`. Baseten Model APIs use `BASETEN_API_KEY` and
`baseten:<model-slug>` at `https://inference.baseten.co/v1`; custom deployment
URLs are not supported for council judges. The direct API runner remains
available as `--runner api`; the default is `deepagent`.

Actual rubric text, rules, models, and request settings are saved. Preview
settings may change before voting; after votes exist, changing judging instructions
or request settings requires a fresh checkpoint. Concurrency (at most 16) and
retry settings may change. Successful durable votes are never repeated. A response
lost before saving may require another paid call. Coordinator/skill software
versions do not gate resume.

Results print kept/dropped/incomplete counts, exclusions, and reasons. Individual
current votes are in `triage.jsonl`. There are no separate labels or report files.
Use `smithtune skill export --output ./skills` for the portable skill and rubric.
See the [audit](trace-labeling-audit.md) for evidence and limitations.

## Portable tool bindings

One LangSmith example holds the whole unchanged `inputs.messages`, null `outputs`,
and source metadata. Bindings live in the same example:

```json
{
  "metadata": {
    "source_workspace_id": "<workspace>",
    "source_project_id": "<project>",
    "source_scope": "thread",
    "source_scope_id": "<thread>",
    "smithtune_source": {
      "schema_version": 1,
      "assistant_runs": [
        {"message_index": 1, "run_id": "<producing-run>", "trace_id": "<trace>", "tools": []}
      ]
    }
  }
}
```

Every assistant occurrence has exactly one binding indexed by its original
zero-based message position. `tools` contains all available function definitions,
including unused tools. `[]` means verified empty availability. Missing evidence
is not empty. Different turns may add/remove tools or change descriptions and
schemas; definitions are never merged across turns.

Capture requires a unique stable assistant message ID found in an LLM run's
output, plus matching normalized visible content and calls. Message IDs are not
run IDs. Supplied input history and trajectory UI `metadata.run_id` can point to
a consuming run; neither proves production. Order, timestamps, and fuzzy text
matching are not used. StandardMessage output lists, LangChain ChatGeneration
messages, and identified OpenAI choice messages are understood. Explicit
`extra.invocation_params.tools` (including `[]`) is required on the producing run.
Unsupported formats, repeated/ambiguous output identities, and modified outputs
exclude the whole conversation with the message position and candidate run IDs.
If these fields are absent, supply stable producing-output identities and explicit
tool availability through instrumentation, then pull a new checkpoint.

Preparation consumes exported bindings without the original curation directory
or source-run access. For unbound exports it attempts the same verified capture;
a legacy union cannot reconstruct historical availability. Triaged metadata also
binds the council decision to both messages and tool evidence.

External consumers must read the full example, including metadata, and select
the target's binding. Passing only `example.inputs` to an arbitrary LangSmith
evaluator does not automatically supply tools. See [target rendering and replay](reference.md#per-assistant-tools-and-target-rendering).

## Upload and extend

Use `--dataset-id '<id>'` instead of `--name` to extend an existing dataset in the
source workspace. Once bound, the destination is fixed; flagless reruns or the
same name/ID continue it. A new selection or extended thread needs a fresh
checkpoint, with fresh whole-conversation triage if the destination was triaged.

Push uses saved conversations and votes only; there are no source/tool reads.
Incomplete councils block upload. Sources match by workspace, project, scope,
and scope ID. Equivalent messages, bindings, and triage provenance are skipped.
An extension must preserve the exact old message prefix and bindings. Changed
prior tools conflict even if text is unchanged. Updates preserve the example ID
and unrelated destination metadata. Legacy destinations without bindings need
an explicit validated metadata upgrade outside this workflow, or a new dataset.

Push indexes a pinned destination version in memory and counts all its examples.
Writes are sequential with deterministic example IDs and a saved pending write.
Timeouts and conflicts are reconciled by reading and comparing remote evidence.
Pending dataset creation records its requested UUID before sending; name equality
alone never permits adoption of an unrelated dataset. No rollback or deletion is
performed. Run only one import per destination at a time.

Command results include `existing_before`, created/updated/skipped/rejected counts
for the attempt, and `final_size`. Prior landed writes appear as skipped on retry.
A final size over `--limit` is reported with a warning; counts assume no concurrent
external writer. Before triage finishes the kept count is unknown.

## Recovery and compatibility

```bash
smithtune dataset resume data/datasets/my-sft
smithtune dataset resume data/datasets/my-sft --confirm
```

Resume previews pending recorded stages; confirmation continues them in order.
It never pushes after incomplete triage. A completed resume makes no network
calls; explicit repeated push may reconcile again. A standalone completed pull
has no pending stages until triage or push is requested. Missing destinations
return a concrete push command.

The curation footprint is:

```text
my-sft/
  checkpoint.json
  triage.jsonl                 # created when judging begins
  conversations/<sha256>.json  # one unchanged conversation and its bindings
  .smithtune.lock
```

The command acquires one directory lock. Conversation hashes cover messages and
bindings; changing either invalidates votes. Never edit or refetch judged evidence.
Failures return the checkpoint path, source identity, and resume command. Earlier
successful writes stay in place. An interruption between publishing a file and
recording completion can cause that one conversation to be fetched again; an
unreferenced file is never judged.

Exit codes: 0 for ordinary previews/completion; 1 for incomplete work (including a
resume preview with pending work); 2 for usage or incompatible directories.

Old `selection.json`/`snapshot.json` layouts are not migrated. Start `dataset create
NEW_DIR` with source flags and `--dataset-id EXISTING_ID` to extend a compatible
destination. Removed `--triage-dir`, `--run-dir`, `--output`, and triage source flags
print replacement hints. Hidden `triage --output-dir DIR` and `--dry-run` aliases
remain when unambiguous. Preparation/training artifacts have their own formats
and are separate from the curation checkpoint.

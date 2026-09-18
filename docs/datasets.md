# Dataset curation

[Back to the workflow](../README.md)

## Create a dataset from trajectories

Create a dataset directly from a tracing project and filters:

```bash
smithtune dataset create \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name my-sft-dataset \
  --limit 100 \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'
```

Each matching root selects its whole thread when it has a thread ID, otherwise
its single trace. Thread examples include turns outside the filter window. Use
the returned dataset ID in `prepare`.

`--end-time` defaults to now, and `--start-time` defaults to 24 hours before the
resolved end. Omit both for the last 24 hours, or pass either or both as ISO 8601
timestamps with timezones. These defaults also apply to a new `dataset triage` run.

- Filters apply to trace root runs. The example selects correctness feedback of at least 0.9; see [filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax)
- `--limit` is required, at most 2000. Querying stops once that many distinct trajectories are found, in the order LangSmith returns roots; no sampling is applied
- Each trajectory is fetched with the trajectory API and stored as one example; `--concurrency` imports up to 4 at once (the default). Transient fetch failures are retried up to three times; example writes are never retried
- Use `--name` for a new dataset or `--dataset-id` for an existing dataset in the same workspace. If an import fails, inspect the returned receipt before retrying; uploads do not resume automatically

Direct creation and triage both save complete examples under `conversations/` in a local run
directory, defaulting to `data/datasets/<generated-id>/`. Creation saves each
trajectory before uploading it, alongside the selection and import receipt.
Use `dataset create --run-dir <directory>` to choose a location. The returned
`run_dir` identifies the saved files; they remain on disk after upload or failure.

Before each example upload, creation validates the saved messages and captures
the tool union from all source LLM runs, including tools that were not called.
It excludes whole trajectories with malformed tool pairs, repeated tool-call IDs,
unsupported content such as images, conflicting tool schemas, unknown tools,
invalid arguments, or system messages after the first position. A leading system
message is allowed and preserved; messages are never repaired or truncated.
Compatible optional tool arguments and description changes use the same merge
rules as `prepare`.

The JSON result includes `rejected`; new-dataset receipts include `rejections`
with reason codes, source identities, and saved conversation paths. Rejected
trajectories are not replaced with additional selections, so the uploaded count
can be less than `--limit`. If all selected trajectories are rejected, the new
dataset is empty and the receipt still records every rejection. Source-read or
schema-resolution failures stop the import rather than counting as rejections.
These checks need LangSmith access, but no training provider or tokenizer;
reasoning policy, rendering compatibility, and context limits remain in `prepare`.

To add trajectories to an existing dataset, use the same source flags with
`--dataset-id` instead of `--name`:

```bash
smithtune dataset create \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --dataset-id '<dataset-id>' \
  --start-time 2026-09-08T00:00:00Z --end-time 2026-09-15T00:00:00Z \
  --limit 100
```

Sources match by workspace, project, scope and scope ID. New trajectories are
added; unchanged ones are skipped. Longer snapshots update the existing example
only when its saved messages are an exact prefix, preserving its ID and unrelated
metadata. Shorter or conflicting snapshots, or duplicate sources already in the
destination, stop the import. Extending a triaged example requires fresh passing
triage. Run only one import into a dataset at a time.

Existing-dataset imports download with bounded concurrency and write sequentially.
The receipt records created, updated, skipped and rejected counts, an action log,
and any pending write whose outcome needs checking. Rejected actions record their
reason and saved conversation path without creating or updating an example.
Existing remote examples are never deleted. Earlier successful writes remain if a
later trajectory fails.

## Label full trajectories with an agent council

Optionally use `dataset triage` to select training trajectories with model judges before
creating a dataset. Each judge sees the full trajectory; a majority keep vote
selects it for import once all judges finish. Ties are dropped.

The default council is DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks, plus
GPT-5.6 Terra on OpenAI. Set `LANGSMITH_API_KEY`, `FIREWORKS_API_KEY`, and
`OPENAI_API_KEY`, then install the optional agent support:

```bash
uv tool install --upgrade --python 3.12 \
  'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git'
```

**1. Download and preview:**

```bash
smithtune dataset triage data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --limit 100
```

This saves trajectories and a judging plan locally, without calling judges.
Each judge receives the full saved message list, including system messages.
`--limit` samples root traces; roots in the same thread become one trajectory,
including history outside the time window. Repeat an interrupted download to
reuse saved progress. Omit the directory to generate one under `data/datasets/`.
Select runs that contain trajectories. For example, use
`--filter 'eq(run_type,"chain")'` for agent runs in a project that also records
standalone prompt-rendering runs. A source with no messages stops the download
and reports its ID.

**2. Run the judges:**

```bash
smithtune dataset triage data/datasets/my-sft --confirm
```

Judging uses paid inference. The CLI saves each vote; repeat this command to
retry incomplete work with the saved settings. Completed runs make no new model
calls. Read `report.md` for results, `labels.jsonl` for keep/drop labels, and
`judgments.jsonl` for individual votes.

- Multimodal trajectories are excluded before judging.
- If any judge rejects a trajectory as too long, the whole trajectory is
  excluded without truncation.
- Other request failures remain incomplete and can be retried.

**3. Import accepted trajectories:**

```bash
smithtune dataset create --triage-dir data/datasets/my-sft --name selected-sft --confirm
```

Import uses the saved messages and tool schemas; unsupported training content
is excluded. Pass the returned dataset ID to `prepare`. To add to an existing
dataset, replace `--name` with `--dataset-id '<dataset-id>'`.

Import validates the saved snapshot once and releases its run evidence before
accessing the destination. Snapshot fingerprints are computed incrementally,
without making additional whole-snapshot JSON copies. Hashes and saved votes
remain compatible with earlier runs; the parsed snapshot still needs to fit in
memory.

Use a new triage directory when trajectories or judging settings change.
Extended trajectories need fresh passing triage. After a partial import,
inspect `dataset-import.json` before retrying; uploads do not resume automatically.

### Change the council or selection rules

Add `--judges` and `--rule` to the preview command:

```bash
smithtune dataset triage data/datasets/custom-council \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra \
  --rule 'Drop answers that claim an action succeeded without evidence.'
```

Use any subset of these aliases, or `provider:model` for other models, such as
`anthropic:claude-sonnet-5`. The first model also runs the coordinator. Direct
Anthropic uses `ANTHROPIC_API_KEY`; `anthropic-gateway:<model-id>` uses
`LANGSMITH_GATEWAY_API_KEY`. The council and rules are saved for resume.

For Baseten judges, set `BASETEN_API_KEY` and use `baseten:<model-slug>` from
[Baseten Model APIs](https://docs.baseten.co/inference/model-apis/overview).
For example, this council mixes Baseten, Fireworks, and OpenAI:

```bash
--judges baseten:deepseek-ai/DeepSeek-V4.1-Flash,glm-5.3-flash,gpt-5.6-terra
```

Baseten uses `https://inference.baseten.co/v1`; no deployment is needed.
The first model also runs the coordinator. Baseten calls request reasoning off,
except GLM-5.3 variants, which require low reasoning. This supports managed
Model APIs; custom Baseten deployment URLs are not accepted.

Use `smithtune dataset triage --help` for filters, concurrency, and other options.
For coding agents, `smithtune skill export --output ./skills` exports the triage
skill and rubric. See the [trace-labeling audit](trace-labeling-audit.md)
for detailed behavior and limitations.

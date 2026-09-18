# Dataset curation

Use one directory throughout curation. Each trajectory becomes one LangSmith
dataset example. `create` combines the stages; use them separately when you want
to inspect or change the plan before continuing.

| Command | Behavior |
| --- | --- |
| `dataset pull DIR` | Download trajectories and per-assistant tool lists |
| `dataset triage DIR` | Preview council judging; `--confirm` runs it |
| `dataset push DIR` | Preview upload; `--confirm` uploads |
| `dataset create DIR` | Download, optionally judge, and upload |
| `dataset resume DIR` | Show pending work; `--confirm` continues it |

Without `--confirm`, `create` may download but never judges or uploads. `resume`
without `--confirm` reads only local state. A directory is generated if omitted
from a new `pull` or `create`; keep the returned `run_dir` for subsequent commands.
`dataset publish-splits` remains a separate operation on prepared data.

## Choose filters or council judging

Use existing feedback, metadata, tags, and errors when they express your criteria:

```bash
smithtune dataset create data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name my-sft-dataset \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'

smithtune dataset create data/datasets/my-sft --confirm
```

An explicit `--filter` with no council criteria takes the path with **no model
calls**. The CLI uses the expression you or your coding agent supply; it does not
translate natural language or guess whether a score means success. Use your
project's actual fields and [LangSmith filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax).

For criteria requiring trajectory content, add `--rule`:

```bash
smithtune dataset create data/datasets/grounded \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name grounded-trajectories \
  --filter 'eq(error, false)' \
  --rule 'Keep answers supported by the retrieved documentation.'
```

The filter narrows the source query first; the council then judges those
trajectories against the rule. Repeat `--rule` for multiple criteria. `--judges`
also requests council judging. Without a filter, `create` defaults to council
review. Use `--no-triage` to explicitly download and upload without review;
it cannot discard council rules or bypass an existing council plan.

The preview reports the selected path, pending stages, and the next command.
Confirm the workflow after reviewing it. Flagless reruns use the saved settings.

## Source selection

- Filters apply to **root runs**. A matching root selects its whole thread when
  it has one, otherwise its trace. Earlier turns outside the filter window remain
  part of the trajectory.
- `--end-time` defaults to now; `--start-time` defaults to 24 hours before it.
  Explicit times must use ISO 8601 with a timezone. Resume reuses the original bounds.
- `--limit` defaults to 100, up to 2000 distinct trajectories. Selection follows
  the order LangSmith returns roots and stops paging once the limit is reached.
  Rejections are not replaced, and the limit is not a target dataset size.
- `--concurrency` defaults to 4; downloads cap at 4 workers and uploads are sequential.
- Source settings are frozen in the directory. Use a new directory to select a
  different project, time window, filter, or limit.

Whole trajectories with invalid messages or unsupported tool evidence are
excluded before council calls and upload. The triage preview lists rejection
reasons and counts only eligible judge tasks. Recorded messages are preserved. The result includes
`eligible` and `rejected` counts; saved units retain validation errors. Model-specific
rendering and context checks remain in `prepare`.

## Label full trajectories with an agent council

```bash
smithtune dataset pull data/datasets/reviewed \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --limit 100

smithtune dataset triage data/datasets/reviewed \
  --rule 'Keep complete, useful solutions.'
smithtune dataset triage data/datasets/reviewed --confirm

smithtune dataset push data/datasets/reviewed --name reviewed-sft
smithtune dataset push data/datasets/reviewed --confirm
```

`triage` reads only downloaded trajectories. Its default council is DeepSeek
V4.1 Flash and GLM-5.3-Flash on Fireworks plus GPT-5.6 Terra on OpenAI, managed by a
Deep Agent. Install the optional `deepagents` extra from the README and configure
credentials for the selected providers. Use `--judges` to choose aliases or
`provider:model`, and `--concurrency` to set concurrent judge tasks (default 4,
maximum 16). Direct Anthropic uses `ANTHROPIC_API_KEY`; the Anthropic gateway uses
`LANGSMITH_GATEWAY_API_KEY`.

Every council member judges the full trajectory. All votes must finish; a strict
majority keeps it, and ties drop it. Multimodal and provider context-window
rejections are excluded without truncation. Request failures remain incomplete.
`labels.jsonl` contains `trajectory_id`, `keep`, and `reason`; `report.md` summarizes
them. Rules and models can change during preview, but changing them after votes
requires a new directory.

Push respects any council plan attached to the directory and waits for judging
to finish. Pull followed directly by push uses structural validation without a
council. No eligible examples means no empty remote dataset is created.

## Upload and recover

Use `--name` for a new dataset, or `--dataset-id` for an existing dataset in the
source workspace. The destination is saved for subsequent commands.

Sources match by workspace, project, scope, and scope ID. Unchanged examples are
skipped. Longer trajectories extend the existing example only when its messages
are an exact prefix, retaining its ID and unrelated metadata. Conflicting or
shorter histories stop the import. Existing tool lists and producing-run identities must also match the message
prefix. Use a new dataset when the destination lacks this per-assistant evidence.
Run one import per destination at a time.

```bash
smithtune dataset resume data/datasets/my-sft
smithtune dataset resume data/datasets/my-sft --confirm
```

Resume uses the original selection, completed downloads, votes, and upload
receipts. Unfinished trajectory downloads restart from their beginning. Uncertain
writes are checked by saved IDs before retrying; changed remote content stops
recovery for inspection. Preserve the directory and its content-verified files.

The positional directory replaces `--run-dir`, `--output`, and `--triage-dir`.
For older checkpoints, `resume DIR --confirm` can finish direct imports; saved
triage snapshots can use `triage`, `push`, and `resume`. Receipts predating the
recovery format still require inspection and a new directory with `--dataset-id`.

After upload, pass the returned dataset ID to `smithtune prepare`. Saved tool
availability travels with the examples; see [per-assistant tools](reference.md#per-assistant-tools-and-training-targets).

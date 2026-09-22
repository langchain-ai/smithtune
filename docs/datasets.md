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

For example, select initial reviewer runs with
`--filter 'and(eq(name,"reviewer"),eq(metadata_key,"re_review"),eq(metadata_value,false))'`.
To select a known root, use `--filter 'eq(id,"<root-run-id>")'` instead of a narrow
time window; it still selects that root's full thread. Set time bounds that include the root.

For criteria requiring trajectory content, add `--rule`:

```bash
smithtune dataset create data/datasets/grounded \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name grounded-trajectories \
  --filter 'eq(error, false)' \
  --rule 'Keep answers supported by the retrieved documentation.'
```

The filter narrows the source query first; the council then judges those
trajectories against the rule. Repeat `--rule` for multiple criteria, or pass a
file with `--rubric` as described below. `--rubric` and `--judges`
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
  Use `--concurrency 1` to reduce memory peaks for very large trajectories.
- Oversized trajectory response pages are retried at the same cursor with `page_size=1`.
  All messages and per-assistant tool metadata are preserved. If the smallest page
  still exceeds the server limit, the download stops without saving a partial trajectory.
- Source settings are frozen in the directory. Use a new directory to select a
  different project, time window, filter, or limit.

Whole trajectories with invalid messages or unsupported tool evidence are
excluded before council calls and upload. The triage preview lists rejection
reasons and counts only eligible judge tasks. Recorded messages are preserved. The result includes
`eligible` and `rejected` counts; saved units retain validation errors. After downloading,
`pull` and `create` also report selected roots, full threads, total traces, and
structural exclusion counts by reason, in stderr and the JSON `download_summary`.
The existing `downloaded` count includes excluded trajectories. Inspect files
listed in `snapshot.json` for each trajectory's error and source IDs. Model-specific
rendering and context checks remain in `prepare`.

Saved trajectories are read individually during judging and upload; queued judge
work holds IDs, not message bodies. Downloads retrieve messages and tool availability
from the trajectory endpoint without fetching raw run trees. Hashing and file
writes avoid whole-document copies. Each active full trajectory must still fit in memory. Legacy monolithic
snapshots remain readable but must fit in memory; new downloads use individual files.

## Label full trajectories with an agent council

```bash
smithtune dataset pull data/datasets/reviewed \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --limit 100

smithtune dataset triage data/datasets/reviewed \
  --rubric ./rubric.md
smithtune dataset triage data/datasets/reviewed --confirm

smithtune dataset push data/datasets/reviewed --name reviewed-sft
smithtune dataset push data/datasets/reviewed --confirm
```

Write the task description, keep/drop criteria, and concrete examples in a
UTF-8 `rubric.md` file. `--rubric` works with both `triage` and `create`; it
requests council judging even when a source filter is set. It can accompany
short additional `--rule` criteria and cannot be combined with `--no-triage`.

The preview saves the exact text as `selection_rubric` in `plan.json` and the
workflow checkpoint. Each judge receives it alongside the standard quality
checks and JSON output format. Confirm and resume use the saved text even if
the original file changes or is deleted. Before scoring, preview again with
`--rubric` to replace it. After votes start, changed criteria need a new directory.

For help choosing criteria, `smithtune skill export --output ./skills` exports
the general-purpose triage skill and discovery guide. The agent reads varied
traces, discusses concrete examples with the user, writes an agreed rubric,
and passes the file to the CLI. Inspect a small council batch before scoring
the larger pool. Domain-specific rules and private examples stay in local run
files; the CLI saves the rubric but does not verify human agreement.

`triage` reads only downloaded trajectories. Its default council is DeepSeek
V4.1 Flash and GLM-5.3-Flash on Fireworks plus GPT-5.6 Terra on OpenAI, managed by a
Deep Agent. The [README installation](../README.md#setup) includes the `deepagents` extra. Configure
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

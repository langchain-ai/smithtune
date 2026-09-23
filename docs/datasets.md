# Dataset curation

Use one directory throughout curation. Each trajectory becomes one LangSmith
dataset example. The recommended workflow is to download with `pull`, review
training-example quality with `triage`, and upload with `push`. Inspect each
stage’s result before continuing.

| Command | Behavior |
| --- | --- |
| `dataset pull DIR` | Download trajectories and per-assistant tool lists |
| `dataset triage DIR` | Review trajectory quality with an agent council; preview before confirming |
| `dataset push DIR` | Preview upload; `--confirm` uploads |
| `dataset resume DIR` | Show pending work; `--confirm` continues it |

`pull` downloads without model calls. `triage` and `push` preview without
`--confirm`; `resume` without `--confirm` reads only local state. A directory is
generated if omitted from a new `pull`; keep the returned `run_dir` for subsequent commands.
`dataset publish-splits` remains a separate operation on prepared data.

## Select candidates and review quality

Use source filters to select relevant candidates, then review their quality with
an agent council before uploading:

```bash
smithtune dataset pull data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --filter 'eq(name,"reviewer")'

smithtune dataset triage data/datasets/my-sft \
  --rule 'Keep reviews with actionable findings supported by the code.'
smithtune dataset triage data/datasets/my-sft --confirm

smithtune dataset push data/datasets/my-sft --name my-sft-dataset
smithtune dataset push data/datasets/my-sft --confirm
```

Filtering by agent name selects relevant trajectories; it does not establish
training quality. The council reviews the saved trajectories against your
criteria. Repeat `--rule` for multiple criteria, or pass a file with `--rubric`.
Choose models with `--judges`. Inspect the labels before pushing; council review
helps assess quality but does not guarantee good training data. Review stops once
the cumulative council-approved target is reached; remaining candidates stay saved.
If the pool is exhausted below target, repeat `pull DIR` and then `triage DIR --confirm`
to collect and review new candidates. Completed votes are reused. Small approved datasets
receive a reliability advisory after review and in the upload summary; it does
not block upload.

Use `--no-triage` on the first `pull` when trusted feedback or quality labels already
establish which trajectories meet your training criteria. For example, filter
on a validated correctness score using
`--filter 'and(eq(feedback_key,"correctness"),gte(feedback_score,0.9))' --no-triage`, then
proceed directly from `pull` to `push` without model calls. This mode counts
structurally usable trajectories toward the target and stops downloading when it
is met. The mode is fixed for that directory. Use thresholds suited
to your project; ordinary metadata or the absence of errors alone is not evidence
of quality. An attached council plan must finish before upload.

Use your project's actual fields and
[LangSmith filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax).
The CLI uses the expression you or your coding agent supply; it does not translate
natural language or guess whether a score means success. To select a known root,
use `--filter 'eq(id,"<root-run-id>")'` and time bounds that include it; this still
selects that root's full thread.

Previews report pending stages and the next command. Flagless reruns use the saved settings.

## Source selection

- Filters apply to **root runs**. A matching root selects its whole thread when
  it has one, otherwise its trace. Earlier turns outside the filter window remain
  part of the trajectory.
- `--end-time` defaults to now; `--start-time` defaults to 24 hours before it.
  Explicit times must use ISO 8601 with a timezone. Resume reuses the original bounds.
- `--target-count` is the cumulative goal (default 100): council-approved
  trajectories by default, or structurally usable trajectories with `--no-triage`.
  `--max-candidates` caps **new** candidates per round (default 1000, maximum 2000).
  These replace `--limit`. A round can collect fewer candidates if the source is
  exhausted, or if a `--no-triage` pull reaches its target.
- Council mode downloads the candidate pool before review; the council stops at
  the approved target. When a fully reviewed pool falls short, another explicit
  `pull DIR` collects unseen candidates. Neither review nor resume automatically
  starts another collection round. Previously encountered threads, including
  rejected and structurally excluded ones, are skipped.
- Up to three collection rounds are allowed. Interruptions and judge errors resume
  the same round. Source exhaustion stops collection earlier. After the limit or
  source exhaustion, the eligible subset can still be uploaded; broader source
  criteria require a new directory. Filters, time bounds, review mode, and limits
  stay fixed within a directory.
- Summaries report candidate and usable counts, review progress, the current round,
  and why collection stopped. Resume reuses candidate pages, downloaded evidence,
  and completed votes. Existing completed trajectory content is never refreshed.
- `--concurrency` defaults to 4; downloads cap at 4 workers and uploads are sequential.
  Use `--concurrency 1` to reduce memory peaks for very large trajectories.
- Oversized trajectory response pages are retried at the same cursor with `page_size=1`.
  All messages and per-assistant tool metadata are preserved. If the smallest page
  still exceeds the server limit, the whole trajectory is excluded and downloading
  continues. The saved rejection is reused on resume; partial messages are discarded.
- Source settings are frozen in the directory. Use a new directory to select a
  different project, time window, filter, or limit.

Whole trajectories with invalid messages or unsupported tool evidence are
excluded before council calls and upload. The triage preview lists rejection
reasons and counts only eligible judge tasks. Recorded messages are preserved. The result includes
`eligible` and `rejected` counts; saved units retain validation errors. After downloading,
`pull` also reports selected roots, full threads, total traces, and
structural exclusion counts by reason, in stderr and the JSON `download_summary`.
The existing `downloaded` count includes excluded trajectories. Inspect files
listed in `snapshot.json` for each trajectory's error and source IDs. Model-specific
rendering and context checks remain in `prepare`.

Saved trajectories are read individually during judging and upload; queued judge
work holds IDs, not message bodies. Downloads retrieve messages and tool availability
from the trajectory endpoint without fetching raw run trees. Hashing and file
writes avoid whole-document copies. Each active full trajectory must still fit in memory. Legacy monolithic
snapshots remain readable but must fit in memory; new downloads use individual files.

## Review training examples with an agent council

```bash
smithtune dataset pull data/datasets/reviewed \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --target-count 100 --max-candidates 1000

smithtune dataset triage data/datasets/reviewed \
  --rubric ./rubric.md
smithtune dataset triage data/datasets/reviewed --confirm

smithtune dataset push data/datasets/reviewed --name reviewed-sft
smithtune dataset push data/datasets/reviewed --confirm
```

Write the task description, keep/drop criteria, and concrete examples in a
UTF-8 `rubric.md` file. Pass it to `triage` with `--rubric`; it can accompany
short additional `--rule` criteria.

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
rejections are excluded without truncation. A recognized context-limit rejection
is saved without retrying that request and remains excluded on resume. Request
failures such as timeouts and rate limits remain incomplete after retries.
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

Threads can gain traces while downloading. Pull saves the returned trajectory
and its source evidence; growth alone does not exclude it. Structural and tool
validation still apply. Saved trace IDs describe the downloaded content, and
resume reuses that local snapshot without incorporating later thread activity.

The positional directory replaces `--run-dir`, `--output`, and `--triage-dir`.
Saved triage snapshots can use `triage`, `push`, and `resume`. Receipts predating the
recovery format still require inspection and a new directory with `--dataset-id`.

After upload, pass the returned dataset ID to `smithtune prepare`. Saved tool
availability travels with the examples; see [per-assistant tools](reference.md#per-assistant-tools-and-training-targets).

---
name: sft-trace-triage
description: Label full LangSmith conversations for SFT with a Deep Agent council. Dispatch model judges with code mode, save one 1/0 and reason per trajectory, and explain the results.
---

# SFT trajectory selection

Reuse the CLI's saved conversation messages. Each full conversation is one
trajectory and one possible training example. Each council member judges that
same full trajectory once. Do not split it into turns or replay cases.

## Coordinator inside the CLI

Dispatch every pending trajectory/judge pair. The CLI supplies the source,
models, and [judge rubric](judge.md). It filters media before dispatch. Each
judge subagent makes one model request with the full messages and returns a
score and reason. The CLI saves the result and computes the majority label.
Do not judge or rewrite the trajectory yourself.

Use `code_mode` with:

- `pending_tasks(limit=128)`: up to 128 unattempted `{trajectory_id, judge}` pairs.
- `judge_batch(tasks)`: dispatch those pairs with the configured concurrency.
  Results are saved before the function returns.

```python
jobs = pending_tasks()
results = judge_batch(jobs) if jobs else []
{"finished_batch": len(results), "next_tasks": pending_tasks()}
```

Repeat until there are no pending tasks. Each code call has fresh state. A
code error can occur after results were saved; check pending tasks again.
For a single task, use `task` with `subagent_type="trajectory-judge"` and a
JSON description: `{"trajectory_id":"<saved-id>","judge":"judge-1"}`.

Code can dispatch only planned tasks. It has no shell, host files, network,
or environment access. Do not add filtered conversations back or invent votes.
Failed requests remain incomplete for resume. A provider context-window
rejection filters the whole trajectory with 0 and a reason. Do not shorten,
summarize, page, or split the input to make it fit.

Every council member must finish for a quality label. A strict majority gives
1; a tie gives 0. The CLI combines the reasons for that label. Your final text
should explain the saved counts and remaining failures to the user.

## Agent helping a user

Use [discovery.md](discovery.md) to inspect varied full trajectories with the
user, agree on useful training examples, and write a selection rubric. Learn
the application's task and tools from its traces; keep domain-specific rules
and evidence in local run files. The coordinator above only dispatches saved
tasks; it must not start an interview or change the rubric.

Run `smithtune doctor` and `smithtune dataset pull --help` for setup. Use one
saved directory across `pull`, `triage`, `push`, and `resume`.

First map the user's selection criteria to the project's actual feedback,
metadata, tags, and error fields. Use `--filter` for criteria those fields express.
Do not invent feedback keys, thresholds, or the meaning of missing values.
Recommend council review of training-example quality. Write `rubric.md` with the
task, keep/drop criteria, and concrete examples.
Review it with the user, then pass `--rubric ./rubric.md` for council judging.
Use `--rule` as a shortcut for short additional criteria.

- Use `dataset pull DIR` with source IDs, time window, and optional `--filter` / `--limit`.
  Inspect its download summary, exclusion reasons, and representative saved trajectories.
  Pull makes no model calls. Selecting an agent by name or filtering out errors
  alone does not establish training quality.
- Agree on the review rubric and run
  `dataset triage DIR --rubric ./rubric.md` to preview. Check the saved text in
  `plan.json`, then use `dataset triage DIR --confirm` when judging is authorized.
  Inspect a small batch's decisions before scoring the larger pool.
  Read `labels.jsonl`, individual votes in `judgments.jsonl`, and `report.md`
  and explain the counts and reasons. Failed judge requests remain incomplete.
- Skip council review only when trusted feedback or quality labels already establish
  which trajectories meet the training criteria. Explain the evidence and decision
  to the user. Council review helps assess quality; it does not guarantee it.
- Preview upload with `dataset push DIR --name NAME` (or `--dataset-id ID`).
  Add `--confirm` when upload is authorized. Push respects any council plan
  already attached to the directory. Pull followed directly by push uses the
  source filters and structural checks without model calls.
- `dataset resume DIR` shows pending stages without network calls. With
  `--confirm`, it continues the saved workflow and reuses completed work.
- On failure, identify the stage and saved directory, inspect pending work, and resume
  that directory. Completed downloads, votes, and upload receipts are reused.
- Pass the resulting dataset ID to `prepare -> plan -> train`.

The default council is DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks plus
GPT-5.6 Terra on OpenAI. Choose models with `--judges`; other models use
`provider:model`. The rubric and rules apply to whole trajectories. Confirm and
resume use the saved rubric text even if its original file changes or is deleted.
Source and destination
are frozen; council rules can change before judging starts. Use a new directory
to change rules after votes or to review a different source selection.

Preserve recorded messages and saved per-assistant tool availability. Labels remain local; these
commands do not write trace feedback. Replay evaluation is separate under
`evaluate`. Fireworks judging uses its official API. Direct Anthropic uses
`ANTHROPIC_API_KEY`; `anthropic-gateway` uses `LANGSMITH_GATEWAY_API_KEY`.

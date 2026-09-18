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
judge subagent receives full unchanged messages and per-assistant tool bindings
as untrusted evidence, not executable tools. It makes a model request and returns a
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

Run `smithtune doctor` and the relevant command's `--help`. The default council
is DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks, plus GPT-5.6 Terra on OpenAI.
Use `--judges` aliases or `provider:model`, and `--rule` for project rules.

1. Download: `smithtune dataset pull DIR --workspace-id WORKSPACE --project-id
   PROJECT` with optional time bounds, `--limit`, and `--filter`. Selection samples
   distinct threads/traces. Each saved conversation includes all messages and
   verified producing-run tool bindings. Missing provenance excludes it.
2. Preview local candidates and votes: `smithtune dataset triage DIR`.
   When paid judging is authorized, add `--confirm`. Successful votes are durable
   in `triage.jsonl`; repeat to retry incomplete pairs with saved settings.
3. Explain counts and keep/drop reasons from the JSON command result. Request
   errors are incomplete work, not quality votes. No separate labels/report file
   is needed.
4. Preview uploads: `smithtune dataset push DIR --name NAME` (or `--dataset-id ID`).
   When writes are authorized, add `--confirm`. The CLI reconciles uncertain
   writes and preserves prior successes. Resume with `dataset resume DIR --confirm`.
5. Pass the dataset ID to `prepare -> plan -> train`.

`dataset create DIR` with source/destination flags composes all three stages.
Without confirmation it downloads and previews. `--confirm` authorizes judging
and uploads; `--no-triage` chooses pull and push only when that is the user's intent.

Keep `checkpoint.json`, `triage.jsonl`, and `conversations/` together. Never refetch
or edit judged messages or bindings. Frozen source selections and post-vote
rubric/model changes require a fresh checkpoint. Coordinator or skill software
changes do not invalidate durable votes. A response lost before saving may need
another paid request. Labels remain local; no trace feedback is written.
Replay is a separate flow under `evaluate`.

Direct Anthropic uses `ANTHROPIC_API_KEY`; `anthropic-gateway` uses
`LANGSMITH_GATEWAY_API_KEY`.

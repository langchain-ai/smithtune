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

Run `smithtune doctor` and `smithtune dataset triage --help` for setup and source
options. The default council is DeepSeek V4.1 Flash and Muse Glimmer 30B on Fireworks,
plus GPT-5.6 Terra on OpenAI. Use one `--judges` list to choose models; other
models use `provider:model`. Use `--rule` for project selection rules.

1. Download and preview: `smithtune dataset triage <directory>` with source IDs,
   time window, and optional `--limit` / `--filter`. Roots from the same thread
   form one full trajectory. Review the conversation and vote counts.
2. When paid judging is authorized: `smithtune dataset triage <directory> --confirm`.
   Repeat this command to resume. Completed votes are retained.
3. Read `labels.jsonl` and `report.md`. Each row has `trajectory_id`, `keep`
   (1 or 0), and `reason`. Explain the counts and main reasons to the user.
   Request errors have a clear incomplete reason and do not count as votes.
4. When upload is authorized: `smithtune dataset create --triage-dir <directory>
   --name <name> --confirm`. This imports kept conversations with the exact saved
   messages and tool schemas. Inspect `dataset-import.json` after a partial write.
5. Pass the dataset ID to the existing `prepare -> plan -> train` flow.

Do not refetch or edit messages after judging. Changed source, rubric, or models
need a new run. Labels remain local; this command does not write trace feedback.
Fireworks calls always use its official API. Replay evaluation is a separate
flow under `evaluate`.

Direct Anthropic uses `ANTHROPIC_API_KEY`; `anthropic-gateway` uses
`LANGSMITH_GATEWAY_API_KEY`.

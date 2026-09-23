---
name: trajectory-coordinator
description: Internal instructions for the Deep Agent inside `smithtune dataset triage`. Dispatch planned trajectory/judge pairs with code mode and report the saved counts. Not for operators; see the smithtune workflow skill instead.
---

# SFT trajectory selection

Reuse the CLI's saved conversation messages. Each full conversation is one
trajectory and one possible training example. Each council member judges that
same full trajectory once. Do not split it into turns or replay cases.

Dispatch every pending trajectory/judge pair. The CLI supplies the source,
models, and judge instructions. It filters media before dispatch. Each
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

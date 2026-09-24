---
name: smithtune
description: Run the smithtune fine-tuning flow end to end — select LangSmith trajectories and upload them as a dataset (pull, optional council triage, push), prepare it for Fireworks or Baseten, plan and train with base-vs-tuned evaluation, and optionally deploy. Use when a user wants to fine-tune a model on their LangSmith traces, build an SFT dataset from a tracing project, train or evaluate with smithtune, or resume/debug a smithtune run.
---

# smithtune workflow

smithtune turns LangSmith trajectories into a fine-tuned model:

```text
dataset pull → (dataset triage) → dataset push    build a LangSmith dataset
prepare → plan → train --evaluate                 train and compare base vs tuned
deploy → undeploy                                 optional endpoint
```

Installation and credentials live in the
[README](https://github.com/langchain-ai/smithtune/blob/main/README.md). This
skill covers how to operate the flow: which command to run, in what order, what
to check after each one, and when you are done. Run `smithtune <command> --help`
before an unfamiliar command.

## Rules that apply to every step

- **Parse the JSON.** Every command prints a JSON result on stdout and progress
  on stderr. Keep the IDs and paths it returns, and follow its `next_command`
  when present instead of guessing the next step.
- **Preview before paying.** `triage`, `resume`, `push`, `train`, `evaluate`,
  `deploy`, and `undeploy` do nothing billable or remote-changing without
  `--confirm`. Run each one without `--confirm` first, show the user the
  preview, and add `--confirm` only when the user has authorized that specific
  operation. Authorization for one step does not cover the next.
- **One directory per stage, reused throughout.** The same dataset directory for
  `pull`/`triage`/`push`/`resume`; the same `--data-dir` for
  `prepare`/`plan`/`train`/`evaluate`; a new or empty `--run-dir` per training
  run. Settings are frozen into these directories once work starts. To change
  source, filter, time window, limits, review mode, or rubric, use a new
  directory.
- **Same provider throughout.** One `--provider` (`fireworks` or `baseten`)
  from `prepare` through `deploy`.
- **Never guess IDs, fields, or thresholds.** Ask for workspace, project, and
  dataset IDs. Build filters only from fields you have seen in the project.
- **Never print credential values.** Check presence with `smithtune doctor`.

## 0. Before anything

```bash
smithtune doctor
smithtune models list --provider "$provider"
```

Check:
- `doctor` shows the keys the user's path needs as `set` (presence only, not
  validity); `credentials_required_for` says what each key is for:

| Path | Keys |
| --- | --- |
| Baseten | `LANGSMITH_API_KEY`, `BASETEN_API_KEY` (Model API access covers the default council and evaluation judge) |
| Fireworks, with a Baseten key | `LANGSMITH_API_KEY`, `FIREWORKS_API_KEY`, `BASETEN_API_KEY` for the default judges |
| Fireworks only | `LANGSMITH_API_KEY`, `FIREWORKS_API_KEY`; then pass `--judges deepseek-v4.1-flash,glm-5.3-flash` to `triage` and set `judge_model=accounts/fireworks/models/deepseek-v4p1-flash` |

- The requested model is in `models list`.
- Data rights are acknowledged. If not, the user must run
  `smithtune acknowledge-data-rights` in an interactive terminal; you cannot do
  it for them and `--confirm` does not replace it.

Set the values used below once, and reuse them in every command:

```bash
provider=baseten                 # or fireworks; --provider defaults to fireworks, so always pass it
model=qwen3p8-27b                # from `smithtune models list`
workspace_id='<workspace-id>'
project_id='<project-id>'
dataset_dir=./data/datasets/my-sft
data_dir=./data/my-sft
run_dir=./runs/my-sft
judge_model=baseten/zai-org/GLM-5.3-Flash   # default; see the table above for Fireworks only
start_time=2026-09-01T00:00:00Z  # ISO 8601 with timezone
end_time=2026-09-22T00:00:00Z
```

Pick the starting point:

| The user has | Start at |
| --- | --- |
| Traces in a tracing project | Step 1 |
| A LangSmith trajectory dataset | Step 5 (`prepare`) |
| A prepared `--data-dir` | Step 6 (`plan`) |
| A finished `--run-dir` | Step 8 (`evaluate`) or step 9 (`deploy`) |

Continue from existing directories when they match the task. For an existing
LangSmith dataset that smithtune did not create, `prepare` recovers per-turn
tool data from the source traces only when the example's messages match the
source trajectory exactly; otherwise it rejects those examples.

## 1. Choose the source and write the filter

The filter decides what the model learns from. Do this carefully and with the
user; `pull` freezes it into the directory.

**1a. Identify the project.** Get the workspace ID and project ID from the
user. If they only know the name, list projects:

```bash
langsmith project list --workspace "$workspace_id" --format json
```

**1b. Look at real root runs.** Filters match **root runs** (traces), so
inspect roots, not child LLM calls:

```bash
langsmith trace list --workspace "$workspace_id" --project-id "$project_id" \
  --since 2026-09-01T00:00:00Z --limit 20 --full --format json
```

From the output, write down what actually exists:
- root `name` values (which agent or graph produced the trace);
- `tags` and `custom_metadata` keys and typical values (environment, version,
  customer tier, etc.);
- `feedback_stats` keys and their score ranges (e.g. `correctness` 0–1,
  `user_thumbs` 0/1) and how many roots have each key;
- how many roots errored.

**1c. Agree with the user what "good training data" means**, then map it to
those fields. Ask:
- Which agent (root name) should the model imitate?
- Is there feedback that means the run was good? Which key, and what score
  counts as good? What fraction of traces have it?
- Any environment, version, or tag restrictions (e.g. production only, after a
  prompt change)?
- What time window?

Do not invent feedback keys, thresholds, or what a missing score means. If
there is no trustworthy quality signal, say so; that is the case for council
review in step 3.

**1d. Write the filter** in [LangSmith filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax).
Common building blocks:

| Intent | Filter |
| --- | --- |
| One agent | `eq(name, "support-agent")` |
| Has a tag | `has(tags, "production")` |
| Metadata value | `and(eq(metadata_key, "env"), eq(metadata_value, "prod"))` |
| Good feedback | `and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))` |
| One known root | `eq(id, "<root-run-id>")` (time bounds must include it) |
| Combine | `and(eq(name, "support-agent"), has(tags, "production"))` |

**1e. Test the filter before pulling.** `langsmith trace list --filter` takes
the same syntax and costs nothing. Use the same window you will pull:

```bash
langsmith trace list --workspace "$workspace_id" --project-id "$project_id" \
  --filter "$filter" --since "$start_time" --before "$end_time" \
  --limit 50 --full --format json
```

Check that:
- the matches are the agent and behavior the user wants (open a few);
- there are enough of them for the target (the smithtune default target is 100);
- nothing obviously wrong slipped in (errors, test traffic, other agents).

Adjust and re-test until the user agrees. Only then pull.

Know what the filter does **not** limit: each matching root pulls in its
**whole thread**, including earlier turns and turns outside the filter and time
window. The training example is the full trajectory.

## 2. Pull

**Ask the user whether to review with a council before pulling.** The choice is
fixed for the directory, so settle it now. Explain both options:

| | No council review | Council review |
| --- | --- | --- |
| Use when | The filter already selects on a trusted quality signal (validated feedback score, human labels) | There is no trustworthy quality signal, or the user wants a second check |
| Pull flag | `--no-triage` | none (this is the CLI default) |
| What the target counts | Structurally usable trajectories | Council-approved trajectories |
| Extra keys | None | Judge keys (`BASETEN_API_KEY` for the default council) |
| Extra work | None; go straight to push | Write a rubric with the user (step 3) |
| Cost | No model calls | One judge call per trajectory per council member |

Recommend one based on what you found in step 1: if the user's quality signal
is real and well covered, suggest no review; if not, suggest council review.
An agent-name filter or "no errors" alone is not a quality signal. Let the user
decide, and record the choice.

Because omitting the flag means council review, always pass `--no-triage`
explicitly when the user chose no review.

```bash
smithtune dataset pull "$dataset_dir" \
  --workspace-id "$workspace_id" --project-id "$project_id" \
  --start-time "$start_time" --end-time "$end_time" \
  --filter "$filter" \
  --target-count 100 --max-candidates 1000 \
  --no-triage   # omit to use council review
```

- Always pass explicit `--start-time`/`--end-time` (ISO 8601 with timezone).
  Without them the window is the last 24 hours.
- `--target-count`: how many trajectories you want in the final dataset.
- `--max-candidates`: cap on **new** candidates per pull round (max 2000). In
  council mode, set it well above the target, since some will be rejected.
- Pull makes no model calls. Re-running the same command resumes it.

Check:
- `download_summary`: selected roots, threads, traces, and exclusion counts by
  reason. Exclusions come from missing tool data, multimodal content, provider
  built-in tools, or oversized pages. If a large share is excluded, report the
  reasons before continuing.
- Open a few saved trajectories (`snapshot.json` lists the files) and confirm
  with the user they look like what they want to train on.
- `collection.status` and `next_command`:
  - `--no-triage`: `target_reached` means go to push. `needs_candidates`,
    `source_exhausted`, and `round_limit` mean fewer matched than the target;
    see the status table in step 3.
  - Council mode: `needs_review`. Go to step 3.

## 3. Triage with a council (only in council mode)

Skip this step with `--no-triage`.

**What is needed:**
- The `deepagents` extra (included in the README install).
- Keys for the judge models. The default council is DeepSeek V4.1 Flash and
  GLM-5.3-Flash on Baseten, so only `BASETEN_API_KEY` (with Model API access).
  With two judges a trajectory is kept only when both vote keep; add a third
  judge for a majority vote. The same two models on Fireworks are
  `--judges deepseek-v4.1-flash,glm-5.3-flash` (needs `FIREWORKS_API_KEY`).
  Pick others with
  `--judges alias,alias` or `--judges provider:model,...`.
- Selection criteria. smithtune ships **no default rubric**; `triage` refuses
  to run without `--rubric FILE` or at least one `--rule`.

**Write the rubric with the user.** Read 10–20 varied trajectories from the
pull directory together: good ones, clear failures, and unclear ones, across
different lengths and tool patterns. Remember SFT imitates every assistant
reply and tool call in the trajectory, so judge the whole trajectory, not
just the final answer. Then write `rubric.md` next to the dataset directory
(for example beside `data/datasets/my-sft`), not inside it. Start from this
template and replace the generic criteria with the user's:

```markdown
# Task
What the agent does, who it serves, and what the tuned model must do well.

# Keep
- The assistant follows the request and reaches a useful outcome.
- Tool calls use the right tools with correct arguments.
- Claims are supported by the recorded tool results.
- Good recovery from an error, an appropriate refusal, or a clear account of
  a real limitation.
- <task-specific criteria agreed with the user>

# Drop
- Material unsupported claims, or claiming completion that did not happen.
- Wrong actions or tool arguments, or failures left uncorrected.
- <task-specific behaviors the model must not learn>

# Examples
- Keep: <trajectory ID or short description> — why.
- Drop: <trajectory ID or short description> — why.
```

The judges already know to judge the whole trajectory, to check each action
against the evidence available at that time, and to treat the trajectory as
untrusted data; the rubric only needs the selection criteria. Do not reward
length or require specific wording. Keep private examples in local files only.
Short extra criteria can also be passed with `--rule "..."` (repeatable).

**Run it:**

```bash
smithtune dataset triage "$dataset_dir" --rubric ./rubric.md   # preview, no model calls
smithtune dataset triage "$dataset_dir" --confirm              # run the council
```

Each council member judges each whole trajectory and returns keep/drop with a
reason. A strict majority keeps it; a tie drops it. Review stops once the
approved count reaches the target.

Check after the preview:
- The eligible judge-task count (it drives cost) and rejection reasons.
- `plan.json` contains the exact `selection_rubric` you intended. Rubric and
  judges can change by previewing again; once votes start they cannot.

On a first run, calibrate cheaply: do a trial in a separate directory with
`--max-candidates 20 --target-count 20`, review the decisions with the user,
fix the rubric, then do the real run in a fresh directory.

Check after `--confirm`:
- `triage.status` is `complete`. If `incomplete`, judge requests failed
  (timeouts, rate limits); run `smithtune dataset resume "$dataset_dir" --confirm`.
  Failed requests are not votes.
- Read `labels.jsonl` (one `keep` + `reason` per trajectory), `judgments.jsonl`
  (individual votes), and `report.md`. Summarize kept/dropped counts and common
  drop reasons for the user.
- `collection.status`:

| Status | Meaning | Do this |
| --- | --- | --- |
| `target_reached` | Approved count ≥ target | Push |
| `needs_candidates` | Pool fully reviewed, below target, rounds remain | Run `next_command` (`dataset pull DIR`), then `triage DIR --confirm` again |
| `source_exhausted` | No unseen matches in the window | Push what you have, or new directory with broader criteria |
| `round_limit` | 3 pull rounds used | Push what you have, or new directory with broader criteria |
| `downloading` | A pull round is unfinished | `dataset resume DIR --confirm` |

- An `advisories` entry means the approved set is small (under 100). It does
  not block upload; relay it and let the user decide.

## 4. Push to LangSmith

```bash
smithtune dataset push "$dataset_dir" --name my-sft-dataset   # preview
smithtune dataset push "$dataset_dir" --confirm               # upload
```

Use `--dataset-id ID` instead of `--name` to extend an existing dataset.

Check:
- The preview's example count matches the approved (council) or usable
  (`--no-triage`) count. Push refuses while council judging is incomplete.
- After `--confirm`, `status` is `complete` and the result has `dataset_id`.
  Keep it; it is the input to `prepare`.
- After upload starts, the directory cannot pull more candidates; use a new
  directory for more data.

## 5. Prepare

```bash
smithtune prepare \
  --provider "$provider" --model "$model" \
  --workspace-id "$workspace_id" --dataset-id "$dataset_id" \
  --data-dir "$data_dir"
```

Prepare checks provider support for the model, formats each trajectory with the
model's tokenizer and per-assistant tools, splits about 80/10/10 into
train/validation/test (a whole trajectory stays in one split), and publishes the
splits to the LangSmith dataset.

Check:
- Split counts. Validation and test must be non-empty for training and
  evaluation to mean anything; tiny datasets can produce empty splits.
- `"$data_dir/prepared/rejected.json"`: trajectories dropped as unsupported or
  over the context limit (never truncated). If many are too long, ask the user
  before lowering `--max-seq-len` or choosing a longer-context model.
- Split publication succeeded. If only publication failed, run
  `smithtune dataset publish-splits --data-dir "$data_dir"`; do not re-prepare.

Re-running with `--no-fetch` reuses the downloaded data (splits still sync).
`--no-sync-splits` prepares locally only; publish the splits with
`smithtune dataset publish-splits --data-dir "$data_dir"` before evaluation.

## 6. Plan

```bash
smithtune plan \
  --provider "$provider" --data-dir "$data_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2
```

`plan` is free and saves nothing for `train`. `--judge-model` accepts
`baseten/<model-id>` (default `baseten/zai-org/GLM-5.3-Flash`), a Fireworks model
ID such as `accounts/fireworks/models/deepseek-v4p1-flash`, or
`anthropic/<model-id>` (needs `ANTHROPIC_API_KEY`).

Check:
- Training example and token counts, epochs, and hyperparameters.
- Evaluation case and call counts (base response + tuned response + judge per
  action, plus calibration). `--max-points-per-trajectory` caps actions per
  test trajectory; omitting it evaluates every action.
- Show the user these numbers and get authorization for the spend.

## 7. Train (and evaluate)

Repeat **every** custom flag from `plan` exactly:

```bash
smithtune train \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2 \
  --confirm
```

- `--run-dir` must be new or empty. Omit `--evaluate` and its options to train
  only.
- Baseten's optional spend guard needs both `--max-spend-usd` and
  `--hourly-rate-usd`.

Check:
- Share the LangSmith comparison link printed when evaluation starts.
- `"$run_dir/result.json"` records the best checkpoint (lowest validation loss).
  In `epochs.json`, flag validation loss that rises steadily or never improves.
- The final JSON has `langsmith.comparison_url`; `"$run_dir/replay/summary.json"`
  has base vs tuned scores. `teacher_agreement` is per action,
  `trajectory_teacher_agreement` the per-trajectory mean. They measure agreement
  with recorded behavior; tools are not executed.

## 8. Evaluate separately (resume or re-run)

Use when training ran without `--evaluate` or evaluation was interrupted. Keep
the original provider, directories, and replay settings:

```bash
smithtune eval-plan --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --max-points-per-trajectory 2
smithtune evaluate --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --judge-model "$judge_model" --max-points-per-trajectory 2 --confirm
```

Rerunning reuses completed predictions and judgments. Fireworks samples the
saved serverless training checkpoint (a promoted model ID alone cannot be
sampled); Baseten starts temporary samplers and deactivates them on exit. To change the judge, cap,
or sampling settings, pass a fresh `--output-dir` to both commands.

## 9. Deploy (optional)

Only when the user wants a standing endpoint; evaluation does not need one.
`deploy` and `undeploy` work for both providers with provider-specific flags.
See the [deployment guide](https://github.com/langchain-ai/smithtune/blob/main/docs/deployment.md)
for details.

```bash
# Fireworks (firectl 1.8.5+): promotes the best checkpoint, matches a validated
# deployment shape, and waits for a ready replica. --account-id must own the
# checkpoint; a saved promotion is reused. Omit --confirm first to preview the shape.
smithtune deploy --provider fireworks --run-dir "$run_dir" \
  --account-id "$account_id" --output-model-id my-tuned-model \
  --deployment-id my-endpoint --confirm

# Baseten: needs the baseten-deploy extra; choose GPUs explicitly.
smithtune deploy --provider baseten --run-dir "$run_dir" \
  --accelerator H200:1 --max-seq-len 32768 --confirm
```

**Fireworks blocks agents from creating or deleting deployments.** firectl refuses
mutating commands when it detects an AI agent, so `deploy --confirm` and
`undeploy` stop with a message containing the exact `smithtune` command to run.
As the agent: run the preview (no `--confirm`) to show the shape, run
`--confirm` once (it promotes the checkpoint and then stops), and give the user
the printed command to run in their own terminal. If the user confirms their
team allows it, `FIRECTL_AGENT_SAFE_ACCOUNTS=<account>` (firectl's allowlist)
lets you create the deployment; `undeploy` always needs the user. Never hide
the agent environment to get past the block. Baseten deploys are not affected.

Stop serving when the user is done:

```bash
smithtune undeploy --provider fireworks --account-id "$account_id" --deployment-id my-endpoint --confirm
smithtune undeploy --provider baseten --run-dir "$run_dir" --confirm
```

`undeploy` stops serving capacity only; the Fireworks model and Baseten
checkpoint are kept. Tell the user the deployment exists, that it bills until
removed, and the exact `undeploy` command to stop it.

## End state

The flow is done when you can hand the user:

- the LangSmith **dataset ID**, its trajectory count, the filter and window used,
  and (council mode) kept/dropped counts with common drop reasons;
- the **data directory**, split counts, and rejected count from `prepare`;
- the **run directory** and the selected checkpoint and epoch from `result.json`;
- the **LangSmith comparison URL** with base vs tuned `teacher_agreement`, noting
  what the score does and doesn't measure;
- if deployed, the **deployment ID** and its `undeploy` command.

Nothing should be left pending or running without the user knowing:
`dataset resume DIR` shows no pending stages, Baseten samplers are cleaned up
(`sampler.json`), and any deployment is acknowledged.

## Debugging

Identify the stage and directory, then look at saved state before re-running.
Do not delete directories or edit saved files; they hold the recovery state.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| "Data Rights and Permitted Use has not been acknowledged" | First-run acknowledgment missing | User runs `smithtune acknowledge-data-rights` interactively |
| 401/403 from LangSmith, Fireworks, Baseten, OpenAI, or Anthropic | Key missing, wrong, or for another workspace/account; Baseten judges also need Model API access on the key | Check `doctor`, then the key's workspace; see README credentials |
| "BASETEN_API_KEY is not set for the judge" on a Fireworks run | The default council and evaluation judge run on Baseten | Set `BASETEN_API_KEY`, or use the Fireworks judges from step 0 |
| Pull downloads 0 or very few candidates | Filter fields wrong, or window too narrow | Re-test the filter with `langsmith trace list` over the same window; pull into a new directory |
| Most trajectories excluded at pull | Missing per-assistant tool data, multimodal content, provider built-in tools | Read reasons in `download_summary` and per-trajectory errors |
| Triage `incomplete` | Judge timeouts or rate limits | `dataset resume DIR --confirm`; lower `--concurrency` if rate-limited |
| "council review needs selection criteria" | No `--rubric` or `--rule` given | Write `rubric.md` with the user (step 3) and pass `--rubric` |
| Triage fails immediately with a key error | Missing judge key or `deepagents` extra | Set the judge keys, or choose judges you have keys for with `--judges` |
| "council judging is incomplete" on push | Votes pending | `dataset resume DIR --confirm`, then push |
| "cannot collect more candidates after upload started" | Pull after push in the same directory | New directory |
| Changing filter/rubric/limits/review mode is rejected | Settings are frozen | New directory |
| Conflicting history on push | Destination dataset diverged from saved receipts | Stop and inspect; push to a fresh `--name` if the user agrees |
| Prepare rejects many trajectories for length | Longer than the model's context | Longer-context model, or accept the rejections; never truncate |
| Prepare: split publication failed | LangSmith write error after local success | `smithtune dataset publish-splits --data-dir DIR` |
| Prepare: tool data missing for an older dataset | Examples uploaded before per-assistant tool capture | Re-pull and push a fresh dataset |
| "Baseten workspace does not advertise MODEL" | Model not available to the Baseten workspace | Confirm model ID via `models list`; contact Baseten for access |
| "Baseten workspace context limit is below N" | Sequence length above Baseten's limit | Lower `--max-seq-len` at prepare, or pick another model |
| Train refuses the run directory | `--run-dir` not empty | New run directory; never clear an old one |
| Evaluation stops before paid work on resume | Settings differ from the saved run | Re-run with the original settings |
| "Fireworks blocks firectl from changing resources inside an AI agent" | firectl refuses deploy/undeploy under an agent | Give the user the printed `smithtune` command to run in their own terminal |
| "automatic deployment shape selection needs firectl 1.8.5 or newer" | Old firectl | `firectl upgrade`, or pass `--deployment-shape` |
| Process killed during Baseten evaluation or deploy | Samplers or endpoints may still be running | Cleanup commands in `sampler.json`, or `undeploy --provider baseten --run-dir DIR --confirm` |

When reporting a failure, include the command, stage, directory, error text, and
any example/run IDs from the output.

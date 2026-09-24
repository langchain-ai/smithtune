# Preparation and evaluation reference

[Back to the workflow](../README.md)

## Prepare data

Preparation validates the trajectories and formats them for the chosen model.
Recorded system messages are preserved; Qwen requires them at the start.

## Supported models

Run `smithtune models list --provider fireworks` or
`smithtune models list --provider baseten` to list supported model aliases and IDs.
Only listed models are supported. Preparation and training check provider
availability and select the tokenizer and renderer for the model.

A Baseten workspace can be unapproved for a listed model, or approved for it only
up to a shorter sequence length. When Baseten reports either, preparation and
training stop before any paid work and repeat Baseten's own explanation and
remediation instead of reporting the model as unavailable.

## Preparation defaults

- One complete trajectory per dataset example
- LoRA training on text and tool trajectories; images are unsupported
- Approximately 80% training, 10% validation, and 10% replay test, keeping each source trajectory in one split
- All assistant messages are training targets, including earlier turns
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; lower the limit with `--max-seq-len`, for example `--max-seq-len 32768`

## Splits and dataset versions

Preparation saves trajectory assignments in `prepared/split_assignments.json`
and reuses them as the dataset grows. Keep the same fractions when rerunning;
small datasets may have empty splits, which are reported without reshuffling.

When continuing training in a new data directory, add
`--split-from <previous-data-dir>` to `prepare` to preserve prior assignments.
Keep the assignments file, including entries for removed trajectories.

Preparation also publishes these splits onto the original LangSmith dataset and
records its version for evaluation. Existing conflicting assignments stop
preparation. If local preparation succeeds but split publication fails, publish
only the saved memberships without fetching or rendering again:

```bash
smithtune dataset publish-splits --data-dir "$data_dir"
```

`prepare --no-fetch` still contacts LangSmith to synchronize splits; use
`--no-sync-splits` for local-only preparation.

## Reuse downloaded data

Use `prepare --no-fetch` with the same settings and data directory to reuse the
downloaded data. Provider checks, tokenizer loading, and split synchronization
still run.

## Per-assistant tools and training targets

Whole trajectories remain the dataset examples and split rows. Their metadata
stores `smithtune_source.assistant_runs`: the message position, producing run and
trace IDs, and complete tools recorded for each assistant call. Tools can be
added, removed, or change descriptions and schemas between calls. An empty list
means no tools were offered; missing availability is not treated as an empty list.

Pull requests `/v1/trajectory` in `ui` format with system messages enabled. Each
assistant item's `message.available_tools` supplies its complete tool list, and
`metadata.run_id` / `metadata.trace_id` identify the producing call. No additional
LLM-run lookups are needed. Missing availability or provenance and unsupported
provider built-ins exclude the trajectory with a recorded reason.

Preparation reuses saved bindings. For unbound, unjudged exports, it can retrieve
them from the source trajectory only when its messages match the saved example
exactly. Interrupted preparation retains completed captures. New source downloads
require a LangSmith instance that supplies `available_tools`; existing bound
datasets and explicit contract overrides remain usable.

Fireworks and Baseten render each supported assistant answer with its preceding
messages and its tool list. Only that answer receives training loss; earlier
assistant answers serve as context. Replay uses the same per-call tools. This
preserves the recorded trajectory prefix; it does not reconstruct hidden prompt
rewrites, context compaction, or routing between agents.

`--inference-contract FILE` remains an explicit global tool-schema override.
Run `prepare` again for older prepared artifacts. Older council datasets need a
fresh pull and review to attach per-call evidence, or an explicit global override.

## Model-specific formatting

Muse Glimmer requires an explicit system message. It rejects assistant messages
that combine visible text with tool calls, or make tool calls immediately before
another assistant message.

## Training options and artifacts

Training prints a generated run ID and saves artifacts to `./runs/<run-id>`.
Override either with `--run-id` or `--run-dir`; the folder must be new or empty.
Repeat customized training settings on both `plan` and `train`; the preview does not save settings for the training command.

The best checkpoint is selected by validation loss and recorded in `<run-dir>/result.json`.
Training artifacts also include `plan.json`, `run-state.json`, and `epochs.json` in that directory.
Use `--init-from-checkpoint '<checkpoint-uri>'` to initialize a new training run from a saved checkpoint.
Baseten's optional spend guard requires both `--max-spend-usd` and `--hourly-rate-usd`.

## Sampler behavior

Fireworks reuses its serverless training session for `train --evaluate`.
Baseten shuts down its trainer, then starts dedicated Loops samplers for the
best checkpoint and base model; both samplers are deactivated on exit.
Sampler and judge costs are separate from Baseten's training spend guard.
Baseten sampler replay needs no `baseten-deploy` extra or Fireworks key with the
default Baseten judge or an Anthropic judge. Periodic replay during training is not implemented.

Fireworks requires its saved serverless training checkpoint; a promoted model ID
alone cannot be sampled. Baseten requires saved Loops sampler weights. Keep the
training run directory so another team member can resolve the selected checkpoint.
See [Fireworks in-session sampling](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models#in-session-sampling-serverless-training).

## Replay options

| Flag | Use |
| --- | --- |
| `--judge-model` | Judge route; default: `baseten/zai-org/GLM-5.3-Flash` |
| `--max-points-per-trajectory` | Cap assistant actions per trajectory; omitted by default, which scores every eligible action |
| `--max-output-tokens` | Maximum generated response tokens |
| `--concurrency` | Concurrent evaluation cases; default: 4 |

With `plan` and `train`, replay options require `--evaluate`. Repeat the same
options on the preview and paid command. `eval-plan` accepts the case and token
caps; judge and concurrency options belong on `evaluate`.

The default judge runs on Baseten Model APIs and needs `BASETEN_API_KEY` with
Model API access. Other Baseten models use `--judge-model baseten/<model-id>`, for
example `baseten/deepseek-ai/DeepSeek-V4.1-Flash`. For a Fireworks judge, select
`--judge-model accounts/fireworks/models/deepseek-v4p1-flash` and set
`FIREWORKS_API_KEY`. For direct Anthropic, select
`--judge-model anthropic/claude-sonnet-5` and set `ANTHROPIC_API_KEY`. Judge
credentials do not replace the LangSmith API key used to read the dataset and
publish experiments.

Replay checks tool selection, JSON arguments, argument schemas, and reference
arguments. Ambiguous text-encoded argument types fail format validation.
Tool-validation metrics are recorded in run outputs; they are separate from the
judge's agreement score. Each case starts from recorded context, and generated
tool calls are never executed.

## Experiment details

`train --evaluate` and standalone `evaluate` publish a base and tuned experiment
when comparing both models, or one experiment when evaluating only a tuned endpoint.
Names follow `smithtune-base-<short-model-name>-<evaluation-id>` and
`smithtune-tuned-<short-model-name>-<same-evaluation-id>`. Both use the base-model
name; exact model and checkpoint identifiers remain in metadata.
Each completed trajectory has a root run, with a child LLM run for each generated
action. Finished children and their feedback upload incrementally. The parent,
its complete outputs, and its aggregate score publish only after all selected
actions in that trajectory finish. Until then its experiment row is not shown;
the comparison link is available and local receipts track publication progress.
This publisher posts completed run outputs rather than updating them.
Experiment metadata identifies the provider and whether
predictions came from a sampler or deployed endpoint. When a saved training run
is available, `parent_training_run_id` records the smithtune run ID and
`checkpoint_epoch` records the selected checkpoint’s epoch. Both experiments in
a base/tuned comparison carry this training context; the epoch describes the
tuned checkpoint. Endpoints without a saved training run omit these fields.

- `teacher_agreement`: each action's judge pass/fail, with its explanation.
- `trajectory_teacher_agreement`: fraction of evaluated actions that passed in a
  trajectory. Averaging this score weights trajectories equally; local summary
  rates weight individual actions equally.

These scores measure agreement with the recorded response, not independently
verified task success. Tool-validation metrics remain in the saved run outputs
and local replay files rather than separate feedback columns.

## Resume an interrupted evaluation

Run standalone `evaluate` again with the same provider, data directory, training
run, output directory, and replay settings. Completed predictions and judgments
are reused; saved generations can be judged without sampling again. A fully
completed evaluation makes no model calls. Training checkpoints survive replay
failures. Use a new training run directory only when training again.

If the process was killed or Baseten cleanup failed, use the deployment IDs and
cleanup instructions in `sampler.json` to stop paid capacity before resuming.

| Artifact | Contents |
| --- | --- |
| `cases.jsonl` | Frozen replay inputs and references |
| `generations.jsonl` | Responses saved before judging |
| `results.jsonl` | Completed per-case responses, checks, and judgments |
| `summary.json` | Aggregate scores and base-versus-tuned differences |
| `sampler.json` | Sampler identities and cleanup status |
| `langsmith-experiments.json` | Experiment IDs, comparison link, and publication status |

Results publish incrementally. If publication fails, rerun `evaluate` with the
same settings to resume from saved results. The publication receipt records
progress and any upload error.

An older CLI may have already published a finished parent containing only some
actions. The new publisher detects these legacy partial/open parents on resume
and stops before paid work, with the run and experiment IDs. It does not patch,
delete, or silently replace them. Preserve the original artifacts and resolve
that existing experiment explicitly; ordinary resume cannot repair this legacy
state. Fully completed, matching parents remain reusable without another upload.

Evaluation always publishes to LangSmith. Local files are recovery artifacts;
an upload failure leaves evaluation incomplete until publication succeeds.
Data prepared with `--no-sync-splits` must have its splits published before
evaluation, using `smithtune dataset publish-splits --data-dir "$data_dir"`.

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
- All assistant messages are targets unless you [choose a different supervision scope](datasets.md#choose-assistant-teaching-targets)
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; lower the limit with `--max-seq-len`, for example `--max-seq-len 32768`

For Fireworks GLM-5.3-Flash, `prepare --reasoning-effort low|high|max` selects
the native thinking-effort prompt used by training and replay. The default is
`max`; all three choices keep thinking enabled. The saved model configuration
and training plan record this choice. Training and evaluation reuse it without
additional flags; changing it requires preparing a new data directory. For a
deployed model, pass the matching `reasoning_effort` to Fireworks' chat API.
Other training models currently reject this flag.

`--reasoning-policy` is separate: it controls whether recorded reasoning is
included in the source examples. Omitting recorded reasoning does not disable
the model's thinking mode.

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
stores `smithtune_source.assistant_runs`: the message position, source run and
trace IDs, and tool availability recorded for each assistant call. Tools can be
added, removed, or change descriptions and schemas between calls. An empty list
means no tools were offered; missing availability is not treated as an empty list.

Pull requests `/v1/trajectory` in `ui` format with system messages enabled. Each
assistant item's `message.available_tools` supplies its complete tool list, and
`metadata.run_id` / `metadata.trace_id` identify its source observation (which may be an input occurrence for supplied
history). They do not independently establish authorship. No additional
LLM-run lookups are needed. Selected assistant targets require complete, supported
tool definitions; missing availability or unsupported provider built-ins on a
target exclude the trajectory with a recorded reason. Context-only bindings retain
recorded tool metadata without requiring a valid inference contract; absent
availability is saved as `tools: null`, and context bindings may be omitted.
Run/trace provenance is still required on captured assistant items.

Preparation reuses saved bindings. For unbound, unjudged exports, it can retrieve
them from the source trajectory only when its messages match the saved example
exactly. Interrupted preparation retains completed captures. New source downloads
require a LangSmith instance that supplies `available_tools`; existing bound
datasets and explicit contract overrides remain usable.

Fireworks and Baseten render each selected supported assistant answer with its preceding
messages and its tool list. Only that answer receives training loss; earlier
assistant answers serve as context. Replay uses the same selection and per-call tools. This
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
Training artifacts also include `plan.json`, `run-state.json`, `epochs.json`, and
`validations.json` (each completed validation and its checkpoint) in that directory.
Use `--init-from-checkpoint '<checkpoint-uri>'` to initialize a new training run from a saved checkpoint.
Baseten's optional spend guard requires both `--max-spend-usd` and `--hourly-rate-usd`.

### Remote controller recovery

Use the [remote machine workflow](../README.md#run-on-a-remote-machine) to launch
`train --detach` on an existing VM when the selected workflow's plan reports
`execution.requires_active_client: true`. Detachment keeps the controller alive
independently of the launching terminal. It does not restart a failed controller
or resume the same logical training run. The VM must allow the detached process
to outlive its SSH session, and any supervisor must leave automatic restarts of
the training command disabled.

On the VM, inspect the original run:

```bash
smithtune status --run-dir /persistent/runs/my-sft
smithtune logs --run-dir /persistent/runs/my-sft --tail 200
```

`controller_active` uses the worker's local process lock, so a stale or reused
PID cannot make a dead controller appear alive. `running` means the worker
holds that lock, not that the provider is making progress. `completed` means
the training command returned successfully, including evaluation if requested.
`failed` records a command error. `interrupted` means the worker received an
interruption or disappeared after starting. `unknown` means it has not recorded
startup and holds no lock; it may still be starting, or may have died before its
first receipt. Inspect the log before deciding to recover. `launch_failed`
means the launcher could not start the process. These states do not establish
whether provider resources are still running or billing.

Preserve the data directory, run directory, and `.controller` sibling before
retrying. `status` lists the available top-level run artifacts without assuming
provider-specific filenames. Inspect those artifacts and the log, then confirm
remote state with the recorded provider before submitting more work. Remote
operations may have succeeded before their IDs or checkpoints were saved locally.
Recovery options depend on the provider and training workflow; use the matching
procedure below rather than inferring them from controller health.

#### Fireworks Training API and Baseten Loops

These integrations recover unfinished training by initializing a new run from a
saved training-state checkpoint. They do not resume the same logical Smithtune
run. Sampler-only weights cannot initialize training.

1. **Reconcile paid resources first.** Inspect the saved session/run IDs,
   `run-state.json` (Baseten), `session.json` (Fireworks), and any
   `replay/sampler.json` receipt. Confirm resource state with the provider and
   stop remaining owned training/sampling capacity. An interrupted creation
   can have succeeded before its ID was saved; inspect the provider's console
   or management API when receipts are incomplete. Closing a client or deleting
   a VM does not establish that GPU capacity was released. Baseten's local spend
   guard cannot enforce a limit after the controller dies; sampler and judge
   costs are separate from its training budget.
2. **If training is unfinished, inspect `validations.json` for a saved resumable
   checkpoint** (`epochs.json` for older runs). Baseten also records
   `last_resumable_state_uri` in `run-state.json`; Fireworks records
   `resume_checkpoint` with each completed validation. Work since that checkpoint
   may need repeating. If local receipts are incomplete, inspect the provider's
   checkpoint records too. Use the base model only if no usable training-state
   checkpoint can be recovered.
3. **Initialize a new run with explicit remaining settings.** The new run
   starts its epoch and step numbering, early-stopping history, and budget afresh.
   Keep the earlier artifacts and checkpoints for comparison; the new run does not
   select across the earlier run's validation history. Account for previous
   work and spend when choosing the remaining epochs and budget.

```bash
smithtune plan --provider baseten --data-dir /persistent/data/my-sft --max-epochs 2
smithtune train --provider baseten --data-dir /persistent/data/my-sft \
  --run-dir /persistent/runs/my-sft-recovery --max-epochs 2 \
  --init-from-checkpoint '<saved-resumable-checkpoint>' --detach --confirm
```

Use your original provider and chosen remaining settings; `2` is illustrative.
Checkpoint availability is subject to provider retention. Do not clear the old
run directory or its `.controller` sibling to force a retry.

4. **If training completed and evaluation failed, resume evaluation.** Keep
   the original provider, data/run/output directories, and replay flags. The
   existing [evaluation recovery](#resume-an-interrupted-evaluation) reuses
   saved generations and completed results; it can run in a persistent terminal
   or service on the VM. Detached training itself is never resubmitted.

Back up the complete data directory, run directory, and `.controller` sibling
if the VM or its disk might be deleted. Atomic local artifact writes protect
against interrupted process writes; preserving a remote provider checkpoint
does not preserve the local configuration, tool evidence, or evaluation results.

## Training optimizations

### Validation cadence and checkpoint selection

By default, validation runs at epoch end. Add `--validation-every-steps N` to
both `plan` and `train` to also validate every N completed optimizer updates,
counted across epochs. Epoch ends are always checked, including successful
training completion and early stopping; coinciding checks run once. This uses
the held-out validation split, separately from replay evaluation.

`plan.training_schedule` shows steps per epoch and expected validation passes
through `--max-epochs`; early stopping or spend limits can reduce the count.
Fireworks batches training trajectories; Baseten Loops batches rendered assistant
targets through gradient accumulation. Baseten's preview is an estimate; training
recalculates it from the retained rendered sequences before provisioning.

The lowest validation loss across all checks selects the best checkpoint,
recording its epoch and global optimizer step. `validations.json` preserves
completed checks incrementally. Early stopping is **off by default**. Opt in
with `--early-stopping-patience N` to stop after N epochs without sufficient
epoch-end loss improvement. `--early-stopping-min-delta` controls that comparison;
checkpoint selection always uses the lowest loss across all checks.

Choose an interval using the planned step count and validation cost. Each check
runs the validation split and saves checkpoints, adding provider charges. A
failure or exhausted spend budget preserves completed checks without forcing
another validation pass.

### Dedicated training throughput

Size `prepare --max-seq-len` to cover `prepared/manifest.json`'s
`audit.max_context_tokens`, the largest rendered sequence across accepted splits,
including history and tools. Re-prepare to change the cap and inspect
`prepared/rejected.json` for exclusions; over-limit trajectories are excluded,
never truncated. Fireworks uses the prepared cap for trainer context and dedicated
shape selection. Baseten Loops already sizes trainer context to the measured
maximum, bounded by its training limit. A lower cap alone does not guarantee
higher throughput.

Review capacity in `plan`; pin Fireworks' versioned `--training-shape` for
comparisons. Automatic shape selection checks compatibility, without benchmarking
throughput or cost.

| Workflow | Tuning controls |
| --- | --- |
| Fireworks dedicated | Increase `--batch-size` gradually, then tune `--pipeline-depth` separately to overlap more submitted training steps. Batches count trajectories, each potentially producing multiple assistant targets. |
| Baseten Loops | Increase `--microbatch-token-budget` to fit more tokens per forward/backward call; keep it large enough for the longest sequence. `--batch-size` controls the effective batch through gradient accumulation; increasing it alone can just add accumulation work. |

Use short trials on representative data within authorized spend, each in a new
run directory. Change one setting at a time, holding data, model, capacity, LoRA
rank, learning rate, and seed fixed. Compare rendered tokens per second and total
capacity cost, including provisioning and validation. Stop increasing a setting
when throughput plateaus or memory/provider limits are reached; keep a stable
setting with headroom. Larger batches also change optimization, so check
validation quality before committing to a full run.

## Sampler behavior

Replay uses `SamplingClient` for both providers; serving capacity depends on the
training mode:

| Training mode | Replay capacity |
| --- | --- |
| Fireworks serverless (default) | Shared serverless sampling for base and tuned models |
| Fireworks dedicated | Two temporary dedicated deployments, for base and tuned models |
| Baseten Loops | Two temporary dedicated samplers, after training resources shut down |

Fireworks requests deletion of owned dedicated resources on exit; Baseten
deactivates its samplers. Dedicated capacity is billed for allocated time; judge
calls cost extra. Both are outside Baseten's training spend guard. Cleanup
failures are reported with resource IDs.

`evaluate --run-dir ...` uses the saved checkpoint and training mode. Standalone
dedicated Fireworks replay starts a temporary trainer to restore weights. Dedicated
checkpoints cannot use serverless sampling, even for the same base model.
Keep the run directory; a promoted Fireworks model ID alone cannot be sampled.
Baseten replay needs no `baseten-deploy` extra. Replay manages temporary serving;
use `deploy` separately for a standing endpoint.

## Fireworks dedicated LoRA training

Add `--training-mode dedicated` to both `plan` and `train`. SmithTune automatically
selects a validated LoRA shape with sufficient context; `plan` shows its hardware
and exact version. Dedicated planning requires `FIREWORKS_API_KEY` but creates no
resources. Serverless remains the default.

| Option | Use |
| --- | --- |
| `--training-shape accounts/<account>/trainingShapes/<shape>[/versions/<version>]` | Override automatic selection; include a version to pin the reviewed shape |
| `--provisioning-timeout SECONDS` | Placement/readiness budget per stage; default 1800, range 1–7200 |

Runs record the resolved shape in `plan.json`. Add `--evaluate` for dedicated
replay; training alone provisions no inference capacity.

Initialize a new run with `--init-from-checkpoint <trainer-job>:<checkpoint>`.
Inspect `session.json` and replay `resources.json` if cleanup fails or the process
is killed; delete remaining capacity in Fireworks. Checkpoints remain subject
to Fireworks retention.

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
`checkpoint_epoch` and, when available, `checkpoint_step` identify the selected
checkpoint. Both experiments in a base/tuned comparison carry this context for
the tuned checkpoint. Endpoints without a saved training run omit these fields.

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

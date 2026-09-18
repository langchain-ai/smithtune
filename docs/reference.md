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

## Preparation defaults

- One complete trajectory per dataset example
- LoRA training on text and tool trajectories; images are unsupported
- Approximately 80% training, 10% validation, and 10% replay test, keeping each source trajectory in one split
- Every supported assistant message is a target once, with its recorded tools and zero loss on history
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; use `--max-seq-len 32768` to lower the limit

## Splits and dataset versions

Preparation saves trajectory assignments in `prepared/split_assignments.json`
and reuses them as the dataset grows. Use `--validation-fraction 0.1` and
`--test-fraction 0.1` to set the fractions for either provider; set test to 0 to
omit replay data. Keep the same fractions when rerunning;
small datasets may have empty splits, which are reported without reshuffling.

When continuing training in a new data directory, add
`--split-from <previous-data-dir>` to `prepare` to preserve prior assignments.
Keep the assignments file, including entries for removed trajectories.

Preparation also publishes these splits onto the original LangSmith dataset and
records its version for evaluation. Existing conflicting assignments stop
preparation. If publication fails, run `dataset publish-splits --data-dir DIR`
to publish and verify the saved memberships without rerendering or repartitioning.
`prepare --no-fetch` is for re-preparation and still contacts LangSmith for publication; use
`--no-sync-splits` for local-only preparation.

## Reuse downloaded data

Use `prepare --no-fetch` with the same settings and data directory to reuse the
downloaded data. Provider checks, tokenizer loading, and split synchronization
still run.

## Per-assistant tools and target rendering

Each parent example contains unchanged `inputs.messages`, null `outputs`, and
`metadata.smithtune_source.assistant_runs`. See the [binding schema](datasets.md#portable-tool-bindings).
Preparation validates every recorded call against its producing run's tools,
including unused declarations. It preserves additions, removals, descriptions,
and schema changes across turns. Missing provenance never becomes an empty list.
Unbound exports need unique stable LLM output-message IDs and explicit recorded
tool availability; legacy triage evidence must be pulled and judged again.

Prepared schema version 2 keeps whole conversations and original positions in
canonical split files. Rendering derives one prefix plus assistant target at a
time, supplying exactly that target's tools. Fireworks and Baseten training and
validation mask all historical turns to zero loss. Unsupported or overlong
targets exclude the entire conversation without truncation. Conversation counts,
rendered datum counts, context tokens (including repeated prefixes), and target
tokens are reported separately. This reconstructs recorded prefixes, not hidden
prompt rewrites in the original model request.

`--inference-contract FILE` explicitly applies a global tool policy instead.
`capture-contract` can collect that alternate contract from a sample conversation;
its union behavior is not historical per-message capture. The prepared artifacts
record `global_override` and keep original source metadata unchanged. Old prepared
formats require re-preparation before training or replay.

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
Baseten sampler replay needs no `baseten-deploy` extra or Fireworks key when using
an Anthropic judge. Periodic replay during training is not implemented.

Fireworks requires its saved serverless training checkpoint; a promoted model ID
alone cannot be sampled. Baseten requires saved Loops sampler weights. Keep the
training run directory so another team member can resolve the selected checkpoint.
See [Fireworks in-session sampling](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models#in-session-sampling-serverless-training).

## Replay options

| Flag | Use |
| --- | --- |
| `--judge-model` | Judge route; default: `anthropic/claude-sonnet-5` |
| `--max-points-per-trajectory` | Cap assistant actions per trajectory; omitted by default, which scores every eligible action |
| `--max-output-tokens` | Maximum generated response tokens |
| `--concurrency` | Concurrent evaluation cases; default: 4 |

With `plan` and `train`, replay options require `--evaluate`. Repeat the same
options on the preview and paid command. `eval-plan` accepts the case and token
caps; judge and concurrency options belong on `evaluate`.

For a Fireworks judge, select
`--judge-model accounts/fireworks/models/deepseek-v4p1-flash` and set
`FIREWORKS_API_KEY`. For the internal Anthropic gateway, select
`--judge-model anthropic-gateway/claude-sonnet-5` and set
`LANGSMITH_GATEWAY_API_KEY`. Gateway credentials do not replace the LangSmith
API key used to read the dataset and publish experiments.

Replay checks tool selection, JSON arguments, argument schemas, and reference
arguments. Ambiguous text-encoded argument types fail format validation.
Tool-validation metrics are recorded in run outputs; they are separate from the
judge's agreement score. Invalid candidate schemas (also on text-reference turns)
and incorrect parallel-call sets force a failing agreement score. Each case starts from recorded context, and generated
tool calls are never executed. Both models receive the same target-specific
schemas, including an empty set. Earlier calls are checked with their own tools.
The judge receives definitions and future reference tool results as untrusted
evidence; future results never enter the candidate prefix.

## Experiment details

`train --evaluate` and standalone `evaluate` publish a base and tuned experiment
when comparing both models, or one experiment when evaluating only a tuned endpoint.
Names follow `smithtune-base-<short-model-name>-<evaluation-id>` and
`smithtune-tuned-<short-model-name>-<same-evaluation-id>`. Both use the base-model
name; exact model and checkpoint identifiers remain in metadata.
Each trajectory has a root run referencing the original LangSmith example, with
a child LLM run for each action. Children record actual input tools, original
message position, producing source run ID, and source trace ID.
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
are reused only for the same cases, bindings, models, settings, and pinned dataset
version. A later dataset version leaves the pinned evaluation intact. Saved generations can be judged without sampling again. A fully
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

Evaluation always publishes to LangSmith. Local files are recovery artifacts;
an upload failure leaves evaluation incomplete until publication succeeds.
Data prepared with `--no-sync-splits` must have its splits synchronized before
evaluation, using `dataset publish-splits --data-dir DIR`.

## Older curation directories

Old `selection.json`/`snapshot.json` directories are incompatible. Start a new
checkpoint with `dataset create NEW_DIR` and source flags. To extend the same
destination, pass `--dataset-id`; existing messages and historical bindings must
be an exact prefix. A legacy destination lacking bindings requires a separately
validated metadata upgrade or a fresh dataset. See [curation recovery](datasets.md#recovery-and-compatibility).

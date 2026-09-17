# Preparation and evaluation reference

[Back to the workflow](../README.md)

## Prepare data

Preparation collects each conversation's tools, including tools that were never
called. Tools added mid-run appear from the start of the training example.
Optional top-level arguments are combined when the rest of the tool definition
matches; the expanded schema applies to the whole conversation. Provider built-ins
(such as tool search) and incompatible tool definitions remain unsupported.

Existing datasets need `source_scope` (thread or trace), `source_scope_id`, and
`source_project_id` in each example's metadata; CLI-created datasets
include these automatically. Preparation errors if the source project ID is missing.
Recorded system messages are preserved; Qwen requires them at the start.

## Supported models

Run `smithtune models list --provider fireworks` or
`smithtune models list --provider baseten` to list supported model aliases and IDs.
Only listed models are supported. Preparation and training check provider
availability and select the tokenizer and renderer for the model.

| Provider | Model alias | Provider model ID | Training context limit |
| --- | --- | --- | --- |
| Baseten Loops | `qwen3p8-27b` | `Qwen/Qwen3.8-27B` | 262,144 |
| Baseten Loops | `kimi-k3` | `moonshotai/Kimi-K3` | 131,072 |
| Baseten Loops | `qwen3p5-9b` | `Qwen/Qwen3.5-9B` | 131,072 |
| Baseten Loops | `glm-5p3-flash` | `zai-org/GLM-5.3-Flash` | 131,072 |
| Fireworks serverless Training API | `qwen3p8-27b` | `accounts/fireworks/models/qwen3p8-27b` | 131,072 |
| Fireworks serverless Training API | `kimi-k3` | `accounts/fireworks/models/kimi-k3` | 196,608 |
| Fireworks serverless Training API | `deepseek-v4-flash-0731` | `accounts/fireworks/models/deepseek-v4-flash-0731` | 262,144 |
| Fireworks serverless Training API | `muse-glimmer-30b` | `accounts/fireworks/models/muse-glimmer-30b` | 131,072 |

## Preparation defaults

- One complete trajectory per source conversation; repeated source identities fail validation before tool capture, including with `--no-fetch`
- LoRA training on text and tool conversations; images are unsupported
- Tool definitions are combined by name across each conversation, using the latest recorded description and compatible optional arguments; earlier turns see the combined definitions
- Approximately 80% training, 10% validation, and 10% replay test, keeping each source conversation in one split
- All assistant messages are training targets, including earlier turns
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; use `--max-seq-len 32768` to lower the limit

## Splits and dataset versions

Preparation saves conversation assignments in `prepared/split_assignments.json`
and reuses them as the dataset grows. Existing prepared splits are recovered
from their saved rows and source provenance. Keep the same fractions when rerunning;
small datasets may have empty splits, which are reported without reshuffling.

When continuing training in a new data directory, add
`--split-from <previous-data-dir>` to `prepare` to preserve prior assignments.
Keep the assignments file, including entries for removed conversations.

Preparation also publishes these splits onto the original LangSmith dataset and
records its version for evaluation. Existing conflicting assignments stop
preparation. If split publication fails, rerun `prepare --no-fetch` with the same
settings to finish. `--no-fetch` still contacts LangSmith for this step; use
`--no-sync-splits` for local-only preparation.

## Source tools and workspaces

If source traces live in another workspace, add
`--source-workspace-id '<traces-workspace-id>'` to `prepare`; `--workspace-id`
still identifies the dataset workspace. Per-example `metadata.source_workspace_id`
takes precedence over this flag, which defaults to the dataset workspace. Your
LangSmith API key must have access to both. `dataset create` saves the source
workspace automatically; existing examples still need valid source scope
and project IDs.

Description changes are reported in `prepared/tool_description_replacements.json`
without rejecting examples. Incompatible argument schemas still fail preparation.

Interrupted tool capture resumes automatically when you rerun the same command with
the same data directory. Completed examples are checkpointed in
`raw/example_contracts.partial.json`; remove that file to restart capture from scratch.

Use `--no-fetch` to reuse downloaded data and completed tool schemas. Provider checks and
tokenizer loading still run. To supply the same tools for every example, use
`--inference-contract path/to/contract.json` instead of automatic tool capture.

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
| `--max-points-per-trajectory` | Cap assistant actions per conversation; omitted by default, which scores every eligible action |
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
judge's agreement score. Each case starts from recorded context, and generated
tool calls are never executed.

## Experiment details

`train --evaluate` and standalone `evaluate` publish a base and tuned experiment
when comparing both models, or one experiment when evaluating only a tuned endpoint.
Names follow `smithtune-base-<short-model-name>-<evaluation-id>` and
`smithtune-tuned-<short-model-name>-<same-evaluation-id>`. Both use the base-model
name; exact model and checkpoint identifiers remain in metadata.
Each conversation has a root run, with a child LLM run for each generated action.
Experiment metadata identifies the provider and whether
predictions came from a sampler or deployed endpoint. When a saved training run
is available, `parent_training_run_id` records the smithtune run ID and
`checkpoint_epoch` records the selected checkpoint’s epoch. Both experiments in
a base/tuned comparison carry this training context; the epoch describes the
tuned checkpoint. Endpoints without saved training provenance omit these fields.

- `teacher_agreement`: each action's judge pass/fail, with its explanation.
- `trajectory_teacher_agreement`: fraction of evaluated actions that passed in a
  conversation. Averaging this score weights conversations equally; local summary
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

Background publication sends up to 10 comparisons per batch, starting with the
first completed pair and then every five seconds or when a batch fills. Each
batch includes both models when comparing base and tuned responses.

Publication retries and indexing waits do not block model workers. If publication
fails, evaluation continues saving results locally and reports the upload failure.
On completion or Ctrl-C, the publisher gets up to 75 seconds to flush saved results;
model workers and owned serving resources still follow their normal cleanup.
Run uploads and resume checks use batches and the LangSmith SDK's native retries
for rate limits, transient server errors, and connection failures. Errors include the request method,
endpoint, status, and valid `Retry-After` timing. Known rate/usage-limit messages
are shown; other response bodies are omitted. Publication errors are also
recorded in `langsmith-experiments.json`. After upload, verification allows up
to one minute of backoff for runs and scores to become searchable, printing
progress while waiting. A timeout preserves the saved results for resumption.

Evaluation always publishes to LangSmith. Local files are recovery artifacts;
an upload failure leaves evaluation incomplete until publication succeeds.
Data prepared with `--no-sync-splits` must have its splits synchronized before
evaluation, using `prepare --no-fetch` with the original settings.

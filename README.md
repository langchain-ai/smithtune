<h1 align="center">smithtune</h1>

Fine-tune models on LangSmith trajectories with Fireworks or Baseten.
Both providers support training, sampler replay evaluation, and deployment.

## Setup

Install directly from GitHub using [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Git:

```bash
uv tool install --python 3.12 \
  'git+https://github.com/langchain-ai/smithtune.git'
smithtune doctor
smithtune --help
```

No local GPU or repository checkout is required. To upgrade, repeat the install
command with `--upgrade`.

Install these companion tools for the operations you use:

| Tool | Required for |
| --- | --- |
| [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) | Trace selection, dataset creation, contract capture, and fetching data during preparation |
| [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) | Fireworks deployment and undeployment |

Install the LangSmith CLI with the official installer, then start a new shell
or refresh your PATH so `langsmith` is available:

```bash
curl -fsSL https://cli.langsmith.com/install.sh | sh
langsmith --help
```

Follow each tool's installation and authentication instructions, then run
`smithtune doctor` to check local setup. It does not validate credentials.

Run from a writable directory. Data defaults to `./data/`; use `--data-dir` to
choose another location.

Commands show a music-note spinner and elapsed time in interactive terminals.
Routine library logs and Python warnings are suppressed; errors remain visible.
When output is redirected, they print a plain status line; JSON results stay on stdout.

Configure credentials in your environment:

| Task | Variable |
| --- | --- |
| Read LangSmith data and publish splits and evaluation experiments | `LANGSMITH_API_KEY` |
| Fireworks preparation, training, and inference | `FIREWORKS_API_KEY` |
| Baseten preparation, training, deployment, inference, and council judging | `BASETEN_API_KEY` |
| Direct Anthropic judging (triage and replay) | `ANTHROPIC_API_KEY` |
| Optional LangSmith gateway judging | `LANGSMITH_GATEWAY_API_KEY` |

Fireworks calls use `https://api.fireworks.ai` for training and deployment control,
and `https://api.fireworks.ai/inference/v1` for inference. Temporary evaluation
uses the REST API directly and does not require `firectl`; manual cleanup does.

## Using with a coding agent

Give your agent this prompt, replacing the placeholders:

```text
Help me <task> with smithtune using <provider>.
My data: <workspace/project/dataset IDs or prepared-data directory>.
Follow https://github.com/langchain-ai/smithtune/blob/main/AGENTS.md.
```

## Create a dataset from conversations

Create a dataset directly from a tracing project and filters:

```bash
smithtune dataset create \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --name my-sft-dataset \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --limit 100 \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'
```

Each matching root selects its whole thread when it has a thread ID, otherwise
its single trace. Thread examples include turns outside the filter window. Use
the returned dataset ID in `prepare`.

- Filters apply to trace root runs. The example selects correctness feedback of at least 0.9; see [filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax)
- `--limit` is required, at most 2000. Querying stops once that many distinct conversations are found, in the order LangSmith returns roots; no sampling is applied
- Each conversation is fetched with the trajectory API and stored as one example; `--concurrency` imports up to 4 at once (the default). Transient fetch failures are retried up to three times; example writes are never retried
- Use `--name` for a new dataset or `--dataset-id` for an existing dataset in the same workspace. If an import fails, inspect the returned receipt before retrying; uploads do not resume automatically

Both dataset paths save complete examples under `conversations/` in a local run
directory, defaulting to `data/datasets/<generated-id>/`. Creation saves each
conversation before uploading it, alongside the selection and import receipt.
Use `dataset create --run-dir <directory>` to choose a location. The returned
`run_dir` identifies the saved files; they remain on disk after upload or failure.

To add conversations to an existing dataset, use the same source flags with
`--dataset-id` instead of `--name`:

```bash
smithtune dataset create \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --dataset-id '<dataset-id>' \
  --start-time 2026-09-08T00:00:00Z --end-time 2026-09-15T00:00:00Z \
  --limit 100
```

Sources match by workspace, project, scope and scope ID. New conversations are
added; unchanged ones are skipped. Longer snapshots update the existing example
only when its saved messages are an exact prefix, preserving its ID and unrelated
metadata. Shorter or conflicting snapshots, or duplicate sources already in the
destination, stop the import. Extending a triaged example requires fresh passing
triage. Run only one import into a dataset at a time.

Existing-dataset imports download with bounded concurrency and write sequentially.
The receipt records created, updated and skipped counts, an action log, and any
pending write whose outcome needs checking. Earlier successful writes remain if a
later conversation fails.

## Label full trajectories with an agent council

Use `dataset triage` to select training conversations with model judges before
creating a dataset. Each judge sees the full conversation; a majority keep vote
selects it for import once all judges finish. Ties are dropped.

The default council is DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks, plus
GPT-5.6 Terra on OpenAI. Set `LANGSMITH_API_KEY`, `FIREWORKS_API_KEY`, and
`OPENAI_API_KEY`, then install the optional agent support:

```bash
uv tool install --upgrade --python 3.12 \
  'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git'
```

**1. Download and preview:**

```bash
smithtune dataset triage data/datasets/my-sft \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --limit 100
```

This saves conversations and a judging plan locally, without calling judges.
Messages come from `POST /v1/trajectory` with system messages included, for both
threads and standalone traces. Source runs supply tool schemas and media checks;
the judges receive the full saved message list.
`--limit` samples root traces; roots in the same thread become one conversation,
including history outside the time window. Repeat an interrupted download to
reuse saved progress. Omit the directory to generate one under `data/datasets/`.
Select runs that contain conversations. For example, use
`--filter 'eq(run_type,"chain")'` for agent runs in a project that also records
standalone prompt-rendering runs. A source with no messages stops the download
and reports its ID.

**2. Run the judges:**

```bash
smithtune dataset triage data/datasets/my-sft --confirm
```

Judging uses paid inference. The CLI saves each vote; repeat this command to
retry incomplete work with the saved settings. Completed runs make no new model
calls. Read `report.md` for results, `labels.jsonl` for keep/drop labels, and
`judgments.jsonl` for individual votes.

- Multimodal conversations are excluded before judging.
- If any judge rejects a conversation as too long, the whole conversation is
  excluded without truncation.
- Other request failures remain incomplete and can be retried.

**3. Import accepted conversations:**

```bash
smithtune dataset create --triage-dir data/datasets/my-sft --name selected-sft --confirm
```

Import uses the saved messages and tool schemas; unsupported training content
is excluded. Pass the returned dataset ID to `prepare`. To add to an existing
dataset, replace `--name` with `--dataset-id '<dataset-id>'`.

Use a new triage directory when conversations or judging settings change.
Older snapshots from the V2 message readers must be downloaded and judged again
to include system messages.
Extended conversations need fresh passing triage. After a partial import,
inspect `dataset-import.json` before retrying; uploads do not resume automatically.

### Change the council or selection rules

Add `--judges` and `--rule` to the preview command:

```bash
smithtune dataset triage data/datasets/custom-council \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra \
  --rule 'Drop answers that claim an action succeeded without evidence.'
```

Use any subset of these aliases, or `provider:model` for other models, such as
`anthropic:claude-sonnet-5`. The first model also runs the coordinator. Direct
Anthropic uses `ANTHROPIC_API_KEY`; `anthropic-gateway:<model-id>` uses
`LANGSMITH_GATEWAY_API_KEY`. The council and rules are saved for resume.

For Baseten judges, set `BASETEN_API_KEY` and use `baseten:<model-slug>` from
[Baseten Model APIs](https://docs.baseten.co/inference/model-apis/overview).
For example, this council mixes Baseten, Fireworks, and OpenAI:

```bash
--judges baseten:deepseek-ai/DeepSeek-V4.1-Flash,glm-5.3-flash,gpt-5.6-terra
```

Baseten uses `https://inference.baseten.co/v1`; no deployment is needed.
The first model also runs the coordinator. Baseten calls request reasoning off,
except GLM-5.3 variants, which require low reasoning. This supports managed
Model APIs; custom Baseten deployment URLs are not accepted.

Use `smithtune dataset triage --help` for filters, concurrency, and other options.
For coding agents, `smithtune skill export --output ./skills` exports the triage
skill and rubric. See the [trace-labeling audit](docs/trace-labeling-audit.md)
for detailed behavior and limitations.

## Prepare data

Preparation collects each conversation's tools, including tools that were never
called. Tools added mid-run appear from the start of the training example.
Optional top-level arguments are combined when the rest of the tool definition
matches; the expanded schema applies to the whole conversation. Provider built-ins
(such as tool search) and incompatible tool definitions remain unsupported.

Existing datasets need `source_scope` (thread or trace), `source_scope_id`, and
`source_project_id` in each example's metadata; CLI-created datasets
include these automatically. Preparation errors if the source project ID is missing.
Recorded system messages are preserved; the default
Qwen renderer requires them at the start.

Choose a provider and model, then prepare your dataset:

```bash
provider=fireworks # or baseten

smithtune prepare \
  --provider "$provider" \
  --workspace-id '<workspace-id>' \
  --dataset-id '<dataset-id>' \
  --model qwen3p8-27b
```

`--model` is required and accepts an alias or provider model ID from the list below.

List the models supported by smithtune:

```bash
smithtune models list --provider baseten
smithtune models list --provider fireworks
smithtune models list  # both providers
```

Only listed models are supported. Preparation and training check provider availability
automatically and select the appropriate tokenizer and formatting.

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

Preparation uses these defaults:

- One complete trajectory per source conversation; repeated source identities fail validation before tool capture, including with `--no-fetch`
- LoRA training on text and tool conversations; images are unsupported
- Tool definitions are combined by name across each conversation, using the latest recorded description and compatible optional arguments; earlier turns see the combined definitions
- Approximately 80% training, 10% validation, and 10% replay test, keeping each source conversation in one split
- All assistant messages are training targets, including earlier turns
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; use `--max-seq-len 32768` to lower the limit

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

## Plan and train

Review the plan before running `train`. Training is billed by the provider and requires `--confirm`.

```bash
smithtune plan --provider "$provider"
```

```bash
smithtune train \
  --provider "$provider" \
  --confirm
```

Training prints a generated run ID and saves artifacts to `./runs/<run-id>`.
Override either with `--run-id` or `--run-dir`; the folder must be new or empty.
Repeat customized training settings on both `plan` and `train`.

The best checkpoint is selected by validation loss and recorded in `<run-dir>/result.json`.
Training artifacts also include `plan.json`, `run-state.json`, and `epochs.json` in that directory.
Use `--init-from-checkpoint '<checkpoint-uri>'` to initialize a new training run from a saved checkpoint.
Baseten's optional spend guard requires both `--max-spend-usd` and `--hourly-rate-usd`.

## Deploy a Baseten checkpoint

For replay before deployment, use the [sampler workflow](#evaluate-a-trained-model) below.

Install the optional deployment tools:

```bash
uv tool install --upgrade --python 3.12 \
  'smithtune[baseten-deploy] @ git+https://github.com/langchain-ai/smithtune.git'
```

Set `BASETEN_API_KEY` and, for replay judging, `ANTHROPIC_API_KEY`.
Supported Baseten models are public and do not require a Hugging Face token.
Choose GPUs explicitly: `H200:1`
below is an example, not a verified allocation for every model.

Preview cases and deployment settings, then run a temporary evaluation:

```bash
run_dir='<run-dir printed by train>'
smithtune eval-plan --provider baseten --serving-mode temporary \
  --run-dir "$run_dir" --data-dir data/qwen3p8-27b --output-dir "$run_dir/replay" \
  --accelerator H200:1 --max-seq-len 32768
smithtune evaluate --provider baseten --serving-mode temporary \
  --run-dir "$run_dir" --data-dir data/qwen3p8-27b --output-dir "$run_dir/replay" \
  --accelerator H200:1 --max-seq-len 32768 --deployment-timeout 1800 --confirm
```

The plan makes no deployment or inference calls. Evaluation creates or activates
an endpoint from Baseten's official generated Loops serving template, checks it,
runs replay, and deactivates serving replicas on completion or failure. It preserves
the model and checkpoint. Repeat with the same directories to resume; hardware
and context flags can be omitted once the deployment receipt exists.
Temporary mode does not support `--base-model`. Use a separate output directory.

For an endpoint that stays running, deploy and evaluate using its saved receipt:

```bash
smithtune deploy --provider baseten --run-dir "$run_dir" \
  --accelerator H200:1 --max-seq-len 32768 --confirm
smithtune evaluate --provider baseten --serving-mode existing --run-dir "$run_dir" \
  --data-dir data/qwen3p8-27b --output-dir "$run_dir/replay" --confirm
smithtune undeploy --provider baseten --run-dir "$run_dir" --confirm
```

`--max-seq-len` is an evaluation cap verified against the live server; it does
not configure serving context. Replay also respects the prepared-data limit.
`deploy` waits up to 1800 seconds; temporary evaluation defaults to 600, adjustable
with `--deployment-timeout`. Endpoint IDs are saved even if smoke checks fail.
A killed process or cleanup failure may leave paid capacity running; use the
`undeploy` command above to deactivate its replicas. If creation has an unknown
outcome without saved IDs, inspect Baseten before retrying. Keep the receipt.

### Evaluate an existing Baseten endpoint

After [deploying your Loops checkpoint](https://docs.baseten.co/loops/deploy-checkpoints),
evaluate its dedicated chat endpoint using `BASETEN_API_KEY` and, for the default
judge, `ANTHROPIC_API_KEY`:

```bash
smithtune evaluate \
  --provider baseten \
  --data-dir data/qwen3p8-27b \
  --output-dir runs/my-sft/replay \
  --model-id '<baseten-model-id>' \
  --deployment-id '<baseten-deployment-id>' \
  --tuned-model '<checkpoint-name>' \
  --max-seq-len 32768 \
  --confirm
```

Use data prepared with Baseten and an endpoint serving the same base model and
compatible chat template, with tool/reasoning parsing configured for your model.
`--tuned-model` is the served checkpoint **name**, not its globally unique ID.
Set `--max-seq-len` to the endpoint's configured context limit; replay uses the
lower of that limit and the preparation limit, including the output budget.

Use `eval-plan` with the same data and endpoint options, without `--confirm`, to
preview cases. Add `--base-model '<served-base-model-name>'` to compare a base
route available on the **same endpoint**. Results go to `summary.json`; rerun
the same command to resume. No Fireworks key is needed with an Anthropic judge.
This path uses an existing deployment and leaves it running; manage externally
created deployments in Baseten. Training support alone does not verify a model's
serving configuration.

## Evaluate a trained model

Both providers can compare the base model and best checkpoint through their
training API samplers. No deployment command or model promotion is needed.

To run replay after training:

```bash
provider=baseten # or fireworks
smithtune plan --provider "$provider" --data-dir data --evaluate
smithtune train --provider "$provider" --data-dir data \
  --run-dir runs/my-sft --evaluate --confirm
```

`--evaluate` adds model and judge inference to the training run. The plan shows
case counts, prompt tokens, and call counts. Fireworks reuses its serverless
training session. Baseten shuts down its trainer, then starts dedicated Loops
samplers for the saved best weights and base model. Both samplers are deactivated
on exit; sampler and judge costs are separate from Baseten's training spend guard.
Baseten sampler replay needs no `baseten-deploy` extra or Fireworks key when using
an Anthropic judge. Repeat custom settings on `plan` and `train`.

```text
prepare
  +-- train.jsonl       ~80%: training
  +-- validation.jsonl  ~10%: checkpoint selection
  +-- test.jsonl        ~10%: replay

train --evaluate
  +-- build replay cases from test.jsonl
  +-- check the judge on known positive and negative examples
  +-- start training
  +-- train, validate, and save a checkpoint after each epoch
  +-- select the epoch with the lowest validation loss
  +-- open samplers for that checkpoint and the base model
  +-- create experiments and show the comparison link
  +-- sample base and tuned responses to the same test cases
  +-- score, save, and publish completed comparisons during replay
  +-- clean up sampler resources and finish pending publication
```

You do not write separate replay examples. The CLI makes a case before each
visible assistant message in the test conversations. The model sees the recorded
prefix and tool definitions. The reference is the next recorded assistant text
or tool call. Each case starts from recorded history, including recorded tool
results; generated tool calls are never executed.

Validation loss selects the checkpoint. Replay scores do not influence training
or checkpoint selection. The full-trajectory triage council is a separate step
that decides which conversations enter the dataset.

For a completed training run:

```bash
smithtune eval-plan --provider "$provider" --data-dir data --run-dir runs/my-sft
smithtune evaluate --provider "$provider" --data-dir data --run-dir runs/my-sft --confirm
```

`eval-plan` builds and previews cases without opening a session or calling models.
`evaluate` reads the best checkpoint from `result.json` and uses the same sampler
and scorer as `train --evaluate`, without training steps. Base-model comparison
is automatic. Periodic replay during training is not implemented.

Both paths support `--judge-model`, `--concurrency`, `--max-output-tokens`, and
`--max-points-per-trajectory` (an optional cap; the default scores every assistant
action). With `train`, these options require `--evaluate`. Results go to
`<run-dir>/replay`; standalone evaluation can use `--output-dir` to start another
comparison with different settings.

Replay checks tool selection, JSON arguments, argument schemas, and reference
arguments. Ambiguous text-encoded argument types fail format validation.
A calibrated judge also scores whether each response matches the
recorded behavior. Results include base and tuned pass rates, paired wins and
regressions, and scores by text/tool-call case type.

- `cases.jsonl`: frozen replay inputs and references.
- `generations.jsonl`: generated responses saved before judging.
- `results.jsonl`: per-case responses, checks, and judgments.
- `summary.json`: aggregate scores and base-versus-tuned differences.
- `sampler.json`: sampler identities and cleanup status.

Repeat standalone `evaluate` with the same data, run directory, output directory,
and settings to finish missing results. Saved generations can be judged without
sampling again. A fully completed evaluation makes no model calls. Training
results and checkpoints survive a replay failure; the command reports the
incomplete replay and exits with an error. Use a new training run directory to
train again.

Fireworks requires its saved serverless training checkpoint; a promoted model ID
alone cannot be sampled. Baseten requires saved Loops sampler weights. If a
process is killed or Baseten cleanup fails, use the deployment IDs and cleanup
instructions in `sampler.json` to stop paid capacity before resuming.
See [Fireworks in-session sampling](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models#in-session-sampling-serverless-training).

Replay judging calls Anthropic directly by default with `ANTHROPIC_API_KEY`.
For a Fireworks judge, pass `--judge-model accounts/fireworks/models/deepseek-v4p1-flash`.
To use the internal Anthropic gateway, explicitly select
`--judge-model anthropic-gateway/claude-sonnet-5` and set
`LANGSMITH_GATEWAY_API_KEY`. Fireworks training and sampling always use the
official Fireworks API.

### Review replay results in LangSmith

`train --evaluate` and standalone `evaluate` publish a base and tuned experiment
when comparing both models, or one experiment when evaluating only a tuned endpoint.
Names follow `smithtune-base-<short-model-name>-<evaluation-id>` and
`smithtune-tuned-<short-model-name>-<same-evaluation-id>`. Both use the base-model
name; exact model and checkpoint identifiers remain in metadata.
The CLI prints one comparison link before generation starts. Open it to watch
completed results appear side by side on the original dataset's test split.
For tuned-only evaluation,
the same link opens that experiment. Each conversation has a root run, with a
child LLM run for each generated action. Experiment metadata identifies the provider and whether
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

LangSmith's pinned dataset version is verified before paid work. For older
prepared data, rerun `prepare --no-fetch` with its original settings to register
the splits first. `plan` and `eval-plan` remain local previews.

Each completed action comparison is saved locally, then queued for background
publication. The publisher sends up to 10 comparisons per batch, starting with
the first completed pair and then every five seconds or when a batch fills.
Each batch includes both models. Conversation outputs show completed and total
actions; their running average includes only completed judgments. Refresh the
comparison view to see new results. The final JSON also contains the same
`langsmith.comparison_url`.

Publication retries and indexing waits do not block model workers. If publication
fails, evaluation continues saving results locally and reports the upload failure.
On completion or Ctrl-C, the publisher gets up to 75 seconds to flush saved results;
model workers and owned serving resources still follow their normal cleanup.
If publication remains pending, repeat standalone `evaluate` with the same
directories and settings. Completed predictions and judgments are reused;
unfinished cases still require evaluation. Keep the local files for recovery.
Run uploads and resume checks use batches and the LangSmith SDK's native retries
for rate limits, transient
server errors, and connection failures. Errors include the request method,
endpoint, status, and valid `Retry-After` timing. Known rate/usage-limit messages
are shown; other response bodies are omitted. Publication errors are also
recorded in `langsmith-experiments.json`. After upload, verification allows up
to one minute of backoff for runs and scores to become searchable, printing
progress while waiting. A timeout preserves the saved results for resumption.

Evaluation always publishes to LangSmith. Local files are recovery artifacts;
an upload failure leaves evaluation incomplete until publication succeeds.
Data prepared with `--no-sync-splits` must have its splits synchronized before
evaluation, using `prepare --no-fetch` with the original settings.

### Keep an endpoint running with `deploy`

Use `deploy` when you want an endpoint for repeated use. It starts the endpoint;
it does not run the evaluation. The endpoint stays available and can incur
charges until you run `undeploy`:

```text
promote -> deploy -> use endpoint -> undeploy
```

```bash
run_dir='runs/my-sft'
account_id='<your-fireworks-account>'
run_id='my-sft'
deployment_id='my-sft'
deployment_shape='<compatible-deployment-shape>'

smithtune promote --run-dir "$run_dir" --output-model-id "$run_id" --confirm
smithtune deploy \
  --run-dir "$run_dir" --account-id "$account_id" \
  --output-model-id "$run_id" --deployment-id "$deployment_id" \
  --deployment-shape "$deployment_shape" --confirm
```

For replay, use `evaluate --run-dir` as described above. The serverless sampler
uses the saved training checkpoint independently of this production endpoint.

```bash
smithtune undeploy --account-id "$account_id" --deployment-id "$deployment_id" --confirm
```

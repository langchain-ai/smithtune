<h1 align="center">smithtune</h1>

Fine-tune models on LangSmith trajectories with Fireworks or Baseten.
Fireworks supports training, deployment, and replay evaluation; Baseten supports SFT checkpoints.

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

Configure credentials in your environment:

| Task | Variable |
| --- | --- |
| Read LangSmith datasets and runs | `LANGSMITH_API_KEY` |
| Fireworks preparation, training, and inference | `FIREWORKS_API_KEY` |
| Baseten preparation and training | `BASETEN_API_KEY` |
| Replay judge | `ANTHROPIC_API_KEY` containing a **LangSmith gateway key**, or `ANTHROPIC_CUSTOM_HEADERS` |

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
- Choose a new dataset name. If an import fails, inspect its receipt in `data/selections/` before retrying

## Label traces with an agent council

`dataset triage` downloads traces locally and labels each one for SFT.
It starts a Deep Agent coordinator, which uses Python code to launch judge
subagents. The default council uses three different models:

| Judge | API provider | Model ID |
| --- | --- | --- |
| DeepSeek V4.1 Flash | Fireworks | `accounts/fireworks/models/deepseek-v4p1-flash` |
| GLM-5.3-Flash | Fireworks | `accounts/fireworks/models/glm-5p3-flash` |
| GPT-5.6 Terra | OpenAI | `gpt-5.6-terra` |

Each judge gets fresh context. DeepSeek also runs the coordinator.
This selects training examples; `evaluate` tests a trained model.

Install the optional agent support and set `FIREWORKS_API_KEY`, `OPENAI_API_KEY`,
and `LANGSMITH_API_KEY` in your environment:

```bash
uv tool install --upgrade --python 3.12 \
  'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git'
```

**1. Download and preview.** Supply the source only on the first run:

```bash
smithtune dataset triage data/triage \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --limit 100
```

This saves the messages and full run trees in `snapshot.json`, and the council
settings and vote count in `plan.json`. It makes no judge calls. Selected
threads expand to their full history, including turns outside the time window.
Thus the number of traces to judge can exceed `--limit`.
Successful read responses are saved under `download/`. If downloading stops,
repeat the command to reuse them. The CLI waits and retries when LangSmith
returns a rate limit. Trace run data comes from the V2 endpoint,
`GET /api/v2/traces/{trace_id}/runs`. Empty turns remain in the saved evidence;
training-format checks do not stop the download.
If a root run is missing, the saved trace includes a warning for the judges.
That conversation cannot enter training through this import flow.

Before judging, the CLI filters traces with multimodal content in their
messages, run inputs/outputs, or media attachments. Those traces get `0` with
a filter reason and incur no judge calls. Every remaining trace is sent to
the three default judges, or to your chosen council.
The check includes conversation history supplied to the judge. It does not
remove media blocks and then judge an altered trace.

**2. Label, or resume an interrupted run:**

```bash
smithtune dataset triage data/triage --confirm
```

The CLI reuses the saved source and settings. New defaults apply only to new
plans; existing plans retain their selected models. It saves each vote as it finishes.
Repeating the command retries incomplete votes; a completed run makes no new
agent calls. The directory defaults to `data/triage` if omitted.

```text
LangSmith traces
      | V2 download; save messages, runs, and tool schemas
      v
Local snapshot
      |
Media filter ---- media found ----> 0 + reason; no judge calls
      | text only
      v
Deep Agent coordinator -> Python code -> independent judge subagents
                                               | read saved evidence
                                               v
                                  Validated votes -> majority label
                                               |
                                   labels.jsonl: 1/0 + reason
                                               |
                                  dataset create --triage-dir
                                               | upload all-pass conversations
                                               v
                                     prepare -> plan -> train
```

Judges read the conversation and a run index. Long messages carry a `read_full`
reference. Judges use read-only Python with `read_message(index)` and
`read_run(id)` to inspect original messages and run inputs/outputs. Previews
are marked as incomplete; the full saved content remains available through
code. Media references are saved as JSON, not rendered for the judges. Judges
cannot run recorded tools, access host files or secrets, or delegate further.
The coordinator cannot write a verdict in place of a judge.

The result is one line per trace in `labels.jsonl`:

```json
{"trace_id":"...","keep":1,"reason":"2/3 judges voted 1. The answer completes the request and the tool results support it."}
```

`1` means use for SFT; `0` means do not use. A majority of the council decides
the label; a tie gives `0`. The reason combines the reasons from judges who
voted for that label. The CLI ends with counts and the result path. `report.md`
explains each label in plain text. Detailed votes and source quotes stay in
`judgments.jsonl` for inspection. No extra model call is needed for the report.

If a judge cannot finish, the reason says "Labeling incomplete" and the trace
has `0` until a retry completes it. The command reports the unfinished count
and exits with code 1. Repeat the same command to retry. Labels stay local;
triage does not write feedback to LangSmith.

**3. Create a dataset from accepted conversations:**

```bash
smithtune dataset create --triage-dir data/triage --name selected-sft --confirm
```

Every trace in a conversation must pass. This prevents a passing turn from
bringing rejected earlier behavior into training. Unsupported content and tool contracts
are excluded and counted in the summary. Import uses the saved messages and
tool schemas without fetching the source again. Pass the returned dataset ID
to `prepare` below. Keep a separate test set for model comparisons.

After a partial import, inspect `dataset-import.json` before retrying. Dataset
imports do not resume automatically. Other complete conversations can still
be imported when some trace labels are incomplete.

### Change the council or selection rules

Use one `--judges` list. Omit it to use these three defaults. Use `--rule`
to add project rules:

```bash
smithtune dataset triage data/custom-council \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra \
  --rule 'Drop answers that claim an action succeeded without evidence.'
```

Choose any subset, or repeat a model to give it independent judge slots.
For other models, use `provider:model` in the same list, for example
`--judges openai:<model-id>,fireworks:accounts/fireworks/models/<model-id>`.
The list replaces the council and is saved for confirm and resume; you do not
need to repeat it. The first judge model also runs the
coordinator. Terra uses OpenAI Responses for reasoning with tools. The Fireworks
adapter preserves reasoning fields between tool calls. Fireworks uses its official API and `FIREWORKS_API_KEY`; OpenAI
uses `OPENAI_API_KEY`. Direct Anthropic uses `SMITHTUNE_ANTHROPIC_API_KEY`.
`anthropic-gateway` uses the LangSmith Anthropic gateway credential.

`--concurrency` sets the maximum active judge tasks (default 4). Advanced
options remain supported: `--config` for a judge JSON file, `--runner api` for
direct calls without agents, `--attempts` (default 3), `--max-input-chars`
(default 200,000), and `--max-output-tokens` (default 4,096). Agent input limits
apply to the conversation, run index, and rubric; direct calls include full
run payloads. Oversize inputs remain incomplete. Each judge attempt has at most
24 graph steps. Coordinator calls add to judge cost. `agent-state.json` records
its status; votes include judge code-call counts and run IDs read.

The source defaults to a seeded sample of 100 matching roots before thread
expansion. `--filter` accepts LangSmith root-run filters. Once judging starts,
changed evidence, rules, models, skill, or input/output limits require a new
run directory. One process can use a directory at a time.

Any CLI-capable agent can load the same portable skill:

```bash
smithtune skill export --output ./skills
```

This exports `sft-trace-triage/SKILL.md`, the judge rubric, and an optional
config example. The skill uses the CLI for download, labels, resume, and import.

See the [trace-labeling audit](docs/trace-labeling-audit.md) for live checks,
packaging coverage, and current quality and scaling limits.

## Prepare data

Preparation collects each conversation's tools, including tools that were never
called. Tools added mid-run appear from the start of the training example.
Optional top-level arguments are combined when the rest of the tool definition
matches; the expanded schema applies to the whole conversation. Provider built-ins
(such as tool search) and incompatible tool definitions remain unsupported.

Existing datasets need `source_scope` (thread or trace), `source_scope_id`, and
`source_project_id` in each example's metadata; CLI-created datasets
include these automatically. Recorded system messages are preserved; the default
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

- LoRA training on text and tool conversations; images are unsupported
- Tool definitions are combined by name across each conversation, using the latest recorded description and compatible optional arguments; earlier turns see the combined definitions
- 80% training, 10% validation, and 10% replay test, keeping each source conversation in one split
- All assistant messages are training targets, including earlier turns
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; use `--max-seq-len 32768` to lower the limit

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

## Evaluate a trained model (Fireworks)

`deploy` starts a serving endpoint so a model can answer requests. `evaluate`
sends test prompts to a model and scores its answers. A trained checkpoint
needs running compute before it can answer a prompt.

For a temporary evaluation, let `evaluate` manage that compute:

```text
promote -> eval-plan -> evaluate --serving-mode preemptible
                              |
                              +-- create temporary serving capacity
                              +-- wait until ready
                              +-- generate and score responses
                              +-- save results and delete the deployment
```

**You do not run `deploy` or `undeploy` yourself for this path.** Preemptible
capacity uses idle GPUs that Fireworks can reclaim. If that interrupts the run,
repeat the evaluation command to finish missing cases.

First, promote the selected checkpoint to a Fireworks model ID:

```bash
account_id='<fireworks-account-id>'
run_id='<run-id printed by train>'
run_dir='<run-dir printed by train>'

smithtune promote \
  --run-dir "$run_dir" \
  --output-model-id "$run_id" \
  --confirm

eval_shape='<full-compatible-fireworks-deployment-shape-resource>'
```

The deployment shape specifies compatible serving hardware. Use a shape for
the promoted model, in the form
`accounts/<account>/deploymentShapes/<shape>` (optionally with `/versions/<version>`).
See [Fireworks evaluation paths](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models).
Use the data directory from `prepare` (`data` in these examples).
`eval-plan` previews the held-out cases and deployment settings. It does not
start a deployment or run model inference:

```bash
smithtune eval-plan \
  --data-dir data \
  --output-dir "$run_dir/replay" \
  --tuned-model "accounts/$account_id/models/$run_id" \
  --serving-mode preemptible --account-id "$account_id" \
  --deployment-id "$run_id-eval" --deployment-shape "$eval_shape"
```

Run `evaluate` with the same settings. It checks the judge, creates one
temporary replica, waits for readiness, and runs the evaluation:

```bash
smithtune evaluate \
  --data-dir data \
  --output-dir "$run_dir/replay" \
  --tuned-model "accounts/$account_id/models/$run_id" \
  --serving-mode preemptible --account-id "$account_id" \
  --deployment-id "$run_id-eval" --deployment-shape "$eval_shape" \
  --confirm
```

For each case, the evaluator:

1. Takes recorded conversation context from the held-out data.
2. Asks the tuned model for its next response or tool call.
3. Checks the response and asks a judge model to score it against the recorded behavior.
4. Saves the result. It does not execute generated tool calls.

The CLI then deletes the temporary deployment and confirms deletion. Results
are saved to `<run-dir>/replay/summary.json`.
Add `--base-model '<deployed-base-model-route>'` for a before/after comparison.
The base route must already be available; the temporary deployment serves only
the tuned model. Model and judge inference use current provider rates.

The default readiness timeout is 600 seconds; use `--deployment-timeout` to
change it. Capacity loss leaves the evaluation interrupted, rather than scoring
a model failure. Repeat the command with the same settings and output directory
to finish missing cases. Completed cases are retained. Completed runs do not
repeat inference, but still check deployment cleanup.

Ownership and cleanup are recorded in `deployments/<deployment-id>.json` under
the evaluation directory. The CLI refuses to use or delete an unrelated
deployment. It attempts cleanup after success, failure, or a keyboard interrupt.
A killed process or failed API request can leave capacity behind. The receipt
contains the exact `smithtune undeploy ... --confirm` recovery command; a cleanup
failure is reported as an error. Inspect that receipt before deleting capacity.

This mode requires a promoted model ID. It does not open an in-session sampling
client from an active training checkpoint.

### Keep an endpoint running with `deploy`

Use `deploy` when you want an endpoint for repeated use. It starts the endpoint;
it does not run the evaluation. The endpoint stays available and can incur
charges until you run `undeploy`:

```text
promote -> deploy -> evaluate -> undeploy
```

```bash
smithtune deploy \
  --run-dir "$run_dir" --account-id "$account_id" \
  --output-model-id "$run_id" --deployment-id "$run_id" \
  --deployment-shape "$eval_shape" --confirm
```

To evaluate a route that is already serving, omit `--serving-mode` (its default
is `existing`) and all temporary deployment flags:

```bash
smithtune eval-plan --output-dir "$run_dir/existing-replay"
smithtune evaluate \
  --output-dir "$run_dir/existing-replay" \
  --tuned-model "accounts/$account_id/models/$run_id#accounts/$account_id/deployments/$run_id" \
  --confirm
```

Remove the endpoint when finished to stop deployment billing:

```bash
smithtune undeploy \
  --account-id "$account_id" \
  --deployment-id "$run_id" \
  --confirm
```

Use `smithtune <command> --help` for more options. See [Contributing](CONTRIBUTING.md)
for local development, tests, and releases.

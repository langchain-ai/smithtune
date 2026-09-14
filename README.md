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
| [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) | Dataset creation, contract capture, and fetching data during preparation |
| [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) | Fireworks deployment and undeployment |

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
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'
```

Each example contains a whole conversation, including turns outside the filter
window. Use the returned dataset ID in `prepare`.

- Filters apply to trace root runs. The example selects correctness feedback of at least 0.9; see [filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax)
- All matching threads are included; use `--limit 100` to sample up to 100
- Choose a new dataset name. If an import fails, inspect its receipt in `data/selections/` before retrying

## Prepare data

Preparation collects each conversation's tools, including tools that were never
called. Tools added mid-run appear from the start of the training example.
Optional top-level arguments are combined when the rest of the tool definition
matches; the expanded schema applies to the whole conversation. Provider built-ins
(such as tool search) and incompatible tool definitions remain unsupported.

Existing datasets need source thread/trace and project IDs; CLI-created datasets
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
| Baseten Loops | `qwen3p8-27b` | `Qwen/Qwen3.8-27B` | 131,072 |
| Baseten Loops | `kimi-k3` | `moonshotai/Kimi-K3` | 131,072 |
| Baseten Loops | `qwen3p5-9b` | `Qwen/Qwen3.5-9B` | 131,072 |
| Baseten Loops | `glm-5p3-flash` | `zai-org/GLM-5.3-Flash` | 131,072 |
| Fireworks serverless Training API | `qwen3p8-27b` | `accounts/fireworks/models/qwen3p8-27b` | 131,072 |
| Fireworks serverless Training API | `kimi-k3` | `accounts/fireworks/models/kimi-k3` | 196,608 |
| Fireworks serverless Training API | `deepseek-v4-flash-0731` | `accounts/fireworks/models/deepseek-v4-flash-0731` | 262,144 |
| Fireworks serverless Training API | `muse-glimmer-30b` | `accounts/fireworks/models/muse-glimmer-30b` | 131,072 |

Preparation uses these defaults:

- LoRA training on text and tool conversations; images are unsupported
- 80% training, 10% validation, and 10% replay test, keeping each source conversation in one split
- All assistant messages are training targets, including earlier turns
- Reasoning is omitted; add `--reasoning-policy preserve` to retain it
- Examples over the context limit are rejected without truncation; use `--max-seq-len 32768` to lower the limit

Use `--no-fetch` to reuse downloaded data and tool schemas. Provider checks and
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

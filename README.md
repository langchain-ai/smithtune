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

## Select SFT traces with model judges

Use `dataset triage` to label traces before creating the training dataset.
A judge is a model that checks recorded behavior against the selection rules.
Each trace gets `keep: 1` or `keep: 0`, a completion status, reasons, and evidence.
This checks the training examples; `evaluate` checks the trained model.

This adds a judging step between local trace capture and dataset creation:

```text
LangSmith traces -> local snapshot -> Deep Agent coordinator
                                           |
                                      Python code mode
                                           |
                                    judge subagents
                                           |
                                    checked 0/1 labels
                                           |
                    dataset create --triage-dir -> prepare -> plan -> train
```

First download and save the source evidence without calling a judge:

```bash
smithtune dataset triage \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --limit 100 --output-dir data/triage --dry-run
```

Review `data/triage/plan.json`. Then repeat the same command with `--confirm`
in place of `--dry-run` to run paid judging. The default is one
`claude-sonnet-5` judge through the Anthropic gateway. Use `--config judges.json`
on both commands to set 1–16 named judge slots and optional rules:

```json
{
  "judges": [
    {"name": "judge-1", "provider": "anthropic-gateway", "model": "claude-sonnet-5"},
    {"name": "judge-2", "provider": "anthropic-gateway", "model": "claude-sonnet-5"},
    {"name": "judge-3", "provider": "anthropic-gateway", "model": "claude-sonnet-5"}
  ],
  "rules": ["Drop answers that claim an action succeeded without evidence."]
}
```

Each slot makes a fresh call for each trace. Slots can use the same model or
different models. All slots must return a valid vote. A strict majority keeps
the trace; ties drop it. An invalid or failed vote makes the label incomplete
with `keep: 0`. It is not counted as a quality failure.

| Judge provider | Credential | Endpoint |
| --- | --- | --- |
| `fireworks` | `FIREWORKS_API_KEY` | Official Fireworks inference API |
| `openai` | `OPENAI_API_KEY` | Official OpenAI API |
| `anthropic` | `SMITHTUNE_ANTHROPIC_API_KEY` | Official Anthropic API |
| `anthropic-gateway` | `ANTHROPIC_API_KEY` or `ANTHROPIC_CUSTOM_HEADERS` | LangSmith Anthropic gateway |

Use a model ID available to the selected provider. Direct Anthropic uses a
separate variable to avoid sending the existing gateway key to another service.

Selection and recovery:

- The time window and `--filter` select root traces. `--limit` defaults to 100 roots, sampled with `--seed 42` before thread expansion.
- Selected threads expand to all turns, including earlier turns outside the window. Every expanded trace is judged in its recorded context. Standalone traces are supported.
- Messages and the full run tree are saved in `snapshot.json`. Missing pages, active traces, and unsupported message formats stop the snapshot before paid judging. Queries have a 1,000-page limit; thread expansion has a 10,000-trace limit.
- Evidence is treated as data. Judges cannot execute recorded tools. Quotes must match an actual message or run.
- Default limits are 4 concurrent tasks, 3 attempts per task, 200,000 input characters, and 4,096 output tokens. Use `--concurrency`, `--attempts`, `--max-input-chars`, and `--max-output-tokens` to change them. Input includes the rubric and full evidence; it is never shortened to fit.
- Read `summary.json`, `report.md`, `labels.jsonl`, and `judgments.jsonl`. The summary includes the number of eligible training conversations. An incomplete run prints its summary and exits with status 1.
- Repeat the same command and output directory to retry failed votes. Successful votes are reused. Source, rubric, model, runner, and input/output limits must match. Changed settings require a new output directory. One process can use the directory at a time.

Create a dataset from the accepted saved conversations:

```bash
smithtune dataset create \
  --triage-dir data/triage --name selected-sft --confirm
```

Only complete conversations whose **every trace passes** are eligible. This
matters because preparation trains on all assistant messages in each example.
One passing turn cannot admit a rejected turn from the same thread. Conversations
with unsupported tool schemas are excluded and counted in the summary.

The import uses the saved messages and tool schemas. It does not fetch the
live source again. The returned dataset ID goes to `prepare` below. Preparation
checks the saved message hash, even with `--no-fetch` or a global contract.
Review `dataset-import.json` after a partial write; imports do not resume or
repeat automatically. Other complete conversations can be imported while some
labels remain incomplete. Keep an independent test set for model comparisons.

### Deep Agents and the portable skill

Install the optional [Deep Agents](https://github.com/langchain-ai/deepagents)
extra to run a coordinator with code mode and judge subagents:

```bash
uv tool install --upgrade --python 3.12 \
  'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git'
```

Label an existing local snapshot with:

```bash
smithtune dataset triage --output-dir data/triage \
  --runner deepagent --config judges.json --confirm
```

Once `snapshot.json` exists, source flags can be omitted. The CLI reads the
saved local evidence without querying LangSmith again. Labels stay local;
triage does not write feedback to the tracing project.

One Deep Agent loads the packaged skill and uses Python code to list pending
trace/judge pairs and dispatch batches of judge subagents. It can also dispatch
an individual judge with the `task` tool. The first configured judge model
serves as the coordinator. Each slot uses its configured model for judging.
The default `--runner api` remains available without the optional agent runtime.

Code runs in a Monty sandbox with `pending_tasks`, `read_trace`, and
`judge_batch` functions. It has no shell, network, environment, or host file
access. Each code call has 32 MiB of memory, a 5-second execution limit, and at
most 256 host calls; judge batches have at most 128 pairs. `--concurrency`
limits active judge tasks across batches. Duplicate dispatches cannot repeat
paid votes within the same run.

Each judge gets a fresh Deep Agent with the full saved trace, preceding context,
and run tree. Judges read the rubric but cannot execute recorded tools or
delegate further. Automatic summarization is disabled. Inputs above the limit
remain incomplete. A judge can also report missing evidence as incomplete.
Each judge attempt has at most 12 graph steps. The coordinator has at most
`min(1000, 24 + 4 * pending_tasks)` graph steps, so its calls add to judge cost.

Only validated subagent votes enter the label files. The coordinator cannot
replace them with its own final answer. Votes are saved as subagents finish;
`agent-state.json` records coordinator status and code-call counts. Resume
reuses successful votes; a fully completed run starts no agent. Coordinator
skill changes require a new output directory, like rubric or model changes.

Any agent that can run the CLI can use the same portable skill:

```bash
smithtune skill export --output ./skills
```

This writes `skills/sft-trace-triage/SKILL.md`, the judge rubric, and an example
config. Point your agent at that directory or use its normal skill loader.
The skill uses the CLI for fetching, labels, resume, and dataset creation.

## Prepare data

Preparation collects each conversation's tools, including tools that were never
called. Tools added mid-run appear from the start of the training example.
Provider built-ins (such as tool search) and conflicting definitions of the same
tool are unsupported.

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

## Deploy and evaluate (Fireworks)

Promote the selected checkpoint, then deploy it. The endpoint incurs charges until removed.

```bash
account_id='<fireworks-account-id>'
run_id='<run-id printed by train>'
run_dir='<run-dir printed by train>'

smithtune promote \
  --run-dir "$run_dir" \
  --output-model-id "$run_id" \
  --confirm

smithtune deploy \
  --run-dir "$run_dir" \
  --account-id "$account_id" \
  --output-model-id "$run_id" \
  --deployment-id "$run_id" \
  --deployment-shape '<compatible-fireworks-shape>' \
  --confirm
```

Review the replay cases, then evaluate with the gateway credentials above:

```bash
smithtune eval-plan --output-dir "$run_dir/replay"
```

```bash
smithtune evaluate \
  --output-dir "$run_dir/replay" \
  --tuned-model "accounts/$account_id/models/$run_id#accounts/$account_id/deployments/$run_id" \
  --confirm
```

Results are saved to `<run-dir>/replay/summary.json`. Replay scores agreement with recorded actions without executing tools.
Add `--base-model '<deployed-base-model-route>'` for a before/after comparison. Reuse the output directory to resume an interrupted evaluation.

Remove the endpoint when finished to stop deployment billing:

```bash
smithtune undeploy \
  --account-id "$account_id" \
  --deployment-id "$run_id" \
  --confirm
```

Use `smithtune <command> --help` for more options. See [Contributing](CONTRIBUTING.md)
for local development, tests, and releases.

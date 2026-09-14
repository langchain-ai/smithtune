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

For an existing local checkout, see [Contributing](CONTRIBUTING.md).

uv manages an isolated environment and can provision Python 3.12. No repository
checkout or environment activation is needed. It fetches smithtune and the pinned
Fireworks cookbook automatically. Both providers and their Python dependencies
are included. GitHub access and Git are needed during installation. Uninstall
with `uv tool uninstall smithtune`.

To upgrade, repeat the installation command with `--upgrade`. For a reproducible
release, append a tag or full commit SHA as `@<ref>` to the Git URL. A release tag
must exist before it can be installed. There is no PyPI publication step.

Install these companion tools for the operations you use:

| Tool | Required for |
| --- | --- |
| [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) | Dataset creation, contract capture, and fetching data during preparation |
| [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) | Fireworks deployment and undeployment |

Follow their official installation/authentication instructions and ensure their
commands are on `PATH`. `smithtune doctor` reports installation versions, tool
availability, and whether credential variables are set. It makes no network calls,
does not validate credentials, and never prints their values. Missing prerequisites
only affect operations that need them.

Run smithtune from a writable working directory of your choice. Data defaults to
`./data/`; use `--data-dir` to select a different location. The training runtime
includes PyTorch, so installation is substantial, but no local GPU is required.
Preparation can download model tokenizer files into the Hugging Face cache.
Public tokenizer repositories such as Qwen's can be downloaded without a Hugging
Face token. Gated or private repositories require an authorized `HF_TOKEN` or an
existing Hugging Face login. Only tokenizer assets are needed, not model weights.

Configure credentials in your environment:

| Task | Variable |
| --- | --- |
| Read LangSmith datasets and runs | `LANGSMITH_API_KEY` |
| Fireworks preparation, training, and inference | `FIREWORKS_API_KEY` |
| Baseten preparation and training | `BASETEN_API_KEY` |
| Replay judge | `ANTHROPIC_API_KEY` containing a **LangSmith gateway key**, or `ANTHROPIC_CUSTOM_HEADERS` |

## Using with a coding agent

Agents working in this checkout can use [AGENTS.md](AGENTS.md); Claude loads the
same guidance through `CLAUDE.md`.

Give your agent the following prompt, replacing the placeholders:

```text
Help me use smithtune for this task: <desired outcome and provider>.
My starting point is <tracing project, trajectory dataset, or prepared data>,
with these source IDs or paths: <workspace/project/dataset IDs or data directory>.

Read the operating guidance and workflow documentation:
https://github.com/langchain-ai/smithtune/blob/main/AGENTS.md
https://github.com/langchain-ai/smithtune/blob/main/README.md

Check setup with smithtune doctor and the relevant command's --help, then
start from the data I already have. Ask for any missing source information.
Use the documented workflow and keep paid operations within what I authorize.
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

The command filters trace root runs, deduplicates their threads, and imports one complete
conversation per dataset example, entirely server-side. It prints the dataset ID for `prepare`.
Recorded messages are preserved, including earlier turns and turns outside the filter window.

- `--filter` accepts [LangSmith API filter expressions](https://docs.langchain.com/langsmith/trace-query-syntax). The example selects root runs with correctness feedback of at least 0.9.
- The time window uses an inclusive start and exclusive end. Feedback, metadata, tag, and error filters apply to root runs.
- All matching threads are included by default. Use `--limit 100` to sample up to 100 threads; `--seed` defaults to 42.
- Roots without a thread ID are excluded and counted. If no threads match, no dataset is created.
- Selected IDs and an import receipt are saved automatically under `data/selections/`. Use `--output path.json` to choose the selection file location.

Existing dataset names are rejected. On failure, the receipt records confirmed imports and any
pending write; inspect the partial dataset before rerunning. Imports are not automatically retried.
This command requires current LangSmith run-query and thread-import APIs.

## Prepare data

Preparation automatically collects tools from **all LLM runs in each example's source
thread** (or source trace for older trace datasets). Each training row gets its own combined
list of tool names, descriptions, and argument schemas, including tools that were available
but never called. Tools introduced later are included from the start of that example.

The dataset needs its source thread/trace ID and project ID. Datasets created by this CLI
already include these. Native `source_session_id` and `source_trace_id` are also supported.
Schemas are saved with the raw export and prepared data; `--no-fetch` reuses that snapshot,
and replay uses the matching example's saved schemas.

Preparation fails if any scanned call lists a provider built-in (such as Anthropic tool search
or OpenAI web search), even if it was never called. Conflicting definitions for the same tool
name within an example also fail, with the example and source run IDs.

System messages come from each trajectory and are preserved during preparation and replay.
The default Qwen renderer supports a system message only as the first message.

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

Training generates a run ID such as `sft-20260914-213000-a1b2c3d4e5f6` and writes
artifacts to `./runs/<run-id>`. It prints the ID and output directory before
training starts and includes `run_id` and `run_dir` in the final JSON output.
Use `--run-id my-sft` to choose a name, `--run-dir ./my-output` to choose a folder,
or both. The output directory must be new or empty. A plan is a preview and does
not reserve a run ID or save settings for training; repeat any customized
training settings on both commands.

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

## Options and tests

Data defaults to `data/` in your current working directory; the examples above
save checkpoints and reports under `runs/`. Those directories are ignored by
Git in this repository. No artifacts are written into the installed package.
See command help for supported model selection, split fractions, and provider-specific training settings.

```bash
smithtune --help
smithtune dataset create --help
smithtune prepare --help
smithtune train --help
smithtune --version
```

See [Contributing](CONTRIBUTING.md) for development, tests, and releases.

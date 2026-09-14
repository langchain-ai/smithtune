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

Configure credentials in your environment:

| Task | Variable |
| --- | --- |
| Read LangSmith datasets and runs | `LANGSMITH_API_KEY` |
| Fireworks training and inference | `FIREWORKS_API_KEY` |
| Baseten training | `BASETEN_API_KEY` |
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

Choose a provider and prepare your dataset:

```bash
provider=fireworks # or baseten
run_id=my-sft

smithtune prepare \
  --provider "$provider" \
  --workspace-id '<workspace-id>' \
  --dataset-id '<dataset-id>' \
  --model-profile qwen3p8-27b
```

For an existing global contract, `--inference-contract path/to/contract.json` explicitly
uses its schemas for every example and skips automatic capture. Legacy contract files with
system-prompt metadata still work; that prompt is not injected or compared. `capture-contract`
remains available to create a global contract from a sample thread.

- Default split: 80% training, 10% validation, 10% replay test, grouped by source thread or standalone trace.
- SFT trains on all supported assistant messages, including earlier turns.
- Reasoning is omitted by default. Use `--reasoning-policy preserve` with a supported model and renderer to retain it.
- Unsupported content fails validation; examples over the preparation context limit are rejected without truncation.

## Plan and train

Review the plan before running `train`. Training is billed by the provider and requires `--confirm`.

```bash
smithtune plan --provider "$provider" --run-id "$run_id"
```

```bash
smithtune train \
  --provider "$provider" \
  --run-id "$run_id" \
  --run-dir "runs/$run_id" \
  --confirm
```

The best checkpoint is selected by validation loss and recorded in `runs/$run_id/result.json`.
Training artifacts also include `plan.json`, `run-state.json`, and `epochs.json` in that directory.
Use `--init-from-checkpoint '<checkpoint-uri>'` to initialize a new training run from a saved checkpoint.
Baseten's optional spend guard requires both `--max-spend-usd` and `--hourly-rate-usd`.

## Deploy and evaluate (Fireworks)

Promote the selected checkpoint, then deploy it. The endpoint incurs charges until removed.

```bash
account_id='<fireworks-account-id>'

smithtune promote \
  --run-dir "runs/$run_id" \
  --output-model-id "$run_id" \
  --confirm

smithtune deploy \
  --run-dir "runs/$run_id" \
  --account-id "$account_id" \
  --output-model-id "$run_id" \
  --deployment-id "$run_id" \
  --deployment-shape '<compatible-fireworks-shape>' \
  --confirm
```

Review the replay cases, then evaluate with the gateway credentials above:

```bash
smithtune eval-plan --output-dir "runs/$run_id/replay"
```

```bash
smithtune evaluate \
  --output-dir "runs/$run_id/replay" \
  --tuned-model "accounts/$account_id/models/$run_id#accounts/$account_id/deployments/$run_id" \
  --confirm
```

Results are saved to `runs/$run_id/replay/summary.json`. Replay scores agreement with recorded actions without executing tools.
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
See command help for model profiles, custom models, split fractions, and provider-specific training settings.

```bash
smithtune --help
smithtune dataset create --help
smithtune prepare --help
smithtune train --help
smithtune --version
```

See [Contributing](CONTRIBUTING.md) for development, tests, and releases.

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
| [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) | Trace selection, dataset creation, contract capture, and fetching data during preparation |
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

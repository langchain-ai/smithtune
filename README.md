<h1 align="center">smithtune</h1>

Fine-tune models on LangSmith trajectories with Fireworks or Baseten.
Fireworks supports training, deployment, and replay evaluation; Baseten supports SFT checkpoints.

## Setup

Requires Python 3.12, `git`, `uv`, `sfw`, the [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli), and `firectl`.

```bash
git clone https://github.com/langchain-ai/smithtune.git
cd smithtune
./bootstrap.sh
source .venv/bin/activate
```

Configure credentials in your environment:

| Task | Variable |
| --- | --- |
| Read LangSmith datasets and runs | `LANGSMITH_API_KEY` |
| Fireworks training and inference | `FIREWORKS_API_KEY` |
| Baseten training | `BASETEN_API_KEY` |
| Replay judge | `ANTHROPIC_API_KEY` containing a **LangSmith gateway key**, or `ANTHROPIC_CUSTOM_HEADERS` |

## Create a dataset from conversations

Select whole conversations (threads) by filtering trace root runs in a project:

```bash
python pipeline.py dataset select \
  --workspace-id '<workspace-id>' --project-id '<project-id>' \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))' \
  --limit 100 --seed 42 \
  --output data/selection.json

python pipeline.py dataset create \
  --selection data/selection.json --name my-sft-dataset
```

`select` applies LangSmith filters to root runs within the time window (start inclusive, end exclusive),
previews matches, and saves their thread IDs. A matching root selects its entire thread, including
earlier turns and turns outside the window. Roots without a thread ID are excluded and counted
in the preview. Threads are deduplicated before sampling; `--limit` counts threads, and omitting
it keeps all matching threads.

`create` imports one full conversation per example into a new dataset, entirely server-side,
and prints its ID for `prepare` below. Messages, including recorded system prompts, are preserved.
The selection file fixes the thread IDs; source content can still change before import.
Existing thread selection files still work. The `--scope` flag has been removed; old trace
selections must be regenerated with `dataset select`.

An existing selection, receipt, or dataset name is rejected. Imports stop on the first error;
`data/selection.import.json` records confirmed examples and any pending write. A timeout can leave
the last write's outcome unknown. Inspect the partial dataset before starting a new attempt with
a new selection path and dataset name. Automatic retry/resume and appending are not supported.
These commands require current LangSmith run-query and thread-import APIs.

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

python pipeline.py prepare \
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
python pipeline.py plan --provider "$provider" --run-id "$run_id"
```

```bash
python pipeline.py train \
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

python pipeline.py promote \
  --run-dir "runs/$run_id" \
  --output-model-id "$run_id" \
  --confirm

python pipeline.py deploy \
  --run-dir "runs/$run_id" \
  --account-id "$account_id" \
  --output-model-id "$run_id" \
  --deployment-id "$run_id" \
  --deployment-shape '<compatible-fireworks-shape>' \
  --confirm
```

Review the replay cases, then evaluate with the gateway credentials above:

```bash
python pipeline.py eval-plan --output-dir "runs/$run_id/replay"
```

```bash
python pipeline.py evaluate \
  --output-dir "runs/$run_id/replay" \
  --tuned-model "accounts/$account_id/models/$run_id#accounts/$account_id/deployments/$run_id" \
  --confirm
```

Results are saved to `runs/$run_id/replay/summary.json`. Replay scores agreement with recorded actions without executing tools.
Add `--base-model '<deployed-base-model-route>'` for a before/after comparison. Reuse the output directory to resume an interrupted evaluation.

Remove the endpoint when finished to stop deployment billing:

```bash
python pipeline.py undeploy \
  --account-id "$account_id" \
  --deployment-id "$run_id" \
  --confirm
```

## Options and tests

Data is saved under `data/`; checkpoints and reports under `runs/`. Both directories are ignored by Git.
See command help for model profiles, custom models, split fractions, and provider-specific training settings.

```bash
python pipeline.py --help
python pipeline.py dataset select --help
python pipeline.py dataset create --help
python pipeline.py prepare --help
python pipeline.py train --help
python -m pytest
```

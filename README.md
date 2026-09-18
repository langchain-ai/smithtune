<h1 align="center">smithtune</h1>

Fine-tune models on LangSmith trajectories with Fireworks or Baseten

```text
Tracing project → dataset → prepare → plan → train + evaluate → LangSmith comparison
```

| What you have | Start here |
| --- | --- |
| Trajectories in a tracing project | [Create a dataset](#create-a-dataset-from-trajectories) |
| A LangSmith trajectory dataset | [Prepare data](#prepare-data) |
| Prepared smithtune data | [Plan and train](#plan-and-train) |
| A completed smithtune training run | [Evaluate a trained model](#evaluate-a-trained-model) |

## Setup

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.

```bash
uv tool install --python 3.12 \
  --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/main/overrides.txt \
  'git+https://github.com/langchain-ai/smithtune.git'
```

To upgrade, repeat the command with `--upgrade`.

The override selects the patched Transformers version tested by smithtune while
the upstream Fireworks and Tinker cookbook metadata still pins an affected
release. Keep the override URL and smithtune Git ref aligned when installing a
release tag or commit.

Install the [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) for
fetching traces and datasets:

```bash
curl -fsSL https://cli.langsmith.com/install.sh | sh
```

Start a new shell or refresh your PATH, then follow the LangSmith CLI's
installation and authentication instructions. Training and sampler evaluation
use the provider APIs directly. `firectl` is only needed for
[Fireworks deployment and undeployment](docs/deployment.md#deploy-a-fireworks-checkpoint).

### Credentials

For the workflow below, configure **a LangSmith key, your chosen training
provider's key, and a judge key** in your environment:

| Variable | Needed for |
| --- | --- |
| `LANGSMITH_API_KEY` | Reading the dataset and publishing splits and evaluation results in LangSmith |
| `FIREWORKS_API_KEY` | Preparing, training, and sampling with Fireworks |
| `BASETEN_API_KEY` | Preparing, training, and sampling with Baseten |
| `ANTHROPIC_API_KEY` | The default replay judge, Claude Sonnet 5 |

You need the key for the training provider you choose. The default judge calls
Anthropic. [Other judge routes](docs/reference.md#replay-options) include
Fireworks and the internal LangSmith gateway (`LANGSMITH_GATEWAY_API_KEY`).
The default [dataset curation council](docs/datasets.md#label-full-trajectories-with-an-agent-council)
uses Fireworks and OpenAI, requiring both `FIREWORKS_API_KEY` and `OPENAI_API_KEY`
and the optional Deep Agents extra. Use `create --no-triage` to skip this review.

Check local setup:

```bash
langsmith --help
smithtune doctor
smithtune --help
```

`doctor` checks installed tools and whether credentials are set; it does not
validate service access. Only the prerequisites for your chosen operations matter.

### Choose your provider and directories

Run the following commands from one writable directory, using the same variables
throughout. Replace the workspace ID with the workspace containing your dataset.

```bash
provider=fireworks # or baseten; keep the same provider for prepare, train, and evaluate
model=qwen3p8-27b
workspace_id='<workspace-id>'
data_dir='./data/my-sft'
run_dir='./runs/my-sft'
judge_model='anthropic/claude-sonnet-5'

smithtune models list --provider "$provider"
```

`qwen3p8-27b` is supported by both providers. Choose another model from the list
if preferred. Preparation selects its tokenizer and formatting automatically.
If you already have prepared data or a training run, set these variables to its
provider and existing directories instead.

## Create a dataset from trajectories

Skip this step if you already have a LangSmith dataset. Otherwise, select
trajectories from a tracing project in the workspace above. Replace the project
ID with your source, and choose a feedback filter your project uses:

```bash
project_id='<project-id>'

smithtune dataset create data/datasets/my-sft \
  --workspace-id "$workspace_id" --project-id "$project_id" \
  --name my-sft-dataset \
  --limit 100 \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'
```

Without time flags, creation selects roots from the last 24 hours: `--end-time`
defaults to now and `--start-time` defaults to 24 hours before the resolved end.
Pass either or both flags to choose another ISO 8601 window.

The filter matches feedback on trace root runs. Each matching root selects its
whole thread when it has one, otherwise its single trace. Each trajectory
becomes one dataset example, including earlier turns outside the time window.
`--limit 100` selects at most 100 distinct trajectories. Remove `--filter` to
select without a feedback threshold; see [filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax).

This first command downloads and previews the default council work. It makes
no judge calls or LangSmith writes. Review the candidates and pending votes,
then authorize judging and upload using the saved settings:

```bash
smithtune dataset create data/datasets/my-sft --confirm
```

Creation includes triage by default. Add `--no-triage` to the first command to
select only training-valid conversations without council review. The default
council needs the optional Deep Agents extra and Fireworks/OpenAI credentials;
see [dataset curation](docs/datasets.md) for setup, staged commands, and recovery.

Selection is a seeded sample of distinct conversations (default 100, maximum
2000, seed 42). Invalid messages, media, or missing producing-run/tool provenance
exclude whole conversations before paid judging. Exclusions are reported with
source identities and reasons. Zero eligible conversations create no dataset.
Model-specific rendering and context limits are checked by `prepare`.

Continue interrupted work with `smithtune dataset resume data/datasets/my-sft
--confirm`. Completed downloads and votes are reused, and uncertain uploads are
reconciled. Use a new checkpoint to select a different source or time window.

## Prepare data

Set `dataset_id` to the ID returned by creation, or to your existing dataset ID:

```bash
dataset_id='<dataset-id>'

smithtune prepare \
  --provider "$provider" --model "$model" \
  --workspace-id "$workspace_id" --dataset-id "$dataset_id" \
  --data-dir "$data_dir"
```

Preparation downloads whole conversations with their saved per-assistant tool
bindings and validates the training format. Exported bound examples need no
source-run reads; unbound examples require verifiable producing-run evidence. It creates approximately 80% training, 10% validation, and 10%
held-out test data, keeping each source trajectory in one split. These splits
are also registered on the original LangSmith dataset for evaluation. If local
preparation succeeds but publication does not, publish and verify only the saved
memberships without rerunning preparation:

```bash
smithtune dataset publish-splits --data-dir "$data_dir"
```

The main data requirements are:

- Text and tool trajectories; images are unsupported
- Recorded system messages are preserved; Qwen requires them at the start
- Every supported assistant message is trained once, using its own available tools
  and recorded prefix; earlier assistant context has zero loss

Whole malformed or incompatible trajectories are excluded unchanged. `prepare`
prints a warning and records each exclusion in `prepared/warnings.json` and
`prepared/rejected.json`, including a stable reason code and source identity.

Examples over the model's context limit are rejected without truncation.
Reasoning is omitted by default. See the [reference](docs/reference.md) for
`--validation-fraction`, `--test-fraction`, sticky `--split-from` assignments,
model selection, and reasoning options.

## Plan and train

**Check the evaluation size before starting paid work.** Replay evaluates the
next assistant action with its recorded tool definitions at multiple points in each test trajectory. For example,
20 trajectories with 30 eligible actions each produce 600 comparisons. Each
comparison generates and judges a base response and a tuned response, with
additional judge calibration calls.

For a smaller first evaluation, the commands below cap replay at **2 actions per
test trajectory**. This cap changes evaluation coverage, not training data.
Omit it from both commands to evaluate every eligible assistant action.

Preview the training and evaluation plan:

```bash
smithtune plan \
  --provider "$provider" --data-dir "$data_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2
```

Review the case and call counts, then start training:

```bash
smithtune train \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2 \
  --confirm
```

Training and evaluation incur provider charges. The run directory must be new or
empty. Repeat customized settings on both `plan` and `train`; the plan is a
preview and does not carry settings into the training command.

The CLI trains, selects the checkpoint with the lowest validation loss, and
compares it with the base model on the held-out test data. Both providers use
training API samplers, so this workflow needs no `promote` or `deploy` command.
Replay scores do not affect checkpoint selection.

## Review results in LangSmith

When replay begins, the CLI prints **one comparison link** for the base and tuned
experiments. Open it to view results on the original LangSmith dataset. Completed
comparisons publish in the background; refresh the view as evaluation progresses.

Each trajectory groups its independent next-action predictions:

- `teacher_agreement`: whether an action passed the judge, with an explanation
- `trajectory_teacher_agreement`: the trajectory's average over completed actions

Replay predicts the next response or tool call from recorded context. Generated
tool calls are **not executed**. Scores measure agreement with recorded behavior,
so they do not establish whether the agent would complete a task end to end.

The final JSON includes `langsmith.comparison_url`. Keep the local data and run
directories for recovery.

## Evaluate a trained model

Use standalone evaluation when training is already complete or when resuming an
interrupted replay. Set `provider`, `data_dir`, and `run_dir` to the original run.
The commands below match the first-run replay settings above:

```bash
smithtune eval-plan \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --max-points-per-trajectory 2

smithtune evaluate \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --judge-model "$judge_model" --max-points-per-trajectory 2 --confirm
```

`eval-plan` previews cases without model calls. `evaluate` loads the saved best
checkpoint and automatically compares it with the base model. Results default
to `"$run_dir/replay"`.

To resume, keep the original directories and settings, including any replay
cap. Completed predictions and judgments are reused. To change the judge, cap,
or sampling settings, add `--output-dir "$run_dir/replay-new"` to **both** commands
and choose a fresh directory. Training checkpoints survive evaluation failures.

Evaluation requires LangSmith publication to finish successfully. If uploads
fail, rerun standalone `evaluate` with the same settings. See
[recovery guidance](docs/reference.md#resume-an-interrupted-evaluation) for partial
uploads, saved artifacts, and sampler cleanup.

## Other workflows

| Task | Guide |
| --- | --- |
| Judge trajectories before training or extend a dataset | [Dataset curation and triage](docs/datasets.md) |
| Change models, preparation settings, or replay options | [Preparation and evaluation reference](docs/reference.md) |
| Run a trained model in an application or evaluate a Baseten endpoint | [Deployment](docs/deployment.md) |

## Using with a coding agent

Give your agent this prompt, replacing the placeholders:

```text
Help me <task> with smithtune using <provider>.
My data: <workspace/project/dataset IDs or prepared-data directory>.
Follow https://github.com/langchain-ai/smithtune/blob/main/AGENTS.md.
```

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

<h1 align="center">smithtune</h1>

Fine-tune models on [trajectories](https://docs.langchain.com/langsmith/observability-concepts#trajectories)
recorded in LangSmith. Train with Fireworks or Baseten, compare the base and tuned models
in LangSmith, then optionally deploy an endpoint for your application.

smithtune is an early beta project. If you run into issues, please report them
in the [repository](https://github.com/langchain-ai/smithtune/issues).

```text
dataset pull → dataset triage → dataset push
prepare → plan → train --evaluate → deploy (optional)
```

| What you have | Start here |
| --- | --- |
| Trajectories in a tracing project | [Create a dataset](#create-a-dataset-from-trajectories) |
| A LangSmith trajectory dataset | [Prepare data](#prepare-data) |
| Prepared smithtune data | [Plan and train](#plan-and-train) |
| A completed training run | [Evaluate](#evaluate-a-trained-model) or [deploy](#deploy-a-trained-model) |

## Setup

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git:

```bash
uv tool install --python 3.12 \
  --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/main/overrides.txt \
  'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git'
```

This includes using an **Agent Council** for dataset selection. A council is a group of
models that votes on which conversations to keep. Add `--upgrade` to update.
The override is in place to select a more up to date version of the HuggingFace Transformer's package. Keep its Git ref aligned with the package when pinning a release or commit.

Install the [LangSmith CLI](https://github.com/langchain-ai/langsmith-cli) for reading traces and datasets:

```bash
curl -fsSL https://cli.langsmith.com/install.sh | sh
```

## Using with a coding agent

The [smithtune skill](src/smithtune/skills/smithtune/SKILL.md) walks a coding
agent through the whole flow. Give your agent this prompt, replacing the placeholders:

```text
Help me <task> with smithtune using <provider>.
My data: <workspace/project/dataset IDs or prepared-data directory>.
Follow https://github.com/langchain-ai/smithtune/blob/main/src/smithtune/skills/smithtune/SKILL.md.
```

### Credentials and first-use setup

Set these environment variables in the shell where you run smithtune:

| Variable | Needed for |
| --- | --- |
| `LANGSMITH_API_KEY` | Dataset access, split publication, and LangSmith experiments |
| `FIREWORKS_API_KEY` **or** `BASETEN_API_KEY` | Your chosen provider's preparation, training, and evaluation |
| `BASETEN_API_KEY` | The default replay judge, GLM-5.3-Flash on Baseten Model APIs (also the Baseten training key) |

The LangSmith CLI uses `LANGSMITH_API_KEY` for authentication.
Agent Council review with the default models requires only `BASETEN_API_KEY`. The model choices are configurable.

See [other replay judges](docs/reference.md#replay-options) to change the judge.

Read [Data Rights and Permitted Use](docs/data-rights-and-permitted-use.md).
The first workflow requires an interactive acknowledgment, saved locally;
`--confirm` does not replace it. Before running scripts, acknowledge and check setup:

```bash
smithtune acknowledge-data-rights
smithtune doctor
```

`doctor` checks local prerequisites, not service access. Use `smithtune --help`
or `smithtune <command> --help` for command options.

### Confirm paid work

Paid model calls and GPU capacity require `--confirm` on the command to run them.

| Command | Without `--confirm` |
| --- | --- |
| `dataset triage` | Previews council calls |
| `dataset resume` | Shows pending work |
| `train`, `evaluate`, `deploy` | Stops before paid work |
| `dataset push` | Previews the upload |
| `undeploy` | Stops before changing provider resources |

Use `plan` before training and `eval-plan` before evaluation. Neither starts paid
compute. Commands without confirmation can still read remote data or write local
files. `prepare` and `dataset publish-splits` also write LangSmith split metadata;
they do not train or call models. Running endpoints keep incurring charges until stopped.

### Choose your provider and directories

Use the same provider and paths throughout. For an existing run, use its original
provider and directories. Set the workspace to the one containing your dataset:

```bash
provider=fireworks # or baseten
model=qwen3p8-27b
workspace_id='<workspace-id>'
data_dir='./data/my-sft'
run_dir='./runs/my-sft'
judge_model='baseten/zai-org/GLM-5.3-Flash'

smithtune models list --provider "$provider"
```

`qwen3p8-27b` works with both providers. Choose another supported model from the list;
preparation selects its tokenizer and formatting.

## Create a dataset from trajectories

Skip this step if you already have a LangSmith trajectory dataset. Otherwise,
pull trajectories from a tracing project and push them to a LangSmith dataset.
Reviewing them with an agent council in between is optional.

### 1. Write and test a filter

The filter decides what the model learns from, and `pull` saves it for the
directory. Filters use [LangSmith filter syntax](https://docs.langchain.com/langsmith/trace-query-syntax)
and match **root runs**; each match brings in its whole thread. Look at your
project's real root names, tags, metadata, and feedback keys, then test the
filter with the LangSmith CLI over the window you will pull. It uses the same
syntax and costs nothing:

```bash
project_id='<project-id>'
filter='and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))'

langsmith trace list --workspace "$workspace_id" --project-id "$project_id" \
  --filter "$filter" --since 2026-09-01T00:00:00Z --before 2026-09-22T00:00:00Z \
  --limit 50 --full --format json
```

### 2. Pull, then push

When the filter already selects on a trusted quality signal, such as validated
feedback scores or human labels, skip council review with `--no-triage`:

```bash
smithtune dataset pull data/datasets/my-sft \
  --workspace-id "$workspace_id" --project-id "$project_id" \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-22T00:00:00Z \
  --filter "$filter" --target-count 100 --no-triage

smithtune dataset push data/datasets/my-sft --name my-sft-dataset   # preview
smithtune dataset push data/datasets/my-sft --confirm               # upload
```

- `pull` downloads without model calls. Inspect its summary for usable trajectories and exclusion reasons.
- `--target-count` is how many trajectories you want (default 100); `--max-candidates` caps new candidates per pull (default 1,000, maximum 2,000).
- Always set `--start-time` and `--end-time`; the default window is the last 24 hours.
- `push` previews the upload; `--confirm` uploads and returns the dataset ID for `prepare`.

Selecting an agent by name or filtering out errors alone does not establish
training quality. Without a trusted signal, use council review.

### Optional: review with an agent council

Omit `--no-triage` from the first `pull`, then run `triage` before `push`. A
council of models judges each whole trajectory against your rubric and keeps it
on a strict majority. You need:

- the `deepagents` extra (included in the install above);
- keys for the judges: `BASETEN_API_KEY` for the default council (DeepSeek
  V4.1 Flash and GLM-5.3-Flash on Baseten), or pick others with `--judges`;
- a `rubric.md` describing the task, what to keep, what to drop, and a few
  concrete examples of each. smithtune ships no default rubric; `triage`
  requires `--rubric` or `--rule`. The
  [smithtune skill](src/smithtune/skills/smithtune/SKILL.md) has a template. Write it after reading a varied sample of the
  pulled trajectories.

```bash
smithtune dataset triage data/datasets/my-sft --rubric ./rubric.md   # preview, no model calls
smithtune dataset triage data/datasets/my-sft --confirm              # run the council
```

Decisions are saved in `labels.jsonl` and summarized in `report.md`; read them
before pushing. Review stops at the target. If the pool runs out first, run
`pull` and `triage --confirm` again in the same directory to review unseen
candidates (up to three rounds). Council review helps assess quality; it does
not guarantee good training data.

Keep the same directory throughout. The review mode, filter, window, and limits
are fixed once you pull. Use `smithtune dataset resume data/datasets/my-sft` to
inspect pending work and add `--confirm` to continue it. See
[dataset curation](docs/datasets.md) for selection and recovery details.

## Prepare data

Use the dataset ID returned by `push`, or your existing dataset ID:
```bash
dataset_id='<dataset-id>'

smithtune prepare \
  --provider "$provider" --model "$model" \
  --workspace-id "$workspace_id" --dataset-id "$dataset_id" \
  --data-dir "$data_dir"
```

Preparation checks each conversation and the tools available at each assistant turn.
It assigns about 80% to training, 10% to validation, and 10% to testing. Each source trajectory
stays in one split; the memberships are also published to the LangSmith dataset.

Recorded system messages are preserved; reasoning is omitted by default. Unsupported
or overlong trajectories are excluded without truncation and listed in
`"$data_dir/prepared/rejected.json"`. Each supported assistant answer becomes one
training target, with its preceding context and the tools available at that call.
See the [preparation reference](docs/reference.md#prepare-data) for data requirements,
reasoning options, and split recovery.

## Plan and train

Preview the work, then run training and evaluation:
```bash
smithtune plan \
  --provider "$provider" --data-dir "$data_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2

smithtune train \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --evaluate --judge-model "$judge_model" --max-points-per-trajectory 2 \
  --confirm
```

**Training and evaluation incur provider charges.** Review the preview's case and
call counts. The example caps evaluation at two assistant actions per test trajectory;
omit the cap from both commands to evaluate every eligible action. Each comparison
generates and judges a base response and a tuned response, plus judge calibration calls.

Use a new or empty run directory. Repeat custom settings on both commands;
`plan` is a preview and does not save settings for `train`.

The CLI selects the saved model checkpoint with the lowest validation loss and
compares it with the base model on held-out test data. Evaluation uses provider
samplers to generate responses. Fireworks reuses its training session; Baseten starts temporary paid
samplers and deactivates them on exit. You do not need to run `deploy` for this step.
Omit `--evaluate` and its replay options to train only.

## Review results in LangSmith

The CLI prints **one comparison link** for the base and tuned experiments when
evaluation starts. Results publish in the background; each trajectory's row appears
when all its selected comparisons finish. The final JSON includes `langsmith.comparison_url`.

- `teacher_agreement`: whether an assistant action passed the judge, with an explanation.
- `trajectory_teacher_agreement`: the trajectory's average score.

Replay predicts the next response or tool call from recorded context; generated
tool calls are **not executed**. Scores measure agreement with recorded behavior,
not end-to-end task completion, and do not affect checkpoint selection.

## Evaluate a trained model

Use standalone evaluation after training or to resume interrupted evaluation.
Keep the original provider, directories, and replay settings:

```bash
smithtune eval-plan \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --max-points-per-trajectory 2

smithtune evaluate \
  --provider "$provider" --data-dir "$data_dir" --run-dir "$run_dir" \
  --judge-model "$judge_model" --max-points-per-trajectory 2 --confirm
```

`eval-plan` previews without model calls. `evaluate` compares the saved best checkpoint
with the base model and saves results in `"$run_dir/replay"`. Rerunning reuses completed
predictions and judgments and retries LangSmith publication. Keep your local artifacts.

To change the judge, replay cap, or sampling settings, use a fresh
`--output-dir` on both commands. See [evaluation options and recovery](docs/reference.md#replay-options).

## Deploy a trained model

Deploy when you want an endpoint for your application. Both providers
use `deploy`; Fireworks handles promotion automatically and reuses saved promotions.
Run the command for your training provider:

**Fireworks**: install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl)
and use the account owning your checkpoint:

```bash
smithtune deploy --provider fireworks --run-dir "$run_dir" \
  --account-id '<fireworks-account>' --output-model-id my-tuned-model \
  --deployment-id my-endpoint --deployment-shape '<compatible-deployment-shape>' \
  --confirm
```

**Baseten**: install the [deployment extra](docs/deployment.md#deploy-a-baseten-checkpoint)
first. Choose hardware and a context cap suitable for your model; these are example values:

```bash
smithtune deploy --provider baseten --run-dir "$run_dir" \
  --accelerator H200:1 --max-seq-len 32768 --confirm
```

Endpoints stay running and can incur charges. Stop serving when finished:

```bash
# Fireworks
smithtune undeploy --provider fireworks \
  --account-id '<fireworks-account>' --deployment-id my-endpoint --confirm

# Baseten
smithtune undeploy --provider baseten --run-dir "$run_dir" --confirm
```

See the [deployment guide](docs/deployment.md) for setup, recovery, and endpoint evaluation.

## More guides

| Task | Guide |
| --- | --- |
| Filter, judge, resume, or extend a dataset | [Dataset curation](docs/datasets.md) |
| Configure models, splits, reasoning, or evaluation | [Preparation and evaluation reference](docs/reference.md) |
| Deploy and manage endpoints | [Deployment](docs/deployment.md) |
| Contribute to smithtune | [Development setup](CONTRIBUTING.md) |

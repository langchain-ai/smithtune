# Deploy a trained checkpoint

[Back to the workflow](../README.md)

Deploy when you need an endpoint for your application. The standard
[base-versus-tuned evaluation](../README.md#evaluate-a-trained-model) uses training
API samplers and handles their lifecycle without a `deploy` command.

## Deploy a Baseten checkpoint

Use a Baseten training run and its prepared data. Set these paths to the
artifacts from your run; each evaluation mode uses a separate output directory:

```bash
run_dir='./runs/my-sft'
data_dir='./data/my-sft'
```

Add the deployment tools, keeping the council support from the README installation:

```bash
uv tool install --upgrade --python 3.12 \
  --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/main/overrides.txt \
  'smithtune[deepagents,baseten-deploy] @ git+https://github.com/langchain-ai/smithtune.git'
```

Set `BASETEN_API_KEY`. Evaluation also requires `LANGSMITH_API_KEY` for dataset
access and experiment publication, plus `ANTHROPIC_API_KEY` for the default judge.
Supported Baseten models are public and do not require a Hugging Face token.
Choose GPUs explicitly: `H200:1`
below is an example, not a verified allocation for every model.

Preview cases and deployment settings, then run a temporary evaluation:

```bash
smithtune eval-plan --provider baseten --serving-mode temporary \
  --run-dir "$run_dir" --data-dir "$data_dir" --output-dir "$run_dir/replay-temporary" \
  --accelerator H200:1 --max-seq-len 32768
smithtune evaluate --provider baseten --serving-mode temporary \
  --run-dir "$run_dir" --data-dir "$data_dir" --output-dir "$run_dir/replay-temporary" \
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
  --data-dir "$data_dir" --output-dir "$run_dir/replay-endpoint" --confirm
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
evaluate its dedicated chat endpoint using `BASETEN_API_KEY`, `LANGSMITH_API_KEY`,
and, for the default judge, `ANTHROPIC_API_KEY`:

```bash
smithtune evaluate \
  --provider baseten \
  --data-dir "$data_dir" \
  --output-dir ./runs/existing-endpoint-replay \
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
route available on the **same endpoint**. Open the printed LangSmith comparison link to review results; rerun the same
command to resume. No Fireworks key is needed with an Anthropic judge.
This path uses an existing deployment and leaves it running; manage externally
created deployments in Baseten. Training support alone does not verify a model's
serving configuration.

## Deploy a Fireworks checkpoint

Install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) and set
`FIREWORKS_API_KEY`. Use a Fireworks training run.

Use `deploy` when you want an endpoint for repeated use. It automatically promotes
the selected checkpoint to a named Fireworks model, then starts the endpoint;
it does not run the evaluation. The endpoint stays available and can incur
charges until you run `undeploy`:

```text
deploy -> use endpoint -> undeploy
```

```bash
run_dir='runs/my-sft'
account_id='<your-fireworks-account>'
run_id='my-sft'
deployment_id='my-sft'
deployment_shape='<compatible-deployment-shape>'

smithtune deploy --provider fireworks \
  --run-dir "$run_dir" --account-id "$account_id" \
  --output-model-id "$run_id" --deployment-id "$deployment_id" \
  --deployment-shape "$deployment_shape" --confirm
```

`--account-id` must be the account that owns the training checkpoint. A matching
saved promotion is reused. If deployment creation fails after promotion succeeds, retrying does not promote
again. An existing deployment or a promotion with an uncertain outcome still
needs inspection in Fireworks before retrying.

For replay, use [`evaluate --run-dir`](../README.md#evaluate-a-trained-model). The serverless sampler
uses the saved training checkpoint independently of this production endpoint.

```bash
smithtune undeploy --account-id "$account_id" --deployment-id "$deployment_id" --confirm
```

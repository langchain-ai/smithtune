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
uv tool install --force --python 3.12 \
  --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/v0.1.0/overrides.txt \
  'smithtune[deepagents,baseten-deploy] @ git+https://github.com/langchain-ai/smithtune.git@v0.1.0'
```

Set `BASETEN_API_KEY`; the default judge also uses it. Evaluation also requires
`LANGSMITH_API_KEY` for dataset access and experiment publication.
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
evaluate its dedicated chat endpoint using `BASETEN_API_KEY` (also used by the
default judge) and `LANGSMITH_API_KEY`:

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
command to resume. No Fireworks key is needed with the default Baseten judge or an Anthropic judge.
This path uses an existing deployment and leaves it running; manage externally
created deployments in Baseten. Training support alone does not verify a model's
serving configuration.

## Deploy a Fireworks checkpoint

Install [firectl](https://docs.fireworks.ai/tools-sdks/firectl/firectl) **1.8.5 or
newer** (`firectl version`; upgrade with `firectl upgrade`) and set
`FIREWORKS_API_KEY`. Use a Fireworks training run. `smithtune doctor` reports the
installed firectl version and whether automatic shape selection is available.

Use `deploy` when you want an endpoint for repeated use. It automatically promotes
the selected checkpoint to a named Fireworks model, then starts the endpoint;
it does not run the evaluation. The endpoint stays available and can incur
charges until you run `undeploy`:

```text
deploy -> use endpoint -> undeploy
```

```bash
run_dir='./runs/my-sft'
account_id='<your-fireworks-account>'
output_model_id='my-sft'
deployment_id='my-sft'

# Preview: shows the model, whether it is promoted yet, and the deployment shape.
smithtune deploy --provider fireworks \
  --run-dir "$run_dir" --account-id "$account_id" \
  --output-model-id "$output_model_id" --deployment-id "$deployment_id"

smithtune deploy --provider fireworks \
  --run-dir "$run_dir" --account-id "$account_id" \
  --output-model-id "$output_model_id" --deployment-id "$deployment_id" --confirm
```

**Deployment shape.** A shape fixes the hardware and serving configuration. By
default, `deploy` promotes the checkpoint first (the model must exist before
Fireworks can match shapes for it), then runs
`firectl deployment-shape-version match --model accounts/<account>/models/<output-model-id>`
and uses the first validated shape it returns. That list is already restricted to
shapes your account can deploy and excludes Multi-LoRA-only shapes, so it fits a
live-merge deployment. The chosen shape and the alternatives are saved in
`"$run_dir/endpoint.json"`. Once the model is promoted, the preview shows the
shape it will use. To choose yourself, pass `--deployment-shape <shape>`, or
`--deployment-shape default` to let Fireworks pick (firectl 1.8.8+).

**Coding agents.** firectl refuses to create or delete deployments when it runs
inside an AI agent (Claude Code, Cursor, and others). From an agent, `deploy
--confirm` promotes the checkpoint, matches the shape, and then stops with the
exact `smithtune deploy ... --deployment-shape <shape> --confirm` command to run
in your own terminal; `undeploy` does the same. The printed command pins the
matched shape, so it also works with firectl older than 1.8.5.

**Readiness.** A deployment can report `READY` while still waiting for capacity.
`deploy` waits until `replica_stats.ready_replica_count > 0` (up to
`--deployment-timeout`, default 1800 seconds) before its smoke test. If no replica
becomes ready in time, it stops and prints the `undeploy` command.

`--account-id` must be the account that owns the training checkpoint. A matching
saved promotion is reused. If deployment creation fails after promotion
succeeds, retrying does not promote again. An existing deployment or a promotion with an uncertain outcome still
needs inspection in Fireworks before retrying.

For replay, use [`evaluate --run-dir`](../README.md#evaluate-a-trained-model). The serverless sampler
uses the saved training checkpoint independently of this production endpoint.

```bash
smithtune undeploy --provider fireworks --account-id "$account_id" --deployment-id "$deployment_id" --confirm
```

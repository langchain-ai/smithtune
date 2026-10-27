# Experimental SageMaker custom training

This prototype adds customer-account SageMaker Training Jobs to SmithTune. It
currently targets **Qwen3.5-9B, text/tool trajectories, LoRA, one GPU**. No
long-context or production configuration has been validated on AWS; see the
status below. The model registry's context ceiling is not a tested GPU memory
guarantee.

Preparation reuses SmithTune's existing Baseten-compatible tokenizer, native
formatter, reasoning policy, assistant masks and conversation splits. The worker
runs Transformers + PyTorch + Hugging Face PEFT inside a container instead of calling a remote
forward/backward API. An optional TRL `SFTTrainer` backend uses the same rendered
tokens, explicit labels and loss objective. It retains tool-result tokens as context without applying
loss to them. It never truncates examples or concatenates separate conversations.

## Status

Validated so far: short synthetic training runs on one L40S and one A10G (the
TRL backend), adapter reload with generation on one L4, all with a longest
example of 437 tokens, and the full submit/train/export/collect lifecycle on a
CPU instance with the `qwen3p5-0p8b-test` profile (24 synthetic threads, two
epochs, about 15 billable minutes on `ml.m5.2xlarge`). Not validated: longer contexts, hosted endpoints,
cross-job resume, Spot recovery, distributed training, other models, and
LangSmith replay. `train --evaluate`, `deploy` and `evaluate` reject this
provider until replay is wired to its artifacts. Dated run notes are kept
outside the repository.

## Validation workers

Reloading means starting a fresh process, loading the pinned base weights plus
the exported LoRA adapter, and checking that the resulting model can generate.
The experimental `providers/sagemaker/smoke.py` worker verifies exact adapter
tensors, finite probe logits and an adapter effect, then generates base/tuned
responses to three synthetic tool prompts. It reuses the native formatter and
Baseten completion parser. There are no optimizer updates, executed tools,
paid judges, LangSmith publication or endpoint. A short-lived Training Job runs
this inference-only worker; the normal training image still runs SFT.

The smoke stages a checksummed export and its original preparation bundle.
Generation is greedy, capped at 96 new tokens per prompt. Length-limited or
malformed output is retained as a failed format check; successful model loading
does not imply correct tool behavior or an improvement over the base model.
This worker is an experimental validation utility, not a supported deployment
or replay command.

For context sizing, generate full synthetic tool histories and measure their
actual rendered lengths before paid work. Start at 4K, then 8K, with one
optimizer step over all native datums of each full conversation, using the same
BF16, rank-8 LoRA, gradient checkpointing and assistant masks. Attempt 16K only with sufficient
time and GPU-memory headroom; keep 32K and larger tests for a later stage.
Record finite loss, step time, peak allocated/reserved memory and active token
counts. These are capacity probes; task quality needs representative data.

The experimental `providers/sagemaker/benchmark.py` entrypoint implements these
bounded probes. It checksums the original preparation bundle and synthetic
cases, re-renders without truncation, and verifies measured token/target counts.
Each stage resets the rank-8 adapters, random seed and AdamW state, then runs one
optimizer group over every rendered datum of the full conversation. It uses
the unchanged training worker's full-vocabulary loss, token normalization and
gradient clipping. It exports measurements rather than a trained checkpoint.
An OOM or non-finite step records the failure and skips larger cases; a skipped
16K stage is not evidence of an actual 16K failure. The first stage includes
cold CUDA/kernel overhead, so timings are initial capacity measurements rather
than steady-state throughput. There is no public benchmark CLI command yet.

Use one L40S job for the initial probes, a 600-second runtime limit, and stop on
OOM or non-finite loss. Full-vocabulary logits/loss and activations can dominate
memory at longer contexts. Larger L40S instance sizes add host RAM, not GPU
memory. Benchmarking may expose a need for worker optimization or different
hardware before reaching agent-scale contexts.

For future training runs, consider an optional reload check before releasing
the training instance. A fresh model/process can reload the saved adapter on
that same allocation, avoiding a second capacity wait and model download.

## Prerequisites

- Python 3.12 and the optional `aws` extra on the operator's machine. In a
  checkout: `sfw uv sync --locked --extra test --extra aws --python 3.12`.
- A customer AWS identity from the standard SDK credential chain (environment,
  profile/SSO or workload role), region, and execution role in that same account.
  The role must trust `sagemaker.amazonaws.com`.
- Operator permissions for SageMaker create/describe/stop, passing that role,
  and listing/writing the chosen S3 prefix. The execution role needs input/output
  S3, ECR pull and CloudWatch access, plus applicable KMS permissions.
- For hosting: operator permissions to create/describe/delete models, endpoint
  configurations and endpoints, list their tags, and invoke endpoints. Hosting
  also checks the image with ECR `DescribeImages`. GPU quota and capacity are
  separate from training quota/capacity.
- A customer S3 bucket, a versioned ECR training image, GPU job quota and actual
  regional capacity. This prototype does not create IAM roles, buckets or ECR
  repositories. Studio is not required.
- Public Hugging Face access from the container to download the pinned weights
  and tokenizer. VPC settings do not make that access automatic; isolated
  training/model staging is a later extension. No operator API keys are copied
  into the job or image.

## Prepare and preview

Use the normal dataset creation/curation workflow from the [README](../README.md).
For AWS, use a separate data directory and an explicit initial context length:

```bash
smithtune prepare --provider sagemaker --model qwen3p5-9b \
  --workspace-id '<workspace-id>' --dataset-id '<dataset-id>' \
  --data-dir ./data/aws-qwen --max-seq-len 4096
```

The source dataset needs both training and validation conversations. Preparation
publishes the same LangSmith splits as the existing providers. Whole examples
over the selected context are excluded and reported, never shortened. Reusing
an existing raw export is supported with `--no-fetch`; do not relabel an old
provider's prepared manifest as SageMaker.

Create a local `aws-job.json` with real customer resource references:

```json
{
  "region": "us-west-2",
  "profile": "default",
  "role_arn": "arn:aws:iam::<account-id>:role/<execution-role>",
  "training_image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/<repository>@sha256:<image-digest>",
  "s3_prefix": "s3://<customer-bucket>/smithtune",
  "instance_type": "ml.g6e.4xlarge",
  "instance_count": 1,
  "volume_size_gb": 100,
  "max_runtime_seconds": 600
}
```

### Pipeline tests on CPU instances

The `qwen3p5-0p8b-test` profile uses the pinned Qwen3.5-9B tokenizer, chat
template and renderer with the much smaller `Qwen/Qwen3.5-0.8B` weights, so the
whole job lifecycle (staging, submission, training, export, collection) can be
exercised on a CPU instance for cents. Prepare with
`--model qwen3p5-0p8b-test` and set `instance_type` to an `ml.m*`, `ml.c*` or
`ml.r*` type (for example `ml.m5.2xlarge`); the planner then runs the worker on
`cpu`. CPU types are rejected for every other profile, instance preferences
must be all-CPU or all-GPU, and nothing measured on CPU transfers to GPU runs.
`ml.t*` burstable types are not accepted.

The instance above passed the short synthetic smoke; it is **not a long-context
hardware recommendation**. Choose hardware and a runtime budget before paid
execution. The worker uses GPU 0 only, even if the selected instance has more
GPUs. Extra instances are rejected.
Optional fields: `hourly_rate_usd` (per-instance rate for an estimate), `subnets`
and `security_group_ids` together, `output_kms_key` (also used for input uploads),
and `volume_kms_key`. Never put credentials in this file.

```bash
smithtune plan --provider sagemaker --data-dir ./data/aws-qwen \
  --aws-config ./aws-job.json --max-epochs 1 --batch-size 2
```

Planning makes no AWS calls and provisions nothing. It returns the exact
`CreateTrainingJob` request, dataset checksums, worker settings and limitations.
The optional cost estimate uses the supplied rate and runtime; it is not an
invoice or spending cap. Provisioning, storage, logs and any later serving can
add cost. Quota approval is separate from available GPU capacity.

### Try alternative instance sizes when capacity is unavailable

AWS supports an ordered `InstancePreferences` list inside **one training job**.
If the preferred type lacks capacity, SageMaker can select another listed type.
SmithTune does not submit parallel jobs or retry with new job names. This is
capacity selection before training; it does not recover an out-of-memory error,
a failed training loop, or a Spot interruption.

In `aws-job.json`, replace `instance_type` and the optional `hourly_rate_usd`
with the following fields. Keep the role, image, S3 prefix, region, instance count
and runtime settings from the earlier example:

```json
{
  "instance_preferences": [
    {"instance_type": "ml.g6e.2xlarge", "hourly_rate_usd": 2.80},
    {"instance_type": "ml.g6e.4xlarge", "hourly_rate_usd": 3.76}
  ],
  "max_pending_seconds": 1800
}
```

These are observed `us-west-2` on-demand training rates on 2026-10-15, not
automatically fetched prices. Verify the rates for your account/region before
running. Both example types have one L40S GPU; the second has more host RAM.
Only the 4xlarge has completed the full-size short smoke; neither has a
validated long-context profile.

Supply 1–5 unique GPU types in preferred order, each with a positive hourly rate.
The plan shows every option and uses the **highest listed rate** for its headline
runtime estimate (about $0.63 for ten minutes with these choices). Billing uses
the actual selected instance. The worker still uses only one GPU and one
instance, so select hardware that can independently hold the workload. Quotas,
permissions and compatible hardware are still required for the alternatives.

`max_pending_seconds` is required for preferences and optional for a fixed type.
AWS's minimum is **1,800 seconds / 30 minutes**; SmithTune currently accepts up
to one day. It limits total capacity waiting across the list, separately from
`max_runtime_seconds`. Use `aws stop --confirm` for an earlier manual cutoff.
The CLI reports the selected type/count/rate once AWS returns the selection;
this does not mean training has started. No Spot or reserved-plan integration
is exposed by this prototype.

## Submit only after approval

Repeat shared settings from the plan. Submission requires an explicit
`--confirm`, creates paid compute, and returns as soon as AWS accepts the job:

```bash
smithtune train --provider sagemaker --data-dir ./data/aws-qwen \
  --run-dir ./runs/aws-qwen-001 --run-id aws-qwen-001 \
  --aws-config ./aws-job.json --max-epochs 1 --batch-size 2 --confirm

smithtune aws status --run-dir ./runs/aws-qwen-001
smithtune aws wait --run-dir ./runs/aws-qwen-001 \
  --timeout 600 --stop-if-queued --confirm
```

`wait` only stops at its deadline when AWS still reports `Pending` and no training
start. It never submits a replacement or changes hardware. If compute has started,
the job continues under its original AWS runtime limit. Check the returned status;
`Stopping` still requires polling. To explicitly stop any nonterminal job:

```bash
smithtune aws stop --run-dir ./runs/aws-qwen-001 --confirm
```

**Closing the terminal does not stop the job.** AWS enforces the configured
maximum runtime and releases training compute after termination. `stop` requests
termination; poll `status` until terminal. A submission timeout may mean the job
was created: inspect its saved receipt with `status` before retrying. Never
submit a second name just to recover a lost response.

Local files include `plan.json`, `request.json`, a frozen `input/` bundle and
`job.json`. An existing job name or nonempty S3 run prefix is rejected before
upload. S3 input/checkpoint/output objects remain until the customer removes
them; job termination does not delete them.

## Worker and output contract

The cloud-neutral code is in `training/`; the SageMaker entrypoint only resolves
AWS's input/output paths. The [architecture guide](open-training.md) explains what
can be reused by Vertex. Training intent uses the same `SFTJobSpec` contract as
Jake's Vertex scaffold; AWS compute settings remain separate. New bundles include
this description and the worker checks it against the implemented recipe. Older
pilot bundles remain supported. Add `--aws-trainer trl` to **both** `plan` and `train` to
select the experimental `SFTTrainer` backend. The default remains the reference
Torch loop. TRL is configured to consume pre-rendered IDs and labels without
packing, truncation or chat-template rewriting. Small CPU tests establish gradient
and update parity. A full-size GPU smoke established TRL operation,
but numerical parity at that size and equivalence with Fireworks/Baseten's private
optimizers have not been measured.

The worker validates file checksums and the preparation renderer version, then
rerenders the saved canonical conversations. Each microbatch is one rendered
datum; gradients are accumulated to `--batch-size`, normalized over active
assistant tokens in that optimizer group, including its final partial batch.
HF performs the causal label shift once. This specifies our objective; it does
not claim numerical equivalence with providers' private optimizer implementations.

The initial LoRA recipe targets the language model's full-attention
`q_proj/k_proj/v_proj/o_proj` modules, with alpha `2 × rank`, no adapter dropout,
AdamW, constant learning rate, zero weight decay and gradient clipping at 1.0.
Linear-attention and MLP adapters are not included in this initial recipe.
The lowest validation loss chooses the best checkpoint; patience and minimum
delta independently determine when to stop.

SageMaker uploads `/opt/ml/model` as a model artifact archive:

- `adapter/`: selected PEFT adapter, **not merged standalone model weights**.
- `tokenizer/` and `base_config/`: matching tokenizer/config assets.
- `result.json`: model revision, dataset hashes, settings, loss history, selected
  epoch, sequence measurements and peak GPU memory.
- `artifact.json`: versioned per-file SHA-256 manifest used during collection and
  serving. Earlier exports are upgraded locally only after their original training
  provenance is verified.

Completed epochs in `/opt/ml/checkpoints` contain adapters, optimizer/RNG state
and a final `complete.json` marker. SageMaker synchronizes that directory to S3.
Only marked epochs are coherent; mid-epoch progress can be lost. Loading these
checkpoints into a new job is not yet implemented.

`aws status` returns the model artifact S3 URI and CloudWatch log location.
Training creates **no endpoint**. Use the explicit steps below for hosting.
`train --evaluate` and the top-level `deploy`/`evaluate` commands still reject
SageMaker; LangSmith replay is not wired to the AWS sampler yet.

## Collect, host and remove an endpoint

Once training is `Completed`, download and verify the adapter export:

```bash
smithtune aws collect --run-dir ./runs/aws-qwen-001
```

Collection checks the original job's S3 output prefix, bounds archive size,
rejects links/traversal, and validates model identity and saved training inputs.
It writes `artifact/` and `collection.json` under the run directory. Rerunning
collection verifies the local copy rather than replacing it.

Create `aws-endpoint.json` with a real image digest and **current hosting rate**:

```json
{
  "serving_image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/<repository>@sha256:<new-image-digest>",
  "instance_type": "ml.g6.4xlarge",
  "hourly_rate_usd": 1.654,
  "max_context_tokens": 4096,
  "max_new_tokens": 96,
  "startup_timeout_seconds": 600,
  "inference_ami_version": "al2023-ami-sagemaker-inference-gpu-4-1",
  "provision_timeout_seconds": 300
}
```

The rate is illustrative, not fetched pricing. L4 passed the earlier short
reload job; endpoint generation and real 4K workloads remain unvalidated. The updated image understands
SageMaker's `train` and `serve` entrypoints; old training-only images cannot serve.
Hosting loads the pinned base model plus the exported adapter. The image currently
needs outbound Hugging Face access, even with VPC settings. This is not yet a fully
isolated deployment. No model weights are embedded in the image.

`inference_ami_version` is optional and selects the host GPU driver/runtime.
The current Linux image uses PyTorch's CUDA 13 wheel; AWS documents NVIDIA driver
580 and CUDA 13 for `al2023-ami-sagemaker-inference-gpu-4-1`. Older inference AMIs
list CUDA 12 drivers. Local CPU image tests do not establish host-driver compatibility.
`provision_timeout_seconds` optionally sets AWS's capacity-provisioning timeout
(300–3,600 seconds), separately from model-download and container-health timeouts.
It is not a total deployment deadline or spending cap, and deletion may still need
to wait for an in-progress operation. Endpoint status retains AWS's `failure_reason`
when available, even if no log group was created. See AWS's
[ProductionVariant reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html).

For capacity fallback within the same region, replace `instance_type` and
`hourly_rate_usd` with an explicit ordered `instance_preferences` list:

```json
{
  "instance_preferences": [
    {"instance_type": "ml.g6.4xlarge", "hourly_rate_usd": 1.654},
    {"instance_type": "ml.g5.4xlarge", "hourly_rate_usd": 2.03}
  ],
  "provision_timeout_seconds": 300
}
```

Keep the other serving-image/context fields from the complete config above.
This maps to AWS's native `InstancePools`, with priorities in list order and
**one total instance**, not one per pool. Supply 1–5 unique GPU options and their
current hosting rates. Each option must fit the same model and workload; this
does not automatically size GPUs or guarantee capacity. The provisioning timeout
covers all pools together. The plan estimates the highest approved hourly rate;
AWS charges for the instance actually provisioned. Status records the count for
each pool when AWS reports it. See [AWS instance pools](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-heterogeneous.html).

```bash
smithtune aws deploy-plan --run-dir ./runs/aws-qwen-001 \
  --endpoint-config ./aws-endpoint.json --endpoint-name aws-qwen-pilot-001

# Creates a dedicated, continuously billed GPU endpoint after approval.
smithtune aws deploy --run-dir ./runs/aws-qwen-001 \
  --endpoint-config ./aws-endpoint.json --endpoint-name aws-qwen-pilot-001 --confirm

smithtune aws endpoint-status --run-dir ./runs/aws-qwen-001 --timeout 600
```

The offline plan includes the exact CreateModel, CreateEndpointConfig and
CreateEndpoint requests, supplied hourly rate and cleanup command. Deployment
saves an ownership receipt **before** creating resources. After a timeout, use
`endpoint-status` or repeat the identical `deploy` command; it reconciles the
recorded names and ownership. It refuses to guess when a prior creation is still
ambiguous. Without an explicit preference list it requests only the selected
instance type; with a list AWS may use only those approved options. It never
starts another endpoint as a retry.

Use an input JSON file such as:

```json
{
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "tools": [],
  "max_new_tokens": 32,
  "variant": "tuned"
}
```

```bash
smithtune aws invoke --run-dir ./runs/aws-qwen-001 --input ./request.json --confirm
smithtune aws undeploy --run-dir ./runs/aws-qwen-001 --timeout 600 --confirm
```

The pilot endpoint supports one greedy generation at a time, messages/tools in
the native format, and `base`/`tuned` variants on the same loaded model. It preserves
the tool parser and rejects prompt-plus-output lengths above the configured limit.
It does not execute tools, stream, batch, autoscale or provide an OpenAI-compatible
API. The synchronous SageMaker request has a service response deadline; test actual
latency before raising generation/context limits. InService is a health signal,
not a quality or throughput result.

**Billing continues until AWS deletes the endpoint.** Status timeouts and closing
the CLI do not stop billing. Deletion checks receipt ownership, removes the endpoint
first, and then its configuration/model after the endpoint disappears. If the
result is `Deleting`, `CleanupPending`, `CleanupUncertain` or an error, run the cleanup command again
and inspect status. Unknown creation outcomes are never reported as proven cleanup.
AWS can reject deletion while an endpoint is still `Creating` with
`Cannot update in-progress endpoint`. The cleanup loop preserves its dependencies
and retries that response until its wait deadline. A readiness timeout therefore
cannot guarantee immediate cancellation or a fixed hosting cost.
Artifacts and checkpoints in S3 remain. After `Deleted`, a new endpoint name can
be deployed from the same artifact; prior receipts move to `endpoint/history/`.

For an **authorized** live test, use a temporary endpoint:

```bash
smithtune aws endpoint-smoke --run-dir ./runs/aws-qwen-001 \
  --endpoint-config ./aws-endpoint.json --endpoint-name aws-qwen-smoke-001 \
  --input ./request.json --timeout 600 --confirm
```

It creates a fresh endpoint, saves base/tuned generations and requests cleanup in
a `finally` block, including on invocation/readiness failure. It cannot borrow an
active endpoint. Results and cleanup state are saved under `endpoint/`.
`all_formats_valid` reports the parser checks;
`Completed` alone means the lifecycle finished, not that the model improved. The timeout
bounds each readiness/cleanup wait, **not total AWS spend**; process termination or
API failure can leave cleanup pending. No AWS-side expiry is configured. The whole
lifecycle is tested with mocked AWS APIs and tiny real CPU models.

## Container and development checks

The [Dockerfile](../docker/sagemaker/Dockerfile) uses the repository's lockfile and
the optional `aws-training` dependencies. Its build context excludes datasets,
run artifacts and secrets. The Python 3.12 Debian bookworm base is pinned by
digest; update that pin deliberately and build for Linux x86-64:

```bash
docker build --platform linux/amd64 --provenance=false \
  --output type=image,name=smithtune-sagemaker:dev,oci-mediatypes=false \
  -f docker/sagemaker/Dockerfile .
```

Use **Docker V2 schema 2** for an image that will serve an endpoint. A live
CreateModel call rejected an OCI image manifest even though Training Jobs had
accepted that format. The explicit exporter settings above avoid that mismatch.
After pushing, verify ECR's `imageManifestMediaType` is
`application/vnd.docker.distribution.manifest.v2+json` and use the published digest.

The locked PyTorch wheel supplies CUDA user-space libraries; the L40S
SageMaker host passed the GPU smoke. The Dockerfile sets
`TORCH_DISABLE_NATIVE_JIT=1` before Python starts: PyTorch 2.14's eager Triton
overrides otherwise require a C compiler absent from this slim image. An initial
GPU run failed in that path; disabling the override allowed training to finish
with the same data and training objective. This is not a performance comparison.
Optional fast linear-attention/causal-convolution kernels are not bundled; the initial Qwen
worker uses the PyTorch fallback, whose GPU throughput must be benchmarked.
Inspect the image before pushing to the customer's approved ECR repository,
then use its immutable digest
in the job config. No image is automatically built or pushed by SmithTune.

```bash
sfw uv sync --locked --extra test --extra aws --extra aws-training --python 3.12
uv run --no-sync pytest tests/test_sagemaker*.py tests/test_open_training_*.py tests/test_hf_sft.py
```

Start with synthetic data and a short context for the first approved AWS smoke
test. Then test the real required context length and representative trajectories
before expanding the CLI, model lineup or serving integration.

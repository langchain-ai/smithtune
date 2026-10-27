# Shared open training worker

SmithTune owns data preparation and the SFT implementation for custom cloud jobs.
AWS or GCP allocates compute and runs our container. The same local-file worker
should train the model on either cloud; each provider adapter stages files, submits
the job, tracks it and retrieves its output.

```text
LangSmith conversations
  → existing preparation: tools, source splits, native rendering identity
  → provider adapter: stage bundle, choose compute, submit and monitor
      AWS: S3 + SageMaker Training Job     Vertex: future GCS + CustomJob adapter
  → shared training worker in the image
      Transformers → PEFT LoRA → Torch loop or optional TRL SFTTrainer
  → verified adapter export in object storage
  → separate provider deployment adapter → endpoint → invocation → cleanup
```

Fireworks and Baseten retain their existing remote training API paths. This worker
does not replace those adapters or claim identical optimizer implementations.

## Alignment with Jake's branch

The shared boundaries follow
[`jake-add-gcp` at `842d5a9`](https://github.com/langchain-ai/smithtune/tree/842d5a970aaeaf11738e0a7cd5a969bfe18339ce):
`training/spec.py` describes training intent and outputs, `training/hf_sft.py`
owns the Hugging Face recipe, and provider packages own execution settings and
cloud lifecycle. Its token-to-label converter is reused directly. It preserves
the exact token IDs and binary target masks and rejects oversize examples.

AWS plans and new bundles include the shared `SFTJobSpec` description. The worker
checks that description against the model, common settings and implemented LoRA
recipe; older bundles without it remain readable. Compute and backend choices
stay separate from this training description. Artifact collection returns
`TrainingArtifacts` handles to the verified local adapter, metrics and manifest,
alongside the existing collection fields. The source archive's S3 URI stays in
`collection.json`. The resume handle is null because cross-job restart is not
implemented; retained optimizer state alone does not establish restart support.

Jake's Vertex files are included as scaffolding only: no live Vertex submission,
training or inference is available. This alignment does not expose Vertex through the CLI.

## Proposed answers for Jake and Vivek

| Question | Proposed decision and current implementation |
| --- | --- |
| Which training libraries? | Standardize on Transformers + PEFT + TRL SFTTrainer, with Torch underneath. TRL is implemented as an optional backend, tested against the reference loop. Keep the existing pins and switch the default only after a full-size GPU test passes. |
| One worker for AWS and GCP? | Yes. `training/worker.py` takes local input, model-output and checkpoint directories and calls `training/hf_sft.py`. No AWS/GCP SDK calls belong in the optimizer loop. Vertex has execution scaffolding but no live adapter. |
| One image or cloud-specific images? | Share dependency pins, training code and export format. Keep small cloud-specific entrypoints for filesystem paths and upload behavior. AWS currently uses one image for `train` and pilot `serve`; a shared base plus wrappers can follow when Vertex's needs are known. |
| How do we choose compute? | Explicit instance/GPU choices first. Create measured profiles for model, precision, LoRA targets/rank, actual sequence length and microbatch size. Do not infer GPU fit from model parameter count alone. This version uses one GPU; no estimator or FSDP implementation exists. |
| What is the output? | Selected LoRA adapter weights, tokenizer/base config, training provenance and file hashes, saved to object storage. These are **not** standalone merged model weights; serving also needs the exact pinned base checkpoint. |
| What about capacity failures? | Preserve job identity, report allocation phase, use approved instance preferences and bounded queue waits. Quota and capacity are separate. Never silently resize or move regions. Use the native SageMaker path first; Batch/Spot/Modal are later options. |

## Training library responsibilities

Transformers implements the model's computation. A batch of token IDs goes in;
next-token scores come out. PEFT freezes the base weights and inserts trainable
low-rank matrices. Torch computes gradients and updates those trainable parameters.

SFTTrainer organizes this work: dataset iteration, microbatches, gradient
accumulation, optimizer steps and callbacks. Its configuration can change the
learning rate, epochs, precision, packing, tokenization and loss targeting. Those
last choices matter here: handing it raw messages with defaults could change
SmithTune's existing training representation.

Our optional TRL backend therefore receives **already rendered token IDs and
explicit labels**. A label of `-100` means “use this token as context, but do not
score its prediction.” Tool results remain context. Supported assistant outputs,
including tool calls, are training targets under the existing renderer policy.
The causal label shift happens once. The loss is normalized over the actual
assistant target count in each optimizer group, including a final partial group.
No additional truncation, packing or template rewriting happens inside TRL.

The reference Torch loop remains the default. TRL matches its gradients and
parameter updates on small CPU tests and passed an October 20 full-size A10G
training/export/fresh-reload smoke. Full-size numerical parity is still a
required gate before changing the default. Neither backend currently supports
distributed training.

## File and adapter boundaries

| Layer | Files and responsibilities |
| --- | --- |
| Shared input contract | `training/spec.py`: portable job/artifact descriptions. `training/contracts.py`: prepared manifest and splits, checksummed bundle, source/model validation. |
| Shared worker | `training/hf_sft.py`, `trl_trainer.py`: rerender using the pinned native formatter, train, validate, select the best epoch, export. `training/worker.py` provides the CLI entrypoint. |
| AWS submission | `providers/sagemaker/provider.py`: immutable ECR image, explicit hardware/runtime, S3 input, CreateTrainingJob, durable job receipt. |
| AWS execution settings | `providers/sagemaker/execution.py`: explicit region, instance choices, image, IAM, storage and runtime. No resource allocation on import or validation. |
| AWS container wrapper | `providers/sagemaker/worker.py`: map `/opt/ml/input/data/training`, `/opt/ml/model` and `/opt/ml/checkpoints` to the shared worker; it is the image entrypoint. |
| Shared export | `adapter/`, `tokenizer/`, `base_config/`, `result.json`, `artifact.json`. Manifest checksums detect changed/missing files. |
| AWS collection | `providers/sagemaker/artifacts.py`: download the completed job's export, safely extract, verify original run/model/data identity. |
| Pilot serving | `training/serving.py`: reload exact base plus adapter, native prompt rendering and tool parsing, greedy generation, HTTP health/invocation routes. |
| AWS lifecycle | `providers/sagemaker/deployment.py`: plan/create/status/invoke/delete with ownership checks, receipts, explicit billing and temporary smoke cleanup. |

The shared worker can be launched directly after a cloud adapter stages its input:

```bash
python -m smithtune.training.worker \
  --input-dir /input --model-dir /output --checkpoint-dir /checkpoints
```

This command performs training on the available GPU. It does not provision
hardware. Vertex still needs its own staging, IAM, submission, monitoring,
object-store collection and endpoint adapters; accepting its identity in the
worker does not implement that cloud integration.

## Validation gates and remaining work

1. **Local correctness — tested:** real tiny-model optimization through both
   backends; gradient/update comparisons; artifact export/reload and generation;
   HTTP behavior; mocked AWS request validation, ambiguous submission recovery,
   ownership checks and cleanup failures.
2. **Full-size baseline — tested:** Qwen3.5-9B rank-8 LoRA on one L40S,
   437-token longest datum, then separate L4 reload/generation. On October 20,
   the TRL backend trained and reloaded on one A10G: 11 steps, exact adapter
   reload and six valid synthetic generations, about $0.096 compute. This does
   not establish a 4K profile or full-size numerical parity. See the
   [AWS evidence](aws-training.md).
3. **Live endpoint pilot — blocked on capacity:** October 20 Oregon retries
   reused the adapter and image on L4, L40S and A10G, followed by one native
   instance-pool request with five allowed sizes. All failed with AWS
   `InsufficientInstanceCapacity` in roughly 5–6 minutes, before container logs
   or inference. Submission and complete cleanup were verified. The adapter now
   supports explicit GPU inference AMIs, provisioning timeouts, failure reasons
   and ordered endpoint pools. A Virginia H100 attempt also failed for capacity
   after about 317 seconds, before container startup; cleanup was verified.
   Runtime compatibility still needs a host to start.
   Earlier cancelled October 18/19 attempts had no confirmed terminal cause.
   Hosting timeouts do not guarantee immediate cancellation or a fixed cost.
4. **Actual context benchmarks:** prepared 4K/8K and conditional 16K probes still
   need GPU capacity. Record measured memory and step time; stop on OOM/non-finite
   results. Add profiles only from measured runs. Larger agent contexts may need
   memory-efficient loss/kernels, different GPUs or distributed training.
5. **Product integration:** full-size TRL numerical parity, AWS sampler integration with
   existing LangSmith replay, production serving, resumable training and more model
   recipes remain. Reuse the existing replay publication/recovery path when wiring
   AWS; do not create a separate evaluation framework.

No new model support, automatic hardware sizing, private-only networking, Spot
recovery, FSDP or production endpoint throughput is established by this work.

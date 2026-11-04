# Shared open training worker

SmithTune prepares the data and supplies the SFT implementation. SageMaker
allocates compute and runs our container. A second cloud adapter should stage
files, submit the same worker, track the job and collect its output.

```text
LangSmith conversations
  → preparation: source splits, per-assistant tools, native rendering identity
  → SageMaker adapter: S3 bundle, explicit compute, submit and monitor
  → shared training worker
      Transformers + PEFT LoRA + TRL SFTTrainer
  → selected adapter export + optional base/tuned replay answers in S3
  → collect and verify locally
  → existing replay judges and LangSmith comparison
```

Evaluation answers are generated on the training machine after export. No
endpoint is needed. Fireworks and Baseten keep their existing training paths;
this worker does not claim identical optimizers or results.

## Current choices and cross-cloud work

| Question | Current choice |
| --- | --- |
| Which libraries? | Transformers implements the model, PEFT supplies LoRA, and TRL SFTTrainer manages training over PyTorch. TRL is the default; the Torch reference loop remains selectable for tests. |
| One worker for AWS and GCP? | Yes, that is the intended boundary. `training/worker.py` accepts local input, model-output and checkpoint directories. No cloud SDK calls belong in the training loop. |
| One image? | Share dependency pins, training code and export format. Keep cloud entrypoints small. AWS has a training/experimental-serving image; a shared base can follow once Vertex requirements are known. |
| How is compute chosen? | Explicit hardware and runtime settings. Qualify model, precision, LoRA targets/rank, rendered length and microbatch size by measurement. Parameter count alone does not prove GPU fit. |
| What is exported? | The selected LoRA adapter, tokenizer/base config, training provenance and file hashes. These are not merged model weights; inference also needs the exact pinned base checkpoint. |
| What happens when capacity is unavailable? | Preserve the job receipt, report the allocation phase, and use explicit instance preferences and bounded waits. Quota and capacity are separate. No automatic region move or hardware resizing is implemented. |

The training recipe and filesystem boundary can be reused by Jake's Vertex
adapter. This branch contains no Vertex adapter or `training/spec.py` scaffold.
`training/contracts.py` currently validates SageMaker's model profiles and
prepared provider identity. Vertex still needs its own staging, IAM, submission,
monitoring, collection and provider validation. Agree on that shared contract
and ownership with Jake before merging either implementation.

## What the trainer does

Transformers runs the model: token IDs go in and next-token predictions come
out. PEFT freezes the base weights and adds small trainable LoRA matrices.
PyTorch computes gradients and updates those parameters.

SFTTrainer handles dataset iteration, microbatches, gradient accumulation,
optimizer steps and callbacks. Its defaults can also tokenize, pack and truncate
data. We disable those changes and give it already rendered token IDs and
explicit labels, so it preserves SmithTune's preparation behavior.

Each assistant action is a separate target. Earlier messages, tool results and
historical assistant outputs remain context with labels of `-100`, which means
“do not score this position.” The target action uses the tools available at that
call. Source conversations stay separate across train, validation and test.
The causal label shift happens once. Each optimizer group, including the final
partial group, is normalized by its actual count of assistant target tokens.

Both backends use TRL's chunked next-token loss. It skips ignored positions and
projects scored positions through `lm_head` in 256-token checkpointed chunks.
This removes the full-sequence vocabulary-logit allocation. CPU tests compare
losses, gradients, updates and validation against the full-logits objective.
GPU memory is measured through 18,751 tokens on synthetic data; larger contexts
and steady-state GPU speed still need measurement. The
implementation uses a private TRL helper, so keep its version pinned and rerun
the loss tests on upgrades.

Neither backend supports distributed training or cross-job training restart.
Saved optimizer state alone does not establish restart support.

## Files and responsibilities

| Layer | Files and responsibilities |
| --- | --- |
| Shared contracts | `training/contracts.py`: model profiles, worker settings, training description, checksummed input bundles and verified exports. |
| Shared recipe | `training/hf_sft.py`, `trl_trainer.py`: render, train, validate, select the best epoch and export. `training/worker.py` is the local-file CLI. |
| In-job evaluation | `training/evaluation.py`: generate bounded base/tuned answers and save each answer durably. Judging and publication stay outside the job. |
| AWS submission | `providers/sagemaker/provider.py`, `execution.py`: explicit image/hardware/runtime, S3 staging, submission and durable job receipts. |
| AWS container wrapper | `providers/sagemaker/worker.py`: map SageMaker's input, model and checkpoint directories onto the shared worker. |
| AWS collection | `providers/sagemaker/artifacts.py`: download, safely extract and verify run/model/data identity. |
| Saved-answer replay | `providers/sagemaker/sampling.py`: feed in-job answers into the existing replay judges and LangSmith publisher. |
| Experimental hosting | `training/serving.py` and `providers/sagemaker/deployment.py`: native rendering/parsing, HTTP serving and owned endpoint lifecycle. This is separate from evaluation. |

The export contains `adapter/`, `tokenizer/`, `base_config/`, `result.json` and
`artifact.json`, plus `evaluation/` when requested. Collection retains the
source archive's S3 URI in `collection.json`.

After staging the input, launch the worker directly:

```bash
python -m smithtune.training.worker \
  --input-dir /input --model-dir /output --checkpoint-dir /checkpoints
```

The bundle chooses CPU or `cuda:0`; `--device` overrides it. This command does
not provision hardware. CPU is supported only for the small pipeline test profile.

## What has been tested

- **Offline correctness:** tiny-model training through both backends, loss and
  gradient parity, adapter export/reload, native tool parsing, HTTP behavior,
  mocked cloud lifecycle, saved-answer lookup and recovery.
- **GPU pipeline:** Qwen3.5-9B LoRA trained, exported, generated held-out answers
  and fed LangSmith evaluation using TRL and the chunked loss. Later synthetic
  runs reached 439/1,387 tokens on A10G, 2,538 on L4 and 18,751 on L40S. The
  18,751-token run peaked at 34.62 GiB. These are individual pipeline tests,
  not a controlled throughput benchmark or evidence of production quality.
- **Live CPU pipeline:** the 0.8B test profile completed SageMaker training,
  export, collection, in-job replay generation and local judging/publication
  to LangSmith. Evaluation-only jobs also resumed saved answers.

## What still needs testing

- Measure memory and steady-state step time beyond the tested 18,751-token
  synthetic profile, including the required 32K/128K target. A model's context
  ceiling is not a tested hardware profile.
- Run representative real 9B data through training and in-job evaluation,
  then inspect quality and tool-call parsing.
- Qualify accelerated kernels and larger models; add multi-GPU training only
  if measurements and the target context require it.
- Test hosted inference when capacity is available. Earlier endpoint attempts
  failed before container startup; no live endpoint is validated.

See [the AWS guide](aws-training.md) for commands and prerequisites. Automatic
hardware sizing, private-only networking, Spot recovery and production endpoint
throughput remain outside the validated scope.

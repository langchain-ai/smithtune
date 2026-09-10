# Smithtune

Smithtune is a standalone CLI that turns a LangSmith trajectory dataset into a trained
Fireworks LoRA model and a held-out replay evaluation report, or into a
Baseten Loops SFT checkpoint. Fireworks remains the default provider.

## Inputs and outputs

Inputs:

- LangSmith workspace ID and one dataset ID
- An inference contract captured from an approved main-model `llm` run when trajectories use tools
- `LANGSMITH_API_KEY` and `FIREWORKS_API_KEY`
- A reviewed model profile, or equivalent custom model settings
- `ANTHROPIC_CUSTOM_HEADERS` or `ANTHROPIC_API_KEY` with a LangSmith gateway key

Outputs:

- Raw LangSmith export
- Reviewed inference contract with canonical tool schemas and prompt provenance
- Fireworks train, validation, and test JSONL
- Validation, warning, and context-rejection reports
- Resumable epoch checkpoints and the selected best checkpoint
- Promoted Fireworks model and temporary evaluation endpoint
- Judge calibration, paired replay results, and summary report

All generated data and run files stay under `data/` and `runs/`. Git ignores
both directories.

## Module boundaries

`pipeline.py` parses arguments and dispatches commands. Preparation, planning,
and training all go through the adapter selected by `providers/__init__.py`.
Each adapter owns its model profiles, supported settings, validation, and
training lifecycle. Dataset split defaults are shared. Adapters can be used directly without
importing the CLI or loading an optional provider SDK for planning.

| Module | Responsibility |
| --- | --- |
| `providers/base.py` | Common model, option, error, and training contracts |
| `models.py` | Shared named/custom model resolution and prepared-profile compatibility |
| `providers/fireworks.py` | Fireworks profiles, preparation policy, SFT, checkpoint promotion, and deployment |
| `providers/baseten.py` | Baseten profiles, configuration, Loops data conversion, and training lifecycle |
| `dataset.py` | Canonical trajectory conversion, deterministic splits, manifests, and inference-contract capture |
| `rendering.py` | Shared tokenizer/renderer context validation |
| `artifacts.py` | Shared JSON/JSONL artifacts and local command execution |
| `evaluation.py`, `inference.py` | Replay cases, scoring, judging, and inference requests |

Both adapters use the same canonical dataset preparation and an 80/10/10
train/validation/test split by default. Use `--test-fraction 0` to request a
90/10/0 split explicitly. Baseten reserves the test data even though deployment
and replay evaluation remain outside this CLI's supported Baseten workflow.

`plan --help` and `train --help` group the training options by provider and
show their defaults:

| Supported by | Options | Defaults |
| --- | --- | --- |
| Both | `--max-epochs`, `--early-stopping-patience`, `--early-stopping-min-delta` | 5, 1, 0.0 |
| Both | `--learning-rate`, `--batch-size`, `--seed` | 0.0001, 32, 42 |
| Both | `--lora-rank` | Prepared model profile's `default_lora_rank` (8 for built-in profiles) |
| Fireworks | `--lora-alpha`, `--pipeline-depth` | 32, 4 |
| Baseten | `--microbatch-token-budget` | Prepared model's `max_seq_len` |
| Baseten | `--max-spend-usd`, `--hourly-rate-usd` | Unset; supply both to enable the spend guard |
| Baseten | `--replicas`, `--spend-reserve-fraction`, `--max-dropped-training-rows` | 1, 0.1, 1 |

Shared flags retain their meaning across providers: `--lora-rank` configures
LoRA rank in either API, while Baseten accumulates microbatches to reach the
requested effective `--batch-size`. Each adapter rejects options belonging to
the other provider.

Named model profiles are defaults. Both providers also accept
`--model-profile custom` with an explicit base model, tokenizer, pinned
tokenizer revision, renderer, and preparation context limit. The resolved
profile is written to the prepared manifest and supplies the model identity
used by planning, rendering, capability checks, and training. Changing models
does not require editing training code.

Baseten run artifacts retain the full resolved model profile and its hash.
Renderer/history-mode conflicts are rejected during rendering before training
resources are provisioned.

## How it works

### Training

```text
LangSmith dataset
       |
       v
Download every trajectory
       |
       v
Validate messages, matched tool calls, and the inference contract
       |
       v
Convert messages and attach the canonical tools to Fireworks JSONL
       |
       v
Count tokens with the selected tokenizer and Fireworks renderer
       |
       +---- over model context limit ----> reject and report; never truncate
       |
       v
Stable local split by source_thread_id
  80% train | 10% validation | 10% replay test
       |
       v
Fireworks serverless LoRA SFT
       |
       v
Train one epoch -> save checkpoint -> measure validation loss
       |                              |
       |                         loss improved?
       |                         /           \
       |                       yes            no
       |                        |              |
       |                   mark best       stop early
       |
       +---- repeat, up to five epochs
       |
       v
Promote the checkpoint with the lowest validation loss
       |
       v
Deploy the trained model to an on-demand endpoint
```

### Replay evaluation

```text
Prepared replay test trajectories
       |
       v
Select every assistant-message boundary by default
       |
       v
For each boundary, keep:
  full trajectory prefix
  recorded next assistant action
  immediate recorded tool results
       |
       v
Check context length; reject over-limit cases without truncation
       |
       v
Calibrate Claude Sonnet 5
  recorded message    -> must pass
  wrong tool or text  -> must fail
  wrong arguments or empty text -> must fail
       |
       v
For each replay case
       |
       v
Fireworks trained model
       |
       v
candidate next message
       |
       v
Claude Sonnet 5 through the LangSmith gateway
                   |
                   v
Plain prompt -> validate pass/reason JSON
                   |
          invalid? retry, up to three attempts
                   |
                   v
Score agreement with the recorded message
                   |
                   v
summary.json
  combined pass rate
  tool-call pass rate
  text pass rate
  deterministic tool-decision, name, JSON, schema, argument, and call-set rates
```

## Set up

Requirements: Python 3.12, `git`, `sfw`, `uv`, `langsmith`, and `firectl`.

```bash
git clone https://github.com/langchain-ai/smithtune.git
cd smithtune
./bootstrap.sh
source .venv/bin/activate
export LANGSMITH_API_KEY=...
export FIREWORKS_API_KEY=...
```

The setup pins the tested Fireworks cookbook revision, `jsonschema`, and the
optional Baseten Loops SDK. `FIREWORKS_API_KEY` is only needed for Fireworks
training. Baseten preparation needs `LANGSMITH_API_KEY`; Baseten training also
needs `BASETEN_API_KEY`. Reference these variables only—do not write secret
values into files or commands.

## 1. Capture and review the inference contract

Tool-enabled trajectories need the exact tool declarations that accompanied
the production main-model call. Capture them from one approved LangSmith
`llm` run:

```bash
python pipeline.py capture-contract \
  --workspace-id <langsmith-workspace-id> \
  --run-id <approved-main-model-llm-run-id> \
  --output inference_contract.json
```

The run query reads `extra.invocation_params.tools`, normalizes supported
OpenAI-, Anthropic-, and Gemini-style declarations into canonical OpenAI-style
function tools, and records the run's system prompt and safe generation
settings. It does not copy credentials or provider/runtime payloads. Review the
contract and its provenance before preparation.

The contract validates every tool JSON Schema and hashes the schemas, system
prompt, and semantic inference settings. Preparation fails if a trajectory's
first system message differs from the contract or a recorded tool name or
argument does not validate. Tool declarations remain separate from the system
message; the selected Fireworks renderer serializes them in its model-specific
format.

## 2. Download, validate, and split

```bash
python pipeline.py prepare \
  --workspace-id <langsmith-workspace-id> \
  --dataset-id <langsmith-dataset-id> \
  --inference-contract inference_contract.json \
  --model-profile qwen3p8-27b
```

The pipeline uses the LangSmith CLI to read the dataset count, export the
dataset, and page through every example. It preserves recorded message order.
Message IDs are optional and are preserved when present.

It preserves supported messages, system prompts, tool calls, tool results, and
source metadata. Content may be a string, text blocks, or tool-call blocks.
Native reasoning blocks are accepted and omitted from derived training and
replay messages by default (`--reasoning-policy omit`). A message left empty
by reasoning removal is removed; an example without a remaining assistant
target fails preparation. Raw exports and the source LangSmith dataset are
unchanged. The manifest records the effective policy, source/preserved/omitted
reasoning-block counts, and removed-message count before context rejection.

To include readable reasoning explicitly, pass `--reasoning-policy preserve`.
This requires `ModelSpec.supports_reasoning_content=True` and a verified
renderer (`qwen3_8_preserved` or `kimi_k3`). The built-in Qwen and Kimi profiles
declare this capability; custom profiles must opt in with
`--supports-reasoning-content`. Without reasoning in the source, either policy
works with any supported model. A reasoning-capable profile still defaults to
omission. `thinking_trace_history_mode` controls how retained history is
rendered and does not enable preservation by itself.

Preservation maps readable native `reasoning` text to `reasoning_content`,
separate from the visible answer. Encrypted reasoning state and signatures
are never training text. OpenAI reasoning blocks may contain summaries rather
than full reasoning; preservation uses only the text actually recorded.
Reasoning appearing after visible content fails preservation because the
target format places reasoning first. Omission does not infer reasoning from
text or change the target model's generation settings; model templates may
still emit empty thinking delimiters. Replay uses the prepared test messages
and the same policy.
Reasoning-only messages remain context in preserve mode but are not scored as
answer/tool replay cases. The action judge compares visible answers and tool
calls, with the same conversation prefix the candidate received.

Unsupported multimodal and non-standard blocks fail validation instead of
being silently dropped. It does not synthesize prompts or apply quality
filters.

For tool-enabled data, the canonical tools are stored separately on every
prepared row and passed to the same renderer used for training. Preparation
fails when the selected profile does not require tool declarations or its
renderer cannot serialize them. The built-in Qwen and Kimi profiles require
tools. A compatible custom profile must opt in with
`--requires-tool-declarations`.

The model tokenizer and Fireworks renderer calculate the exact training length,
including rendered tool declarations. A row above the selected model context
limit is rejected and reported. It is never truncated.

The stable split groups by `source_thread_id`:

- 80% train
- 10% validation
- 10% replay test

The train fraction is the remainder after validation and test. Fractions may be
zero. To reserve one fixed dataset entirely for replay evaluation, prepare it in
a separate data directory:

```bash
python pipeline.py prepare \
  --data-dir data/replay-test \
  --workspace-id <langsmith-workspace-id> \
  --dataset-id <test-dataset-id> \
  --validation-fraction 0 \
  --test-fraction 1 \
  --inference-contract inference_contract.json \
  --model-profile qwen3p8-27b
```

This produces an empty train and validation split and places every trajectory
in `data/replay-test/prepared/test.jsonl`. Pass `--data-dir data/replay-test` to
`eval-plan` and `evaluate`. Training refuses a preparation without train and
validation rows; replay evaluation refuses one without test rows. The pipeline
always expects complete trajectories and generates replay boundaries locally.

## 3. Baseten Loops SFT

Baseten is an SFT-only path. Its default profile is Qwen 3.8; dataset size and
split counts come from the prepared files and manifest. Training requires
nonempty training and validation splits. It keeps system, user, and
tool-result tokens as context while learning only from assistant text and
tool calls.

Prepare the approved 100-row dataset with the shared 80/10/10 split defaults:

```bash
python pipeline.py prepare \
  --provider baseten \
  --workspace-id <langsmith-workspace-id> \
  --dataset-id <dataset-id> \
  --inference-contract inference_contract.json \
  --model-profile qwen3p8-27b
```

To prepare a different model, supply its configuration explicitly:

```bash
python pipeline.py prepare \
  --provider baseten \
  --workspace-id <langsmith-workspace-id> \
  --dataset-id <dataset-id> \
  --inference-contract inference_contract.json \
  --model-profile custom \
  --base-model <baseten-model-name> \
  --tokenizer-model <tokenizer-repository> \
  --tokenizer-revision <pinned-revision> \
  --renderer <compatible-renderer> \
  --max-seq-len <preparation-context-limit> \
  --trainer-max-seq-len <trainer-context-limit> \
  --default-lora-rank <rank> \
  --requires-tool-declarations
```

Use the renderer and thinking-history options appropriate for the selected
model. The trainer limit defaults to the preparation limit when omitted.
Before provisioning, the adapter checks that the workspace advertises the
selected base model with enough context capacity. Preparation split fractions
can be set with `--validation-fraction` and `--test-fraction`.

Review the no-side-effect plan before authorizing the paid training command:

```bash
run_id=my-baseten-sft

python pipeline.py plan --provider baseten --run-id "$run_id"

python pipeline.py train \
  --provider baseten \
  --run-id "$run_id" \
  --run-dir "runs/$run_id" \
  --confirm
```

Baseten defaults to five epochs with patience one, an effective batch size of
32, and token-budgeted microbatches. LoRA rank defaults to the selected model's
rank; the microbatch token budget defaults to its preparation context limit.
Set `--max-epochs`, `--early-stopping-patience`, `--batch-size`, `--lora-rank`,
`--microbatch-token-budget`, `--seed`, or `--replicas` to configure the run.
It provisions a trainer with
`with_sampler=False`; saving sampler-ready weights does not require running
a sampler. It saves resumable optimizer
state after every epoch, saves sampler-ready weights whenever validation loss
improves, and selects the best sampler weights rather than merely the final
epoch. To initialize a later run from a saved resumable state, add
`--init-from-checkpoint <state-uri>` to `train`.

The default Qwen profile allows 262,144 preparation tokens and 131,072 trainer
tokens. These limits belong to the model profile. Training defaults to dropping
at most one complete training row above the trainer limit and records each
exclusion in `plan.json`; set `--max-dropped-training-rows 0` to reject every
oversized training row, or supply another explicit allowance. It rejects
oversized validation rows and an empty remaining training split before
provisioning. It never truncates trajectories.

Each Baseten run directory contains four operator-facing artifacts:

- `plan.json` records the reviewed model, training, split, checkpoint, and
  budget configuration authorized for the run.
- `run-state.json` is the durable lifecycle record for auditing progress,
  identifying the exact prepared data, and locating the latest checkpoints.
- `epochs.json` lists completed epoch losses, optimizer-step counts, and the
  resumable and sampler checkpoint written for each epoch.
- `result.json` records the terminal outcome, selected best checkpoint, budget
  status, and whether client closure and resource deactivation succeeded.

Budgeting is optional. To set a client-side guard, pass both a $75 maximum and
a verified hourly rate:

```bash
python pipeline.py train \
  --provider baseten \
  --run-id "$run_id" \
  --run-dir "runs/$run_id" \
  --max-spend-usd 75 \
  --hourly-rate-usd <verified-rate> \
  --confirm
```

The guard defaults to reserving 10 percent of the budget for provisioning and
shutdown; configure this with `--spend-reserve-fraction`. If
you omit both flags, budgeting is disabled; supplying only one is rejected.
Regardless of success, failure, interruption, or a budget stop, the pipeline
automatically closes the trainer and deactivates Baseten resources. If
deactivation cannot be confirmed, the error reports the exact manual cleanup
command: `baseten loops run deactivate --run-id <run-id> --yes`.

This Baseten path does not require Hugging Face, deployment, replay evaluation,
or a separate paid smoke job. These commands document preparation and the
reviewed training workflow; they do not start the complete paid run without
the required final approval.

## 4. Review and run Fireworks training

```bash
run_id=my-qwen-sft

python pipeline.py plan --run-id "$run_id"

python pipeline.py train \
  --run-id "$run_id" \
  --run-dir "runs/$run_id" \
  --confirm
```

Defaults are five epochs, patience-one early stopping, learning rate `1e-4`,
batch size 32, LoRA rank 8, and LoRA alpha 32. System, user, and tool-result
tokens are context. Loss applies to all assistant text and tool calls.

Each epoch resumes from the prior checkpoint. The pipeline measures validation
loss after each epoch and selects the lowest-loss checkpoint.

## 5. Promote and deploy the best Fireworks checkpoint

```bash
python pipeline.py promote \
  --run-dir "runs/$run_id" \
  --output-model-id "$run_id" \
  --confirm

python pipeline.py deploy \
  --run-dir "runs/$run_id" \
  --account-id <fireworks-account-id> \
  --output-model-id "$run_id" \
  --deployment-id "$run_id" \
  --deployment-shape <compatible-fireworks-shape> \
  --confirm
```

Deployment uses the official Fireworks API and must pass a real inference smoke
test before evaluation.

## 6. Run held-out Fireworks replay evaluation

```bash
python pipeline.py eval-plan --output-dir runs/replay-eval

python pipeline.py evaluate \
  --output-dir runs/replay-eval \
  --tuned-model '<full-model-path>#<full-deployment-path>' \
  --confirm
```

This scores the trained model against the recorded held-out messages. To also
measure a before-versus-after delta, deploy the base model and pass its serving
route with `--base-model`.

Evaluation runs four cases at a time by default and saves each completed case.
Use `--concurrency` to change the limit. A stopped run resumes from the same
output directory.

The default judge is `anthropic/claude-sonnet-5`. It calls the Anthropic
Messages API through the LangSmith gateway. Set `ANTHROPIC_API_KEY` to your
LangSmith gateway key. Existing Claude Code gateway setups can use
`ANTHROPIC_CUSTOM_HEADERS`. You can override the model with `--judge-model`.

The evaluator selects every assistant-message boundary by default. Use
`--max-points-per-trajectory` to set an optional cost limit. Each case contains
the full prefix, recorded next message, and immediate recorded tool results.
The judge payload labels the prefix as the only history visible to the candidate
and labels the recorded tool results as future outcomes of the reference action.
The judge may use those future results to assess the reference action, but must
not treat them as prior context or reject a candidate for repeating them.
The selected renderer includes tool declarations when counting the replay
prompt. A case is rejected when its rendered prefix plus the output allowance
exceeds the profile context limit; it is never truncated.
Before scoring, the judge must pass controls for both tool-call and text cases.
The judge uses a plain prompt. Invalid `pass` and `reason` JSON is retried up to
three times.

### Judge example

Suppose the held-out trajectory has this next recorded action:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call-recorded",
      "type": "function",
      "function": {
        "name": "search_docs",
        "arguments": "{\"query\":\"configure retries\"}"
      }
    }
  ]
}
```

The next recorded tool result is:

```json
{
  "role": "tool",
  "tool_call_id": "call-recorded",
  "content": "Retry failed requests with exponential backoff."
}
```

The base or trained model proposes this candidate:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call-new",
      "type": "function",
      "function": {
        "name": "search_docs",
        "arguments": "{\"query\":\"configure retries\"}"
      }
    }
  ]
}
```

The judge receives the full trajectory prefix plus the three objects above. It
does not receive the name of the model that made the candidate. A valid result
is:

```json
{"pass": true, "reason": "The tool and material arguments match the recorded action."}
```

The new tool-call ID does not matter. If the candidate instead calls
`search_docs` with `{"query":"configure authentication"}`, a valid result is:

```json
{"pass": false, "reason": "The query does not match the recorded action."}
```

This score measures agreement with the recorded trajectory. The evaluator does
not run the proposed tool.

For a text case, suppose the recorded assistant message is `Use exponential
backoff for retries.` A candidate that says `Retry with increasing delays.` can
pass because the meaning agrees. A candidate that says `Disable retries.` must
fail because it contradicts the recorded response.

`runs/replay-eval/summary.json` reports base and tuned pass rates, delta, paired
wins, regressions, and ties. This measures agreement with the recorded trace. It
does not execute tools.

For every candidate, deterministic metrics also report:

- Whether the model correctly chose tool call versus text
- Whether the tool-name multiset matches the reference
- Whether every argument string parses as JSON
- Whether every call validates against its captured JSON Schema
- Whether canonical arguments match the recorded calls
- Whether a parallel call set matches independent of call ordering

The summary contains base and tuned rates for each metric and tuned-minus-base
deltas. The LLM judge remains useful for semantic text equivalence and material
argument differences; deterministic schema and call checks do not depend on the
judge. Base and tuned Fireworks requests use the same messages, tools,
generation settings, and output allowance. Only the model route differs.

## 7. Stop deployment billing

```bash
python pipeline.py undeploy \
  --account-id <fireworks-account-id> \
  --deployment-id "$run_id" \
  --confirm
```

## Public Python flow

```python
contract = load_inference_contract(Path("inference_contract.json"))
dataset, examples = download_dataset(workspace_id, dataset_id, raw_dir)
audit = validate_trajectories(examples, dataset["example_count"])
rows = prepare_sft_rows(examples, contract, model=model, reasoning_policy="omit")
accepted, rejected, token_audit = validate_model_context(rows, model)
train, validation, test = split_rows(accepted)
```

`prepare_dataset`, `run_early_stopping`, and `run_replay_evaluation` compose the
same steps for direct Python use.

## Test

```bash
python -m pytest
```

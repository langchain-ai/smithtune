# AGENTS.md

Smithtune prepares LangSmith trajectories for SFT with Fireworks or Baseten.

## Start here

- For operating the CLI, read [README.md](README.md) for setup, commands, and provider support.
- For changing the implementation, also read [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency compatibility, and checks.
- Use the installed CLI for operating tasks; make source changes when the task calls for them.
- Run `smithtune doctor` and the relevant command's `--help` before an unfamiliar workflow. Doctor checks local setup and credential presence, not credential validity or workspace access.

## Choose the starting point

- Tracing project: use `dataset create DIR` to preview the saved workflow, then repeat with `--confirm`. Explicit `--filter` without council criteria skips model calls; `--rule` applies council judging after filtering. Without a filter, create defaults to council review; `--no-triage` explicitly skips it.
- For staged curation: `dataset pull DIR` downloads; `dataset triage DIR` previews local judging, and `--confirm` runs it; `dataset push DIR --name NAME` (or `--dataset-id ID`) previews upload, and `--confirm` uploads. Source flags belong to pull/create. Pass the resulting dataset ID to prepare. See [dataset curation](docs/datasets.md) for council models, rules, and labels; explain the saved counts and reasons after judging.
- Existing dataset: start at `prepare`; reuse `metadata.smithtune_source` per-assistant tool evidence. Unbound exports need source scope/project metadata, stable assistant output-message IDs, and producing LLM runs with recorded tools for automatic capture.
- Prepared data: start at `plan`, then `train` using the same provider and data directory.
- Continue from existing artifacts when they match the task. Ask for missing source IDs rather than guessing them. New dataset pulls and creation default to the last 24 hours when time bounds are omitted; use explicit bounds when the user specifies another window.
- Use LangSmith API filter expressions from the README and linked syntax reference.

## Run and recover

- Repeat customized shared training options, such as `--learning-rate`, on both `plan` and `train`. Plan is a preview and does not carry settings forward.
- Keep an explicit working directory or `--data-dir`; default data paths follow the current directory. Training requires a new or empty `--run-dir`.
- Parse successful command results as JSON and retain returned IDs and artifact paths for the next step.
- Use `dataset resume DIR` to inspect pending work and `dataset resume DIR --confirm` to continue the saved workflow. See [dataset recovery guidance](docs/datasets.md) for older receipts or conflicting remote content.
- `prepare --no-fetch` reuses the raw export and saved tool schemas; it still needs compatible tokenizer dependencies and cache access.
- Preparation publishes dataset splits by default, including with `--no-fetch`. Replay verifies that pinned dataset version before paid work and returns one LangSmith comparison link. `--no-sync-splits` only skips publication during preparation; splits must be synchronized before evaluation. LangSmith publication is required for evaluation to complete.
- Report failures with the relevant example/run IDs and artifact paths. Preserve recorded data and validation while diagnosing the cause.
- Paid training, evaluation, and deployment must be within the user's authorized scope. Honor authorization already given; obtain it before adding `--confirm` for an operation that has not been authorized.
- Use credentials through environment variables; keep their values out of messages, logs, and committed files.
- Report any provisioned deployment and its cleanup command; deployment charges continue until it is removed.
- Fireworks replay always uses the official serverless Training API sampler. Use `train --evaluate` to train and replay in one session, or `eval-plan --run-dir` and `evaluate --run-dir` to restore a completed run's best training checkpoint. Both compare the matching base model and tuned checkpoint. No evaluation deployment is created. Keep the same data, checkpoint, and sampling settings when resuming; generated responses are saved before judging. A promoted model ID alone cannot be sampled.

## Preserve the data behavior

- Triage judges each full saved conversation once per council member. Each trajectory gets one majority label. When a council is attached, import requires completed votes and a keep label for that exact trajectory. Replay splitting belongs to `evaluate`, not triage. Judge errors stay incomplete; they are not quality votes. Do not refetch or edit messages after judging. Use saved per-assistant tool evidence from triage.
- Pull reads full messages through `/v1/trajectory` with system messages enabled and downloads run evidence through the V2 trace-runs endpoint. Completed trajectories and their per-assistant tool evidence are saved in content-verified files for resume; raw API pages and run trees are not retained in new snapshots. Before judging, it filters multimodal content in messages (including supplied history), run inputs/outputs, and media attachments. Filtered whole trajectories get 0 with a reason and no judge calls. Each remaining full trajectory is sent unchanged to each council model. A provider context-window rejection filters the whole trajectory with 0 and a reason; never truncate or split it to fit. Judges return only keep and reason. Upload selected trajectories with `dataset push DIR --confirm`.
- Dataset creation imports whole conversations: a root's thread when it has one, otherwise its single trace. Thread examples include earlier turns and turns outside the selection window.
- Preparation preserves per-assistant tool availability, including additions, removals, and schema/description changes. Capture matches recorded assistant output IDs and content to producing LLM runs; input history is not production evidence. Missing availability is not an empty list. Provider built-ins remain unsupported even when unused. `--inference-contract` explicitly overrides tools globally.
- Canonical dataset/split rows remain whole trajectories. Rendering derives each supported assistant target once with its preceding messages and tools; earlier assistant context receives zero loss. Keep original indices through reasoning conversion and use matching tools in replay. Reject the whole trajectory when a required target cannot render or exceeds context. Older prepared artifacts require preparation again.
- Fireworks and Baseten support deployment and replay evaluation. Baseten deploys saved sampler checkpoints through the optional `baseten-deploy` extra; `undeploy` deactivates only the recorded deployment and preserves its checkpoint. Temporary Baseten evaluation deactivates its owned deployment on exit. Replay compares responses against recorded context without executing tools.
- LangSmith replay experiments group independent next-action predictions by source conversation. `teacher_agreement` is per action; `trajectory_teacher_agreement` is the conversation's mean. Experiment metadata includes the parent smithtune run ID and selected checkpoint epoch when saved training provenance is available. Publish completed action pairs in the background and show the comparison link before generation. Conversation averages include completed judgments only. Upload failures resume from saved predictions and judgments; release owned serving resources before the final publication wait.

## Change the repository

- Follow CONTRIBUTING for Python 3.12, uv, dependency installation, and checks. Use nearby implementations before adding abstractions.
- Keep the README's commands aligned with CLI behavior. Keep the main workflow in the README, detailed usage in its linked guides, and development procedures in CONTRIBUTING.
- Run checks proportional to the change. If local prerequisites are missing, report the limitation and defer those checks to CI unless setup repair is requested.
- Keep credentials, generated datasets, run artifacts, and private planning notes out of commits. Commit only task-related source, tests, and maintained documentation.
- `CLAUDE.md` imports this file; update shared instructions here so both agents receive the same guidance.

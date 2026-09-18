# AGENTS.md

Smithtune prepares LangSmith trajectories for SFT with Fireworks or Baseten.

## Start here

- For operating the CLI, read [README.md](README.md) for setup, commands, and provider support.
- For changing the implementation, also read [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency compatibility, and checks.
- Use the installed CLI for operating tasks; make source changes when the task calls for them.
- Run `smithtune doctor` and the relevant command's `--help` before an unfamiliar workflow. Doctor checks local setup and credential presence, not credential validity or workspace access.

## Choose the starting point

- Tracing project: preview `dataset create DIR` with the intended workspace, project, time window, root filters, and destination. It downloads without judge calls or writes. Use the saved directory with `--confirm` to run the default triage and push; `--no-triage` chooses pull and push only. Pass the returned dataset ID to `prepare`.
- For staged work, use `dataset pull DIR` with source flags, `dataset triage DIR` to preview, `dataset triage DIR --confirm` to judge, then `dataset push DIR --name NAME --confirm`. The default is a Deep Agent coordinator with DeepSeek V4.1 Flash and GLM-5.3-Flash on Fireworks and GPT-5.6 Terra on OpenAI. Set a `--judges` list of aliases or `provider:model`, and `--rule` for project rules. Votes live in `triage.jsonl`; labels, counts, and reasons are derived in command results. Explain them after dispatch. Export the portable skill with `skill export`.
- Existing dataset: start at `prepare`. Examples carry `metadata.smithtune_source.assistant_runs` indexed by original message position. Unbound examples need source workspace/project/scope metadata and stable producing-output message IDs for automatic capture; missing provenance is an exclusion, never an implicit tool union.
- Prepared data: start at `plan`, then `train` using the same provider and data directory.
- Continue from existing artifacts when they match the task. Ask for missing source IDs rather than guessing them. New pulls and dataset creation default to the last 24 hours when time bounds are omitted; use explicit bounds when the user specifies another window.
- Use LangSmith API filter expressions from the README and linked syntax reference.

## Run and recover

- Repeat customized shared training options, such as `--learning-rate`, on both `plan` and `train`. Plan is a preview and does not carry settings forward.
- Keep an explicit working directory or `--data-dir`; default data paths follow the current directory. Training requires a new or empty `--run-dir`.
- Parse successful command results as JSON and retain returned IDs and artifact paths for the next step.
- Resume curation with `dataset resume DIR --confirm`. Keep one checkpoint directory with `checkpoint.json`, `triage.jsonl`, and one content-verified file per conversation. Successful downloads, durable votes, and reconciled writes are reused. Run only one import per destination at a time. Use a fresh checkpoint to change frozen source selection.
- `prepare --no-fetch` reuses the raw export and saved tool schemas; it still needs compatible tokenizer dependencies and cache access.
- Preparation publishes dataset splits by default, including with `--no-fetch`. Retry publication alone with `dataset publish-splits --data-dir DIR`; reserve `prepare --no-fetch` for re-preparation. Set `--validation-fraction`, `--test-fraction`, and `--split-from` explicitly when changing or carrying forward split settings. Replay verifies that pinned dataset version before paid work and returns one LangSmith comparison link. `--no-sync-splits` only skips publication during preparation; splits must be synchronized before evaluation. LangSmith publication is required for evaluation to complete.
- Report failures with the relevant example/run IDs and artifact paths. Preserve recorded data and validation while diagnosing the cause.
- Paid judging, training, evaluation, deployment, and dataset writes must be within the user's authorized scope. Honor authorization already given; obtain it before adding `--confirm` for an operation that has not been authorized.
- Use credentials through environment variables; keep their values out of messages, logs, and committed files.
- Report any provisioned deployment and its cleanup command; deployment charges continue until it is removed.
- Fireworks replay always uses the official serverless Training API sampler. Use `train --evaluate` to train and replay in one session, or `eval-plan --run-dir` and `evaluate --run-dir` to restore a completed run's best training checkpoint. Both compare the matching base model and tuned checkpoint. No evaluation deployment is created. Keep the same data, checkpoint, and sampling settings when resuming; generated responses are saved before judging. A promoted model ID alone cannot be sampled.

## Preserve the data behavior

- Triage judges the entire saved conversation unchanged with bindings as untrusted evidence, once per council member. A keep requires every council slot to complete and a strict majority; ties drop. Judge errors stay incomplete. Provider context rejections exclude the whole conversation without truncation. Do not refetch or edit judged messages or bindings. File changes invalidate votes; coordinator/skill software changes do not.
- Pull reads full messages through `/v1/trajectory` with system messages enabled and continuation cursors, and supporting evidence through V2 trace runs. Group roots into distinct threads/traces before seeded sampling (default 100, seed 42, maximum 2000). Save selection before download. Check thread membership before/after, and filter media in messages, run inputs/outputs, and attachments. Keep runs only in memory; interrupted conversations restart, completed conversations are never fetched again.
- Upload whole conversations, including earlier turns outside the selection window, with `outputs: null` and bindings in metadata. Empty eligibility creates no dataset. Deterministic IDs and pending writes support reconciliation; extensions must preserve exact previous messages and bindings. Never silently upgrade a legacy union or adopt an unrelated dataset with the same name.
- Each assistant message must map unambiguously to its producing LLM run through a unique stable output-message ID and matching content/calls. Supplied input history, trajectory UI run attribution, order, and timestamps cannot prove production. Save all available tools (including unused tools) with that message. Explicit `tools: []` is verified empty; missing availability is an exclusion. Preserve changing schemas and removed tools without merging definitions across turns.
- Preparation consumes embedded bindings without source reads; automatic capture for unbound exports uses the same provenance requirements. A global inference contract is an explicit alternate policy and preserves original evidence. Unsupported tools or invalid calls exclude the whole conversation.
- Canonical split rows remain whole conversations. At training/validation rendering, derive each supported assistant target once with its history and that target's tools; only that target receives loss. Preserve original positions through reasoning conversion. Reject the whole conversation if any required target cannot render or exceeds context. Both providers' actual loaders follow this policy; old prepared formats require re-preparation.
- Replay uses the target binding for both models, candidate validation, and judge evidence. Validate history with its own bindings. Keep parent reference-example IDs, pinned dataset verification, case identities, saved-prediction resume, and conversation-level reporting. A later dataset version does not invalidate a still-readable pinned version.
- Fireworks and Baseten support deployment and replay evaluation. Baseten deploys saved sampler checkpoints through the optional `baseten-deploy` extra; `undeploy` deactivates only the recorded deployment and preserves its checkpoint. Temporary Baseten evaluation deactivates its owned deployment on exit. Replay compares responses against recorded context without executing tools.
- LangSmith replay experiments group independent next-action predictions by source conversation. `teacher_agreement` is per action; `trajectory_teacher_agreement` is the conversation's mean. Experiment metadata includes the parent smithtune run ID and selected checkpoint epoch when saved training provenance is available. Publish completed action pairs in the background and show the comparison link before generation. Conversation averages include completed judgments only. Upload failures resume from saved predictions and judgments; release owned serving resources before the final publication wait.

## Change the repository

- Follow CONTRIBUTING for Python 3.12, uv, dependency installation, and checks. Use nearby implementations before adding abstractions.
- Keep the README's commands aligned with CLI behavior. Keep the main workflow in the README, detailed usage in its linked guides, and development procedures in CONTRIBUTING.
- Run checks proportional to the change. If local prerequisites are missing, report the limitation and defer those checks to CI unless setup repair is requested.
- Keep credentials, generated datasets, run artifacts, and private planning notes out of commits. Commit only task-related source, tests, and maintained documentation.
- `CLAUDE.md` imports this file; update shared instructions here so both agents receive the same guidance.

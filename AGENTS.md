# AGENTS.md

Smithtune prepares LangSmith trajectories for SFT with Fireworks or Baseten.

## Start here

- For operating the CLI, read [README.md](README.md) for setup, commands, and provider support.
- For changing the implementation, also read [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency compatibility, and checks.
- Use the installed CLI for operating tasks; make source changes when the task calls for them.
- Run `smithtune doctor` and the relevant command's `--help` before an unfamiliar workflow. Doctor checks local setup and credential presence, not credential validity or workspace access.

## Choose the starting point

- Tracing project: use `dataset create` with the intended workspace, project, time window, and root-run filters, then pass the returned dataset ID to `prepare`.
- Existing dataset: start at `prepare`; source thread/trace and project information is needed for automatic tool capture.
- Prepared data: start at `plan`, then `train` using the same provider and data directory.
- Continue from existing artifacts when they match the task. Ask for missing source information rather than guessing IDs or a time window.
- Use LangSmith API filter expressions from the README and linked syntax reference.

## Run and recover

- Repeat customized shared training options, such as `--learning-rate`, on both `plan` and `train`. Plan is a preview and does not carry settings forward.
- Keep an explicit working directory or `--data-dir`; default data paths follow the current directory. Training requires a new or empty `--run-dir`.
- Parse successful command results as JSON and retain returned IDs and artifact paths for the next step.
- After a partial dataset import, inspect its receipt and the existing dataset before retrying. Dataset imports have no automatic resume.
- `prepare --no-fetch` reuses the raw export and saved tool schemas; it still needs compatible tokenizer dependencies and cache access.
- Report failures with the relevant example/run IDs and artifact paths. Preserve recorded data and validation while diagnosing the cause.
- Paid training, evaluation, and deployment must be within the user's authorized scope. Honor authorization already given; obtain it before adding `--confirm` for an operation that has not been authorized.
- Use credentials through environment variables; keep their values out of messages, logs, and committed files.
- Report any provisioned deployment and its cleanup command; deployment charges continue until it is removed.

## Preserve the data behavior

- Dataset creation imports whole conversations, including earlier turns and turns outside the selection window.
- Preparation preserves recorded messages and gathers each example's tool union from all its source LLM runs. Automatic capture is the normal path; a global inference contract is an explicit override.
- Preparation combines optional top-level tool arguments when shared arguments and other schema fields match. The expanded definition applies to the whole example. Provider built-ins and incompatible definitions still fail, even when the tools were not called.
- SFT targets all supported assistant messages, including earlier turns. Keep source conversations separate across train, validation, and test splits.
- Fireworks supports deployment and replay evaluation; Baseten currently produces training checkpoints. Replay compares responses against recorded context without executing tools.

## Change the repository

- Follow CONTRIBUTING for Python 3.12, uv, dependency installation, and checks. Use nearby implementations before adding abstractions.
- Keep the README's commands aligned with CLI behavior. Keep detailed usage in the README and development procedures in CONTRIBUTING.
- Run checks proportional to the change. If local prerequisites are missing, report the limitation and defer those checks to CI unless setup repair is requested.
- Keep credentials, generated datasets, run artifacts, and private planning notes out of commits. Commit only task-related source, tests, and maintained documentation.
- `CLAUDE.md` imports this file; update shared instructions here so both agents receive the same guidance.

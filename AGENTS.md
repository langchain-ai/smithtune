# AGENTS.md

Smithtune prepares LangSmith trajectories for SFT with Fireworks or Baseten.

## Start here

- For operating the CLI, read [README.md](README.md) for setup, commands, and provider support.
- For changing the implementation, also read [CONTRIBUTING.md](CONTRIBUTING.md) for development, dependency compatibility, and checks.
- Use the installed CLI for operating tasks; make source changes when the task calls for them.
- Run `smithtune doctor` and the relevant command's `--help` before an unfamiliar workflow. Doctor checks local setup and credential presence, not credential validity or workspace access.

## Choose the starting point

- Tracing project: use `dataset create` with the intended workspace, project, time window, and root-run filters, then pass the returned dataset ID to `prepare`.
- To label full trajectories for SFT, preview with `dataset triage <directory>` and source flags. Then use `dataset triage <directory> --confirm` to label or resume with saved settings. The default is a Deep Agent coordinator and DeepSeek V4.1 Flash and GLM-5.3-Flash judge subagents on Fireworks, plus GPT-5.6 Terra on OpenAI, with Python code mode. Set models with one `--judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra` list; other models use `provider:model`. Use `--rule` for project rules. Each line in `labels.jsonl` has only `trajectory_id`, `keep` (1 or 0), and `reason`. Explain the counts and reasons to the user after dispatch. Detailed votes stay in `judgments.jsonl`. Import selected whole conversations with `dataset create --triage-dir`. Export the portable skill with `skill export`.
- Existing dataset: start at `prepare`; each example's metadata needs `source_scope`, `source_scope_id`, and `source_project_id` for automatic tool capture.
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
- For a promoted Fireworks model, prefer `eval-plan` and `evaluate --serving-mode preemptible` for temporary evaluation. Use the same model, account, shape, deployment ID, and limits on resume. Check the deployment receipt after an interruption or cleanup failure. This path uses the official Fireworks REST API.

## Preserve the data behavior

- Triage judges each full saved conversation once per council member. Each trajectory gets one majority label. Import requires completed council votes and a keep label for that exact conversation. Replay splitting belongs to `evaluate`, not triage. Judge errors stay incomplete; they are not quality votes. Do not refetch or edit messages after judging. Use saved tool schemas from triage.
- Triage downloads run evidence through the V2 trace-runs endpoint. Completed read responses are cached for download resume. Before judging, it filters multimodal content in messages (including supplied history), run inputs/outputs, and media attachments. Filtered whole trajectories get 0 with a reason and no judge calls. Each remaining full trajectory is sent unchanged to each council model. A provider context-window rejection filters the whole trajectory with 0 and a reason; never truncate or split it to fit. Judges return only keep and reason. Upload selected text conversations with `dataset create --triage-dir`.
- Dataset creation imports whole conversations: a root's thread when it has one, otherwise its single trace. Thread examples include earlier turns and turns outside the selection window.
- Preparation preserves recorded messages and gathers each example's tool union from all its source LLM runs. Automatic capture is the normal path; a global inference contract is an explicit override.
- Preparation combines tools by name, keeps the latest description by source run timestamp (run ID breaks ties), and combines optional top-level arguments when shared arguments and other schema fields match. Description replacements are reported in `prepared/tool_description_replacements.json`. The combined definition applies to the whole example. Provider built-ins and incompatible definitions still fail, even when the tools were not called.
- SFT targets all supported assistant messages, including earlier turns. Keep source conversations separate across train, validation, and test splits.
- Fireworks supports deployment and replay evaluation; Baseten currently produces training checkpoints. Replay compares responses against recorded context without executing tools.

## Change the repository

- Follow CONTRIBUTING for Python 3.12, uv, dependency installation, and checks. Use nearby implementations before adding abstractions.
- Keep the README's commands aligned with CLI behavior. Keep detailed usage in the README and development procedures in CONTRIBUTING.
- Run checks proportional to the change. If local prerequisites are missing, report the limitation and defer those checks to CI unless setup repair is requested.
- Keep credentials, generated datasets, run artifacts, and private planning notes out of commits. Commit only task-related source, tests, and maintained documentation.
- `CLAUDE.md` imports this file; update shared instructions here so both agents receive the same guidance.

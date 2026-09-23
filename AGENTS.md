# AGENTS.md

Smithtune prepares LangSmith trajectories for SFT with Fireworks or Baseten.

## Start here

- To operate smithtune (curate a dataset, prepare, train, evaluate, deploy), follow the
  [smithtune skill](src/smithtune/skills/smithtune/SKILL.md). It is the single source of
  operating guidance for agents; keep it aligned with CLI behavior.
- To change the implementation, read [CONTRIBUTING.md](CONTRIBUTING.md) for development,
  dependency compatibility, and checks, and preserve the behavior below.
- `src/smithtune/triage_prompts/` holds internal prompts for the council inside
  `dataset triage`. They are not operator guidance; smithtune ships no default rubric.

## Preserve the data behavior

- Triage reviews whole trajectories until the cumulative approved target is reached. Each reviewed trajectory requires one vote per council member and receives one majority label. When a council is attached, import requires completed votes and a keep label for that exact trajectory. Replay splitting belongs to `evaluate`, not triage. Judge errors stay incomplete; they are not quality votes. Do not refetch or edit messages after judging. Use saved per-assistant tool evidence from triage.
- Pull reads messages, per-assistant `available_tools`, and run/trace metadata through `/v1/trajectory` in `ui` format with system messages enabled. Completed trajectories and their per-assistant tool evidence are saved in content-verified files for resume; raw API pages and run trees are not retained in new snapshots. Before judging, it filters multimodal content in the returned messages (including supplied history). Filtered whole trajectories get 0 with a reason and no judge calls. Each trajectory selected for review is sent unchanged to each council model. A provider context-window rejection filters the whole trajectory with 0 and a reason; never truncate or split it to fit. Judges return only keep and reason. Upload selected trajectories with `dataset push DIR --confirm`.
- Dataset pull downloads whole trajectories: a root's thread when it has one, otherwise its single trace. Thread examples include earlier turns and turns outside the selection window.
- Preparation preserves per-assistant tool availability, including additions, removals, and schema/description changes. Availability is read directly from each assistant trajectory item; no output-ID matching or raw-run fetching is needed. Missing availability is not an empty list. Provider built-ins remain unsupported even when unused. `--inference-contract` explicitly overrides tools globally.
- Canonical dataset/split rows remain whole trajectories. Rendering derives each supported assistant target once with its preceding messages and tools; earlier assistant context receives zero loss. Keep original indices through reasoning conversion and use matching tools in replay. Reject the whole trajectory when a required target cannot render or exceeds context. Older prepared artifacts require preparation again.
- Fireworks and Baseten support deployment and replay evaluation. Baseten deploys saved sampler checkpoints through the optional `baseten-deploy` extra; `undeploy` deactivates only the recorded deployment and preserves its checkpoint. Temporary Baseten evaluation deactivates its owned deployment on exit. Replay compares responses against recorded context without executing tools.
- LangSmith replay experiments group independent next-action predictions by source conversation. `teacher_agreement` is per action; `trajectory_teacher_agreement` is the conversation's mean. Experiment metadata includes the parent smithtune run ID and selected checkpoint epoch when saved training provenance is available. Publish completed action pairs in the background and show the comparison link before generation. Conversation averages include completed judgments only. Upload failures resume from saved predictions and judgments; release owned serving resources before the final publication wait.

## Change the repository

- Follow CONTRIBUTING for Python 3.12, uv, dependency installation, and checks. Use nearby implementations before adding abstractions.
- Keep the README's commands and the smithtune skill aligned with CLI behavior. Keep setup and the main workflow in the README, detailed usage in its linked guides, agent operating steps in the skill, and development procedures in CONTRIBUTING.
- Run checks proportional to the change. If local prerequisites are missing, report the limitation and defer those checks to CI unless setup repair is requested.
- Keep credentials, generated datasets, run artifacts, and private planning notes out of commits. Commit only task-related source, tests, and maintained documentation.
- `CLAUDE.md` imports this file; update shared instructions here so both agents receive the same guidance.

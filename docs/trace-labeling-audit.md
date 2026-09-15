# Trace labeling audit — 2026-09-15

The CLI flow works with the default mixed council: download real LangSmith
traces, judge them, import an accepted conversation, and prepare it for SFT.
The audit found and fixed provider compatibility, evidence citation, default
configuration, and recovery issues. Label accuracy has not been calibrated
against a human-reviewed reference set.

## Default council

| Role | Provider | API model ID |
| --- | --- | --- |
| Coordinator and judge 1 | Fireworks | `accounts/fireworks/models/deepseek-v4p1-flash` |
| Judge 2 | Fireworks | `accounts/fireworks/models/glm-5p3-flash` |
| Judge 3 | OpenAI | `gpt-5.6-terra` |

All three IDs were returned by the providers' authenticated model catalogs.
Each model also completed a live code-tool round trip. Fireworks uses its
official inference endpoint. Terra uses OpenAI Responses with medium reasoning
and `store=false`; encrypted reasoning state stays in the local agent context.

## Scope and findings

| Area | Finding and result |
| --- | --- |
| Command surface | One labeling command: `dataset triage <directory>`. Without `--confirm`, it downloads/previews. With it, it labels/resumes. Source flags are needed only for the first download. One comma-separated `--judges` list sets models; `--rule` adds project rules. The three default models have short names; other models use `provider:model`. Older advanced flags remain accepted but hidden from normal help. |
| Result surface | `labels.jsonl` has only `trace_id`, `keep` (1/0), and `reason` per trace. Reasons combine the votes for the final label. `report.md` explains every label; the command ends with counts and the result path. Detailed votes stay in `judgments.jsonl`. Incomplete rows use 0 with an explicit retry reason, and the command exits 1. |
| Defaults | Fixed three different definitions of the default council. The CLI and Python helper now load the packaged config; skill export copies the same file. Existing plans retain their selected models. `models list` remains the training-model registry, not a judge catalog. |
| Provider compatibility | Live Terra tool calls failed on Chat Completions with reasoning. Routing Terra through Responses fixed this. The generic OpenAI adapter dropped Fireworks `reasoning_content`; a small adapter now preserves it between tool calls. Streaming is disabled so that this preservation path is always used. No new dependencies were added. |
| Evidence citations | Live judges sometimes counted message indexes incorrectly. One Terra result cited index 14 when both exact quotes were in message 10 of an 11-message conversation. Judge inputs now carry explicit indexes. Retry prompts give validation feedback. Source messages and exact-quote checks are unchanged. |
| Source capture | Root selection uses `POST /api/v2/runs/query`; run trees use `GET /api/v2/traces/{trace_id}/runs`. Whole threads expand beyond the source window; standalone traces are supported. Successful reads are cached for resume. Messages, run trees, and tool schemas are frozen locally. Empty queries fail clearly. Empty turns remain available for judging. A missing root is marked in the evidence and blocks conversation import. |
| Media filter | Before scheduling any votes, the CLI checks messages (including supplied history), run inputs/outputs, and media attachments. Filtered traces get 0 with a reason and no judge calls. Their conversations cannot enter training. Text that merely mentions media is not filtered. |
| Long messages | Large messages have marked previews and a `read_message(index)` reference. Judges can page the full original content through read-only Python, alongside `read_run(id)`. Full evidence stays in the snapshot; no summary replaces it. |
| Agent boundaries | The coordinator dispatches planned trace/judge pairs through bounded Python batches. Each judge has fresh context and read-only run access. Code has no host filesystem, environment, shell, or network access. The CLI validates and saves votes; coordinator prose cannot assign labels. |
| Recovery | Missing credentials no longer write the paid-run identity before work starts. Failed votes record safe failure categories and retry counts. Completed runs need no agent runtime or provider keys. A partial live resume preserved all five completed votes and retried only the remaining judge. |
| Aggregation | All configured votes are required; strict majority keeps, ties drop. Missing evidence remains incomplete. Mixed-quality or incomplete conversations cannot enter training through a passing neighbor. |
| Dataset handoff | Live import and `prepare` passed. Downloaded messages and prepared tool schemas matched the frozen source exactly. Each accepted conversation is one dataset example. Labels remain local; no tracing feedback is written. |
| Packaging | The wheel and source archive include all six triage modules, the skill, rubric, and default config. Clean installs test the default package and optional agent runtime outside the checkout. Discovery and help do not import the agent or training runtimes or use the network. |

## Live evidence

The final mixed-council tests used three real traces in two conversations.
No verdict was edited to force acceptance.

| Check | Observed result |
| --- | --- |
| One complete conversation | Three valid votes; two keep and one drop. The strict-majority label kept it. Judging completed in 32.53 seconds, including one successful citation retry. |
| A conversation with an interrupted turn | Five valid votes and one explicit insufficient-evidence result. The CLI exited 1 and excluded the conversation. Judging took 44.61 seconds. |
| Partial resume | Reused all five completed votes unchanged. Retried only GLM's incomplete vote, which again reported missing evidence. No attempt was made to convert that abstention into a quality vote. |
| Completed resume | Passed with Fireworks, OpenAI, and LangSmith keys removed and PATH limited to the virtual environment. Snapshot, votes, labels, and agent state were unchanged. |
| Live dataset and preparation | Imported the accepted conversation and prepared it for Fireworks Qwen3.8-27B: 11 messages, four tool calls, four tool results, no removed messages. Messages and tool schemas matched the snapshot. |
| Simple model list and output | A fresh CLI download with `--judges deepseek-v4.1-flash,glm-5.3-flash,gpt-5.6-terra` completed all three votes in 34.47 seconds. Votes were 1, 1, 0. The label file contained only `trace_id`, `keep`, and `reason`. Resume without keys left source, votes, labels, and agent state unchanged. Live dataset import passed with the compact labels and preserved the saved messages. |

The one-conversation preparation test used zero validation/test fractions.
It checks the handoff, not a training evaluation. No training or deployment
was started. Private source data, provider responses, and test artifacts are
excluded from Git.

Local checks passed: 781 tests, with 11 optional tokenizer tests skipped;
Ruff; skill validation; and clean distribution checks. The distribution checks
passed 758 default wheel tests (13 skipped), 81 optional agent tests, and 23
tests of the wheel rebuilt from the source archive. They also cover
dependency compatibility, isolated CLI installation, and skill
export. CI also tests the pinned public tokenizers on Linux and macOS.

## Remaining limits

- **Quality needs calibration.** The judges disagreed on all three sampled
  traces among the valid votes. Exact quotes prove that cited text exists;
  they do not prove that a verdict follows from it. This small test cannot
  estimate precision or recall. Review a representative labeled sample and
  define project rules for interrupted or tool-only traces before a large run.
- **Installation is still heavy.** The optional `deepagents` extra adds the
  agent runtime, but the base package also installs the existing Fireworks
  and Baseten training stacks. The development environment with agent and test
  extras was about 1.2 GB on macOS. A labeling-only dependency split would be
  a separate package-contract change. The base training stack also retains the
  documented Transformers advisory and upstream version constraint described
  in [CONTRIBUTING](../CONTRIBUTING.md#dependency-compatibility). The LangSmith
  CLI remains an external prerequisite for download and dataset import.
- **Large-project throughput is not proven.** Root selection reads all matching
  result pages before seeded sampling. Successful source responses are cached
  for download resume. The snapshot stays in memory and the vote file is
  atomically rewritten on each save. Page and trace caps bound work, but there
  is no high-volume load test. Narrow the source window/filter for large projects.
- **Costs are bounded, not totaled.** The plan lists task counts, concurrency,
  retries, and limits. It does not provide a complete token/cost ledger across
  all failed and successful model calls. The coordinator adds inference cost.
- **Upgrade and import recovery are conservative.** Changed evidence, rubric,
  agent behavior, or model settings require a new run directory once judging
  has started. Dataset imports have receipts but no automatic resume; inspect
  a partial import before retrying. Windows is not in the supported test matrix.

Sources: [DeepSeek V4.1 Flash announcement](https://www.deepseek.com/en/news/deepseek-v4-1-flash/),
[OpenAI Terra model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-terra),
and [Fireworks reasoning documentation](https://docs.fireworks.ai/guides/reasoning).

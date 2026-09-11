# Dataset Curation Implementation Plan

> **For agentic workers:** Use the `executing-plans` skill to implement this plan task by task. Track completion with the checkboxes below.

**Goal:** Select traces from a LangSmith project, create a trajectory dataset, and feed it into the existing SFT `prepare` command.

**Architecture:** Extend the existing argument parser with two dataset commands. Put selection and import orchestration in one new `curation.py` module; reuse the installed LangSmith CLI for authentication and API requests. Make the existing dataset consumer accept real trace and thread source identities.

**Tech stack:** Python 3.12, argparse, standard library, existing `langsmith` CLI and pytest; no new dependency.

**Spec:** [Dataset Curation](https://app.notion.com/p/3d7808527b178094ad8fec7fc8a648e1) and the concrete v1 decisions below.

## Scope and commands

Work in `/Users/jake/Documents/ChatGPT/LangChain __ Baseten/smithtune`. At planning time it is clean on `fix/transformers-path-traversal` (`b4067bd`). Implementation should use a fresh `jake/dataset-curation` branch/worktree from the current main, preserving the unrelated dependency fix.

Keep the current invocation; adding an installed `smithtune` entry point is deferred.

```bash
uv run python pipeline.py dataset select \
  --workspace-id "$WORKSPACE_ID" --project-id "$PROJECT_ID" \
  --start-time 2026-09-01T00:00:00Z --end-time 2026-09-08T00:00:00Z \
  --filter 'and(eq(feedback_key, "correctness"), gte(feedback_score, 0.9))' \
  --scope thread --limit 100 --seed 42 \
  --output data/selection.json

uv run python pipeline.py dataset create \
  --selection data/selection.json --name chat-langchain-sft
```

- `select`: workspace ID, project ID, start/end, scope and output are required. `--filter` is optional raw LangSmith syntax; `--limit` is optional and positive; `--seed` defaults to 42. Timestamps require a timezone; the window is start-inclusive and end-exclusive, applied to root-run **start time**.
- `--scope trace`: one example per matching trace. `--scope thread`: one example per distinct containing thread, including its full trajectory outside the selection window. Exclude roots without a thread ID from thread scope and report their count.
- Query all matching roots, deduplicate by final scope, then sample. With no limit, keep all. Sort candidates before seeded sampling so page order cannot change the result.
- `select` prints counts and up to 20 matching roots with IDs, thread IDs, feedback statistics, and selection status. The file contains the complete selection and matching-root provenance. Zero matches saves an empty selection; `create` rejects it before any write.
- `create` uses the saved workspace/project/IDs without rerunning filters. It creates a **new dataset**; an existing name is an error. IDs are fixed, but source messages can change before import. This is not a content snapshot.
- Preserve trajectory messages as returned, including system messages when present. Do not require system prompts, compare prompts, rewrite messages, truncate, or change SFT training targets.
- First release targets deployments with the current run-query, trajectory, and thread-import APIs. No old-server fallback, cross-workspace import, append mode, automatic resume, or provider changes.

## Files

| File | Change |
| --- | --- |
| `curation.py` (new) | Root selection, selection JSON, preview, trace relay and native thread import |
| `pipeline.py` | Nested `dataset select` / `dataset create` parser and dispatch |
| `artifacts.py` | Allow `_run` to send JSON through standard input |
| `dataset.py` | Preserve source fields on download; accept and split trace/thread sources |
| `rendering.py`, `evaluation.py` | Include trace identity in rejection and replay provenance |
| `tests/test_curation.py` (new), `tests/test_pipeline.py` | Selection/import behavior and preparation compatibility |
| `README.md` | Short select → create → prepare walkthrough and failure behavior |

## Task 1: Select and save source IDs

**Files:** `curation.py`, `artifacts.py`, `tests/test_curation.py`.

**Interface:** `select_dataset(*, workspace_id: str, project_id: str, start_time: str, end_time: str, scope: str, output: Path, filter: str | None = None, limit: int | None = None, seed: int = 42, runner=_run) -> dict` returns the printable preview and writes the selection file.

- [x] Extend `_run(command, *, capture=False, input: str | None = None)` to forward `input` to `subprocess.run`. Existing callers remain unchanged. Inside `curation.py`, use one private `_api(workspace_id, method, path, body=None, *, runner=_run)` helper that parses JSON and raises `PipelineError` for command/response failures. Send bodies via stdin, never command arguments or temporary message files:

  ```python
  command = ["langsmith", "api", path, "--workspace", workspace_id, "--method", method]
  if body is not None:
      command += ["--input", "-"]
  result = runner(command, capture=True,
                  input=json.dumps(body) if body is not None else None)
  ```

- [x] Query `POST /api/v2/runs/query` with `project_ids: [project_id]`, `is_root: true`, `min_start_time`, `max_start_time`, `page_size: 100`, and `selects: ["ID", "TRACE_ID", "THREAD_ID", "START_TIME", "FEEDBACK_STATS"]`. Combine the user filter with `lt(start_time, <JSON-encoded end timestamp>)` to enforce the exclusive upper bound. Follow `next_cursor` over `items` until exhausted; reject a repeated cursor or malformed page. Never apply `--limit` to the root query.
- [x] Read the real returned `thread_id`; do not derive one from a trace ID. Deduplicate roots by trace ID, then candidates by trace or thread scope. Build a sorted candidate list; use `random.Random(seed).sample(candidates, min(limit, len(candidates)))` only when a limit is supplied, and sort the selected result for stable output.
- [x] Save with existing JSON helpers. Reject an existing output path rather than overwriting a reviewed selection. Use this versioned shape (illustrative IDs):

  ```json
  {
    "schema_version": 1,
    "created_at_utc": "2026-09-10T18:00:00+00:00",
    "workspace_id": "workspace-uuid",
    "project_id": "project-uuid",
    "scope": "thread",
    "query": {
      "start_time": "2026-09-01T00:00:00Z",
      "end_time": "2026-09-08T00:00:00Z",
      "filter": "and(eq(feedback_key, \"correctness\"), gte(feedback_score, 0.9))",
      "limit": 100,
      "seed": 42
    },
    "matches": [
      {"trace_id": "trace-uuid", "thread_id": "conversation-1",
       "start_time": "2026-09-02T12:00:00Z", "feedback_stats": {}}
    ],
    "selected_ids": ["conversation-1"]
  }
  ```

  Store no messages or credentials. Preview counts distinguish matching roots, distinct threads, excluded unthreaded roots, eligible examples and selected examples. Display the returned feedback statistics without inventing a single score from multiple feedback entries.
- [x] Add fake-runner tests for cursor pagination, unchanged raw filter semantics/root restriction, exclusive end time, missing thread IDs, empty matches, no-limit behavior, invalid arguments and refusal to overwrite. Prove deduplication precedes sampling: roots A/B in thread 1 and C in thread 2 yield two examples in thread scope and three in trace scope. With a limit, reversing pages yields the same selected IDs.
- [x] Run `uv run --no-sync python -m pytest tests/test_curation.py -q` once after the changes; resolve code failures before continuing.

## Task 2: Create the dataset from the selection

**Files:** `curation.py`, `tests/test_curation.py`.

**Interface:** `create_dataset(*, selection: Path, name: str, runner=_run) -> dict` returns the dataset ID, confirmed example count and receipt path.

- [x] Load and validate version 1, workspace/project IDs, scope, unique selected IDs, and their membership in the saved matches. Reject an empty selection. Do not query source runs. Create a plain dataset with `POST /api/v1/datasets`, body `{"name": name, "data_type": "kv"}`; do not attach transformations or a custom schema.
- [x] Process selected IDs sequentially, **one example per request** for v1. Common example metadata is `trajectory_format: "messages"`, `conversation_scope: "root"`, `source_project_id`, and `selection_scope`. Here `root` describes the conversation projection, while `selection_scope` records trace versus thread.
- [x] For trace scope, call `POST /v1/trajectory` with:

  ```python
  body = {"project_id": project_id, "trace_id": trace_id,
          "format": "messages", "include": {"system_messages": True}}
  ```

  Require a nonempty `messages` list. Trace-ID requests currently return one complete page; fail on unexpected continuation cursors rather than silently importing a partial result. Pass the message objects unchanged to `POST /api/v1/examples` as `{"dataset_id": dataset_id, "inputs": {"messages": messages}, "outputs": None, "metadata": metadata}`. Add `source_trace_id` to metadata and the actual `source_thread_id` when the selected root has one. Hold only the current trajectory in memory; never write its contents during curation.
- [x] For thread scope, call `POST /v1/platform/datasets/{dataset_id}/examples/thread-imports` with `{"project_id": project_id, "thread_ids": [thread_id], "metadata": common_metadata}`. The server retrieves the full trajectory and sets the dedicated `source_thread_id` field. Require `count == 1` and one returned example ID. Do not fetch messages locally. Surface API size/unavailable-source errors without rewriting or skipping examples.
- [x] Write a small receipt beside the selection (`selection.import.json`): selection path, dataset name/ID, `status`, confirmed example IDs, and the current source ID. Refuse an existing receipt. Write `status: "in_progress"` before the first remote write, update after each confirmed success, and use `complete` only when all selected IDs succeeded. An interrupted receipt remains visibly incomplete.
- [x] Stop on the first failure and report the receipt, dataset ID when known, confirmed count and current source ID. Distinguish confirmed successes from the current write, whose outcome can be unknown after a timeout or invalid response. Do not retry writes, delete the partial dataset, or resume into it. A new attempt uses a reviewed new selection path and dataset name; the user can inspect or remove the previous dataset in LangSmith. Dataset-creation timeouts report the requested name even if no ID was returned.
- [x] Test both API paths with a recording runner: `create` makes no run queries; trace messages including absent/varying system prompts remain identical; thread import makes no trajectory download; metadata and output shape are correct. Check existing-name rejection, missing/empty trajectory, unexpected cursors, a failure after one success, and an ambiguous write that is called only once. Ensure receipts contain no message bodies.
- [x] Run `uv run --no-sync python -m pytest tests/test_curation.py -q` after this task's changes.

## Task 3: Make both dataset sources work with preparation

**Files:** `dataset.py`, `rendering.py`, `evaluation.py`, `tests/test_pipeline.py`.

Native imports set a top-level `source_thread_id`. The current `langsmith example list` JSON output discards that field, and smithtune currently requires it inside metadata. Trace-only examples also need a real trace identity instead of a fabricated thread ID.

- [x] Change `_langsmith_page_command` to call raw `GET /api/v1/examples?dataset=<id>&limit=<limit>&offset=<offset>` through `langsmith api`, with the explicit workspace. Preserve full returned example objects, existing pagination/count checks, and existing raw artifact names. Keep the separate dataset export behavior unchanged.
- [x] Add `_source_identity(example) -> dict` in `dataset.py`: resolve `source_thread_id` from the dedicated field or legacy metadata; resolve `source_trace_id` from metadata. Reject conflicting nonempty thread identities and require at least one source identity. Keep `trajectory_format`, `conversation_scope`, message and tool-pair validation as they are.
- [x] In `prepare_sft_rows`, carry both source fields into `_source`, using `None` for an absent identity. Add `_source_group(row) -> tuple[str, str]` and use it in both splitting and isolation checks:

  ```python
  source = row["_source"]
  if source.get("source_thread_id"):
      return ("thread", source["source_thread_id"])
  return ("trace", source["source_trace_id"])
  ```

  Keep the existing hash input `f"{SPLIT_SEED}:{id}"` for thread groups so existing thread datasets retain their split assignment. Hash standalone traces with `f"{SPLIT_SEED}:trace:{id}"`; keep typed group keys in sets. Update insufficient-group errors and the manifest description to explain the trace fallback. Keep the legacy manifest method for thread-only datasets.
- [x] Include `source_trace_id` alongside the nullable thread ID in rendering rejection reports and replay cases. Leave conversion, reasoning policy, training targets and split fractions unchanged.
- [x] Extend tests with native top-level thread sources, legacy metadata sources, standalone traces, conflicting identities and traces sharing a real thread. Prove shared-thread rows stay together, standalone traces form separate groups, reversed input stays deterministic, and legacy split assignment is unchanged. Exercise the raw-download → validation → preparation chain with native import fixtures, plus trace provenance in rejection/replay output.
- [x] Run `uv run --no-sync python -m pytest tests/test_pipeline.py -q` after these changes.

## Task 4: Expose the commands and verify the workflow

**Files:** `pipeline.py`, `README.md`, `tests/test_curation.py`.

- [x] Add nested `dataset` subparsers with exactly the flags above; dispatch to `curation.select_dataset` / `curation.create_dataset` and retain JSON output and `PipelineError` handling. No provider settings belong on these commands.
- [x] Add parser/dispatch tests for required flags, explicit scope, defaults and passing the saved selection to creation. Add a fixture-backed select → create → download → prepare test for each scope. The resulting examples must pass the existing preparation contract; use enough independent source groups for the requested split and mock renderer/network boundaries.
- [x] Add a short README walkthrough ending with `uv run python pipeline.py prepare --workspace-id "$WORKSPACE_ID" --dataset-id "$DATASET_ID"`, retaining existing inference-contract requirements for tool datasets. Explain full-thread scope, local trace relay, fixed IDs versus mutable content, partial receipts, and current all-assistant-message training in a few sentences.
- [x] Run the complete existing suite once: `uv run --no-sync python -m pytest`. Check `uv run --no-sync python pipeline.py dataset select --help` and `dataset create --help`.
- [x] In a configured development workspace, smoke-test each scope using a few approved traces and a new dataset, then run existing `prepare`. Verify imported system/tool messages and source identities. Do not launch training. If credentials, CLI or dependencies are missing, report that live verification is deferred; do not install or repair the environment as part of this task.

**Done:** Both scopes complete select → create → existing prepare; selections are inspectable and reproducible from saved IDs; failed imports are visible; existing thread datasets still prepare identically.

## Verified references and planning limits

- [LangSmith CLI raw request bodies](https://github.com/langchain-ai/langsmith-cli/blob/main/internal/cmd/api/request.go): supports `--input -`; the raw client currently has a 30-second timeout and no application-level retry loop.
- [Run-query contract](https://github.com/langchain-ai/langchainplus/blob/main/smith-go/runs/v2/query/types.go): root filtering, selected fields and cursor pagination.
- [Thread importer](https://github.com/langchain-ai/langchainplus/blob/main/smith-go/examples/thread_examples.go): server-side full trajectories, dedicated source field, fresh example IDs; no request idempotency key. One-thread requests avoid introducing batching policy in v1.
- [System-message support](https://github.com/langchain-ai/langchainplus/pull/37841): explicit inclusion for trace relay; messages format includes systems by default.

Source contracts were inspected on September 10, 2026. At planning time, the LangSmith CLI was not on PATH; it has now been installed and exercised live. Implementation completed on `jake/dataset-curation`, based on current main `f90dc82` (which includes the dependency fix). Verification: 268 tests passed, both command help checks passed, and independent code review found no actionable issues. Live verification subsequently passed with LangSmith CLI v0.2.54; see [the verification report](../../verification/2026-09-10-dataset-curation.md) for source comparisons, preparation results and the existing tool-contract limitation.

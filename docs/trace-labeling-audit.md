# Trajectory council audit

`dataset triage` selects whole training conversations. `evaluate` scores model
responses through replay. These are separate flows.

## Current path

1. Pull groups distinct threads/traces before seeded selection, freezes source
   IDs, and fetches full trajectory pages with system messages.
2. V2 supporting runs are inspected in memory for media and producing-output
   identities. A unique stable output-message ID plus matching content/calls
   binds each assistant to its producing run and explicit recorded tools.
3. Each unchanged conversation and its bindings are saved together. Media,
   malformed structures, unsupported tools, and unverifiable mappings are
   terminal exclusions. Source service failures remain retryable incomplete work.
4. Triage dispatches missing conversation/judge pairs from local files, through
   a fresh Deep Agent coordinator with Python code mode or the direct runner.
   Judges receive full messages and bindings as untrusted evidence, no executable
   tools, and return only `keep` (0/1) and `reason`.
5. Every slot must succeed before a strict majority can keep a conversation;
   ties drop. Provider context rejection excludes it without truncation. Other
   request failures remain incomplete.
6. Sequential push uploads one whole example with bindings and evidence-bound
   triage provenance. Deterministic IDs and pending writes support reconciliation.

`checkpoint.json` and `triage.jsonl` are the only bookkeeping files. Completed
conversations and votes resume without source reads or repeated durable judgments.
Hashes cover messages and bindings; there are no raw caches, retained run trees,
software/skill identity gates, or coordinator state files. A lost unsaved response
may require another paid call. Code upgrades do not promise bit-for-bit judging
reproducibility.

## Verification and limitations

Offline regressions cover selection, source mismatch, interrupted download/write,
changed threads, media/history checks, vote reuse, ties, context rejection, and
exact messages/bindings through upload and fresh export. The roundtrip test
continues through preparation, pinned split publication/verification, both
providers' actual sampling adapters with fake services, schema scoring, judge
evidence, parent reference IDs, child provenance, and publication-only resume.
Installed renderer interfaces test per-target tools and zero loss on history;
pinned real-tokenizer checks are opt-in. Optional Deep Agents/Monty tests exercise
the real graph with deterministic local models when that extra is installed.

Mapping support comes from inspected API schemas and backend fixtures, reproduced
as synthetic test evidence; this change has no live source/write/inference smoke.
UI run attribution alone does not prove production because it also attributes
supplied input history to consumers. Missing or ambiguous stable output identities
and missing explicit tool availability are excluded, not guessed. Instrumentation
must supply those fields before such conversations can be used automatically.
Thread membership checks do not atomically freeze all live run payloads.

Earlier live council checks used the retired union/snapshot format. Their counts
and outcomes do not validate this new storage or per-assistant pipeline. Label
accuracy still needs a human-reviewed sample; valid JSON and completed votes do
not establish the correctness of a model's quality decision.

# Contributing

Use Python 3.12, uv, and Git. The Fireworks cookbook is a direct Git dependency,
pinned to a full upstream commit in `pyproject.toml`. No source snapshot or fork
is maintained here, and there is no bootstrap step.

Provider-specific training, sampling, and deployment modules live under
`src/smithtune/providers/`. Shared rendering stays in `src/smithtune/`; replay orchestration lives in
`src/smithtune/evaluation/replay.py`.

`evaluation/langsmith.py` publishes versioned splits, saved replay predictions,
and judge feedback through the LangSmith SDK. Keep run and feedback IDs stable
across upload retries. Verify snapshots and create the experiments before paid
replay work. Publish saved action pairs in a separate worker; indexing and retries
must not block inference. Keep conversation outputs and aggregate scores current
as actions finish, and leave immutable children unchanged. Release owned serving
resources before the final publication wait. Bounded publisher shutdown must
retain its output lock until any in-flight request returns. Tests marked
`sdk_integration` use an in-memory service boundary; other unit tests mock LangSmith I/O. Production
evaluation always verifies its dataset snapshot and publishes results.
Cover both sampler providers, endpoint cleanup, and saved-generation recovery
when changing this integration.

`providers/fireworks_training.py` keeps one serverless session across epochs and optional
replay. It uses the pinned cookbook's rendering, data loader, validation,
optimizer, and checkpoint helpers. `providers/fireworks_sampling.py` uses the official
Training API sampler and the same renderer for replay. When changing these
adapters, check checkpoint selection, session cleanup, tool parsing, and replay
recovery from saved generations.

`providers/baseten_sampling.py` uses the Loops sampler REST endpoint to save resource IDs
before the SDK readiness wait, then samples through `baseten-loops`. Closing the
SDK client does not release GPUs: deactivate each owned deployment explicitly.
`providers/baseten_sampling_formats.py` parses the pinned official model formats without a
Fireworks renderer dependency. Test native-tokenizer roundtrips, malformed tool
calls, best-checkpoint identity, cleanup failures, and cached-generation resume
when changing either adapter. These offline checks do not replace a paid sampler
smoke test.

```bash
sfw uv sync --locked --extra test --python 3.12
uv run --no-sync smithtune --help
uv run --no-sync pytest
uv run --no-sync ruff check
```

For the optional Deep Agents judge runner:

```bash
sfw uv sync --locked --extra test --extra deepagents --python 3.12
uv run --no-sync pytest tests/test_triage_agent.py tests/test_triage_coordinator.py tests/test_triage.py
```

Deep Agents, its OpenAI adapter, and Monty are pinned in the optional `deepagents` extra.
Tests use the actual coordinator graph with a deterministic local model. They
check skill loading, Python dispatch, bounded concurrency, one full-message
request per judge, provider context rejection, and resume.
Monty is the Pydantic project's MIT-licensed Python sandbox. Version 0.0.23
was checked against its source, PyPI metadata, and OSV on 2026-09-14; no published
advisories were returned. Code gets no host mounts or OS handlers. Only the
reviewed trace/task functions cross the sandbox boundary. The coordinator can dispatch bounded judge batches. Judges receive full
conversation messages and have no tools.
Provider transport tests replace HTTP requests at the service boundary; no
test uses paid inference or creates a live deployment.

The default council is read from the packaged `config.example.json`; CLI,
Python, and exported skill defaults must agree. Terra uses OpenAI Responses
with `reasoning.effort=none`; Fireworks calls set `reasoning_effort=none`.
GLM-5.3-Flash requires reasoning, so selecting it uses `low`; the request policy
records that exception.
Keep the request checks when changing either transport. Durable votes resume from
saved rubric, models, request settings, and conversation hashes. Coordinator and
skill revisions do not gate resume; do not add software identity checks or an
agent-state file.

`sfw` is used for contributor dependency installation. It is not a smithtune runtime
prerequisite. Companion CLIs and provider credentials are only needed for live
operations; the automated tests do not provision training or deployments.

CI runs `ruff check` with the rules in `pyproject.toml`; it does not enforce
`ruff format`. Run `uv run --no-sync ruff check --fix` to apply the safe autofixes.

To install your checkout as an isolated CLI:

```bash
sfw uv tool install --python 3.12 --overrides overrides.txt .
```

For Baseten checkpoint deployment, install the optional `baseten-deploy` extra.
It pins Truss 0.18.30 because the adapter uses its internal request builder and
raw creation response to save deployment IDs before parsing generated config.
Distribution checks exercise the released builder with mocked provider I/O.
When updating Truss, verify request shape, exact GPU allocation, error logging,
and receipt persistence. No model/GPU serving profile has been validated by
these offline tests; deployment runs text and tool-call smoke tests before
marking an endpoint ready.

## Dependency compatibility

Transformers is pinned to the patched `5.10.4`. The upstream Fireworks cookbook
still requires `5.5.4`, and its Tinker cookbook dependency caps Transformers at
`5.5.4`, so `overrides.txt` and the matching `tool.uv.override-dependencies`
setting replace those constraints without patching or copying upstream source.
Customers must pass the override file to `uv tool install`; contributor sync uses
the project setting automatically. Tests verify the installed upstream Git
revision and Transformers version, permit only these two recorded metadata
conflicts, and compare renderer outputs to a pinned reference using a
deterministic test tokenizer. Remove both overrides when upstream supports the
patched version.

Baseten uses Transformers' assistant-token masks and TRL `1.13.0` training
templates. Keep TRL pinned: its template and mask conventions are part of the
prepared-data identity. Native annotated templates pass through unchanged;
unrecognized unannotated templates fail preparation. The model support registry
remains separate from upstream template coverage.

The additional Baseten models use `native_rendering.py`: it calls the official
formatter and verifies each target response against its inference prompt. The
training boundary renders each assistant target separately with zero history loss. Keep its implementation version in the prepared
identity when changing formatting or masks. Test model-specific stop tokens,
empty reasoning, history changes, and Unicode against the pinned real tokenizers.

CI also downloads the exact tokenizer revisions for each supported provider/model
configuration and checks tools, reasoning, and loss masks without provisioning
training or downloading model weights. Run those checks locally with:

```bash
SMITHTUNE_TOKENIZER_TESTS=1 uv run --no-sync pytest tests/test_tokenizer_integration.py tests/test_baseten_sampling_formats.py
```

The ordinary test suite uses synthetic tokenizers and requires no Hub access.

Transformers `5.10.4` includes the fix for
[CVE-2026-9856](https://osv.dev/vulnerability/GHSA-xrqw-3rrv-vx5w). Version
`5.10.0`, the first fixed release, was withdrawn from PyPI; keep the tested,
non-yanked patch release unless compatibility checks establish a newer version.

To update the cookbook, change its commit in `pyproject.toml`, inspect the upstream
diff, run `sfw uv lock`, and run the distribution checks below. Retain the renderer
reference unless a reviewed upstream behavior change requires updating it. Security
minimums for Pillow and Datasets are declared in smithtune's dependencies.

## Distribution checks

```bash
python scripts/check_dist.py
```

This builds smithtune's wheel and source distribution, installs dependencies from
their declared sources in clean environments, and runs tests outside the checkout.
It also tests `uv tool install` and rebuilding the wheel from the source archive.
The default wheel is tested first, then the optional Deep Agents extra. Both
distributions must include the portable skill and support `skill export` outside
the checkout.

CI performs these checks on Linux x86-64 and macOS ARM64 with Python 3.12. Windows
and other Python versions are not yet part of the supported test matrix. CI saves
the smithtune artifacts and matching `overrides.txt` for inspection.

## Releases from GitHub

Merge the changes and let CI pass. Set the version in `pyproject.toml`, regenerate
`uv.lock`, and tag the tested commit (for example, `v0.1.0`). Customers install that
tag directly:

```bash
uv tool install --python 3.12 \
  --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/v0.1.0/overrides.txt \
  'git+https://github.com/langchain-ai/smithtune.git@v0.1.0'
```

The example tag must be created before this command works. Repeat the command with
the next tag to upgrade; `smithtune --version` reads the installed version metadata.
No PyPI projects, publishing environments, or second-package releases are needed.
PyPI does not accept the direct Git dependency in smithtune's package metadata, so
these distributions are intended for GitHub/direct installation.

## Curation and target bindings

Use `checkpoint.py` for small curation bookkeeping and one command-level lock.
Keep each unchanged conversation and its per-assistant bindings in one file;
source run trees and API pages stay in memory. `bindings.py` is shared by source
capture, preparation, and replay. Missing provenance is not an empty tool set.
See `tests/fixtures/README.md` for the inspected wire schemas and conservative
output-identity mapping. Never use run order, timestamps, or input-history IDs
as producing-run evidence.

Prepared version 2 keeps parent conversations in split files. Both actual provider
training/validation loaders derive target-only datums at the rendering boundary.
When changing rendering, test changing/removed/empty tools and verify previous
assistant turns receive zero loss. `test_assistant_bindings.py` checks installed
renderer interfaces; `test_curation_roundtrip.py` crosses fake upload/export,
fresh prepare, pinned split publication, both samplers, scoring, reporting, and
publication-only resume. Real tokenizers remain the opt-in checks above.
Run `git diff --check` with pytest and Ruff before submitting changes.

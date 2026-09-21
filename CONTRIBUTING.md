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
must not block inference. Upload finished children and their feedback as actions
finish. Publish each conversation parent and its aggregate score only once all
selected actions are complete, rather than updating a partial parent. Release owned serving
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
GLM-5.3-Flash requires reasoning, so selecting it uses `low`; the saved plan
records that exception.
Keep the request checks when changing either transport. Bump the triage agent version
when model transport or evidence presentation changes.

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

`bindings.py` normalizes each trajectory UI item's `available_tools` and run/trace
metadata into the saved per-assistant tool representation. Training,
validation, and replay must consume the same tool list for each assistant target.
The Fireworks loader uses smithtune's target renderer with the official cookbook's
JSONL dataset and batching; changing only preparation masks is insufficient.
Regression coverage includes actual loader masks, tool additions/removals/schema
changes, local upload/prepare roundtrips, and capture checkpoint recovery.

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
formatter, verifies each response against its inference prompt, and coalesces
only identical token prefixes. The training boundary requests each final assistant
target separately, with history loss masked out. Keep its implementation version in the prepared
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

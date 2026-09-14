# Contributing

Use Python 3.12, uv, and Git. The Fireworks cookbook is a direct Git dependency,
pinned to a full upstream commit in `pyproject.toml`. No source snapshot or fork
is maintained here, and there is no bootstrap step.

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
Tests use the actual agent graph with a deterministic local model. They check
coordinator skill loading, Python sandbox execution, subagent dispatch, bounded
concurrency, fresh judge context, full evidence retention, and resume.
Monty is the Pydantic project's MIT-licensed Python sandbox. Version 0.0.23
was checked against its source, PyPI metadata, and OSV on 2026-09-14; no published
advisories were returned. Code gets no host mounts or OS handlers. Only the
three reviewed trace/task functions cross the sandbox boundary.
Provider transport tests replace HTTP requests at the service boundary; no
test uses paid inference or creates a live deployment.

`sfw` is used for contributor dependency installation. It is not a smithtune runtime
prerequisite. Companion CLIs and provider credentials are only needed for live
operations; the automated tests do not provision training or deployments.

CI runs `ruff check` with the rules in `pyproject.toml`; it does not enforce
`ruff format`. Run `uv run --no-sync ruff check --fix` to apply the safe autofixes.

To install your checkout as an isolated CLI:

```bash
sfw uv tool install --python 3.12 .
```

## Dependency compatibility

Transformers is pinned to `5.5.4`, matching the upstream Fireworks cookbook and
satisfying its Tinker cookbook dependency. No overrides or source patches are needed.
Distribution checks run `uv pip check`; tests verify the installed upstream Git
revision and Transformers version, and compare renderer outputs to a pinned
reference using a deterministic test tokenizer.

Baseten uses Transformers' assistant-token masks and TRL `1.13.0` training
templates. Keep TRL pinned: its template and mask conventions are part of the
prepared-data identity. Native annotated templates pass through unchanged;
unrecognized unannotated templates fail preparation. The model support registry
remains separate from upstream template coverage.

The additional Baseten models use `native_rendering.py`: it calls the official
formatter, verifies each response against its inference prompt, and coalesces
only identical token prefixes. Keep its implementation version in the prepared
identity when changing formatting or masks. Test model-specific stop tokens,
empty reasoning, history changes, and Unicode against the pinned real tokenizers.

CI also downloads the exact tokenizer revisions for each supported provider/model
configuration and checks tools, reasoning, and loss masks without provisioning
training or downloading model weights. Run those checks locally with:

```bash
SMITHTUNE_TOKENIZER_TESTS=1 uv run --no-sync pytest tests/test_tokenizer_integration.py
```

The ordinary test suite uses synthetic tokenizers and requires no Hub access.

Known security tradeoff: `5.5.4` is affected by
[CVE-2026-9856](https://osv.dev/vulnerability/GHSA-xrqw-3rrv-vx5w), fixed in
Transformers `5.10.0`. Malicious chat-template dictionary keys can cause arbitrary
file writes when tokenizer/processor `save_pretrained()` is called. No explicit
calls were found in smithtune or the installed Fireworks/Tinker cookbook code;
this is not a proof of unreachability. Reassess this choice when upstream permits
a fixed version or tokenizer loading/saving paths change.

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
the smithtune artifacts for inspection.

## Releases from GitHub

Merge the changes and let CI pass. Set the version in `pyproject.toml`, regenerate
`uv.lock`, and tag the tested commit (for example, `v0.1.0`). Customers install that
tag directly:

```bash
uv tool install --python 3.12 \
  'git+https://github.com/langchain-ai/smithtune.git@v0.1.0'
```

The example tag must be created before this command works. Repeat the command with
the next tag to upgrade; `smithtune --version` reads the installed version metadata.
No PyPI projects, publishing environments, or second-package releases are needed.
PyPI does not accept the direct Git dependency in smithtune's package metadata, so
these distributions are intended for GitHub/direct installation.

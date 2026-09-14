# Contributing

Use Python 3.12, uv, and Git. The Fireworks cookbook is a direct Git dependency,
pinned to a full upstream commit in `pyproject.toml`. No source snapshot or fork
is maintained here, and there is no bootstrap step.

```bash
sfw uv sync --locked --extra test --python 3.12
uv run --no-sync smithtune --help
uv run --no-sync pytest
```

`sfw` is used for contributor dependency installation. It is not a smithtune runtime
prerequisite. Companion CLIs and provider credentials are only needed for live
operations; the automated tests do not provision training or deployments.

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

# Contributing

Use Python 3.12 and uv. This repository is a workspace containing the CLI and
`packages/training-runtime`, a separately published dependency. There is no
bootstrap step or external source checkout.

```bash
sfw uv sync --locked --extra test --python 3.12
uv run --no-sync smithtune --help
uv run --no-sync pytest
```

`sfw` is used for contributor dependency installation. It is not a smithtune runtime
prerequisite. Companion CLIs and provider credentials are only needed for live
operations; the automated tests do not provision training or deployments.

## Distribution checks

Build both wheels and source distributions:

```bash
sfw uv build --all-packages
```

CI installs the resulting wheels into a clean environment and runs the tests
outside the checkout, then rebuilds from the source distributions. This catches
missing package resources, accidental imports from the checkout, and unpublished
or local-path dependencies. The runtime's `training` namespace must not coexist
with an independently installed `fireworks-training-cookbook` distribution.

## Releases

The `Release` workflow is manually dispatched from the desired release commit.
Its default is build-and-test only. Enable publishing after the two PyPI projects
and GitHub environments have been configured:

- PyPI projects: `smithtune-training-runtime` and `smithtune`.
- Trusted publishers: this repository, workflow `release.yml`, environments
  `pypi-runtime` and `pypi-smithtune`, respectively.
- Protect both GitHub environments with the required release reviewers.

The workflow publishes the runtime first, waits until its exact version is
available from PyPI, verifies smithtune resolves without workspace sources, and
then publishes smithtune. An unchanged runtime version may already be published;
its existing artifacts must match the build's package metadata and source content
before it can be reused. Failures before CLI publication can be rerun. If the CLI
version has already been uploaded (including a partial upload), bump its version
for a new release; the workflow will not silently skip an existing CLI artifact.

Version changes belong in each package's `pyproject.toml`; when the runtime changes,
also update smithtune's exact runtime dependency and regenerate `uv.lock` with
`sfw uv lock`. `smithtune --version` reads the installed distribution metadata.

Runtime provenance and update instructions are in
[`packages/training-runtime/README.md`](packages/training-runtime/README.md).
Keep renderer equivalence checks and the upstream source checksum manifest current.

The supported release test targets are Linux x86-64 and macOS ARM64 on Python 3.12.
Windows and other Python versions are not yet part of the release support matrix.

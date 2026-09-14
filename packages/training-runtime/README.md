# smithtune-training-runtime

The separately versioned, LangChain-maintained training dependency of
[smithtune](https://github.com/langchain-ai/smithtune). Customers install
`smithtune`; this dependency is installed automatically.

This package preserves the `training` import namespace from Fireworks cookbook
revision `09bbe1170c804eb5e5d7e7286f0feeef4e26055a`. It provides model renderers,
tokenizer utilities, and training recipes. It must not be installed alongside
`fireworks-training-cookbook`, which owns the same import namespace. The
recommended isolated `uv tool install smithtune` environment avoids this conflict.

The runtime includes substantial dependencies, including PyTorch. Training
runs on the provider; customers do not need a local GPU. Preparing data may
download the selected model's tokenizer into the Hugging Face cache.

See `NOTICE`, `LICENSE`, and `upstream.json` for provenance. The latter records
the upstream archive SHA-256 and checksums of imported source/resources. Source
code is preserved with normalized trailing newlines, except that the remaining
`tinker_cookbook.supervised.common` import is routed to a vendored copy of the same
0.4.3 helper (which uses the already-vendored exception types). This avoids that
package's conflicting Transformers requirement. Dependency metadata fixes
the upstream Transformers pin to the version previously selected by smithtune's
uv override, so ordinary wheel installations receive the same requirement.

## Maintenance

Treat upstream updates as explicit source changes: choose a commit, inspect its
diff and dependency changes, preserve third-party notices, update the source
snapshot and `upstream.json`, bump this package's version and smithtune's exact
requirement, regenerate the workspace lock, and run the installation and renderer
tests. Do not add network fetches to package builds or customer startup.

Publish this distribution before publishing a smithtune release that depends on
it. Both distributions are built and tested by the repository release workflow.

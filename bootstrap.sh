#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cookbook_commit="09bbe1170c804eb5e5d7e7286f0feeef4e26055a"
cd "$root"

command -v git >/dev/null
command -v sfw >/dev/null
command -v uv >/dev/null
command -v langsmith >/dev/null
command -v firectl >/dev/null

if [[ ! -d fireworks-cookbook/.git ]]; then
  mkdir -p fireworks-cookbook
  git -C fireworks-cookbook init
  git -C fireworks-cookbook remote add origin https://github.com/fw-ai/cookbook.git
fi

if [[ -n "$(git -C fireworks-cookbook status --short)" ]]; then
  echo "fireworks-cookbook has local changes; refusing to switch revisions" >&2
  exit 1
fi

git -C fireworks-cookbook fetch --depth 1 origin "$cookbook_commit"
git -C fireworks-cookbook checkout --detach "$cookbook_commit"
sfw uv sync --extra test --extra baseten --python 3.12

echo "Ready. Run: source .venv/bin/activate"

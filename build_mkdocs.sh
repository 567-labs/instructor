#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  pipx install uv
fi

uv sync --python 3.13 --extra docs
uv run mkdocs build

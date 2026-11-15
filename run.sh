#!/bin/bash
# Starts Fabric Fitting Room. Needs uv (https://docs.astral.sh/uv/): brew install uv
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:  brew install uv"
  exit 1
fi
exec uv run --python 3.12 server.py

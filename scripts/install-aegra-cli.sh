#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required to install aegra-cli. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

VERSION_FILE="$(dirname "$0")/../aegra-cli-version.txt"
PIN="$(tr -d '[:space:]' < "$VERSION_FILE")"
PIN="${PIN#v}"

uv tool install --force "aegra-cli==${PIN}" \
  --with langchain-openai \
  --with langchain-ollama \
  --with langchain-mcp-adapters \
  --with langfuse

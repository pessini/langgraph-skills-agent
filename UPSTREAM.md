# Upstream Runtime Pin

This repository runs on upstream **Aegra** through its official CLI rather than vendoring the source.

- Source project: <https://github.com/ibbybuilds/aegra>
- CLI pin: see [`aegra-cli-version.txt`](./aegra-cli-version.txt)

## Installation

Install the CLI from the pinned tag:

```bash
# Recommended — via make:
make backend-cli-install

# Manual via uv:
uv tool install "aegra-cli==$(cat aegra-cli-version.txt | tr -d 'v[:space:]')" \
  --with langchain-openai \
  --with langchain-mcp-adapters \
  --with langfuse
```

## Local Overlay Policy

1. **Do not vendor upstream Aegra source in this repository.** The Aegra
   FastAPI server, migrations, auth scaffolding, and database layer all live in
   the installed CLI.
2. Local code is scoped to:
   - `graphs/` — the agent graph(s) and shared `core/` helpers
   - `aegra.json` — graph + (optional) auth registration
   - `tests/` — unit tests for the user-owned graphs only
3. Upgrade process:
   - bump only `aegra-cli-version.txt`,
   - re-run `make backend-cli-install`,
   - run `make test`.

If you ever need to patch Aegra internals, do it upstream and bump the pin —
don't fork the runtime in this repo.

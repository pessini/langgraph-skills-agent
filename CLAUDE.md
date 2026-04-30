# CLAUDE.md

Guidance for Claude Code in this repository.

## What this repo is

A thin overlay on upstream [Aegra](https://github.com/ibbybuilds/aegra). The FastAPI server, persistence, auth, and migrations live in `aegra-cli` (installed at the version pinned in the `Makefile`). **Do not vendor upstream Aegra source here** — patch upstream and bump `AEGRA_CLI_VERSION`.

## Layout

```
aegra.json           graph registration consumed by aegra-cli
graphs/core/         reusable BaseAgent (LLM retry, tool-call pairing)
graphs/skills_agent/ demo agent + progressive-disclosure skills
tests/               unit tests for graphs/ only
```

Imports use top-level `core.*` / `skills_agent.*` because `tests/conftest.py` adds `graphs/` to `sys.path` and `aegra.json` declares `"dependencies": ["graphs"]` for the runtime.

## Commands

```bash
make install   # install pinned aegra-cli + dev deps
make dev       # run aegra dev (manages Postgres, hot-reloads)
make test      # pytest
```

## Out of scope here

HTTP routes, middleware, SSE, auth, Alembic, observability registration, custom-routes mounting — all upstream. If a task needs one, upstream the change in `ibbybuilds/aegra` and bump the pin.

## Tests

Stay isolated: no Postgres, no real LLM, no `aegra-cli`. Mock the LLM and any network calls.

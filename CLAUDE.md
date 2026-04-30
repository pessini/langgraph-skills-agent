# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

`langgraph-skills-agent` is a thin overlay on top of upstream [Aegra](https://github.com/ibbybuilds/aegra). It ships the demo agent for the Medium article *"Stop Stuffing Your System Prompt: Build Scalable Agent Skills in LangGraph."*

**This repo does NOT vendor Aegra.** The FastAPI server, persistence layer, auth, and migrations live in the `aegra-cli` package installed at the pinned version in [`aegra-cli-version.txt`](./aegra-cli-version.txt). See [`UPSTREAM.md`](./UPSTREAM.md) for the upgrade policy.

## Layout

```
graphs/
  core/             reusable BaseAgent (LLM retry, tool-call pairing, error injection)
  skills_agent/     demo agent + progressive-disclosure skills
aegra.json          graph registration consumed by aegra-cli
tests/              unit tests for graphs/ only
```

Imports inside `graphs/` and `tests/` use top-level names (`from core...`, `from skills_agent...`) because:
- `tests/conftest.py` adds `graphs/` to `sys.path`
- `pyproject.toml` declares `where = ["graphs"]` for editable installs
- `aegra.json` declares `"dependencies": ["graphs"]` for the runtime

## Common commands

```bash
# Install the pinned aegra-cli runtime (one-time / after a pin bump)
make backend-cli-install

# Run the dev server (manages Postgres, hot-reloads graphs)
make backend-up

# Tests
uv run pytest

# Lint / format
uv run ruff check .
uv run ruff format .
```

## What lives upstream (do not recreate locally)

- HTTP server, routes, middleware, SSE, auth scaffolding
- Alembic migrations, database manager, connection pooling
- Observability provider registration
- Custom-routes mounting via `aegra.json` `http.app`

If a task requires changing any of the above, the right move is to upstream it in `ibbybuilds/aegra` and bump `aegra-cli-version.txt` here — **not** to vendor source back into this repo.

## What lives here (safe to change)

- The graph definitions under `graphs/skills_agent/`
- Reusable agent helpers under `graphs/core/`
- Skill packs under `graphs/skills_agent/skills/`
- Unit tests for both of the above
- `aegra.json` graph registration

## Testing notes

- Tests must stay isolated: no Postgres, no real LLM, no `aegra-cli`. Mock the LLM and any network calls.
- The `unit` marker is the only one we keep — drop integration/e2e tiers; those belonged to the vendored aegra and now live upstream.

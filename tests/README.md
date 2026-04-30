# Tests

Unit tests for the user-owned graphs (`graphs/core/` and `graphs/skills_agent/`).

The Aegra runtime itself is installed as a pinned tool (`aegra-cli`) and is **not** vendored into this repo, so its tests live upstream — see `UPSTREAM.md`.

## Running

```bash
uv run pytest
```

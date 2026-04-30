# Agent Skills for Production LangGraph Agents

Demo from the Medium article: [Stop Stuffing Your System Prompt: Build Scalable Agent Skills in LangGraph](https://medium.com/@pessini/stop-stuffing-your-system-prompt-build-scalable-agent-skills-in-langgraph-a9856378e8f6).

Demonstrates progressive knowledge loading, skill-based domain modularization, and tool-driven skill activation.

## How it runs

This repo is a thin overlay on top of [Aegra](https://github.com/ibbybuilds/aegra) — the FastAPI server, persistence, and migrations are installed via `aegra-cli` (pinned in the `Makefile`). The local code is just:

```
aegra.json          # graph registration
graphs/core/        # reusable BaseAgent
graphs/skills_agent/ # the demo agent + skills
tests/              # unit tests for graphs/
```

To upgrade Aegra: bump `AEGRA_CLI_VERSION` in the `Makefile` and run `make install`. Don't vendor upstream source here — patch upstream and bump the pin instead.

## Quick start

```bash
make install
cp .env.example .env  # set OPENAI_API_KEY
make dev              # runs on http://localhost:4242
```

## Tests

```bash
make test
```

## Credits

- [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- [Aegra](https://github.com/ibbybuilds/aegra) by Muhammad Ibrahim
- [n8n Skills Repository](https://github.com/haunchen/n8n-skills/) by haunchen

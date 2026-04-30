# Agent Skills for Production LangGraph Agents

This repository contains the example implementation from the Medium article:

Link: [Stop Stuffing Your System Prompt: Build Scalable Agent Skills in LangGraph](https://medium.com/@pessini/stop-stuffing-your-system-prompt-build-scalable-agent-skills-in-langgraph-a9856378e8f6)

It contains example skill definitions and supporting files used to demonstrate:

- Progressive knowledge loading
- Skill-based domain modularization
- Tool-driven skill activation
- Production-focused LangGraph patterns

## How it runs

This repo is a **thin overlay on top of [Aegra](https://github.com/ibbybuilds/aegra)**: the FastAPI server, persistence layer, and migrations all live in the upstream `aegra-cli` package, installed at a pinned version. The local code is intentionally narrow:

```
.
├── aegra.json               # graph registration
├── aegra-cli-version.txt    # pinned upstream runtime
├── graphs/
│   ├── core/                # reusable BaseAgent
│   └── skills_agent/        # the demo agent + skills
├── tests/                   # unit tests for graphs/
├── Dockerfile               # installs aegra-cli, runs `aegra serve`
├── Makefile
└── UPSTREAM.md              # upstream pin + upgrade policy
```

See [`UPSTREAM.md`](./UPSTREAM.md) for the upgrade policy.

## Quick Start

```bash
# 1. Install the pinned Aegra CLI
make backend-cli-install

# 2. Configure your environment
cp .env.example .env
# edit .env and set OPENAI_API_KEY (or your local Ollama settings)

# 3. Run the dev server (manages its own Postgres)
make backend-up
```

The agent will be available on <http://localhost:4242>.

## Tests

```bash
uv sync
uv run pytest
```

## References and Credits

This project builds on and references the following work:

- **LangGraph** by LangChain — <https://github.com/langchain-ai/langgraph>
- **Aegra** — Open-source LangGraph platform by Muhammad Ibrahim — <https://github.com/ibbybuilds/aegra>
- **n8n Skills Repository** by haunchen — <https://github.com/haunchen/n8n-skills/>

All credit for those projects belongs to their respective authors and contributors.

## Purpose

The goal of this repository is to demonstrate how domain expertise can live outside the system prompt and be loaded only when required.

It is intended for experimentation and reference.

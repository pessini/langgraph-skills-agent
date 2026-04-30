# Agent Skills for Production LangGraph Agents

Demo from the Medium article: [Stop Stuffing Your System Prompt: Build Scalable Agent Skills in LangGraph](https://medium.com/@pessini/stop-stuffing-your-system-prompt-build-scalable-agent-skills-in-langgraph-a9856378e8f6).

Demonstrates progressive knowledge loading, skill-based domain modularization, and tool-driven skill activation.

> **Note:** This code has evolved beyond the version published with the article. Notable upgrades:
>
> - Introduced a reusable async `BaseAgent` (`graphs/core/`) with LLM retry classification (transient vs permanent), tool-call pairing safety, and Langfuse-managed prompts.
> - Split `skills_agent` into typed state + slim nodes, replacing the original monolithic `utils/nodes.py`.
> - Stopped vendoring the Aegra runtime — the FastAPI server, persistence, and migrations are now installed via `aegra-cli`, keeping only the graphs and tests in this repo.
>
> The Agent Skills concepts in the article still apply; the surrounding implementation has been hardened and simplified.

## How it runs

This repo is a thin overlay on top of [Aegra](https://github.com/ibbybuilds/aegra) — the FastAPI server, persistence, and migrations are installed via `aegra-cli`. The local code is just:

```text
aegra.json          # graph registration
graphs/core/        # reusable BaseAgent
graphs/skills_agent/ # the demo agent + skills
tests/              # unit tests for graphs/
```

`make install` pulls the latest `aegra-cli`. The repo was last verified against `0.7.2` (see the comment in `pyproject.toml`). Don't vendor upstream source here — patch upstream and re-run `make install` instead.

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

# Agent Skills for Production LangGraph Agents

Demo from the Medium article: [Stop Stuffing Your System Prompt: Build Scalable Agent Skills in LangGraph](https://medium.com/@pessini/stop-stuffing-your-system-prompt-build-scalable-agent-skills-in-langgraph-a9856378e8f6).

Demonstrates progressive knowledge loading, skill-based domain modularization, and tool-driven skill activation.

> **Note:** This code has evolved beyond the version published with the article. Notable upgrades:
>
> - Introduced a reusable async `BaseAgent` (`graphs/core/`) with LLM retry classification (transient vs permanent), tool-call pairing safety, and Langfuse-managed prompts.
> - Split `skills_agent` into typed state + slim nodes, replacing the original monolithic `utils/nodes.py`.
> - The original article described running the agent through Aegra; this repo now runs directly on LangGraph's local dev server for agent testing, with only the graph code and tests kept here.
>
> The Agent Skills concepts in the article still apply; the surrounding implementation has been hardened and simplified.

## How it runs

This repo is a LangGraph application. The local code is:

```text
langgraph.json      # LangGraph dev/server configuration
graphs/core/        # reusable BaseAgent
graphs/skills_agent/ # the demo agent + skills
tests/              # unit tests for graphs/
```

`langgraph.json` registers `skills_agent` from `./graphs/skills_agent/agent.py:graph`
and loads environment variables from `./.env`. The graph code remains pure
LangGraph: `StateGraph`, typed state, runtime context, tool nodes, and
interrupt/resume behavior live under `graphs/`.

## Quick start

```bash
make install
cp .env.example .env  # set OPENAI_API_KEY, or use Ollama defaults
make dev              # runs on http://127.0.0.1:2024
```

LangGraph Studio is available while the dev server is running:

```text
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

## Local smoke checks

With `make dev` running, confirm the graph is registered:

```bash
curl -s -X POST http://127.0.0.1:2024/assistants/search \
  -H 'content-type: application/json' \
  -d '{}'
```

The response should include `"graph_id":"skills_agent"`. For endpoints that
accept a graph ID, use `skills_agent` directly:

```bash
curl -s http://127.0.0.1:2024/assistants/skills_agent/graph
```

Some endpoints require the assistant UUID returned by `/assistants/search`
instead of the graph ID, such as `/assistants/{assistant_id}/schemas`.

To run a live LLM smoke test through the SDK:

```bash
uv run --with langgraph-sdk python - <<'PY'
import asyncio
from langgraph_sdk import get_client


async def main():
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()
    result = await client.runs.wait(
        thread["thread_id"],
        "skills_agent",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello and name one skill you can load.",
                }
            ]
        },
    )
    print(result["messages"][-1]["content"])


asyncio.run(main())
PY
```

## Tests

```bash
make test
```

## Credits

- [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain
- [n8n Skills Repository](https://github.com/haunchen/n8n-skills/) by haunchen

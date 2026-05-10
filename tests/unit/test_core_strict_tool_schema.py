"""Unit tests for the ``strict_tool_schema`` flag on ``BaseAgent``.

Verifies that ``call_llm`` applies per-tool strict mode based on each
tool's ``metadata["strict_schema_compatible"]`` flag (so a single bad
external-MCP schema cannot poison every OpenAI request with a 400),
that the flag is *not* applied when disabled (so providers like Ollama
whose ``bind_tools`` doesn't accept the kwarg keep working), and that
a callable form is re-evaluated on every call (so a mid-thread
provider mutation is reflected immediately).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.base_agent import BaseAgent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


@tool
def strict_compatible_tool(x: str) -> str:
    """A tool whose schema is a strict subset."""
    return x


strict_compatible_tool.metadata = {"strict_schema_compatible": True}


@tool
def opaque_tool(x: str) -> str:
    """A tool that has not opted into strict (e.g. external MCP)."""
    return x


@tool
def nested_meta_tool(x: str) -> str:
    """A tool whose strict opt-in lives under metadata['_meta'].

    This is the shape ``langchain-mcp-adapters`` produces when an MCP
    server tool is declared with ``@mcp.tool(meta={...})`` — the
    server-side meta dict is nested under ``_meta`` rather than merged
    at the top level.
    """
    return x


nested_meta_tool.metadata = {"_meta": {"strict_schema_compatible": True}}


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    return model


def test_call_llm_applies_strict_per_tool_when_enabled() -> None:
    """When strict_tool_schema is on, each tool gets ``strict`` based on
    its own ``metadata["strict_schema_compatible"]``.  External MCP
    tools (no metadata) default off so a bad schema can't 400 the
    whole request.
    """
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[strict_compatible_tool, opaque_tool],
        strict_tool_schema=True,
    )

    asyncio.run(agent.call_llm([], system_prompt=""))

    fake_model.bind_tools.assert_called_once()
    args, kwargs = fake_model.bind_tools.call_args
    # Tools are passed as a list of OpenAI tool dicts (not BaseTool).
    [tool_dicts] = (args or (kwargs.get("tools"),))
    by_name = {td["function"]["name"]: td["function"] for td in tool_dicts}
    assert by_name["strict_compatible_tool"]["strict"] is True
    # Opaque tool keeps strict=False.
    assert by_name["opaque_tool"].get("strict") is not True


def test_call_llm_honors_strict_flag_nested_under_meta() -> None:
    """A tool that opts into strict via ``metadata["_meta"]["strict_schema_compatible"]``
    (the shape ``langchain-mcp-adapters`` produces from FastMCP's
    ``@mcp.tool(meta={...})`` argument) must be bound with ``strict=True``,
    not silently treated as opaque.
    """
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[nested_meta_tool],
        strict_tool_schema=True,
    )

    asyncio.run(agent.call_llm([], system_prompt=""))

    fake_model.bind_tools.assert_called_once()
    args, kwargs = fake_model.bind_tools.call_args
    [tool_dicts] = (args or (kwargs.get("tools"),))
    [td] = tool_dicts
    assert td["function"]["name"] == "nested_meta_tool"
    assert td["function"]["strict"] is True


def test_call_llm_does_not_pass_strict_when_disabled() -> None:
    """Default behavior: bind_tools called with the BaseTool list and no
    strict kwarg, so Ollama and other providers whose ``bind_tools``
    doesn't accept ``strict`` still work.
    """
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[strict_compatible_tool],
    )

    asyncio.run(agent.call_llm([], system_prompt=""))

    fake_model.bind_tools.assert_called_once()
    args, kwargs = fake_model.bind_tools.call_args
    assert args[0] == [strict_compatible_tool]
    assert "strict" not in kwargs


def test_call_llm_evaluates_callable_strict_per_call() -> None:
    """A callable ``strict_tool_schema`` must be re-evaluated on every
    ``call_llm`` so a provider mutation flips strict immediately —
    matching the dynamic-capture pattern of ``model_factory``.
    """
    state = {"strict": True}
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[strict_compatible_tool],
        strict_tool_schema=lambda: state["strict"],
    )

    # Initial call: strict path.
    asyncio.run(agent.call_llm([], system_prompt=""))
    args, _ = fake_model.bind_tools.call_args
    [tool_dicts] = args
    assert tool_dicts[0]["function"]["strict"] is True

    # Provider "switched": strict_tool_schema returns False on next call.
    state["strict"] = False
    asyncio.run(agent.call_llm([], system_prompt=""))
    args, _ = fake_model.bind_tools.call_args
    # bind_tools now receives the BaseTool list directly.
    assert args[0] == [strict_compatible_tool]

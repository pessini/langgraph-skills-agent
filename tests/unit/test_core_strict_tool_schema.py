"""Unit tests for the ``strict_tool_schema`` flag on ``BaseAgent``.

Verifies that ``call_llm`` forwards ``strict=True`` to ``bind_tools``
when the flag is enabled (so providers like OpenAI use constrained
decoding to guarantee tool calls match the JSON schema), and that the
flag is *not* forwarded when disabled (so providers like Ollama whose
``bind_tools`` doesn't accept the kwarg keep working).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.base_agent import BaseAgent
from langchain_core.messages import AIMessage


def _fake_tool() -> MagicMock:
    tool = MagicMock()
    tool.name = "some_tool"
    return tool


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.bind_tools = MagicMock(return_value=model)
    model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    return model


def test_call_llm_passes_strict_when_enabled() -> None:
    """When strict_tool_schema=True, bind_tools must be called with
    strict=True so the provider uses constrained decoding.
    """
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[_fake_tool()],
        strict_tool_schema=True,
    )

    asyncio.run(agent.call_llm([], system_prompt=""))

    fake_model.bind_tools.assert_called_once()
    _, kwargs = fake_model.bind_tools.call_args
    assert kwargs.get("strict") is True


def test_call_llm_does_not_pass_strict_when_disabled() -> None:
    """Default behavior: bind_tools called without strict kwarg so
    Ollama and other providers that don't accept it still work.
    """
    fake_model = _fake_model()

    agent = BaseAgent(
        agent_name="t",
        model_factory=lambda: fake_model,
        tools=[_fake_tool()],
        # strict_tool_schema defaults to False
    )

    asyncio.run(agent.call_llm([], system_prompt=""))

    fake_model.bind_tools.assert_called_once()
    _, kwargs = fake_model.bind_tools.call_args
    assert "strict" not in kwargs

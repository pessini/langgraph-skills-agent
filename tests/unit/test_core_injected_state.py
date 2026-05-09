"""Unit tests for InjectedState injection + log redaction in BaseAgent."""

from __future__ import annotations

from typing import Annotated

import pytest
from core.base_agent import BaseAgent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState


@tool
def needs_state(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Test tool that asks for the full graph state."""
    return f"q={query} keys={sorted(state.keys())}"


@tool
def no_state_arg(query: str) -> str:
    """Test tool that doesn't ask for state."""
    return f"q={query}"


def _make_agent() -> BaseAgent:
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=[needs_state, no_state_arg],
    )


class TestInjectedState:
    def test_state_arg_populated_when_declared(self) -> None:
        agent = _make_agent()
        args: dict = {"query": "hi"}
        injected = agent._inject_state_args(needs_state, args, {"messages": [], "x": 1})
        assert injected == ["state"]
        assert "state" in args
        assert args["state"] == {"messages": [], "x": 1}

    def test_state_arg_not_populated_when_absent(self) -> None:
        agent = _make_agent()
        args: dict = {"query": "hi"}
        injected = agent._inject_state_args(no_state_arg, args, {"messages": []})
        assert injected == []
        assert "state" not in args

    def test_redact_replaces_state_for_log(self) -> None:
        agent = _make_agent()
        args = {
            "query": "hi",
            "state": {"messages": [1, 2, 3], "tool_call_count": 5},
        }
        out = agent._redact_injected_for_log(args, ["state"])
        assert out["query"] == "hi"
        assert out["state"].startswith("<InjectedState keys=")
        assert "messages" in out["state"]
        assert "tool_call_count" in out["state"]

    def test_redact_no_op_when_nothing_injected(self) -> None:
        agent = _make_agent()
        args = {"query": "hi"}
        assert agent._redact_injected_for_log(args, []) is args

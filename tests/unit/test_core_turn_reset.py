"""Unit tests for ``BaseAgent.is_new_human_turn`` and ``with_turn_reset``."""

from __future__ import annotations

from core.base_agent import BaseAgent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestIsNewHumanTurn:
    def test_empty_messages_returns_false(self) -> None:
        assert BaseAgent.is_new_human_turn([]) is False

    def test_last_human_returns_true(self) -> None:
        msgs = [AIMessage(content="prev"), HumanMessage(content="hi")]
        assert BaseAgent.is_new_human_turn(msgs) is True

    def test_last_ai_returns_false(self) -> None:
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert BaseAgent.is_new_human_turn(msgs) is False

    def test_last_tool_returns_false(self) -> None:
        msgs = [
            AIMessage(content=""),
            ToolMessage(content="result", tool_call_id="x"),
        ]
        assert BaseAgent.is_new_human_turn(msgs) is False


class TestWithTurnReset:
    def test_new_turn_writes_resets_and_extras(self) -> None:
        update: dict = {"messages": ["hi"]}
        out = BaseAgent.with_turn_reset(
            update,
            is_new_human_turn=True,
            extra_reset_fields={"tool_call_count": 0},
        )
        assert out["messages"] == ["hi"]
        assert out["tool_retry_attempts"] == 0
        assert out["tool_feedback"] is None
        assert out["tool_call_count"] == 0

    def test_not_new_turn_writes_nothing_extra(self) -> None:
        update: dict = {"messages": ["hi"]}
        out = BaseAgent.with_turn_reset(
            update,
            is_new_human_turn=False,
            extra_reset_fields={"tool_call_count": 0},
        )
        assert out == {"messages": ["hi"]}
        assert "tool_retry_attempts" not in out
        assert "tool_feedback" not in out
        assert "tool_call_count" not in out

    def test_new_turn_without_extras(self) -> None:
        out = BaseAgent.with_turn_reset({}, is_new_human_turn=True)
        assert out["tool_retry_attempts"] == 0
        assert out["tool_feedback"] is None

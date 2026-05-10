"""Unit tests for ``BaseAgent.is_new_human_turn`` and ``with_turn_reset``."""

from __future__ import annotations

import pytest
from core.base_agent import BaseAgent
from core.feedback import ErrorResponse, ToolFeedback
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


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

    def test_new_turn_resets_tool_feedback_history(self) -> None:
        """A new human turn must clear ``tool_feedback_history`` so prior-
        turn errors don't leak into the next turn's ``previous_errors``
        injection at ``_serialize_previous_errors``.  Preserving the
        history "for analytics" was a false promise — analytics belongs
        in LangFuse traces or Aegra checkpoints, not in-memory state.
        """
        history = [
            ToolFeedback(
                error=ErrorResponse(
                    error_type="DATA_ERROR",
                    message="prior turn",
                    retryable=True,
                    context="ctx",
                )
            )
        ]
        out = BaseAgent.with_turn_reset(
            {"tool_feedback_history": history},
            is_new_human_turn=True,
        )
        assert out["tool_feedback_history"] == []

    def test_not_new_turn_does_not_clear_history(self) -> None:
        """Mid-turn calls (e.g. agent_node re-entry after a tool node) must
        leave ``tool_feedback_history`` alone so retry self-correction
        keeps working within the turn.
        """
        out = BaseAgent.with_turn_reset({}, is_new_human_turn=False)
        assert "tool_feedback_history" not in out


_RECEIVED_PREV_ERRORS: list[str | None] = []


@tool
def echoes_previous_errors_for_turn(
    query: str, previous_errors: str = ""
) -> str:
    """Tool that echoes the ``previous_errors`` it received."""
    _RECEIVED_PREV_ERRORS.append(previous_errors or None)
    return "ok"


@pytest.fixture(autouse=True)
def _reset_received() -> None:
    _RECEIVED_PREV_ERRORS.clear()


class TestCrossTurnErrorIsolation:
    """End-to-end check: errors from turn N must not appear in turn N+1's
    ``previous_errors`` argument injection.

    Today's bug: ``_serialize_previous_errors`` reads
    ``history[-3:]`` and forwards them; with the history preserved
    across human turns, every tool in the next turn sees errors that
    have nothing to do with what the user asked for.
    """

    @pytest.mark.asyncio
    async def test_history_reset_blocks_cross_turn_leakage(self) -> None:
        agent = BaseAgent(
            agent_name="test",
            model_factory=lambda: pytest.fail("model_factory must not run"),
            tools=[echoes_previous_errors_for_turn],
            max_invoke_retries=0,
        )

        # Turn 1: an error accumulates in history, then the turn ends.
        turn1_history = [
            ToolFeedback(
                error=ErrorResponse(
                    error_type="DATA_ERROR",
                    message="turn 1 column missing",
                    retryable=True,
                    context="2025 enterprise sales",
                )
            )
        ]

        # Apply the new-turn reset (simulating agent_node entry on a
        # fresh HumanMessage — the trigger for v0.2.5's cleanup).
        update = BaseAgent.with_turn_reset({}, is_new_human_turn=True)
        next_state = {
            "tool_feedback_history": update.get(
                "tool_feedback_history", turn1_history
            ),
            "tool_retry_attempts": update["tool_retry_attempts"],
        }

        # Turn 2's first tool call.
        await agent.execute_tool_calls(
            [
                {
                    "id": "tc-1",
                    "name": "echoes_previous_errors_for_turn",
                    "args": {"query": "lowest deal count?"},
                }
            ],
            state=next_state,
        )

        assert len(_RECEIVED_PREV_ERRORS) == 1
        prev = _RECEIVED_PREV_ERRORS[0]
        # Pre-fix this would carry "turn 1 column missing" through the
        # injection.  Post-fix, the history was wiped so the tool sees
        # no carry-over.
        assert prev is None

"""State definition for the skills agent.

Inherits ``messages`` from ``MessagesState`` (the ``add_messages`` reducer
appends instead of overwriting).

``tool_retry_attempts`` is incremented per retryable tool error and reset
to 0 on success or on a fresh human turn.

``tool_call_count`` is the total number of tool calls in the conversation;
``should_continue`` routes to ``error_summary_node`` once it hits
``MAX_TOOL_CALLS``. Reset on a fresh human turn so threads don't wedge.

``tool_feedback`` carries the latest structured ``ToolFeedback`` from the
tool node; reset on a fresh human turn.

``tool_feedback_history`` accumulates every feedback within a single
human turn. ``BaseAgent.execute_tool_calls`` reads the last 3 errored
entries and forwards them as the ``previous_errors`` kwarg to any tool
that declares the field — enabling self-correction. ``with_turn_reset``
clears it on every new ``HumanMessage`` so prior-turn errors don't
pollute the next turn's tool calls; longitudinal analytics consume
LangFuse traces or Aegra checkpoints rather than in-memory graph state.
"""

from __future__ import annotations

from core.feedback import ToolFeedback
from langgraph.graph import MessagesState


class State(MessagesState):
    tool_retry_attempts: int = 0
    tool_call_count: int = 0
    tool_feedback: ToolFeedback | None = None
    tool_feedback_history: list[ToolFeedback] = []

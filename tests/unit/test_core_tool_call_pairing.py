"""Unit tests for ``BaseAgent._ensure_tool_call_pairs``.

The repair runs at the LLM boundary so threads resumed from a
checkpointer that was interrupted between assistant and tool nodes
don't blow up on OpenAI's pairing invariant on the next request.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.base_agent import BaseAgent


def _make_agent() -> BaseAgent:
    """Minimal BaseAgent for pure-function tests — model_factory is never called."""
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=[],
    )


def _ai_with_calls(*ids: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"id": tcid, "name": "noop", "args": {}} for tcid in ids
        ],
    )


def _is_synthetic(msg: Any) -> bool:
    return isinstance(msg, ToolMessage) and "TOOL_INTERRUPTED" in (msg.content or "")


class TestEnsureToolCallPairs:
    def test_no_orphans_returns_unchanged(self) -> None:
        agent = _make_agent()
        msgs = [
            HumanMessage(content="hi"),
            _ai_with_calls("tc-1"),
            ToolMessage(content="ok", tool_call_id="tc-1"),
            AIMessage(content="done"),
        ]
        out = agent._ensure_tool_call_pairs(msgs)
        assert len(out) == len(msgs)
        for original, repaired in zip(msgs, out):
            assert original is repaired

    def test_orphan_in_middle_inserts_synthetic_before_next_non_tool(self) -> None:
        agent = _make_agent()
        msgs = [
            HumanMessage(content="q"),
            _ai_with_calls("orphan-A"),
            HumanMessage(content="q2"),  # non-tool message → orphan must flush before
        ]
        out = agent._ensure_tool_call_pairs(msgs)
        # Order: HumanMessage, AIMessage(orphan), synthetic ToolMessage, HumanMessage
        assert isinstance(out[0], HumanMessage)
        assert isinstance(out[1], AIMessage)
        assert _is_synthetic(out[2])
        assert out[2].tool_call_id == "orphan-A"
        assert isinstance(out[3], HumanMessage)
        assert len(out) == 4

    def test_orphan_at_tail_appends_synthetic(self) -> None:
        agent = _make_agent()
        msgs = [
            HumanMessage(content="q"),
            _ai_with_calls("orphan-tail"),
        ]
        out = agent._ensure_tool_call_pairs(msgs)
        assert len(out) == 3
        assert _is_synthetic(out[-1])
        assert out[-1].tool_call_id == "orphan-tail"

    def test_two_groups_first_partly_orphaned(self) -> None:
        """First AI emits two tool_calls; only one resolved before a second AI."""
        agent = _make_agent()
        msgs = [
            HumanMessage(content="q"),
            _ai_with_calls("A", "B"),
            ToolMessage(content="A-result", tool_call_id="A"),
            # B is orphaned; second AI starts a new group
            _ai_with_calls("C"),
            ToolMessage(content="C-result", tool_call_id="C"),
        ]
        out = agent._ensure_tool_call_pairs(msgs)
        # Synthetic for B inserted before the second AIMessage.
        synthetic = [m for m in out if _is_synthetic(m)]
        assert len(synthetic) == 1
        assert synthetic[0].tool_call_id == "B"
        # B's synthetic must be before the second AIMessage.
        ai_indices = [i for i, m in enumerate(out) if isinstance(m, AIMessage)]
        synth_idx = out.index(synthetic[0])
        assert synth_idx < ai_indices[1]

    def test_resolved_then_orphan_only_orphan_synthesized(self) -> None:
        agent = _make_agent()
        msgs = [
            _ai_with_calls("X", "Y"),
            ToolMessage(content="X-ok", tool_call_id="X"),
            HumanMessage(content="continue"),  # forces flush — Y is orphan
        ]
        out = agent._ensure_tool_call_pairs(msgs)
        synthetic = [m for m in out if _is_synthetic(m)]
        assert len(synthetic) == 1
        assert synthetic[0].tool_call_id == "Y"

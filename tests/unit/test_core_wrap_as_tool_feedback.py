"""Unit tests for ``BaseAgent._wrap_as_tool_feedback``.

Covers the MCP content-block branch added so that
``langchain-mcp-adapters`` results (a list of
``[{"type":"text","text":"..."}, ...]`` content blocks) are unwrapped
to their actual text payload instead of being ``str()``-ified to a
Python *repr* that is neither valid JSON nor LLM-friendly.
"""

from __future__ import annotations

import pytest
from core.base_agent import BaseAgent
from core.feedback import ToolFeedback


def _make_agent() -> BaseAgent:
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=[],
    )


def test_wrap_unwraps_mcp_text_block_list() -> None:
    """When raw is the MCP content-block shape, concatenate the text
    payloads so downstream consumers see the actual data.
    """
    agent = _make_agent()

    raw = [{"type": "text", "text": '[{"month": "Jan", "revenue": 100}]'}]
    feedback = agent._wrap_as_tool_feedback(raw)

    assert isinstance(feedback, ToolFeedback)
    assert feedback.success is not None
    assert feedback.success.results == '[{"month": "Jan", "revenue": 100}]'


def test_wrap_joins_multiple_text_blocks_with_newlines() -> None:
    agent = _make_agent()

    raw = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    feedback = agent._wrap_as_tool_feedback(raw)
    assert feedback.success is not None
    assert feedback.success.results == "first\nsecond"


def test_wrap_falls_through_for_non_text_block_lists() -> None:
    """A list of dicts that aren't all text blocks falls back to str()."""
    agent = _make_agent()

    raw = [{"type": "image", "data": "..."}]
    feedback = agent._wrap_as_tool_feedback(raw)
    # Falls back to the legacy str(raw) branch.
    assert feedback.success is not None
    assert "image" in feedback.success.results

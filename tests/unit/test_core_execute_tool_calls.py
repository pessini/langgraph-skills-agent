"""Unit tests for ``BaseAgent.execute_tool_calls`` / ``execute_tool_calls_safe``.

Covers the three return-shape paths (str / dict / ToolFeedback), the
TOOL_SUCCESS / TOOL_ERROR envelope formatting, retry-attempt arithmetic
for retryable vs non-retryable errors, and the catastrophic-fallback
guarantee in execute_tool_calls_safe.
"""

from __future__ import annotations

from typing import Any

import pytest
from core.base_agent import BaseAgent
from core.feedback import ErrorResponse, ToolFeedback
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool


@tool
def returns_string(query: str) -> str:
    """Returns a plain string."""
    return f"hello {query}"


@tool
def returns_feedback_dict(query: str) -> dict:
    """Returns a dict matching ToolFeedback shape."""
    return {
        "success": {"results": f"answered {query}"},
        "query": "select 1",
    }


@tool
def returns_feedback_object(query: str) -> ToolFeedback:
    """Returns a ToolFeedback instance directly."""
    return ToolFeedback(
        error=ErrorResponse(
            error_type="DATA_ERROR",
            message=f"could not process {query}",
            retryable=True,
            context="test",
        )
    )


@tool
def returns_non_retryable_error(query: str) -> ToolFeedback:
    """Returns a ToolFeedback whose error is explicitly non-retryable."""
    return ToolFeedback(
        error=ErrorResponse(
            error_type="PERMISSION_ERROR",
            message="forbidden",
            retryable=False,
            context=None,
        )
    )


@tool
def raises_runtime_error(query: str) -> str:
    """Raises an arbitrary exception to exercise the wrapper's catch path."""
    raise RuntimeError(f"boom {query}")


@tool
def returns_arbitrary_dict(query: str) -> dict:
    """Returns a dict that does NOT match the ToolFeedback schema.

    With Pydantic's default ``extra="ignore"`` this would silently
    construct an empty ToolFeedback and crash the success branch when
    accessing ``feedback.success.results``. ``extra="forbid"`` on the
    model converts the bad parse into a ValidationError that
    ``_wrap_as_tool_feedback`` catches and routes through the plain-
    string success path.
    """
    return {"rows": [1, 2, 3], "answer": query}


def _make_agent(tools: list) -> BaseAgent:
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=tools,
        max_tool_retries=3,
        max_invoke_retries=0,  # disable per-call retry to keep tests fast
    )


def _tc(name: str, args: dict, tool_call_id: str = "tc-1") -> dict[str, Any]:
    return {"id": tool_call_id, "name": name, "args": args}


# ---------------------------------------------------------------------------
# Return-shape coverage
# ---------------------------------------------------------------------------


class TestReturnShapes:
    @pytest.mark.asyncio
    async def test_string_return_is_wrapped_as_success(self) -> None:
        agent = _make_agent([returns_string])
        out = await agent.execute_tool_calls(
            [_tc("returns_string", {"query": "world"})], state={}
        )
        assert len(out["messages"]) == 1
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_SUCCESS" in msg.content
        assert "hello world" in msg.content
        assert "TOOL_QUERY" in msg.content
        assert out["tool_feedback"].is_success()
        assert out["tool_retry_attempts"] == 0

    @pytest.mark.asyncio
    async def test_dict_return_parsed_as_tool_feedback(self) -> None:
        agent = _make_agent([returns_feedback_dict])
        out = await agent.execute_tool_calls(
            [_tc("returns_feedback_dict", {"query": "x"})], state={}
        )
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_SUCCESS" in msg.content
        assert "answered x" in msg.content
        assert "select 1" in msg.content  # query echoed
        assert out["tool_feedback"].query == "select 1"

    @pytest.mark.asyncio
    async def test_arbitrary_dict_falls_through_to_string_success(self) -> None:
        """Regression: dict without 'success'/'error' keys must NOT silently
        produce an empty ToolFeedback (which would crash on
        feedback.success.results in the success branch)."""
        agent = _make_agent([returns_arbitrary_dict])
        out = await agent.execute_tool_calls(
            [_tc("returns_arbitrary_dict", {"query": "hi"})], state={}
        )
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_SUCCESS" in msg.content
        # The dict should be string-wrapped, not unpacked.
        assert "rows" in msg.content
        assert out["tool_feedback"].is_success()
        assert out["tool_feedback"].success.results  # not None

    @pytest.mark.asyncio
    async def test_tool_feedback_object_passthrough(self) -> None:
        agent = _make_agent([returns_feedback_object])
        out = await agent.execute_tool_calls(
            [_tc("returns_feedback_object", {"query": "x"})], state={}
        )
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_ERROR" in msg.content
        assert "type=DATA_ERROR" in msg.content
        assert "retryable=True" in msg.content
        assert "could not process x" in msg.content

    @pytest.mark.asyncio
    async def test_execute_tool_calls_sets_name_on_tool_message(self) -> None:
        """ToolMessage must carry the tool's name so downstream consumers
        can match by name without walking tool_calls on the prior AIMessage.
        """
        agent = _make_agent([returns_string])
        out = await agent.execute_tool_calls(
            [_tc("returns_string", {"query": "hi"})], state={}
        )
        [msg] = out["messages"]
        assert isinstance(msg, ToolMessage)
        assert msg.name == "returns_string"
        assert msg.tool_call_id == "tc-1"
        assert "hello hi" in msg.content


# ---------------------------------------------------------------------------
# Retry-attempt arithmetic
# ---------------------------------------------------------------------------


class TestRetryArithmetic:
    @pytest.mark.asyncio
    async def test_retryable_error_bumps_attempts(self) -> None:
        agent = _make_agent([returns_feedback_object])
        out = await agent.execute_tool_calls(
            [_tc("returns_feedback_object", {"query": "x"})], state={}
        )
        assert out["tool_retry_attempts"] == 1

    @pytest.mark.asyncio
    async def test_non_retryable_snaps_to_max(self) -> None:
        agent = _make_agent([returns_non_retryable_error])
        out = await agent.execute_tool_calls(
            [_tc("returns_non_retryable_error", {"query": "x"})], state={}
        )
        assert out["tool_retry_attempts"] == agent.max_tool_retries

    @pytest.mark.asyncio
    async def test_raised_exception_wrapped_as_execution_error(self) -> None:
        agent = _make_agent([raises_runtime_error])
        out = await agent.execute_tool_calls(
            [_tc("raises_runtime_error", {"query": "x"})], state={}
        )
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_ERROR" in msg.content
        assert "type=EXECUTION_ERROR" in msg.content
        assert out["tool_retry_attempts"] == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_emits_tool_not_found(self) -> None:
        agent = _make_agent([returns_string])
        out = await agent.execute_tool_calls(
            [_tc("does_not_exist", {})], state={}
        )
        msg: ToolMessage = out["messages"][0]
        assert "TOOL_ERROR" in msg.content
        assert "type=TOOL_NOT_FOUND" in msg.content
        # TOOL_NOT_FOUND is non-retryable in BaseAgent → snaps to max.
        assert out["tool_retry_attempts"] == agent.max_tool_retries


# ---------------------------------------------------------------------------
# Catastrophic fallback (execute_tool_calls_safe)
# ---------------------------------------------------------------------------


class TestExecuteSafe:
    @pytest.mark.asyncio
    async def test_catastrophic_failure_yields_one_message_per_call(
        self, monkeypatch
    ) -> None:
        agent = _make_agent([returns_string])

        async def boom(*_a, **_kw):
            raise RuntimeError("wrapper exploded")

        monkeypatch.setattr(agent, "execute_tool_calls", boom)

        tool_calls = [_tc("returns_string", {}, "id-1"), _tc("returns_string", {}, "id-2")]
        out = await agent.execute_tool_calls_safe(tool_calls, state={})

        assert len(out["messages"]) == 2
        for msg in out["messages"]:
            assert isinstance(msg, ToolMessage)
            assert "TOOL_ERROR" in msg.content
            assert "Internal error" in msg.content
        assert out["tool_retry_attempts"] == agent.max_tool_retries

    @pytest.mark.asyncio
    async def test_catastrophic_fallback_carries_name(
        self, monkeypatch
    ) -> None:
        """Fallback ToolMessages emitted by execute_tool_calls_safe must
        also set ``name`` so downstream consumers that match by name
        keep working through the catastrophic path.
        """
        agent = _make_agent([returns_string])

        async def boom(*_a, **_kw):
            raise RuntimeError("wrapper exploded")

        monkeypatch.setattr(agent, "execute_tool_calls", boom)

        out = await agent.execute_tool_calls_safe(
            [_tc("returns_string", {}, "id-1")], state={}
        )
        [msg] = out["messages"]
        assert msg.name == "returns_string"


# ---------------------------------------------------------------------------
# Batch rejection (pause-capable tool mixed with siblings)
# ---------------------------------------------------------------------------


@tool
def pausing_tool(query: str) -> str:
    """Pretends to be pause-capable (metadata is set after definition)."""
    return f"paused: {query}"


pausing_tool.metadata = {"pauses_graph": True}


class TestBatchRejectionCarriesName:
    @pytest.mark.asyncio
    async def test_each_refusal_carries_its_originating_tool_name(self) -> None:
        """When a multi-tool batch mixes a pause-capable tool with siblings,
        BaseAgent emits one refusal ToolMessage per call. Each must carry
        ``name`` so the contract introduced for ToolMessage is consistent.
        """
        agent = _make_agent([pausing_tool, returns_string])
        out = await agent.execute_tool_calls(
            [
                _tc("returns_string", {"query": "x"}, "id-1"),
                _tc("pausing_tool", {"query": "y"}, "id-2"),
            ],
            state={},
        )
        msgs = out["messages"]
        assert len(msgs) == 2
        names = {m.name for m in msgs}
        assert names == {"returns_string", "pausing_tool"}
        for m in msgs:
            assert "TOOL_ERROR" in m.content
            assert "BATCH_REJECTED" in m.content

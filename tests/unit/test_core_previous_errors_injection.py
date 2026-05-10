"""Verify that tools declaring ``previous_errors`` get the last 3 errors."""

from __future__ import annotations

import pytest
from core.base_agent import BaseAgent
from core.feedback import ErrorResponse, ToolFeedback
from langchain_core.tools import tool

_RECEIVED_PREV_ERRORS: list[str | None] = []


@tool
def echoes_previous_errors(query: str, previous_errors: str = "") -> str:
    """Tool that echoes back the previous_errors kwarg it received."""
    _RECEIVED_PREV_ERRORS.append(previous_errors or None)
    return "ok"


@tool
def no_previous_errors_field(query: str) -> str:
    """Tool that does NOT declare previous_errors."""
    return "ok"


def _err(msg: str, ctx: str) -> ToolFeedback:
    return ToolFeedback(
        error=ErrorResponse(
            error_type="DATA_ERROR",
            message=msg,
            retryable=True,
            context=ctx,
        )
    )


def _make_agent(tools: list) -> BaseAgent:
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=tools,
        max_invoke_retries=0,
    )


@pytest.fixture(autouse=True)
def _reset_received() -> None:
    _RECEIVED_PREV_ERRORS.clear()


class TestPreviousErrorsInjection:
    @pytest.mark.asyncio
    async def test_last_three_errors_forwarded_to_declaring_tool(self) -> None:
        agent = _make_agent([echoes_previous_errors])
        history = [
            _err("err 1", "ctx1"),
            _err("err 2", "ctx2"),
            _err("err 3", "ctx3"),
            _err("err 4", "ctx4"),  # only the last 3 should appear
        ]
        await agent.execute_tool_calls(
            [{"id": "tc-1", "name": "echoes_previous_errors", "args": {"query": "q"}}],
            state={"tool_feedback_history": history},
        )
        assert len(_RECEIVED_PREV_ERRORS) == 1
        prev = _RECEIVED_PREV_ERRORS[0]
        assert prev is not None
        # Only last 3 → "err 1: ctx1" must NOT appear.
        assert "err 1" not in prev
        assert "err 2" in prev
        assert "err 3" in prev
        assert "err 4" in prev

    @pytest.mark.asyncio
    async def test_tool_without_field_unaffected(self) -> None:
        agent = _make_agent([no_previous_errors_field])
        # Should not raise (Pydantic would reject unknown kwarg).
        out = await agent.execute_tool_calls(
            [{"id": "tc-1", "name": "no_previous_errors_field", "args": {"query": "q"}}],
            state={"tool_feedback_history": [_err("e", "c")]},
        )
        assert out["tool_feedback"].is_success()

    @pytest.mark.asyncio
    async def test_llm_supplied_previous_errors_is_not_overwritten(
        self,
    ) -> None:
        """If the LLM threads ``previous_errors`` through itself (because
        a future skill tells it to, or because the model decided to), the
        agent must NOT silently override that value with its own injection.

        Today's behaviour is ``args["previous_errors"] = injected_str``
        unconditionally — invisible from logs (the redaction step does
        not diff supplied vs. injected).  The fix uses ``setdefault`` so
        the LLM's value wins.
        """
        agent = _make_agent([echoes_previous_errors])
        history = [_err("history error", "ctx")]
        await agent.execute_tool_calls(
            [
                {
                    "id": "tc-1",
                    "name": "echoes_previous_errors",
                    "args": {
                        "query": "q",
                        "previous_errors": "LLM-supplied wins",
                    },
                }
            ],
            state={"tool_feedback_history": history},
        )
        assert len(_RECEIVED_PREV_ERRORS) == 1
        prev = _RECEIVED_PREV_ERRORS[0]
        assert prev == "LLM-supplied wins"
        # The injected history value must not have leaked through.
        assert "history error" not in (prev or "")

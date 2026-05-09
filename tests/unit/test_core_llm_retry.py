"""Unit tests for ``BaseAgent._ainvoke_with_llm_retry`` / ``call_llm``.

Verifies typed-class retry classification: transient errors (rate
limits, timeouts, 5xx) retry with exponential backoff; permanent errors
(BadRequest, AuthN, AuthZ) re-raise immediately without retrying.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from core.base_agent import BaseAgent
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Fixtures: build openai exceptions with whatever args the SDK requires.
# ---------------------------------------------------------------------------


def _make_rate_limit_error() -> Exception:
    from openai import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    return RateLimitError("rate limited", response=response, body=None)


def _make_bad_request_error() -> Exception:
    from openai import BadRequestError

    response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    return BadRequestError("bad request", response=response, body=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Minimal async stand-in that returns/raises from a scripted list."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0

    def bind_tools(self, _tools: list[Any]) -> _ScriptedModel:
        return self

    async def ainvoke(self, _payload: Any) -> AIMessage:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_agent(model: _ScriptedModel) -> BaseAgent:
    return BaseAgent(
        agent_name="test",
        model_factory=lambda: model,
        tools=[],
        llm_retry_max_attempts=3,
        llm_retry_base_delay_seconds=0.0,  # no real sleep in tests
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallLLMRetry:
    @pytest.mark.asyncio
    async def test_transient_then_success_retries(self) -> None:
        success = AIMessage(content="ok")
        model = _ScriptedModel(
            [_make_rate_limit_error(), _make_rate_limit_error(), success]
        )
        agent = _make_agent(model)

        out = await agent.call_llm(
            [HumanMessage(content="hi")], system_prompt="sys"
        )
        assert out is success
        assert model.calls == 3

    @pytest.mark.asyncio
    async def test_permanent_error_reraises_immediately(self) -> None:
        from openai import BadRequestError

        model = _ScriptedModel([_make_bad_request_error()])
        agent = _make_agent(model)

        with pytest.raises(BadRequestError):
            await agent.call_llm(
                [HumanMessage(content="hi")], system_prompt="sys"
            )
        assert model.calls == 1

    @pytest.mark.asyncio
    async def test_transient_exhaustion_reraises(self) -> None:
        from openai import RateLimitError

        model = _ScriptedModel(
            [_make_rate_limit_error(), _make_rate_limit_error(), _make_rate_limit_error()]
        )
        agent = _make_agent(model)

        with pytest.raises(RateLimitError):
            await agent.call_llm(
                [HumanMessage(content="hi")], system_prompt="sys"
            )
        assert model.calls == 3

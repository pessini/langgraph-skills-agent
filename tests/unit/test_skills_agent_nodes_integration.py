"""End-to-end-ish test of the slim skills_agent nodes wired through ``Context``.

Verifies that:
- ``agent_node`` builds a system prompt and calls the agent's ``call_llm``.
- ``tool_node`` runs the catastrophic-fallback path and bumps
  ``tool_call_count``.
- A new HumanMessage at the tail resets the per-turn counters via
  ``BaseAgent.with_turn_reset``.
- ``SkillsAgent.on_tool_event`` actually forwards events into
  progressive_logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from skills_agent import config as config_module
from skills_agent import nodes as nodes_module
from skills_agent.config import Context
from skills_agent.state import State


@dataclass
class _FakeRuntime:
    context: Context


class _ScriptedModel:
    """Async model double — returns AIMessages from a scripted list."""

    def __init__(self, script: list[AIMessage]) -> None:
        self._script = list(script)
        self.calls = 0

    def bind_tools(self, _tools: list[Any]) -> _ScriptedModel:
        return self

    async def ainvoke(self, _payload: Any) -> AIMessage:
        self.calls += 1
        return self._script.pop(0)


@tool
def echo_tool(query: str) -> str:
    """Returns the query unchanged."""
    return f"echo: {query}"


@pytest.fixture
def patched_context(monkeypatch, tmp_path) -> Context:
    """Build a Context whose model_factory and tools are deterministic."""
    # 1. Patch the chat-model loader so SkillsAgent.model_factory yields our scripted model.
    final_msg = AIMessage(content="all done")
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc-1", "name": "echo_tool", "args": {"query": "hi"}}],
    )
    scripted = _ScriptedModel([tool_call_msg, final_msg])
    monkeypatch.setattr(
        config_module, "_load_chat_model", lambda *_a, **_kw: scripted
    )

    # 2. Build a Context with an empty skills_dir (the real SkillStore.scan() walks
    #    the dir; an empty dir produces an empty catalog which is fine for prompts).
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    ctx = Context(skills_dir=str(skills_dir))

    # 3. Inject our test tool by patching create_skill_tools so initialize() sees it.
    monkeypatch.setattr(
        "skills_agent.tools.create_skill_tools", lambda _store: [echo_tool]
    )
    return ctx


class TestSkillsAgentNodes:
    @pytest.mark.asyncio
    async def test_agent_node_resets_counters_on_new_human_turn(
        self, patched_context
    ) -> None:
        runtime = _FakeRuntime(context=patched_context)
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="hello")],
            "tool_retry_attempts": 2,
            "tool_call_count": 7,
            "tool_feedback": None,
            "tool_feedback_history": [],
        }
        update = await nodes_module.agent_node(state, runtime)
        assert "messages" in update
        assert update["tool_retry_attempts"] == 0
        assert update["tool_call_count"] == 0
        assert update["tool_feedback"] is None

    @pytest.mark.asyncio
    async def test_agent_node_no_reset_when_tail_is_ai(
        self, patched_context
    ) -> None:
        runtime = _FakeRuntime(context=patched_context)
        state: State = {  # type: ignore[typeddict-item]
            "messages": [
                HumanMessage(content="x"),
                AIMessage(content="prev"),
            ],
            "tool_retry_attempts": 2,
            "tool_call_count": 7,
            "tool_feedback": None,
            "tool_feedback_history": [],
        }
        update = await nodes_module.agent_node(state, runtime)
        assert "tool_retry_attempts" not in update
        assert "tool_call_count" not in update

    @pytest.mark.asyncio
    async def test_tool_node_increments_tool_call_count(
        self, patched_context
    ) -> None:
        runtime = _FakeRuntime(context=patched_context)
        # Pre-initialize so ctx.agent exists.
        await patched_context.initialize()

        ai_with_calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc-A", "name": "echo_tool", "args": {"query": "x"}},
                {"id": "tc-B", "name": "echo_tool", "args": {"query": "y"}},
            ],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="go"), ai_with_calls],
            "tool_call_count": 3,
        }
        update = await nodes_module.tool_node(state, runtime)
        assert update["tool_call_count"] == 5  # 3 + 2 calls
        assert len(update["messages"]) == 2

    @pytest.mark.asyncio
    async def test_tool_node_does_not_charge_for_rejected_batch(
        self, patched_context, monkeypatch
    ) -> None:
        """The base-agent batch-rejection path returns one ToolMessage per
        call without executing any tool.  ``tool_node`` must charge
        ``tool_call_count`` only by actual executions — otherwise three
        rejected batches of 5 exhaust MAX_TOOL_CALLS=15 with zero work.
        """
        runtime = _FakeRuntime(context=patched_context)
        await patched_context.initialize()

        # Stub execute_tool_calls_safe to mimic the batch-rejection
        # contract: ToolMessages emitted, but executed_tool_count=0.
        from langchain_core.messages import ToolMessage as TM

        async def fake_safe(_tool_calls, _state):
            return {
                "messages": [
                    TM(content="rejected", tool_call_id="x", name="echo_tool"),
                    TM(content="rejected", tool_call_id="y", name="echo_tool"),
                ],
                "executed_tool_count": 0,
            }

        monkeypatch.setattr(
            patched_context.agent, "execute_tool_calls_safe", fake_safe
        )

        ai_with_calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "x", "name": "echo_tool", "args": {}},
                {"id": "y", "name": "echo_tool", "args": {}},
            ],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="go"), ai_with_calls],
            "tool_call_count": 7,
        }
        update = await nodes_module.tool_node(state, runtime)
        # Pre-fix this would be 7 + 2 = 9 even though no tools ran.
        assert update["tool_call_count"] == 7

    def test_should_continue_blocks_when_projected_exceeds_cap(self) -> None:
        """count=14 + 2 pending tool_calls would push past MAX_TOOL_CALLS=15;
        must route to error_summary instead of letting the cap be busted."""
        cap = nodes_module.MAX_TOOL_CALLS
        ai_with_calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc-1", "name": "echo_tool", "args": {}},
                {"id": "tc-2", "name": "echo_tool", "args": {}},
            ],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [ai_with_calls],
            "tool_call_count": cap - 1,  # 14 + 2 = 16 > 15
        }
        assert nodes_module.should_continue(state) == "error_summary_node"

    def test_should_continue_allows_exactly_at_cap(self) -> None:
        """count=13 + 2 tool_calls = 15 (== cap). Projected does NOT exceed
        the cap (`> MAX`, not `>= MAX`), so routing proceeds to tool_node."""
        cap = nodes_module.MAX_TOOL_CALLS
        ai_with_calls = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc-1", "name": "echo_tool", "args": {}},
                {"id": "tc-2", "name": "echo_tool", "args": {}},
            ],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [ai_with_calls],
            "tool_call_count": cap - 2,  # 13 + 2 = 15 == cap
        }
        assert nodes_module.should_continue(state) == "tool_node"

    def test_should_continue_routes_to_error_summary_on_budget(self) -> None:
        ai_with_calls = AIMessage(
            content="",
            tool_calls=[{"id": "tc-1", "name": "echo_tool", "args": {}}],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [ai_with_calls],
            "tool_call_count": nodes_module.MAX_TOOL_CALLS,
        }
        assert nodes_module.should_continue(state) == "error_summary_node"

    def test_should_continue_ends_when_no_tool_calls(self) -> None:
        state: State = {  # type: ignore[typeddict-item]
            "messages": [AIMessage(content="final")],
        }
        assert nodes_module.should_continue(state) == "__end__"


class TestContextInitDecorator:
    """Regression: every node must self-initialize Context.

    Aegra creates a fresh Context per request. The HITL resume path
    re-enters tool_node (or error_summary_node) directly with that
    fresh Context, never having gone through agent_node — which is
    where init used to live. Without the decorator, the node raises
    ``Context not initialized`` and the run errors silently.
    """

    @pytest.mark.asyncio
    async def test_tool_node_initializes_uninitialized_context(
        self, patched_context
    ) -> None:
        # Critical: do NOT pre-initialize. This mirrors what Aegra does
        # on resume from interrupt — fresh Context, entry at tool_node.
        assert patched_context._skill_store is None

        runtime = _FakeRuntime(context=patched_context)
        ai_with_calls = AIMessage(
            content="",
            tool_calls=[{"id": "tc-resume", "name": "echo_tool", "args": {"query": "x"}}],
        )
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="go"), ai_with_calls],
            "tool_call_count": 0,
        }

        # Pre-fix this raised RuntimeError("Context not initialized").
        update = await nodes_module.tool_node(state, runtime)

        assert patched_context._skill_store is not None  # init happened
        assert update["tool_call_count"] == 1
        assert len(update["messages"]) == 1

    @pytest.mark.asyncio
    async def test_error_summary_node_initializes_uninitialized_context(
        self, patched_context
    ) -> None:
        assert patched_context._skill_store is None

        runtime = _FakeRuntime(context=patched_context)
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="x"), AIMessage(content="prev")],
        }
        update = await nodes_module.error_summary_node(state, runtime)

        assert patched_context._skill_store is not None
        assert update["tool_retry_attempts"] == 0
        assert "messages" in update

    @pytest.mark.asyncio
    async def test_decorator_is_idempotent(self, patched_context) -> None:
        """Re-entry within the same request must reuse the cached store."""
        runtime = _FakeRuntime(context=patched_context)
        await patched_context.initialize()
        cached_store = patched_context._skill_store

        # Run agent_node once; the decorator should NOT re-init.
        state: State = {  # type: ignore[typeddict-item]
            "messages": [HumanMessage(content="hi")],
            "tool_retry_attempts": 0,
            "tool_call_count": 0,
            "tool_feedback": None,
            "tool_feedback_history": [],
        }
        await nodes_module.agent_node(state, runtime)

        assert patched_context._skill_store is cached_store


class TestStrictToolSchemaWiring:
    """``Context`` wires ``strict_tool_schema`` as a callable that reads
    ``self.provider`` dynamically.  Resolving it must reflect the
    *current* provider — not whatever was set at ``initialize()`` time
    — because ``model_factory`` reads ``provider`` dynamically too,
    and a stale ``True`` would feed ``bind_tools(strict=True)`` into a
    freshly-built Ollama model whose ``bind_tools`` raises ``TypeError``.
    """

    @pytest.mark.asyncio
    async def test_strict_enabled_for_openai(self, monkeypatch, tmp_path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr(
            "skills_agent.tools.create_skill_tools", lambda _store: []
        )
        ctx = Context(skills_dir=str(skills_dir), provider="openai")
        await ctx.initialize()
        assert ctx.agent._resolve_strict_tool_schema() is True

    @pytest.mark.asyncio
    async def test_strict_disabled_for_ollama(self, monkeypatch, tmp_path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr(
            "skills_agent.tools.create_skill_tools", lambda _store: []
        )
        ctx = Context(skills_dir=str(skills_dir), provider="ollama")
        await ctx.initialize()
        assert ctx.agent._resolve_strict_tool_schema() is False

    @pytest.mark.asyncio
    async def test_strict_recomputes_when_provider_mutates(
        self, monkeypatch, tmp_path
    ) -> None:
        """If a caller mutates ``ctx.provider`` after ``initialize()``,
        the cached agent's strict resolution must follow on the next
        call.  Otherwise a stale ``strict=True`` would crash Ollama at
        ``bind_tools``.
        """
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        monkeypatch.setattr(
            "skills_agent.tools.create_skill_tools", lambda _store: []
        )
        ctx = Context(skills_dir=str(skills_dir), provider="openai")
        await ctx.initialize()
        assert ctx.agent._resolve_strict_tool_schema() is True

        # Mid-thread mutation — model_factory reads this dynamically too.
        ctx.provider = "ollama"
        assert ctx.agent._resolve_strict_tool_schema() is False

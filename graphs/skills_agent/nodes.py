"""
Graph nodes for the skills agent.

The ReAct mechanism (LLM retry, orphan tool_call repair, ToolFeedback
auto-wrapping, InjectedState injection, previous_errors self-correction)
lives in ``core.BaseAgent``. This file is the thin policy layer:

- renders the system prompt with live data (current time + skill catalog),
- threads ``Runtime[Context]`` into the agent's high-level methods,
- enforces the skills_agent-specific ``MAX_TOOL_CALLS`` budget,
- routes to ``error_summary_node`` when the budget or retry limit is hit.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from skills_agent.config import Context
from skills_agent.prompts import SYSTEM_PROMPT
from skills_agent.state import State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request Context initialization
# ---------------------------------------------------------------------------
#
# Aegra (and any LangGraph platform host) instantiates a fresh ``Context``
# per request, with ``_skill_store`` / ``_tools`` / ``_agent`` all None
# until ``await ctx.initialize()`` runs. The first node entered must do
# that init.
#
# We previously relied on "agent_node always runs first" — true for the
# normal __start__ path, FALSE for resume-from-interrupt: when the user
# resolves a ``request_human_review`` interrupt, the graph re-enters at
# ``tool_node`` with a brand-new uninitialized Context and ``ctx.agent``
# raises ``RuntimeError("Context not initialized")``. Aegra marks the run
# errored and the HITL flow breaks silently from the user's perspective.
#
# The decorator below makes initialization a structural property of every
# graph node rather than something each node author has to remember.

NodeFn = Callable[[State, Runtime[Context]], Awaitable[dict]]


def with_context_init(node_fn: NodeFn) -> NodeFn:
    """Decorator: ensure ``runtime.context`` is initialized before the node body.

    Idempotent — re-entry within the same request reuses the cached
    skill store, tool list, and SkillsAgent. Wrap every graph node so
    no entry path (start, resume, replay) can land on a fresh Context.
    """

    @wraps(node_fn)
    async def wrapper(state: State, runtime: Runtime[Context]) -> dict:
        ctx = runtime.context
        if ctx._skill_store is None:
            await ctx.initialize()
        return await node_fn(state, runtime)

    return wrapper

# Skills-agent policy: hard cap on cumulative tool calls within a single
# human turn (the counter is reset by BaseAgent.with_turn_reset on each
# new HumanMessage). Prevents runaway loops if the LLM keeps calling
# tools without producing a final answer.
MAX_TOOL_CALLS = 15

# Mirrors BaseAgent's default `max_tool_retries`. Kept as a module-level
# constant so should_continue (a simple state-only router) can reference
# it without taking a runtime parameter. If you customize the agent's
# retry budget when constructing SkillsAgent, update this value too.
MAX_TOOL_RETRIES = 3

# Used only by error_summary_node. The LLM is invoked WITHOUT tools so it
# can only produce a plain-text response — prevents infinite retry loops
# when retries are already exhausted.
ERROR_SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant. The tool execution has failed repeatedly "
    "and cannot be retried further. Review the conversation history, "
    "identify what went wrong, and provide a clear, concise summary "
    "to the user explaining the failure and suggesting next steps."
)


@with_context_init
async def agent_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Invoke the LLM with tools bound — the "Reasoning" half of ReAct.

    The ``@with_context_init`` decorator guarantees the per-request
    ``Context`` is initialized before this body runs. Renders the system
    prompt with the current UTC time and the skill catalog, then delegates
    to ``BaseAgent.call_llm`` which handles orphan tool_call repair and
    transient-error retry.

    On a fresh human turn (last persisted message is a HumanMessage), the
    per-turn counters (``tool_retry_attempts``, ``tool_feedback``,
    ``tool_call_count``) are reset so a long-running thread doesn't wedge
    after a previous failure.
    """
    ctx = runtime.context
    agent = ctx.agent
    is_new = agent.is_new_human_turn(state["messages"])
    system_prompt = SYSTEM_PROMPT.format(
        current_time=datetime.now(tz=UTC).isoformat(),
        skill_catalog=ctx.skill_store.get_skill_catalog(),
    )
    response = await agent.call_llm(state["messages"], system_prompt=system_prompt)
    update: dict = {"messages": [response]}
    return agent.with_turn_reset(
        update, is_new, extra_reset_fields={"tool_call_count": 0}
    )


@with_context_init
async def tool_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Execute tool calls from the latest AIMessage — the "Acting" half.

    Delegates to ``BaseAgent.execute_tool_calls_safe`` which guarantees
    that every tool_call gets a paired ToolMessage even on catastrophic
    failure (preserves OpenAI's pairing invariant). The skills_agent-
    specific ``tool_call_count`` is incremented by ``executed_tool_count``
    — i.e. tool calls that *actually* ran — rather than
    ``len(last.tool_calls)``.  The base-agent batch-rejection path and
    the catastrophic-fallback path both emit one ToolMessage per call
    without invoking any tool; counting those would let three rejected
    batches of 5 exhaust ``MAX_TOOL_CALLS=15`` with zero useful work.
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    agent = runtime.context.agent
    update = await agent.execute_tool_calls_safe(last.tool_calls, dict(state))
    executed = update.pop("executed_tool_count", 0)
    update["tool_call_count"] = state.get("tool_call_count", 0) + executed
    return update


@with_context_init
async def error_summary_node(state: State, runtime: Runtime[Context]) -> dict:
    """
    Graceful-degradation node: produce a plain-text failure summary.

    Reached when ``should_continue`` detects either the tool-call budget
    or the retry budget is exhausted. The LLM is invoked without tools
    so it can only produce a final answer. The retry counter is reset
    afterwards so the thread can be reused.
    """
    response = await runtime.context.agent.call_llm_no_tools(
        state["messages"], system_prompt=ERROR_SUMMARY_SYSTEM_PROMPT
    )
    return {"messages": [response], "tool_retry_attempts": 0}


def should_continue(
    state: State,
) -> Literal["tool_node", "error_summary_node", "__end__"]:
    """
    Route after agent_node.

    - tool call budget exhausted → error_summary_node
    - retry budget exhausted     → error_summary_node
    - tool calls present         → tool_node
    - otherwise                  → __end__
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Compare the *projected* total against the cap, not just what
        # has already been executed: the latest AIMessage may carry N
        # tool_calls that haven't run yet. With current=14 and N=2, the
        # bare current-count check would still route to tool_node and
        # let the turn execute 16 calls (cap is 15).
        projected_count = state.get("tool_call_count", 0) + len(last_message.tool_calls)
        if projected_count > MAX_TOOL_CALLS:
            logger.warning(
                "tool_call_budget_exhausted projected=%d max=%d",
                projected_count,
                MAX_TOOL_CALLS,
            )
            return "error_summary_node"
        if state.get("tool_retry_attempts", 0) >= MAX_TOOL_RETRIES:
            return "error_summary_node"
        return "tool_node"
    return "__end__"

"""Graph node functions for the skills agent.

This module contains the three node functions and routing logic for the
ReAct loop, plus all helper functions for tool execution, retry logic,
and LLM model loading.

Graph structure::

    __start__ → agent_node ⟷ tool_node → __end__
                    ↓ (max retries)
              error_summary_node → __end__

Key design decisions:

1. **Single ReAct loop instead of multi-phase pipeline**: An earlier version used
   separate plan/load/execute phases with dedicated nodes. That was replaced with
   a standard ReAct loop because the LLM can naturally chain tool calls (e.g.
   load_skill → read_skill_file) without explicit orchestration. This simplifies
   the graph and makes tool ordering emergent rather than hard-coded.

2. **Retry logic lives in tool_node, not in the graph edges**: The tool_node
   tracks retry attempts in the state and uses exponential backoff for transient
   errors. This keeps the graph structure simple (3 nodes, 2 edges) while still
   handling failures gracefully.

3. **error_summary_node as a graceful degradation path**: When retries are
   exhausted, instead of returning raw error messages, the LLM is called *without
   tools* to produce a user-friendly summary of what went wrong and suggest next
   steps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
import time
from datetime import UTC, datetime
from typing import Any, Literal, Union, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from skills_agent.config import Context
from skills_agent.utils.prompts import SYSTEM_PROMPT
from skills_agent.utils.progressive_logging import log_progressive, summarize_mapping
from skills_agent.utils.state import State

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# These control the retry behaviour for tool execution failures.
MAX_TOOL_RETRIES = 3
MAX_INVOKE_RETRIES = 3
MAX_TOOL_CALLS = 15  # Hard limit on total tool calls to prevent infinite loops

# HTTP status codes that are safe to retry (server overload / rate-limiting).
RETRYABLE_STATUS_CODES = ("429", "502", "503", "504")

# Tool names that are local (in-process) skill operations.
SKILL_TOOLS = ("load_skill", "read_skill_file")

# System prompt used exclusively by error_summary_node.  The LLM is invoked
# *without tools* so it can only produce a plain-text response explaining what
# went wrong.  This prevents the LLM from attempting further tool calls when
# retries are already exhausted.
ERROR_SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant. The tool execution has failed repeatedly "
    "and cannot be retried further. Review the conversation history, "
    "identify what went wrong, and provide a clear, concise summary "
    "to the user explaining the failure and suggesting next steps."
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_message_text(msg: BaseMessage) -> str:
    """Extract text content from a message, handling all LangChain content formats.

    LangChain messages can carry content in three different shapes:
    - ``str`` — simple text (most common).
    - ``dict`` — structured content with a ``"text"`` key (used by some providers).
    - ``list[str | dict]`` — multi-modal content blocks (e.g. text + images).
      We join only the text portions.

    Args:
        msg: A LangChain message object.

    Returns:
        The text content of the message.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    provider: str, model: str, base_url: str | None = None
) -> BaseChatModel:
    """Factory function to create a LangChain chat model instance.

    Called from ``agent_node`` and ``error_summary_node`` on every invocation.
    A new model instance is created each time (these are lightweight wrappers
    around HTTP clients, not persistent connections).

    Note: OpenAI reads ``OPENAI_API_KEY`` from the environment automatically
    (handled by the ``langchain-openai`` package), so no explicit API key
    parameter is needed here.

    Args:
        provider: The LLM provider ('ollama' or 'openai').
        model: The model name (e.g. 'qwen3', 'gpt-4o-mini').
        base_url: Base URL for Ollama server (ignored for OpenAI).

    Returns:
        A configured chat model instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "ollama":
        return ChatOllama(model=model, base_url=base_url)
    elif provider == "openai":
        return ChatOpenAI(model=model)
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Use 'ollama' or 'openai'."
        )


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def _extract_tool_call_id(tc: Union[dict[str, Any], Any]) -> str:
    """Extract tool_call_id from various tool call formats with fallback.

    Handles both dict and object formats. Generates a UUID-based fallback
    if no ID is found, ensuring every tool call gets a unique identifier.
    """
    tool_call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
    if not tool_call_id:
        tool_call_id = f"fallback_id_{uuid.uuid4().hex[:8]}"
    return tool_call_id


def _format_tool_result(result: Any) -> str:
    """Convert a tool result to a string suitable for a ToolMessage."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


def _summarize_tool_result(result: Any) -> str:
    """Create a compact summary for tool results to keep logs readable."""
    if isinstance(result, dict):
        keys = ", ".join(list(result.keys())[:6])
        return f"keys={keys}"
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return "empty"
        try:
            parsed = json.loads(text)
        except Exception:
            return f"chars={len(text)}"
        if isinstance(parsed, dict):
            keys = ", ".join(list(parsed.keys())[:6])
            return f"keys={keys}"
        if isinstance(parsed, list):
            return f"items={len(parsed)}"
        return f"type={type(parsed).__name__}"
    return f"type={type(result).__name__}"


def _create_fallback_error_messages(
    tool_calls: list[dict[str, Any]], error_message: str
) -> list[ToolMessage]:
    """Create error ToolMessages for all tool_calls — catastrophic fallback.

    **Why this exists**: The OpenAI Chat Completions API (and compatible APIs)
    enforces a strict invariant: every ``tool_call`` in an ``AIMessage`` must
    have a matching ``ToolMessage`` with the same ``tool_call_id``.  If the
    tool_node crashes before producing any ToolMessages, the checkpointer would
    persist a broken state that no subsequent request can recover from.

    This function is called in the ``except`` block of ``tool_node`` as a
    last-resort guarantee that the invariant holds, even if the actual tool
    execution threw an unexpected exception.
    """
    messages = []
    for tc in tool_calls:
        tool_call_id = _extract_tool_call_id(tc)
        messages.append(
            ToolMessage(
                content=f"TOOL_ERROR type=internal retryable=false\nmessage: {error_message}",
                tool_call_id=tool_call_id,
            )
        )
        logger.warning(
            "Created fallback error ToolMessage for tool_call_id=%s: %s",
            tool_call_id,
            error_message,
        )
    return messages


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if an exception is a retryable HTTP error (429, 502, 503, 504).

    **Why string matching?**  Some libraries do not expose structured HTTP
    status codes — errors are raised as generic ``Exception`` with the status
    code embedded in the message string.

    **Why recurse into ExceptionGroup?**  Some async transports wrap errors
    in ``ExceptionGroup``.  We must walk the sub-exceptions to find the real
    cause.
    """
    if isinstance(exc, ExceptionGroup):
        return any(_is_retryable_error(sub) for sub in exc.exceptions)
    msg = str(exc)
    return any(code in msg for code in RETRYABLE_STATUS_CODES)


async def _invoke_with_retry(tool: BaseTool, args: dict[str, Any]) -> Any:
    """Invoke a tool with exponential backoff for retryable HTTP errors.

    Retries up to MAX_INVOKE_RETRIES times with delays of 1s, 2s, 4s, etc.
    Non-retryable errors are raised immediately.
    """
    for attempt in range(MAX_INVOKE_RETRIES + 1):
        try:
            return await tool.ainvoke(args)
        except Exception as e:
            if attempt < MAX_INVOKE_RETRIES and _is_retryable_error(e):
                delay = 2**attempt  # 1s, 2s, 4s
                logger.warning(
                    "Retryable error on tool '%s' (attempt %d/%d), "
                    "retrying in %ds: %s",
                    tool.name,
                    attempt + 1,
                    MAX_INVOKE_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
            else:
                raise


async def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools_by_name: dict[str, BaseTool],
    retry_attempts: int,
) -> tuple[list[ToolMessage], int]:
    """Execute a batch of tool calls sequentially and track retry state.

    Returns ``(tool_messages, new_retry_attempts)`` where:

    - ``tool_messages``: One ``ToolMessage`` per tool call (success or error).
    - ``new_retry_attempts``: Reset to 0 on full success, incremented on any
      error.  This counter is persisted in the graph state and checked by
      ``should_continue()`` to decide whether to route to
      ``error_summary_node``.

    **Structured error format**: Error messages follow a parseable format
    (``TOOL_ERROR attempt=N type=... retryable=...``) so the LLM can reason
    about whether to retry or try a different approach.
    """
    tool_messages: list[ToolMessage] = []
    any_error = False

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
        tool_call_id = _extract_tool_call_id(tool_call)
        args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})

        tool = tools_by_name.get(tool_name)
        if not tool:
            any_error = True
            log_progressive({
                "event": "tool.invoke.error",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "summary": "Unknown tool",
            })
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"TOOL_ERROR attempt={retry_attempts + 1} "
                        f"type=unknown_tool retryable=false\n"
                        f"message: Unknown tool '{tool_name}'."
                    ),
                    tool_call_id=tool_call_id,
                )
            )
            continue

        try:
            log_progressive({
                "event": "tool.invoke.start",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "summary": summarize_mapping(args),
            })
            invoke_start = time.perf_counter()
            result = await _invoke_with_retry(tool, args)
            duration_ms = int((time.perf_counter() - invoke_start) * 1000)
            content = _format_tool_result(result)
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
            log_progressive({
                "event": "tool.invoke.end",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "duration_ms": duration_ms,
                "summary": _summarize_tool_result(result),
            })
        except Exception as e:
            any_error = True
            logger.exception("Tool '%s' failed", tool_name)
            new_attempt = retry_attempts + 1
            retryable = new_attempt < MAX_TOOL_RETRIES
            log_progressive({
                "event": "tool.invoke.error",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "summary": str(e),
            })
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"TOOL_ERROR attempt={new_attempt} "
                        f"type=execution_error retryable={retryable}\n"
                        f"message: {e}"
                    ),
                    tool_call_id=tool_call_id,
                )
            )

    new_retry_attempts = (retry_attempts + 1) if any_error else 0
    return tool_messages, new_retry_attempts


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def agent_node(state: State, runtime: Runtime[Context]) -> dict:
    """Invoke the LLM with all available tools (the "Reasoning" half of ReAct).

    This is the central node of the graph.  On every iteration it:

    1. Ensures the ``Context`` is initialized (skills scanned, tools created).
       Initialization is lazy — it happens on the first call and is cached for
       subsequent calls within the same process.
    2. Builds a system prompt that includes:
       - The current UTC timestamp (so the LLM can reason about time).
       - An XML-structured skill catalog (names, descriptions, tags, supporting
         files) produced by ``SkillStore.get_skill_catalog()``.
    3. Binds all tools to the LLM and invokes it with the full conversation
       history.
    4. Returns the LLM's response (an ``AIMessage`` that may or may not contain
       ``tool_calls``).

    The ``should_continue`` router then decides the next step:
    - If the response contains tool calls → route to ``tool_node``.
    - If not → route to ``__end__`` (final answer to the user).
    """
    ctx = runtime.context

    # Lazy initialization on first call — scans skills directory and builds
    # the tools list.  Subsequent calls reuse the cached result.
    if ctx._skill_store is None:
        await ctx.initialize()

    # Use cached tools list (built once during Context.initialize())
    tools: list[BaseTool] = ctx.tools

    # Build system prompt with live data (time, skill catalog).
    # The prompt template lives in utils/prompts.py and uses str.format() placeholders.
    system_message = SYSTEM_PROMPT.format(
        current_time=datetime.now(tz=UTC).isoformat(),
        skill_catalog=ctx.skill_store.get_skill_catalog(),
    )

    model = load_chat_model(
        provider=ctx.provider,
        model=ctx.model,
        base_url=ctx.base_url,
    )
    # bind_tools() attaches the JSON schemas of all tools to the LLM request,
    # enabling the model to produce structured tool_call outputs.
    model_with_tools = model.bind_tools(tools) if tools else model

    response = cast(
        "AIMessage",
        await model_with_tools.ainvoke(
            [{"role": "system", "content": system_message}, *state["messages"]]
        ),
    )
    if response.tool_calls:
        tool_names: list[str] = []
        for tc in response.tool_calls:
            if isinstance(tc, dict):
                tool_names.append(tc.get("name", ""))
            else:
                tool_names.append(getattr(tc, "name", ""))
        log_progressive({
            "event": "agent.response.tool_calls",
            "summary": f"count={len(response.tool_calls)} tools={', '.join([t for t in tool_names if t])}",
        })
    else:
        log_progressive({"event": "agent.response.final"})
    return {"messages": [response]}


async def tool_node(state: State, runtime: Runtime[Context]) -> dict:
    """Execute tool calls from the last AI message (the "Acting" half of ReAct).

    This node picks up tool_calls from the most recent ``AIMessage``, executes
    each one sequentially via ``_execute_tool_calls()``, and returns the
    results as ``ToolMessage`` objects appended to the conversation.

    After this node, the graph always routes back to ``agent_node`` so the LLM
    can inspect the tool results and decide whether to make more tool calls or
    produce a final response.

    **Catastrophic fallback**: The entire execution is wrapped in a try/except.
    If anything unexpected happens (e.g. the tools_by_name dict is corrupted),
    ``_create_fallback_error_messages()`` ensures every tool_call still gets a
    ``ToolMessage``.  Without this, the state would be left in an invalid
    condition where the LLM API rejects follow-up requests.
    """
    ctx = runtime.context
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}

    tool_calls = last_message.tool_calls
    retry_attempts = state.get("tool_retry_attempts", 0)

    try:
        # Use cached tools list (built once during Context.initialize())
        tools_by_name: dict[str, BaseTool] = {
            t.name: t for t in ctx.tools
        }

        tool_messages, new_retry_attempts = await _execute_tool_calls(
            tool_calls, tools_by_name, retry_attempts
        )
        # Increment total tool call count (for loop prevention)
        current_count = state.get("tool_call_count", 0)
        return {
            "messages": tool_messages,
            "tool_retry_attempts": new_retry_attempts,
            "tool_call_count": current_count + len(tool_calls),
        }

    except Exception as e:
        # Catastrophic fallback: ensure every tool_call gets a ToolMessage
        logger.exception("Catastrophic failure in tool_node")
        fallback_messages = _create_fallback_error_messages(
            tool_calls,
            f"Internal error in tool execution: {e}",
        )
        current_count = state.get("tool_call_count", 0)
        return {
            "messages": fallback_messages,
            "tool_retry_attempts": MAX_TOOL_RETRIES,
            "tool_call_count": current_count + len(tool_calls),
        }


async def error_summary_node(state: State, runtime: Runtime[Context]) -> dict:
    """Produce a user-friendly summary when all tool retries are exhausted.

    This node is only reached when ``should_continue()`` detects that
    ``tool_retry_attempts >= MAX_TOOL_RETRIES``.  Instead of silently failing
    or returning raw error text, the LLM is called one more time — crucially
    *without* binding any tools — so it can only produce a natural-language
    response.  This prevents infinite retry loops while still giving the user
    actionable information about what went wrong.

    The retry counter is reset to 0 after generating the summary so the thread
    can be reused for future messages without being permanently stuck.
    """
    ctx = runtime.context

    model = load_chat_model(
        provider=ctx.provider,
        model=ctx.model,
        base_url=ctx.base_url,
    )

    response = cast(
        "AIMessage",
        await model.ainvoke(
            [
                {"role": "system", "content": ERROR_SUMMARY_SYSTEM_PROMPT},
                *state["messages"],
            ]
        ),
    )
    # Reset retry counter so the thread can be reused
    return {"messages": [response], "tool_retry_attempts": 0}


def should_continue(
    state: State,
) -> Literal["tool_node", "error_summary_node", "__end__"]:
    """Route after agent_node.

    - If tool call limit exceeded               → error_summary_node
    - If tool calls present and retries exhausted → error_summary_node
    - If tool calls present and retries available  → tool_node
    - Otherwise                                    → __end__
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check total tool call limit first (prevents infinite loops)
        if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
            logger.warning(
                "Tool call limit reached (%d >= %d), routing to error_summary_node",
                state.get("tool_call_count", 0),
                MAX_TOOL_CALLS,
            )
            return "error_summary_node"
        # Check retry limit (handles repeated failures)
        if state.get("tool_retry_attempts", 0) >= MAX_TOOL_RETRIES:
            return "error_summary_node"
        return "tool_node"
    return "__end__"

"""Reusable async ReAct agent base class.

``BaseAgent`` is decoupled from any specific LLM provider, prompt source,
or graph state schema. Callers supply:

- a ``model_factory`` callable that returns a fresh ``BaseChatModel`` per
  call (matches LangChain's lightweight HTTP-wrapper pattern);
- a ``tools`` list (any LangChain ``BaseTool`` works);
- a system prompt rendered per invocation (the class never imports
  Langfuse or any prompt provider).

The class encapsulates the *mechanism* — orphan tool_call repair, LLM
retry on transient errors, ``ToolFeedback`` auto-wrapping, ``InjectedState``
arg injection, ``previous_errors`` self-correction, structured
``TOOL_SUCCESS`` / ``TOOL_ERROR`` envelope. Per-agent *policy* (max tool
calls, error-summary node, prompt selection) lives in the calling agent
module.

Override ``on_tool_event`` to forward execution events to a logger or
metrics sink (default is no-op).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from core.feedback import ErrorResponse, SuccessResponse, ToolFeedback
from core.llm_errors import PERMANENT_LLM_ERRORS, TRANSIENT_LLM_ERRORS

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(
        self,
        *,
        agent_name: str,
        model_factory: Callable[[], BaseChatModel],
        tools: list[BaseTool],
        max_tool_retries: int = 3,
        max_invoke_retries: int = 3,
        llm_retry_max_attempts: int = 3,
        llm_retry_base_delay_seconds: float = 0.5,
        strict_tool_schema: bool | Callable[[], bool] = False,
    ) -> None:
        self.agent_name = agent_name
        self.model_factory = model_factory
        self.tools = tools
        self.tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
        self.max_tool_retries = max_tool_retries
        self.max_invoke_retries = max_invoke_retries
        self.llm_retry_max_attempts = llm_retry_max_attempts
        self.llm_retry_base_delay_seconds = llm_retry_base_delay_seconds
        # When truthy at call time, the model is bound with per-tool strict
        # schemas so the provider (OpenAI) uses constrained decoding to
        # guarantee tool calls match the JSON schema — required-argument
        # elision becomes impossible.  Accepted as ``bool`` or a callable
        # so callers whose effective provider can change at runtime (e.g.
        # ``Context.provider`` is mutable) stay consistent with
        # ``model_factory``'s dynamic-capture pattern.  Per-tool strict is
        # gated by each tool's ``metadata["strict_schema_compatible"]``;
        # tools without it are bound non-strict so a single bad schema
        # (e.g. an external MCP server tool whose schema isn't a strict
        # subset) cannot poison every request with a 400.
        self.strict_tool_schema = strict_tool_schema

    # ------------------------------------------------------------------
    # Public: LLM invocation
    # ------------------------------------------------------------------

    async def call_llm(
        self,
        messages: list[BaseMessage],
        *,
        system_prompt: str,
    ) -> AIMessage:
        """Invoke the LLM with tools bound and full retry/repair semantics.

        Repairs orphan ``AIMessage.tool_calls`` in the persisted history
        before sending so OpenAI/Anthropic's pairing invariant holds even
        on threads resumed from a checkpointer mid-tool-execution.
        """
        repaired = self._ensure_tool_call_pairs(messages)
        payload = [{"role": "system", "content": system_prompt}, *repaired]
        model = self.model_factory()
        if self.tools:
            if self._resolve_strict_tool_schema():
                model_with_tools = model.bind_tools(self._build_strict_tool_dicts())
            else:
                model_with_tools = model.bind_tools(self.tools)
        else:
            model_with_tools = model
        return await self._ainvoke_with_llm_retry(model_with_tools, payload)

    def _resolve_strict_tool_schema(self) -> bool:
        """Evaluate ``strict_tool_schema`` per-call.

        Bool form is read directly; callable form is invoked so a
        provider mutation (``ctx.provider = "ollama"``) is reflected on
        the very next ``call_llm``.  The model factory captures
        ``self`` dynamically for the same reason — keeping strict
        resolution dynamic too avoids the inconsistency of
        ``bind_tools(strict=True)`` being passed to a freshly-built
        Ollama model whose ``bind_tools`` raises ``TypeError``.
        """
        flag = self.strict_tool_schema
        if callable(flag):
            return bool(flag())
        return bool(flag)

    def _build_strict_tool_dicts(self) -> list[dict[str, Any]]:
        """Convert ``self.tools`` to OpenAI tool dicts with per-tool strict.

        A tool is bound with ``strict=True`` only if its metadata carries
        ``strict_schema_compatible=True``.  This is opt-in because
        OpenAI rejects the *entire* request with a 400 if any single
        tool's JSON schema falls outside strict mode's subset (e.g.
        bare ``dict[str, Any]`` properties from a remote MCP server),
        and external MCP tool schemas are server-defined — outside the
        wiring user's control.
        """
        # Imported lazily so providers/clients that never enable strict
        # don't pay the import cost on every agent construction.
        from langchain_core.utils.function_calling import (  # noqa: PLC0415
            convert_to_openai_tool,
        )

        out: list[dict[str, Any]] = []
        for t in self.tools:
            metadata = getattr(t, "metadata", None) or {}
            tool_strict = bool(metadata.get("strict_schema_compatible"))
            out.append(convert_to_openai_tool(t, strict=tool_strict))
        return out

    async def call_llm_no_tools(
        self,
        messages: list[BaseMessage],
        *,
        system_prompt: str,
    ) -> AIMessage:
        """Invoke the LLM *without* tools — for graceful-degradation paths
        (e.g. error_summary_node) where the model must produce a final
        natural-language response.

        Still repairs orphan tool_calls: this path is reachable directly
        from the assistant node when ``should_continue`` routes to
        error_summary on an exhausted budget — the latest AIMessage may
        carry unresolved tool_calls that would otherwise trigger a
        provider-side 400.
        """
        repaired = self._ensure_tool_call_pairs(messages)
        payload = [{"role": "system", "content": system_prompt}, *repaired]
        model = self.model_factory()
        return await self._ainvoke_with_llm_retry(model, payload)

    # ------------------------------------------------------------------
    # Public: tool execution
    # ------------------------------------------------------------------

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute every tool call sequentially with full state hygiene.

        Reads from ``state``: ``tool_feedback_history``, ``tool_retry_attempts``.
        Returns a dict containing ``messages``, ``tool_feedback`` (latest),
        ``tool_feedback_history``, ``tool_retry_attempts``. Caller composes
        any agent-specific fields (e.g. ``tool_call_count``) on top.
        """
        history: list[ToolFeedback] = list(state.get("tool_feedback_history") or [])
        retry_attempts = int(state.get("tool_retry_attempts") or 0)
        previous_errors_str = self._serialize_previous_errors(history)

        tool_messages: list[ToolMessage] = []
        latest_feedback: ToolFeedback | None = None
        new_retry_attempts = retry_attempts

        # Refuse a multi-tool batch that mixes a pause-capable tool with
        # siblings.  LangGraph replays the entire node on resume, so
        # whatever sibling tools fired before the interrupt would execute
        # again and duplicate non-idempotent side effects (DB writes,
        # external API calls, emails).  Surface a clear TOOL_ERROR for
        # each tool_call so the LLM retries by calling the pausing tool
        # alone — the AIMessage<->ToolMessage pairing invariant is held
        # by emitting one ToolMessage per call.
        if len(tool_calls) > 1:
            pausing = [
                tc for tc in tool_calls if self._tool_pauses_graph(tc)
            ]
            if pausing:
                pausing_names = sorted({
                    tc.get("name") if isinstance(tc, dict)
                    else getattr(tc, "name", "")
                    for tc in pausing
                })
                refusal = (
                    "TOOL_ERROR attempt=0 type=BATCH_REJECTED "
                    "retryable=true\n"
                    f"message: pause-capable tool(s) {pausing_names} "
                    "must be called alone in a turn; LangGraph replays "
                    "the node from the top on resume, so any sibling "
                    "tools that already ran would execute again. "
                    "Re-emit the pause-capable tool as the only "
                    "tool_call in its AIMessage.\n"
                    f"context: batch contained {len(tool_calls)} "
                    "tool_calls"
                )
                refusal_messages: list[ToolMessage] = []
                for tc in tool_calls:
                    tc_id = self._extract_tool_call_id(tc)
                    if tc_id is None:
                        continue
                    tc_name = (
                        tc.get("name") if isinstance(tc, dict)
                        else getattr(tc, "name", None)
                    )
                    refusal_messages.append(
                        ToolMessage(
                            content=refusal,
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                    )
                refusal_feedback = ToolFeedback(
                    error=ErrorResponse(
                        error_type="BATCH_REJECTED",
                        message=(
                            f"pause-capable tool(s) {pausing_names} "
                            "must be called alone"
                        ),
                        retryable=True,
                        context=(
                            f"batch contained {len(tool_calls)} tool_calls"
                        ),
                    )
                )
                history.append(refusal_feedback)
                return {
                    "messages": refusal_messages,
                    "tool_feedback": refusal_feedback,
                    "tool_feedback_history": history,
                    "tool_retry_attempts": min(
                        retry_attempts + 1, self.max_tool_retries
                    ),
                }

        for tc in tool_calls:
            tool_call_id = self._extract_tool_call_id(tc)
            if tool_call_id is None:
                # Skip — emitting a ToolMessage with a fabricated id would
                # not satisfy the AIMessage<->ToolMessage pairing rule.
                # The orphan is surfaced by _ensure_tool_call_pairs on the
                # next assistant turn.
                continue

            tc_name = (
                tc.get("name") if isinstance(tc, dict)
                else getattr(tc, "name", None)
            )

            try:
                feedback = await self._execute_single_tool(
                    tc, state, previous_errors_str
                )
            except GraphBubbleUp:
                # LangGraph signals interrupt() / Command-routed control flow
                # by raising. Must propagate, never wrap as TOOL_ERROR.
                raise
            except Exception as e:
                # Should not normally fire — _execute_single_tool catches
                # tool-side exceptions itself. This is a defensive net for
                # bugs in the wrapper.
                logger.exception(
                    "execute_tool_calls_unexpected agent=%s id=%s",
                    self.agent_name,
                    tool_call_id,
                )
                feedback = ToolFeedback(
                    error=ErrorResponse(
                        error_type="EXECUTION_ERROR",
                        message=str(e),
                        retryable=True,
                        context="execute_tool_calls wrapper",
                    )
                )

            latest_feedback = feedback
            history.append(feedback)

            if feedback.is_error():
                if feedback.error.retryable and retry_attempts < self.max_tool_retries:
                    new_retry_attempts = retry_attempts + 1
                else:
                    new_retry_attempts = self.max_tool_retries
                content = (
                    f"TOOL_ERROR attempt={new_retry_attempts} "
                    f"type={feedback.error.error_type} "
                    f"retryable={feedback.error.retryable}\n"
                    f"message: {feedback.error.message}\n"
                    f"context: {feedback.error.context or ''}"
                )
            else:
                new_retry_attempts = 0
                content = (
                    f"TOOL_SUCCESS\n{feedback.success.results}\n\n"
                    f"TOOL_QUERY\n{feedback.query or ''}"
                )

            # ``name`` is required for downstream consumers that scan the
            # message tree by tool name (e.g. UI builders extracting the
            # query_monthly_sales rows for chart rendering).  LangChain's
            # ``ToolMessage`` accepts it as an optional field; without it,
            # the only way to correlate a ToolMessage back to its origin
            # tool is by walking tool_calls on the prior AIMessage.
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tc_name,
                )
            )

        return {
            "messages": tool_messages,
            "tool_feedback": latest_feedback,
            "tool_feedback_history": history,
            "tool_retry_attempts": new_retry_attempts,
        }

    async def execute_tool_calls_safe(
        self,
        tool_calls: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """``execute_tool_calls`` wrapped in a catastrophic-fallback try/except.

        On unexpected exception emits one error ``ToolMessage`` per tool_call
        (preserves OpenAI's pairing invariant) and snaps
        ``tool_retry_attempts`` to ``max_tool_retries`` so the graph routes
        to its error-summary path on the next turn.
        """
        try:
            return await self.execute_tool_calls(tool_calls, state)
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.exception(
                "execute_tool_calls_catastrophic agent=%s", self.agent_name
            )
            return {
                "messages": self._create_fallback_error_messages(
                    tool_calls, f"Internal error in tool execution: {e}"
                ),
                "tool_retry_attempts": self.max_tool_retries,
            }

    # ------------------------------------------------------------------
    # Public: turn-state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_new_human_turn(messages: list[BaseMessage]) -> bool:
        """True iff the last persisted message is a ``HumanMessage``.

        Used to decide whether retry / feedback state from a prior turn on
        the same thread should be reset before invoking the LLM.
        """
        return bool(messages) and isinstance(messages[-1], HumanMessage)

    @staticmethod
    def with_turn_reset(
        update: dict[str, Any],
        is_new_human_turn: bool,
        *,
        extra_reset_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge per-turn reset fields into ``update`` when a new user turn
        began. ``tool_feedback_history`` is intentionally preserved."""
        if is_new_human_turn:
            update["tool_retry_attempts"] = 0
            update["tool_feedback"] = None
            if extra_reset_fields:
                update.update(extra_reset_fields)
        return update

    # ------------------------------------------------------------------
    # Public: telemetry hook (override to forward to logger / metrics)
    # ------------------------------------------------------------------

    def on_tool_event(self, event: dict[str, Any]) -> None:
        """Default no-op. Subclasses override to forward events.

        Events emitted by the base class:
        - ``{"event": "tool_call_pair_repaired", "orphan_id": str, "position": "mid"|"tail"}``
        - ``{"event": "tool.invoke.start", "tool": str, "tool_call_id": str, "args": dict}``
        - ``{"event": "tool.invoke.end", "tool": str, "tool_call_id": str, "duration_ms": int}``
        - ``{"event": "tool.invoke.error", "tool": str, "tool_call_id": str, "error": str}``
        """

    # ------------------------------------------------------------------
    # Internal: single-tool execution
    # ------------------------------------------------------------------

    def _tool_pauses_graph(self, tc: dict[str, Any] | Any) -> bool:
        """True iff ``tc`` names a tool flagged ``pauses_graph`` in metadata.

        Tools that call ``langgraph.types.interrupt`` (or otherwise raise
        ``GraphBubbleUp``) opt in by setting ``metadata={"pauses_graph":
        True}`` on themselves.  ``execute_tool_calls`` consults this
        before executing a multi-tool batch so a pausing tool can't be
        silently mixed with siblings whose side effects would replay on
        resume.
        """
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        tool = self.tools_by_name.get(name) if name else None
        if tool is None:
            return False
        metadata = getattr(tool, "metadata", None) or {}
        return bool(metadata.get("pauses_graph"))

    async def _execute_single_tool(
        self,
        tc: dict[str, Any] | Any,
        state: dict[str, Any],
        previous_errors_str: str | None,
    ) -> ToolFeedback:
        import time

        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        raw_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        tool_call_id = self._extract_tool_call_id(tc) or "<missing>"

        if not name or name not in self.tools_by_name:
            logger.error(
                "tool_not_found agent=%s tool=%s", self.agent_name, name
            )
            return ToolFeedback(
                error=ErrorResponse(
                    error_type="TOOL_NOT_FOUND",
                    message=f"Unknown tool '{name}'.",
                    retryable=False,
                    context=None,
                )
            )

        tool_func = self.tools_by_name[name]

        # Copy before mutating: tc["args"] is the same dict stored on the
        # persisted AIMessage.tool_calls, so in-place injection would
        # create a state cycle that crashes LangChain's serializer.
        if isinstance(raw_args, dict):
            args: dict[str, Any] = dict(raw_args)
        else:
            args = {"user_question": str(raw_args)}

        injected_names = self._inject_state_args(tool_func, args, state)

        if previous_errors_str and self._tool_accepts(tool_func, "previous_errors"):
            args["previous_errors"] = previous_errors_str

        self.on_tool_event(
            {
                "event": "tool.invoke.start",
                "tool": name,
                "tool_call_id": tool_call_id,
                "args": self._redact_injected_for_log(args, injected_names),
            }
        )

        start = time.perf_counter()
        try:
            raw = await self._ainvoke_tool_with_retry(tool_func, args)
            feedback = self._wrap_as_tool_feedback(raw)
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.on_tool_event(
                {
                    "event": "tool.invoke.end",
                    "tool": name,
                    "tool_call_id": tool_call_id,
                    "duration_ms": duration_ms,
                }
            )
            return feedback
        except GraphBubbleUp:
            # interrupt()/Command propagation: not a tool failure.
            # If the tool raised this without declaring itself
            # pause-capable in its metadata, the upstream batch-refusal
            # guard in execute_tool_calls did not protect siblings — log
            # loudly so the tool author sees the contract violation
            # during development.  We can't *prevent* the resulting
            # side-effect duplication on resume from here (siblings have
            # already run), but a noisy warning is cheap and catches
            # the bug at the earliest possible moment.
            metadata = getattr(tool_func, "metadata", None) or {}
            if not metadata.get("pauses_graph"):
                logger.warning(
                    "undeclared_pausing_tool agent=%s tool=%s id=%s — tool "
                    "raised GraphBubbleUp but its metadata does not declare "
                    "pauses_graph=True. Set metadata={'pauses_graph': True} "
                    "on the tool so BaseAgent.execute_tool_calls can refuse "
                    "multi-tool batches that mix it with siblings (otherwise "
                    "earlier sibling tools' side effects will duplicate on "
                    "resume).",
                    self.agent_name,
                    name,
                    tool_call_id,
                )
            raise
        except Exception as e:
            logger.exception(
                "tool_execution_failed agent=%s tool=%s id=%s",
                self.agent_name,
                name,
                tool_call_id,
            )
            self.on_tool_event(
                {
                    "event": "tool.invoke.error",
                    "tool": name,
                    "tool_call_id": tool_call_id,
                    "error": str(e),
                }
            )
            return ToolFeedback(
                error=ErrorResponse(
                    error_type="EXECUTION_ERROR",
                    message=str(e),
                    retryable=True,
                    context="internal tool wrapper",
                )
            )

    def _wrap_as_tool_feedback(self, raw: Any) -> ToolFeedback:
        """Normalize a tool's return into ``ToolFeedback``.

        Cases:
        - already a ``ToolFeedback`` → passthrough.
        - a ``dict`` matching the shape → ``ToolFeedback(**raw)``.
        - a list of MCP TextContent blocks
          (``[{"type": "text", "text": "..."}, ...]`` per the MCP spec)
          → concatenate the ``text`` payloads.  This is what
          ``langchain-mcp-adapters`` returns when an MCP tool reply has
          no ``structuredContent``; using ``str(raw)`` here would emit
          a Python *repr* (single quotes, escaped strings) that's
          neither valid JSON nor LLM-friendly.  Concatenation gives
          downstream consumers (and the LLM on the next turn) the
          actual text payload.  The shape check requires keys be a
          subset of MCP TextContent's ``{type, text, annotations,
          _meta}`` so a non-MCP tool that returns dicts with extra
          fields (e.g. an ``id`` alongside ``text``) doesn't get its
          fields silently dropped.
        - anything else → ``SuccessResponse(results=str(raw))``.
        """
        if isinstance(raw, ToolFeedback):
            return raw
        if isinstance(raw, dict):
            try:
                return ToolFeedback(**raw)
            except (TypeError, ValidationError):
                # Dict didn't match the schema — treat its repr as a
                # plain success payload rather than failing the call.
                # Other exception types are propagated (a real bug, not
                # a shape mismatch).
                return ToolFeedback(success=SuccessResponse(results=str(raw)))
        if (
            isinstance(raw, list)
            and raw
            and all(self._is_mcp_text_block(b) for b in raw)
        ):
            joined = "\n".join(b["text"] for b in raw)
            return ToolFeedback(success=SuccessResponse(results=joined))
        return ToolFeedback(success=SuccessResponse(results=str(raw)))

    @staticmethod
    def _is_mcp_text_block(b: Any) -> bool:
        """True iff ``b`` matches MCP TextContent strictly.

        Per the MCP spec, TextContent has ``type``, ``text``, and
        optionally ``annotations`` / ``_meta``.  Requiring a strict
        subset prevents accidentally collapsing a non-MCP tool's
        ``[{"type":"text","text":"x","id":"abc"}, ...]`` (which carries
        an ``id`` we'd silently drop) into a flat string.
        """
        return (
            isinstance(b, dict)
            and b.get("type") == "text"
            and isinstance(b.get("text"), str)
            and set(b.keys()) <= {"type", "text", "annotations", "_meta"}
        )

    # ------------------------------------------------------------------
    # Internal: retry helpers
    # ------------------------------------------------------------------

    async def _ainvoke_with_llm_retry(self, model: Any, payload: Any) -> AIMessage:
        """Retry ``model.ainvoke(payload)`` on transient LLM errors.

        Permanent errors (BadRequest, AuthN, AuthZ) are re-raised
        immediately. Transient errors (5xx, rate limit, timeout, network)
        are retried with exponential backoff up to
        ``llm_retry_max_attempts`` times, then the last exception is
        re-raised.
        """
        last_error: BaseException | None = None
        for attempt in range(1, self.llm_retry_max_attempts + 1):
            try:
                return await model.ainvoke(payload)
            except PERMANENT_LLM_ERRORS:
                logger.exception(
                    "llm_permanent_error agent=%s attempt=%d",
                    self.agent_name,
                    attempt,
                )
                raise
            except TRANSIENT_LLM_ERRORS as e:
                last_error = e
                if attempt == self.llm_retry_max_attempts:
                    break
                delay = self.llm_retry_base_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "llm_transient_retry agent=%s attempt=%d/%d error=%s delay_s=%.2f",
                    self.agent_name,
                    attempt,
                    self.llm_retry_max_attempts,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _ainvoke_tool_with_retry(self, tool: BaseTool, args: dict[str, Any]) -> Any:
        """Retry ``tool.ainvoke(args)`` on transient HTTP-class errors.

        Reuses ``_is_retryable_error`` (typed-class check via
        ``TRANSIENT_LLM_ERRORS`` with ``ExceptionGroup`` walk) so any
        LangChain tool that surfaces openai/anthropic exceptions inherits
        the same classification.
        """
        last_error: BaseException | None = None
        for attempt in range(self.max_invoke_retries + 1):
            try:
                return await tool.ainvoke(args)
            except GraphBubbleUp:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_invoke_retries and self._is_retryable_error(e):
                    delay = 2**attempt
                    logger.warning(
                        "tool_transient_retry agent=%s tool=%s attempt=%d/%d delay_s=%d error=%s",
                        self.agent_name,
                        tool.name,
                        attempt + 1,
                        self.max_invoke_retries,
                        delay,
                        type(e).__name__,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        assert last_error is not None
        raise last_error

    def _is_retryable_error(self, exc: BaseException) -> bool:
        """True iff ``exc`` (or any nested exception) is a transient LLM error.

        Walks ``ExceptionGroup`` because some async transports wrap real
        causes in groups.
        """
        if isinstance(exc, ExceptionGroup):
            return any(self._is_retryable_error(sub) for sub in exc.exceptions)
        return isinstance(exc, TRANSIENT_LLM_ERRORS)

    # ------------------------------------------------------------------
    # Internal: history / pairing repair
    # ------------------------------------------------------------------

    def _serialize_previous_errors(
        self, history: list[ToolFeedback]
    ) -> str | None:
        """Build the ``previous_errors`` string from the last 3 errored entries.

        Forwarded to tools that declare a ``previous_errors`` field in
        their args_schema, enabling self-correction (e.g. "previous attempt
        failed because column X did not exist — try column Y").
        """
        errors = [
            f"{h.error.message}: {h.error.context}" for h in history if h.is_error()
        ]
        if not errors:
            return None
        tail = errors[-3:]
        return "\n".join(f"- {m}" for m in tail)

    def _ensure_tool_call_pairs(self, messages: list[Any]) -> list[Any]:
        """Insert synthetic ``ToolMessage`` for any orphan ``tool_calls.id``.

        OpenAI/Anthropic enforce that every ``AIMessage.tool_calls[*].id``
        has a matching ``ToolMessage`` before the next assistant/user
        turn. Threads resumed from a checkpointer that was interrupted
        between the assistant and tool nodes contain orphans; this repair
        runs at the LLM boundary so the next request is well-formed. The
        repair is *not* persisted — the next turn re-repairs
        deterministically.
        """
        repaired: list[Any] = []
        # Tracks (id, name) so synthetic repair messages can carry the
        # tool's ``name`` like every other emitted ToolMessage.
        open_pairs: list[tuple[str, str | None]] = []

        def _flush_orphans(position: str) -> None:
            for orphan_id, orphan_name in open_pairs:
                repaired.append(
                    self._make_synthetic_tool_message(orphan_id, orphan_name)
                )
                logger.warning(
                    "tool_call_pair_repaired agent=%s orphan_id=%s position=%s",
                    self.agent_name,
                    orphan_id,
                    position,
                )
                self.on_tool_event(
                    {
                        "event": "tool_call_pair_repaired",
                        "orphan_id": orphan_id,
                        "position": position,
                    }
                )
            open_pairs.clear()

        for m in messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                _flush_orphans("mid")
                for tc in m.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        open_pairs.append((tc_id, tc.get("name")))
                repaired.append(m)
                continue
            if isinstance(m, ToolMessage):
                for pair in open_pairs:
                    if pair[0] == m.tool_call_id:
                        open_pairs.remove(pair)
                        break
                repaired.append(m)
                continue
            _flush_orphans("mid")
            repaired.append(m)

        _flush_orphans("tail")
        return repaired

    def _make_synthetic_tool_message(
        self, tool_call_id: str, name: str | None = None
    ) -> ToolMessage:
        return ToolMessage(
            tool_call_id=tool_call_id,
            status="error",
            name=name,
            content=(
                "TOOL_INTERRUPTED: No response was recorded for this tool call. "
                "The prior run was terminated before the tool could execute. "
                "Decide whether to retry the call or respond without its output."
            ),
        )

    def _create_fallback_error_messages(
        self, tool_calls: list[dict[str, Any]], error_message: str
    ) -> list[ToolMessage]:
        out: list[ToolMessage] = []
        for tc in tool_calls:
            tool_call_id = self._extract_tool_call_id(tc)
            if tool_call_id is None:
                continue
            tc_name = (
                tc.get("name") if isinstance(tc, dict)
                else getattr(tc, "name", None)
            )
            out.append(
                ToolMessage(
                    content=(
                        f"TOOL_ERROR type=internal retryable=false\n"
                        f"message: {error_message}"
                    ),
                    tool_call_id=tool_call_id,
                    name=tc_name,
                )
            )
            logger.warning(
                "fallback_tool_message agent=%s id=%s msg=%s",
                self.agent_name,
                tool_call_id,
                error_message,
            )
        return out

    # ------------------------------------------------------------------
    # Internal: arg injection (InjectedState + redaction for logs)
    # ------------------------------------------------------------------

    def _inject_state_args(
        self,
        tool_func: BaseTool,
        args: dict[str, Any],
        state: dict[str, Any],
    ) -> list[str]:
        """Populate ``Annotated[..., InjectedState(...)]`` arguments.

        LangGraph's prebuilt ``ToolNode`` normally fulfils this contract.
        We bypass ``ToolNode`` to keep the retry / feedback loop, so we
        replicate it here. Returns the names of fields that were
        populated so callers can redact them from logs.
        """
        injected: list[str] = []
        schema = getattr(tool_func, "args_schema", None)
        if schema is None or not hasattr(schema, "model_fields"):
            return injected

        for field_name, field in schema.model_fields.items():
            for meta in field.metadata or ():
                if meta is InjectedState:
                    args[field_name] = state
                    injected.append(field_name)
                    break
                if isinstance(meta, InjectedState):
                    key = getattr(meta, "field", None)
                    if key is None:
                        args[field_name] = state
                    elif key in state:
                        args[field_name] = state[key]
                    else:
                        raise KeyError(
                            f"InjectedState field '{key}' (for tool arg "
                            f"'{field_name}', agent='{self.agent_name}') is "
                            f"not present in graph state. Available keys: "
                            f"{sorted(state.keys())}"
                        )
                    injected.append(field_name)
                    break
        return injected

    def _tool_accepts(self, tool_func: BaseTool, arg_name: str) -> bool:
        schema = getattr(tool_func, "args_schema", None)
        if schema is None or not hasattr(schema, "model_fields"):
            return False
        return arg_name in schema.model_fields

    def _redact_injected_for_log(
        self, args: dict[str, Any], injected_names: list[str]
    ) -> dict[str, Any]:
        """Return a display-only copy of ``args`` with injected values redacted.

        Injected state is typically the full graph state (messages +
        feedback history) — too noisy to log per call and grows
        unbounded. The key is preserved so injection regressions are
        visible; the value becomes a short shape hint.
        """
        if not injected_names:
            return args
        out: dict[str, Any] = {}
        for k, v in args.items():
            if k not in injected_names:
                out[k] = v
                continue
            if isinstance(v, dict):
                out[k] = f"<InjectedState keys={list(v.keys())}>"
            else:
                out[k] = f"<InjectedState type={type(v).__name__}>"
        return out

    # ------------------------------------------------------------------
    # Internal: misc
    # ------------------------------------------------------------------

    def _extract_tool_call_id(
        self, tc: dict[str, Any] | Any
    ) -> str | None:
        tool_call_id = (
            tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        )
        if not tool_call_id:
            logger.error(
                "tool_call_missing_id agent=%s tc=%r", self.agent_name, tc
            )
            return None
        return tool_call_id

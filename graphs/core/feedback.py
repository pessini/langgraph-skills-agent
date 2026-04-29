"""Structured tool feedback envelope used by ``BaseAgent.execute_tool_calls``.

A tool can return one of three shapes; the orchestrator wraps them all
into ``ToolFeedback``:

1. ``ToolFeedback`` — passthrough.
2. ``dict`` — parsed via ``ToolFeedback(**raw)``; the caller decides
   ``success`` / ``error`` / ``query`` keys.
3. anything else — auto-wrapped as
   ``ToolFeedback(success=SuccessResponse(results=str(value)))``.

This lets existing tools that return plain strings (e.g. JSON payloads)
keep working untouched while letting future tools opt into structured
errors with retryability flags and contextual hints.

Convention values for ``ErrorResponse.error_type`` (free-form ``str`` —
not enforced):

- ``DATA_ERROR`` — data shape / parsing problem
- ``CONNECTION_ERROR`` — network or upstream service unavailable
- ``PERMISSION_ERROR`` — auth / scope problem
- ``EXECUTION_ERROR`` — generic tool execution failure (default)
- ``TOOL_NOT_FOUND`` — tool name not registered
- ``UNKNOWN_TOOL`` — alias for the above
- ``AMBIGUOUS_REQUEST`` — input under-specified
- any tool-specific value
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """Successful tool result. ``results`` is the stringified payload."""

    results: str


class ErrorResponse(BaseModel):
    """Structured tool failure with retryability hint.

    The ``retryable`` flag is consulted by ``BaseAgent.execute_tool_calls``
    to decide whether to bump the retry counter (retryable) or snap it to
    ``max_tool_retries`` to short-circuit further attempts.
    """

    error_type: str
    message: str
    retryable: bool
    context: Optional[str] = None


class ToolFeedback(BaseModel):
    """Either a success or an error, plus an optional ``query`` echo.

    ``query`` is emitted in the success envelope as ``TOOL_QUERY\n{q}`` so
    the LLM can see what action the tool actually took (especially useful
    for SQL / search / parameterised tools).
    """

    success: Optional[SuccessResponse] = None
    error: Optional[ErrorResponse] = None
    query: Optional[str] = None

    def is_success(self) -> bool:
        return self.success is not None

    def is_error(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        if self.is_success():
            return f"ToolFeedback(SUCCESS: {self.success.results} query={self.query})"
        if self.is_error():
            return (
                f"ToolFeedback(ERROR: {self.error.error_type} - "
                f"{self.error.message} query={self.query})"
            )
        return "ToolFeedback(EMPTY)"

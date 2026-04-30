"""Provider-agnostic LLM error classification.

Both the ``openai`` and ``anthropic`` SDKs expose the same exception
hierarchy (``RateLimitError``, ``BadRequestError``, etc.) under their own
top-level modules. We aggregate whichever providers are installed so
callers can catch by *behavior* — transient vs permanent — rather than
by provider.

Usage:

    from core.llm_errors import TRANSIENT_LLM_ERRORS, PERMANENT_LLM_ERRORS

    try:
        response = await chain.ainvoke({})
    except PERMANENT_LLM_ERRORS:
        raise              # bug or operator issue; surface it
    except TRANSIENT_LLM_ERRORS:
        ...                # retry with backoff

Adding a new provider is a single ``try/except ImportError`` block. If
neither SDK is installed the tuples are empty and the ``except`` branches
become no-ops, which is the intended behavior for unit tests with fully
mocked chains.
"""
from __future__ import annotations

_transient: tuple[type[BaseException], ...] = ()
_permanent: tuple[type[BaseException], ...] = ()

try:
    from openai import (
        APIConnectionError as _OAIConnection,
        APITimeoutError as _OAITimeout,
        AuthenticationError as _OAIAuth,
        BadRequestError as _OAIBadRequest,
        InternalServerError as _OAIInternal,
        PermissionDeniedError as _OAIPermission,
        RateLimitError as _OAIRateLimit,
    )

    _transient += (_OAIConnection, _OAITimeout, _OAIRateLimit, _OAIInternal)
    _permanent += (_OAIBadRequest, _OAIAuth, _OAIPermission)
except ImportError:  # pragma: no cover
    pass

try:
    from anthropic import (
        APIConnectionError as _AnthConnection,
        APITimeoutError as _AnthTimeout,
        AuthenticationError as _AnthAuth,
        BadRequestError as _AnthBadRequest,
        InternalServerError as _AnthInternal,
        PermissionDeniedError as _AnthPermission,
        RateLimitError as _AnthRateLimit,
    )

    _transient += (_AnthConnection, _AnthTimeout, _AnthRateLimit, _AnthInternal)
    _permanent += (_AnthBadRequest, _AnthAuth, _AnthPermission)
except ImportError:  # pragma: no cover
    pass


TRANSIENT_LLM_ERRORS: tuple[type[BaseException], ...] = _transient
"""Errors worth retrying with exponential backoff.

Network/timeout failures, rate-limit responses (HTTP 429), and server-side
5xx. Excludes ``BadRequestError`` — post-repair, 400s indicate structural
bugs (malformed history, bad tool schema, missing fields) that retrying
cannot fix.
"""

PERMANENT_LLM_ERRORS: tuple[type[BaseException], ...] = _permanent
"""Errors that must not be silently caught into a user-facing message.

A ``BadRequestError`` here signals a regression in the invariant repair
or tool schema; ``AuthenticationError`` / ``PermissionDeniedError``
require operator action. Re-raising surfaces the failure in telemetry
instead of papering over it.
"""

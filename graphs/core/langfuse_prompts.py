"""Generic Langfuse-prompt fetcher — future drop-in for prompt management.

**Not currently wired in.** The skills_agent uses a static
``SYSTEM_PROMPT`` template formatted at runtime. To switch to Langfuse-
managed prompts, replace the ``SYSTEM_PROMPT.format(...)`` call in
``skills_agent/nodes.py::agent_node`` with::

    from core.langfuse_prompts import get_langfuse_prompt
    system_prompt, _ = get_langfuse_prompt(
        "skills_agent/system",
        compile_vars={
            "current_time": datetime.now(tz=UTC).isoformat(),
            "skill_catalog": ctx.skill_store.get_skill_catalog(),
        },
    )

The ``langfuse`` import is deferred inside the function so this module
imports without the SDK loaded; a clear ``ImportError`` is raised on the
first call instead.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_langfuse_prompt(
    prompt_name: str,
    *,
    compile_vars: dict[str, Any] | None = None,
    label: str = "production",
    cache_ttl_seconds: int = 300,
    custom_suffix: str | None = None,
) -> tuple[str, Any]:
    """Retrieve a prompt from Langfuse, optionally with custom-client fallback.

    Args:
        prompt_name: Base prompt name in Langfuse (e.g. ``"skills_agent/system"``).
        compile_vars: If provided, the template is compiled with these
            variables (Jinja2-style ``{{var}}`` placeholders); otherwise
            the raw prompt body is returned.
        label: Langfuse label / version (default ``"production"``).
        cache_ttl_seconds: Client-side cache TTL.
        custom_suffix: If set, the function first tries
            ``f"{prompt_name}-{custom_suffix}"`` and falls back to
            ``prompt_name`` on miss. Useful for per-tenant overrides.

    Returns:
        ``(rendered_prompt, prompt_template)`` — the second item is the raw
        Langfuse ``TextPrompt`` (used as a callback for trace metadata).

    Raises:
        ImportError: if the ``langfuse`` SDK is not installed.
        Exception: any error from the Langfuse client when the base
            prompt cannot be retrieved.
    """
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise ImportError(
            "core.langfuse_prompts requires the 'langfuse' package. "
            "Install with: uv add langfuse"
        ) from exc

    client = get_client()

    prompt_template = None
    final_name = prompt_name

    if custom_suffix:
        custom_name = f"{prompt_name}-{custom_suffix}"
        try:
            prompt_template = client.get_prompt(
                name=custom_name,
                cache_ttl_seconds=cache_ttl_seconds,
                label=label,
            )
            final_name = custom_name
        except Exception as e:
            # Only swallow "not found" style errors. Auth / network /
            # config failures (AuthenticationError, ConnectionError,
            # ConfigError, etc.) must propagate so they don't get
            # masked as a missing custom prompt — that's the difference
            # between "fall back to base prompt" and "Langfuse is down".
            cls_name = type(e).__name__
            if "NotFound" not in cls_name and "404" not in str(e):
                raise
            logger.warning(
                "Custom prompt '%s' not found (%s), falling back to '%s': %s",
                custom_name,
                cls_name,
                prompt_name,
                e,
            )

    if prompt_template is None:
        prompt_template = client.get_prompt(
            name=prompt_name,
            cache_ttl_seconds=cache_ttl_seconds,
            label=label,
        )
        final_name = prompt_name

    logger.info("Retrieved prompt '%s' (label=%s) from Langfuse", final_name, label)

    if compile_vars is not None:
        # Empty dict still triggers compile() — useful for templates with
        # no placeholders or where the caller wants explicit "render now"
        # semantics. Truthiness would skip the call and return the raw
        # template body, conflicting with the docstring contract.
        return prompt_template.compile(**compile_vars), prompt_template
    return prompt_template.prompt, prompt_template

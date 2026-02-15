"""Helpers for progressive disclosure logging."""

from __future__ import annotations

import logging
from typing import Any, Iterable


LOGGER_NAME = "skills_agent.progressive"

ORDERED_FIELDS: tuple[str, ...] = (
    "tier",
    "event",
    "skill",
    "file",
    "cache",
    "tool",
    "tool_call_id",
    "duration_ms",
    "summary",
)


def get_progressive_logger() -> logging.Logger:
    """Return the dedicated progressive disclosure logger."""
    return logging.getLogger(LOGGER_NAME)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return True
    return False


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch.isspace() for ch in text) or '"' in text:
        escaped = text.replace('"', "'")
        return f'"{escaped}"'
    return text


def format_progressive_event(event: dict[str, Any]) -> str:
    """Format a progressive disclosure event into a readable log line."""
    parts: list[str] = ["[progressive]"]
    for key in ORDERED_FIELDS:
        if key not in event:
            continue
        value = event.get(key)
        if _is_empty(value):
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def log_progressive(event: dict[str, Any], level: int = logging.INFO) -> None:
    """Log a progressive disclosure event using the dedicated logger."""
    logger = get_progressive_logger()
    logger.log(level, format_progressive_event(event))


def summarize_mapping(mapping: dict[str, Any], *, max_items: int = 4, max_value_len: int = 80) -> str:
    """Summarize a mapping into a compact key=value string."""
    if not mapping:
        return ""
    items: Iterable[tuple[str, Any]] = list(mapping.items())[:max_items]
    parts: list[str] = []
    for key, value in items:
        text = str(value)
        if len(text) > max_value_len:
            text = text[: max_value_len - 3] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)

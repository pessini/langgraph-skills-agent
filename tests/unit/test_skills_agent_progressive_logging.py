"""Unit tests for progressive disclosure logging helpers."""

import pytest

from skills_agent.utils.progressive_logging import format_progressive_event


@pytest.mark.unit
def test_format_progressive_event_orders_fields_and_omits_empty() -> None:
    event = {
        "event": "catalog.render",
        "tier": 1,
        "summary": "skills_count=2 chars=120",
        "tool": "",
        "extra": "ignored",
    }
    line = format_progressive_event(event)
    assert line.startswith("[progressive]")
    assert "tier=1" in line
    assert "event=catalog.render" in line
    assert "summary=\"skills_count=2 chars=120\"" in line
    assert "tool=" not in line
    assert "extra=" not in line
    assert line.index("tier=1") < line.index("event=catalog.render") < line.index("summary=")


@pytest.mark.unit
def test_format_progressive_event_handles_file_field() -> None:
    event = {
        "tier": 3,
        "event": "supporting_file.read",
        "skill": "n8n-code-javascript",
        "file": "COMMON_PATTERNS.md",
        "summary": "chars=1200",
    }
    line = format_progressive_event(event)
    assert "tier=3" in line
    assert "event=supporting_file.read" in line
    assert "skill=n8n-code-javascript" in line
    assert "file=COMMON_PATTERNS.md" in line

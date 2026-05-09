"""Human-in-the-loop review tool for the skills agent.

Provides ``request_human_review``, a domain-agnostic tool that any skill
can call to escalate a decision to a human operator. The tool internally
calls ``langgraph.types.interrupt`` to pause the graph; the resume value
is returned to the LLM as a JSON-encoded ``ReviewDecision``.

The interrupt payload is a small generic envelope:

- ``request_id``: tool-generated identifier. UI uses it to correlate the
  displayed card with the resume call.
- ``title`` / ``summary``: human-readable text shown on the review card.
- ``options``: list of ``{"value", "label", "variant"}`` entries — the UI
  renders one button per entry.
- ``payload``: opaque domain dict the UI may render alongside the buttons
  (e.g. anomaly metrics, failed-transaction details).

The skill's ``SKILL.md`` instructs the LLM when to call this tool and what
to put in title/summary/options/payload. The agent code stays generic.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, TypedDict

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt


class ReviewOption(TypedDict):
    value: str
    label: str
    variant: str  # "default" | "secondary" | "destructive" | "outline"


class ReviewRequest(TypedDict):
    request_id: str
    title: str
    summary: str
    options: list[ReviewOption]
    payload: dict[str, Any]


class ReviewDecision(TypedDict):
    request_id: str
    value: str
    note: str | None


def create_review_tool() -> BaseTool:
    """Build the ``request_human_review`` tool.

    The tool is standalone — it does not depend on any other resource —
    so the factory takes no arguments.  Kept as a factory for symmetry
    with ``create_skill_tools()`` so the registration site in
    ``Context.initialize()`` is uniform.

    Returns a single ``BaseTool`` that should be added to the agent's
    tool list alongside ``load_skill`` / ``read_skill_file`` / external
    MCP tools.
    """

    @tool
    def request_human_review(
        title: str,
        summary: str,
        options_json: str,
        payload_json: str = "{}",
    ) -> str:
        """Pause execution and ask a human operator to make a decision.

        Use ONLY when explicitly instructed by a loaded skill that expects
        human-in-the-loop confirmation.  Calling this tool pauses the graph;
        the agent does not continue until a human responds via the UI.

        Args:
            title: Short label for the review card (e.g. "Anomaly review").
            summary: Human-readable paragraph explaining what needs review.
            options_json: JSON-encoded list of options.  Each option must
                have shape ``{"value": "<machine_value>", "label":
                "<button_text>", "variant": "default" | "secondary" |
                "destructive" | "outline"}``.  The UI renders one button
                per option.
            payload_json: JSON-encoded opaque dict of domain data the UI
                may display alongside the buttons (e.g. metrics, IDs).
                Optional; defaults to ``"{}"``.

        Returns:
            JSON-encoded ``ReviewDecision`` carrying the human's response,
            with shape ``{"request_id", "value", "note"}``.  The ``value``
            field will be one of the option values supplied above.
        """
        options = json.loads(options_json)
        payload = json.loads(payload_json) if payload_json else {}
        request_id = uuid.uuid4().hex[:12]
        request: ReviewRequest = {
            "request_id": request_id,
            "title": title,
            "summary": summary,
            "options": options,
            "payload": payload,
        }
        # interrupt() pauses the graph and surfaces ``request`` in the
        # snapshot's interrupts list; the resume value is whatever the
        # caller passes to ``Command(resume=...)``.
        decision: ReviewDecision = interrupt(request)
        return json.dumps(decision)

    return request_human_review  # type: ignore[return-value]

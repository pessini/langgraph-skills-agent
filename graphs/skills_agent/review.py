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

        IMPORTANT — call alone in a turn.  This tool MUST be the only
        ``tool_call`` in its ``AIMessage``.  ``BaseAgent`` rejects any
        batch that mixes it with sibling tool calls (returning
        ``TOOL_ERROR`` for each) because LangGraph replays the node from
        the top on resume — any sibling tool that already ran would
        execute again, duplicating non-idempotent side effects (writes,
        external API calls, emails).  If you need to gather context
        before requesting review, do it in an earlier turn and let the
        review be its own turn.

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
        valid_variants = {"default", "secondary", "destructive", "outline"}
        options = json.loads(options_json)
        if (
            not isinstance(options, list)
            or len(options) == 0
            or not all(
                isinstance(opt, dict)
                and isinstance(opt.get("value"), str)
                and opt["value"]
                and isinstance(opt.get("label"), str)
                and opt["label"]
                and opt.get("variant") in valid_variants
                for opt in options
            )
        ):
            raise ValueError(
                "options_json must decode to a non-empty JSON list of "
                "objects, each with a non-empty string 'value', non-empty "
                "string 'label', and 'variant' in "
                f"{sorted(valid_variants)}"
            )
        payload = json.loads(payload_json) if payload_json else {}
        if not isinstance(payload, dict):
            raise ValueError("payload_json must decode to a JSON object")
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
        raw_decision = interrupt(request)
        if not isinstance(raw_decision, dict):
            raise ValueError("Review decision must be a JSON object")
        allowed_values = {opt["value"] for opt in options}
        if raw_decision.get("value") not in allowed_values:
            raise ValueError(
                f"Review decision value {raw_decision.get('value')!r} is not "
                f"one of the supplied options {sorted(allowed_values)}"
            )
        # On resume LangGraph re-runs this node from the top, so the local
        # ``request_id`` is freshly generated and no longer matches the one
        # the UI saw.  The host-supplied request_id in the resume payload is
        # the authoritative correlation key; preserve it.
        decision: ReviewDecision = {
            "request_id": raw_decision.get("request_id", request_id),
            "value": raw_decision["value"],
            "note": raw_decision.get("note"),
        }
        return json.dumps(decision)

    # Mark the tool as pause-capable so ``BaseAgent.execute_tool_calls``
    # can detect a multi-tool batch that mixes it with siblings and
    # refuse the batch before any side effects fire (LangGraph replays
    # the node from the top on resume — sibling tools would duplicate).
    request_human_review.metadata = {"pauses_graph": True}
    return request_human_review  # type: ignore[return-value]

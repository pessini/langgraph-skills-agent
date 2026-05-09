"""Test the request_human_review tool's interrupt/resume flow.

The tool is generic (no domain knowledge); its only job is to pause the
graph and surface a structured payload, then return whatever value the
human supplied via ``Command(resume=...)`` as a JSON string the LLM can
parse downstream.

This test wires the tool into a tiny standalone graph (not the full
SkillsAgent graph) so the test stays focused on the interrupt/resume
mechanics rather than the LLM-routing logic.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command

from core.base_agent import BaseAgent
from skills_agent.review import create_review_tool


@pytest.mark.unit
async def test_request_human_review_pauses_then_resumes() -> None:
    """End-to-end: pause on interrupt, inspect the snapshot, resume with a
    decision, assert the tool's return value carries it.
    """
    review_tool = create_review_tool()

    options = [
        {"value": "approve", "label": "Approve", "variant": "default"},
        {"value": "deny", "label": "Deny", "variant": "destructive"},
    ]
    payload = {"foo": "bar", "count": 3}

    async def call_review_tool(state: MessagesState) -> dict[str, Any]:
        # Drive the tool directly with the same shape the LLM would use.
        # The tool will internally call interrupt() and pause execution
        # at this point.
        result = await review_tool.ainvoke({
            "title": "Test review",
            "summary": "Pretend something needs human review.",
            "options_json": json.dumps(options),
            "payload_json": json.dumps(payload),
        })
        return {"messages": [ToolMessage(content=result, tool_call_id="t1")]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_review_tool", call_review_tool)
    builder.add_edge("__start__", "call_review_tool")
    builder.add_edge("call_review_tool", "__end__")
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-review-thread"}}

    # First invocation: should pause at interrupt() inside the tool.
    await graph.ainvoke({"messages": []}, config=config)

    # Inspect the checkpoint snapshot for the interrupt payload.
    snapshot = await graph.aget_state(config)
    interrupts: list[Any] = []
    for task in snapshot.tasks:
        interrupts.extend(task.interrupts)

    assert len(interrupts) == 1, f"expected 1 interrupt, got {len(interrupts)}"
    interrupt_value = interrupts[0].value

    assert interrupt_value["title"] == "Test review"
    assert interrupt_value["summary"] == "Pretend something needs human review."
    assert interrupt_value["options"] == options
    assert interrupt_value["payload"] == payload
    request_id = interrupt_value["request_id"]
    assert isinstance(request_id, str) and len(request_id) > 0

    # Resume the graph with a decision — this becomes the return value
    # of interrupt() inside the tool, which the tool then JSON-encodes
    # and emits as its own return value.
    decision = {"request_id": request_id, "value": "approve", "note": None}
    final_state = await graph.ainvoke(Command(resume=decision), config=config)

    # The tool's return became the ToolMessage content; verify the
    # decision round-tripped intact.
    tool_messages = [
        msg for msg in final_state["messages"] if isinstance(msg, ToolMessage)
    ]
    assert len(tool_messages) == 1
    decoded = json.loads(tool_messages[0].content)
    assert decoded == decision


@pytest.mark.unit
async def test_payload_json_defaults_to_empty_dict() -> None:
    """Skills can omit ``payload_json``; the tool defaults it to ``"{}"``."""
    review_tool = create_review_tool()

    async def call_review_tool(state: MessagesState) -> dict[str, Any]:
        # Note: no payload_json provided.
        result = await review_tool.ainvoke({
            "title": "Minimal review",
            "summary": "No domain payload needed.",
            "options_json": json.dumps(
                [{"value": "ok", "label": "OK", "variant": "default"}]
            ),
        })
        return {"messages": [ToolMessage(content=result, tool_call_id="t1")]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_review_tool", call_review_tool)
    builder.add_edge("__start__", "call_review_tool")
    builder.add_edge("call_review_tool", "__end__")
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-no-payload"}}
    await graph.ainvoke({"messages": []}, config=config)

    snapshot = await graph.aget_state(config)
    interrupts = [iv for task in snapshot.tasks for iv in task.interrupts]
    assert len(interrupts) == 1, f"expected 1 interrupt, got {len(interrupts)}"
    assert interrupts[0].value["payload"] == {}


@pytest.mark.unit
async def test_interrupt_propagates_through_base_agent() -> None:
    """``BaseAgent.execute_tool_calls`` must let LangGraph's ``GraphInterrupt``
    propagate so the graph actually pauses for human input.

    Regression: the generic ``except Exception`` in the tool wrapper used to
    swallow ``GraphInterrupt`` (an ``Exception`` subclass) and convert it into
    a ``TOOL_ERROR``, silently breaking ``request_human_review`` end-to-end.
    """
    review_tool = create_review_tool()

    agent = BaseAgent(
        agent_name="test-interrupt",
        model_factory=lambda: pytest.fail("model_factory must not run"),
        tools=[review_tool],
        max_invoke_retries=0,
    )

    options = [{"value": "approve", "label": "Approve", "variant": "default"}]

    async def call_via_base_agent(state: MessagesState) -> dict[str, Any]:
        # Drive the tool through BaseAgent.execute_tool_calls — the same
        # path the real agent uses. The interrupt() call inside the tool
        # raises GraphInterrupt; if BaseAgent swallowed it, the test would
        # see a TOOL_ERROR ToolMessage instead of a paused snapshot.
        return await agent.execute_tool_calls(
            [
                {
                    "id": "tc-1",
                    "name": "request_human_review",
                    "args": {
                        "title": "Test",
                        "summary": "Pretend.",
                        "options_json": json.dumps(options),
                        "payload_json": "{}",
                    },
                }
            ],
            state={},
        )

    builder = StateGraph(MessagesState)
    builder.add_node("call_via_base_agent", call_via_base_agent)
    builder.add_edge("__start__", "call_via_base_agent")
    builder.add_edge("call_via_base_agent", "__end__")
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-base-agent-interrupt"}}

    await graph.ainvoke({"messages": []}, config=config)

    snapshot = await graph.aget_state(config)
    interrupts = [iv for task in snapshot.tasks for iv in task.interrupts]
    assert len(interrupts) == 1, (
        f"expected graph to pause at interrupt, got {len(interrupts)} "
        f"interrupts; tasks={snapshot.tasks}"
    )
    assert interrupts[0].value["title"] == "Test"


@pytest.mark.unit
def test_request_human_review_signature() -> None:
    """The tool exposes a stable signature the SKILL.md can rely on."""
    review_tool = create_review_tool()

    assert review_tool.name == "request_human_review"

    # The tool should declare title, summary, options_json, payload_json.
    schema_fields = set(review_tool.args_schema.model_fields.keys())  # type: ignore[union-attr]
    assert {"title", "summary", "options_json", "payload_json"} <= schema_fields

    # Description must include the word "review" so loaded skills
    # can refer to the tool in prose.
    assert "review" in (review_tool.description or "").lower()

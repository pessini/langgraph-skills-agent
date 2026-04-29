"""Skills agent — compiled LangGraph graph.

This module assembles and compiles the StateGraph for the Skills Agent.
The actual node implementations live in ``utils/nodes.py``; this file
is purely the graph wiring.

Graph topology::

    __start__ ──→ agent_node ──(should_continue)──→ tool_node ──→ agent_node
                       │                                              (loop)
                       ├──(retries exhausted)──→ error_summary_node ──→ __end__
                       └──(no tool calls)──→ __end__

The ``context_schema=Context`` tells LangGraph to inject a ``Context``
instance into every node via ``runtime.context``.  The Context holds the
skill store and LLM configuration.
"""

from langgraph.graph import StateGraph

from skills_agent.config import Context
from skills_agent.nodes import (
    agent_node,
    error_summary_node,
    should_continue,
    tool_node,
)
from skills_agent.state import State

builder = StateGraph(State, context_schema=Context)
builder.add_node(agent_node)
builder.add_node(tool_node)
builder.add_node(error_summary_node)

builder.add_edge("__start__", "agent_node")
builder.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        "tool_node": "tool_node",
        "error_summary_node": "error_summary_node",
        "__end__": "__end__",
    },
)
builder.add_edge("tool_node", "agent_node")
builder.add_edge("error_summary_node", "__end__")

graph = builder.compile(name="SkillsAgent")

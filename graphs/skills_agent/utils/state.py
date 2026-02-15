"""State definition for the skills agent.

The state is intentionally minimal — just the conversation messages plus a
retry counter.  An earlier version of the agent had additional fields
(``selected_skill``, ``skill_context``, ``execution_phase``) for a
multi-phase pipeline.  These were removed when the architecture was
simplified to a single ReAct loop, since the LLM now manages skill
selection implicitly through its tool calls rather than explicit state.
"""

from langgraph.graph import MessagesState


class State(MessagesState):
    """Agent state — conversation messages plus tool retry/iteration tracking.

    Inherits ``messages`` from ``MessagesState``, which uses the
    ``add_messages`` reducer to append new messages rather than overwrite.

    ``tool_retry_attempts`` is incremented each time ``tool_node`` encounters
    an error and reset to 0 on success.  When it reaches ``MAX_TOOL_RETRIES``
    (defined in ``nodes.py``), ``should_continue()`` routes to
    ``error_summary_node`` instead of looping back to ``agent_node``.

    ``tool_call_count`` tracks the total number of tool calls made during the
    conversation.  This counter is incremented on EVERY tool call (success or
    failure) and is used by ``should_continue()`` to prevent infinite loops
    when the LLM repeatedly calls tools without producing a final answer.
    """

    tool_retry_attempts: int = 0
    tool_call_count: int = 0

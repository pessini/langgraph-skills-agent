"""Define the configurable parameters for the skills agent.

The ``Context`` dataclass is the central configuration object for the Skills
Agent.  It is passed to every graph node via LangGraph's ``Runtime[Context]``
dependency injection mechanism (declared in the ``StateGraph`` constructor as
``context_schema=Context``).

Responsibilities:

1. **Declarative configuration** — all tuneable parameters (LLM provider,
   model names, skills directory, etc.) are declared as dataclass fields
   with sensible defaults.

2. **Environment variable override** — ``__post_init__`` inspects environment
   variables and overrides any field that still has its default value.  This
   allows the same code to run locally with defaults and in Docker with
   env-based configuration, without requiring a separate config file.

3. **Lazy initialization** — expensive resources (skill store) are not created
   at construction time.  The async ``initialize()`` method must be called once
   before the first graph execution.  This is done lazily in ``agent_node`` so
   that the graph can be imported and compiled without side effects.

4. **Caching** — once initialized, the skill store and tools list are cached
   for the lifetime of the Context instance.

5. **External MCP tool composition** — when ``EXTERNAL_MCP_SERVERS_JSON`` is
   set in the environment, ``initialize()`` connects to those MCP servers via
   ``langchain-mcp-adapters`` and merges their tools into the agent's tool
   list.  This is how domain-specific tool servers (e.g. a sales database
   server) plug in without modifying the agent's source.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.base_agent import BaseAgent
from skills_agent.utils.progressive_logging import log_progressive

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from skills_agent.skills import SkillStore

logger = logging.getLogger(__name__)


async def _load_external_mcp_tools() -> list["BaseTool"]:
    """Load tools from external MCP servers declared in the environment.

    Reads ``EXTERNAL_MCP_SERVERS_JSON`` — a JSON object mapping
    server-name → connection-config.  Example:

    .. code-block:: bash

        EXTERNAL_MCP_SERVERS_JSON='{"sales_db": {"url":
        "http://localhost:8765/mcp", "transport": "streamable_http"}}'

    Returns an empty list if the env var is unset, empty, or invalid JSON.
    Connection failures from individual servers propagate — boot fails
    loudly rather than silently shipping a broken tool list.
    """
    raw = os.environ.get("EXTERNAL_MCP_SERVERS_JSON", "").strip()
    if not raw:
        return []
    try:
        connections = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "EXTERNAL_MCP_SERVERS_JSON is not valid JSON; ignoring (%s)", exc
        )
        return []
    if not isinstance(connections, dict):
        logger.warning(
            "EXTERNAL_MCP_SERVERS_JSON must be a JSON object mapping server "
            "name to connection config; got %s. Ignoring.",
            type(connections).__name__,
        )
        return []
    if not connections:
        return []

    # Imported lazily so test suites that don't exercise external MCP
    # don't pay the import cost.
    from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: PLC0415

    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()
    logger.info(
        "Loaded %d external MCP tools from %d server(s): %s",
        len(tools),
        len(connections),
        list(connections.keys()),
    )
    log_progressive({
        "event": "external_mcp.tools.loaded",
        "summary": (
            f"servers={list(connections.keys())} count={len(tools)}"
        ),
    })
    return tools


def _get_default_skills_dir() -> str:
    """Get the default skills directory relative to this file."""
    return str(Path(__file__).parent / "skills")


def _load_chat_model(
    provider: str, model: str, base_url: str | None = None
) -> BaseChatModel:
    """Build a fresh LangChain chat model. Called per agent invocation
    (these are lightweight HTTP-wrapper objects, not persistent connections).

    OpenAI reads ``OPENAI_API_KEY`` from the environment automatically.
    """
    if provider == "ollama":
        return ChatOllama(model=model, base_url=base_url)
    if provider == "openai":
        return ChatOpenAI(model=model)
    raise ValueError(
        f"Unsupported provider: {provider}. Use 'ollama' or 'openai'."
    )


@dataclass(kw_only=True)
class Context:
    """The context configuration for the skills agent.

    This dataclass serves a dual purpose:

    - **Configuration schema**: LangGraph uses it as the ``context_schema`` for
      the graph, meaning it defines the shape of the configuration that callers
      can pass when invoking the graph.
    - **Runtime container**: After ``initialize()`` is called, it also holds
      the live skill store and tool instances used by graph nodes.

    All fields use ``kw_only=True`` so they must be passed by name, preventing
    accidental positional argument mistakes.
    """

    # -- LLM Configuration --------------------------------------------------
    provider: str = field(
        default="ollama",
        metadata={"description": "LLM provider to use: 'ollama' or 'openai'."},
    )

    ollama_model: str = field(
        default="qwen3",
        metadata={"description": "Model name for Ollama."},
    )

    openai_model: str = field(
        default="gpt-5-mini",
        metadata={"description": "Model name for OpenAI."},
    )

    base_url: str = field(
        default="http://localhost:11434",
        metadata={"description": "The base URL for the Ollama server."},
    )

    # -- Skills Configuration -------------------------------------------------
    skills_dir: str = field(
        default_factory=_get_default_skills_dir,
        metadata={"description": "Directory containing skill subdirectories."},
    )

    def __post_init__(self) -> None:
        """Override defaults with environment variables when present.

        The override logic only activates when a field still has its *default*
        value — meaning the caller did not explicitly set it.  This lets
        explicit constructor arguments take precedence over env vars, which in
        turn take precedence over coded defaults.

        Mapping of field names → environment variable names:

        ===============  ====================  ============================
        Field            Env Var               Notes
        ===============  ====================  ============================
        provider         LLM_PROVIDER          "ollama" or "openai"
        ollama_model     OLLAMA_MODEL
        openai_model     OPENAI_MODEL
        base_url         OLLAMA_BASE_URL
        skills_dir       SKILLS_DIR
        ===============  ====================  ============================
        """
        for f in fields(self):
            if not f.init:
                continue

            env_var_name = f.name.upper()
            # Special cases for environment variable names
            if f.name == "provider":
                env_var_name = "LLM_PROVIDER"
            elif f.name == "ollama_model":
                env_var_name = "OLLAMA_MODEL"
            elif f.name == "openai_model":
                env_var_name = "OPENAI_MODEL"
            elif f.name == "base_url":
                env_var_name = "OLLAMA_BASE_URL"
            elif f.name == "skills_dir":
                env_var_name = "SKILLS_DIR"

            current_value = getattr(self, f.name)
            if f.default is not MISSING:
                default_value = f.default
            elif f.default_factory is not MISSING:
                default_value = f.default_factory()
            else:
                continue

            if current_value == default_value:
                env_value = os.environ.get(env_var_name)
                if env_value:
                    setattr(self, f.name, env_value)

        # These are lazily initialized by initialize() and cached for the
        # lifetime of this Context instance.  They are "private" (prefixed
        # with _) because callers should use the corresponding @property
        # accessors which raise a clear error if initialize() hasn't been
        # called yet.
        self._skill_store: SkillStore | None = None
        self._tools: list[BaseTool] | None = None
        self._agent: SkillsAgent | None = None

    @property
    def model(self) -> str:
        """Get the effective model based on provider."""
        if self.provider == "openai":
            return self.openai_model
        return self.ollama_model

    async def initialize(self) -> None:
        """Scan the skill store and build the tools list.

        This method is called lazily from ``agent_node`` on the first graph
        invocation.  It performs two sequential steps:

        1. **Skill store scan** — walks the ``skills_dir`` to find SKILL.md
           files and builds a metadata cache.  This is synchronous and fast
           (only YAML frontmatter is parsed, not the full file body).

        2. **Tools list** — creates skill tools (``load_skill``,
           ``read_skill_file``) that get bound to the LLM via ``bind_tools()``.
        """
        from skills_agent.skills import SkillStore
        from skills_agent.tools import create_skill_tools

        if self._skill_store is None:
            scan_start = time.perf_counter()
            log_progressive({"tier": 1, "event": "catalog.scan.start"})
            self._skill_store = SkillStore(self.skills_dir)
            self._skill_store.scan()
            logger.info(
                "Skill store initialized with %d skills",
                len(self._skill_store.get_skill_names()),
            )
            duration_ms = int((time.perf_counter() - scan_start) * 1000)
            log_progressive({
                "tier": 1,
                "event": "catalog.scan.end",
                "duration_ms": duration_ms,
                "summary": f"skills_count={len(self._skill_store.get_skill_names())}",
            })

        if self._tools is None:
            from skills_agent.review import create_review_tool  # noqa: PLC0415

            skill_tools = create_skill_tools(self._skill_store)
            review_tool = create_review_tool()
            external_tools = await _load_external_mcp_tools()

            # ``BaseAgent.tools_by_name`` is dict-keyed by name, so a later
            # tool with a duplicate name silently overrides an earlier one.
            # Drop external tools that collide with built-ins OR with an
            # already-kept external (two MCP servers exposing the same
            # tool name) so neither path can silently shadow another.
            seen_names = {t.name for t in [*skill_tools, review_tool]}
            kept_external: list[BaseTool] = []
            dropped: list[str] = []
            for ext in external_tools:
                if ext.name in seen_names:
                    dropped.append(ext.name)
                    continue
                seen_names.add(ext.name)
                kept_external.append(ext)
            if dropped:
                logger.warning(
                    "Dropped %d external MCP tool(s) whose names collide "
                    "with a built-in or another external tool: %s",
                    len(dropped),
                    sorted(set(dropped)),
                )

            self._tools = [*skill_tools, review_tool, *kept_external]
            logger.info(
                "Created %d total tools (%d skill, 1 review, %d external)",
                len(self._tools),
                len(skill_tools),
                len(kept_external),
            )
            log_progressive({
                "event": "tools.created",
                "summary": (
                    f"total={len(self._tools)} skill={len(skill_tools)} "
                    f"review=1 external={len(kept_external)}"
                ),
            })

        if self._agent is None:
            # The model_factory closure captures `self` so each call
            # builds a fresh model with the latest provider/model values.
            self._agent = SkillsAgent(
                agent_name="skills_agent",
                model_factory=lambda: _load_chat_model(
                    self.provider, self.model, self.base_url
                ),
                tools=self._tools,
            )

    @property
    def skill_store(self) -> SkillStore:
        """Access the initialized skill store."""
        if self._skill_store is None:
            raise RuntimeError(
                "Context not initialized. Call await context.initialize() first."
            )
        return self._skill_store

    @property
    def tools(self) -> list[BaseTool]:
        """Access the skill tools list (cached)."""
        if self._tools is None:
            raise RuntimeError(
                "Context not initialized. Call await context.initialize() first."
            )
        return self._tools

    @property
    def agent(self) -> SkillsAgent:
        """Access the cached SkillsAgent instance."""
        if self._agent is None:
            raise RuntimeError(
                "Context not initialized. Call await context.initialize() first."
            )
        return self._agent


class SkillsAgent(BaseAgent):
    """Skills-agent specialization of ``BaseAgent``.

    Forwards tool execution events to ``progressive_logging`` so the
    existing log shape is preserved (the BaseAgent itself is logging-
    agnostic).
    """

    def on_tool_event(self, event: dict) -> None:
        log_progressive(event)

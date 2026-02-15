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
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from skills_agent.utils.progressive_logging import log_progressive

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from skills_agent.skills import SkillStore

logger = logging.getLogger(__name__)


def _get_default_skills_dir() -> str:
    """Get the default skills directory relative to this file."""
    return str(Path(__file__).parent / "skills")


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
        from skills_agent.utils.tools import create_skill_tools

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
            self._tools = create_skill_tools(self._skill_store)
            logger.info("Created %d skill tools", len(self._tools))
            log_progressive({
                "event": "tools.created",
                "summary": f"count={len(self._tools)}",
            })

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

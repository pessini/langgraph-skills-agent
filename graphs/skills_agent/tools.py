"""Tool definitions for the skills agent.

This module creates the two LangChain tools that let the LLM interact with
the local skill store:

- ``load_skill(skill_name)`` — returns the full SKILL.md content plus a list
  of available supporting files.
- ``read_skill_file(skill_name, filename)`` — returns the content of a
  specific supporting file from a skill's directory.

**Why are these tools and not just system-prompt context?**  Loading all skills
into the system prompt would consume too many tokens and dilute the LLM's
attention.  By exposing skills as tools, the LLM can selectively load only the
knowledge it needs for the current task (progressive disclosure).

**Why a factory function?**  The tools need a reference to the ``SkillStore``
instance, which is created at runtime.  The ``create_skill_tools()`` factory
captures the store via closure, so each tool invocation reads from the same
cached store.

**Dynamic docstrings**: After creation, each tool's docstring is overwritten
to include the list of available skill names.  This is critical because the
LLM sees tool docstrings as part of the tool schema — without the skill list,
the LLM wouldn't know which skill names are valid arguments.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from skills_agent.utils.progressive_logging import log_progressive

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from skills_agent.skills import SkillStore


def create_skill_tools(store: SkillStore) -> list[BaseTool]:
    """Create skill-loading tools bound to a store instance.

    Returns a list containing ``load_skill`` and ``read_skill_file`` tools
    whose docstrings enumerate the available skill names.  These tools are
    synchronous because the underlying store operations are filesystem
    reads (no async I/O needed).

    Both tools return **JSON strings** rather than raw text.  This gives the
    LLM a structured format to parse and also makes error responses
    distinguishable from success responses (via the ``"error"`` key).
    """
    available = ", ".join(store.get_skill_names()) or "none"

    @tool
    def load_skill(skill_name: str) -> str:
        """Load expert knowledge for a skill."""
        start = time.perf_counter()
        parsed = store.load(skill_name)
        if parsed is None:
            # Return the full list of valid names so the LLM can self-correct
            # (e.g. if it hallucinated a skill name or made a typo).
            names = ", ".join(store.get_skill_names())
            log_progressive({
                "tier": 2,
                "event": "skill.load.error",
                "skill": skill_name,
                "summary": "Skill not found",
            })
            return json.dumps(
                {"error": f"Skill '{skill_name}' not found", "available_skills": names}
            )
        # The response includes:
        # - instructions: the full SKILL.md markdown body
        # - available_files: list of supporting filenames the LLM can request
        #   via read_skill_file() if it needs deeper reference material
        available_files = store.list_supporting_files(skill_name)
        duration_ms = int((time.perf_counter() - start) * 1000)
        log_progressive({
            "tier": 2,
            "event": "tool.load_skill",
            "skill": skill_name,
            "duration_ms": duration_ms,
            "summary": f"instructions_chars={len(parsed.content)} files={len(available_files)}",
        })
        return json.dumps({
            "skill_name": skill_name,
            "description": parsed.metadata.description,
            "instructions": parsed.content,
            "available_files": available_files,
        })

    @tool
    def read_skill_file(skill_name: str, filename: str) -> str:
        """Read a supporting file from a skill folder."""
        try:
            start = time.perf_counter()
            content = store.read_supporting_file(skill_name, filename)
            log_progressive({
                "tier": 3,
                "event": "tool.read_skill_file",
                "skill": skill_name,
                "file": filename,
                "duration_ms": int((time.perf_counter() - start) * 1000),
                "summary": f"chars={len(content)}",
            })
            return json.dumps({
                "skill_name": skill_name,
                "filename": filename,
                "content": content,
            })
        except (FileNotFoundError, ValueError) as e:
            log_progressive({
                "tier": 3,
                "event": "supporting_file.error",
                "skill": skill_name,
                "file": filename,
                "summary": str(e),
            })
            return json.dumps({"error": f"Error reading file: {e}"})

    # Overwrite the docstrings to include the list of available skills.
    # The LangChain @tool decorator uses the docstring as the tool's
    # "description" in the JSON schema sent to the LLM.  By including
    # the skill names here, the LLM knows which values are valid for
    # the skill_name parameter without needing to see the system prompt.
    load_skill.__doc__ = (
        f"Load expert knowledge for a skill. Returns JSON with the skill's "
        f"instructions and a list of available supporting files.\n\n"
        f"Available skills: {available}\n\n"
        f"Args:\n    skill_name: Exact name of the skill to load."
    )
    read_skill_file.__doc__ = (
        f"Read a supporting file from a skill folder. Returns JSON with the "
        f"file content.\n\n"
        f"Use this to access reference documents listed in the "
        f"'available_files' field of a loaded skill.\n\n"
        f"Available skills: {available}\n\n"
        f"Args:\n    skill_name: Name of the skill that owns the file.\n"
        f"    filename: Name of the file to read (from the skill's available_files)."
    )

    return [load_skill, read_skill_file]  # type: ignore[list-item]

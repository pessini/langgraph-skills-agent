"""Skill store and loader for dynamic skill management.

This module implements the **skills system** — the domain-knowledge backbone of
the Skills Agent.  Skills are self-contained packages of expert knowledge stored
as directories on the filesystem.  Each skill directory contains:

- ``SKILL.md`` (required) — the primary document with YAML frontmatter
  (metadata) and a markdown body (instructions).
- Supporting files (optional) — additional ``.md`` files referenced by
  SKILL.md (e.g. ``DATA_ACCESS.md``, ``COMMON_PATTERNS.md``).

The ``SkillStore`` is responsible for:

1. **Discovery** — scanning a directory tree for SKILL.md files.
2. **Metadata caching** — parsing only the YAML frontmatter (fast) and storing
   it in memory so the skill catalog can be generated cheaply.
3. **Lazy content loading** — the full markdown body is parsed only when the
   LLM calls ``load_skill(skill_name)`` via the tool.
4. **On-demand file reading** — supporting files are read only when the LLM
   calls ``read_skill_file(skill_name, filename)`` via the tool.

This multi-tier loading strategy implements **progressive disclosure**: the LLM
sees a lightweight catalog first, loads full instructions only when needed, and
drills into supporting files only if the instructions reference them.  This
keeps token usage low while still providing deep domain knowledge when required.

Security note: ``read_skill_supporting_file()`` validates filenames to prevent
directory traversal attacks (no ``..`` or absolute paths allowed).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skills_agent.utils.progressive_logging import log_progressive

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Regex to split a SKILL.md file into YAML frontmatter and markdown body.
# The pattern expects exactly this structure:
#
#   ---
#   name: my-skill
#   description: A brief summary
#   ---
#   # Markdown body starts here
#
# ``re.DOTALL`` makes ``.`` match newlines so the capture groups can span
# multiple lines.  Group 1 = YAML frontmatter, Group 2 = markdown body.
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


@dataclass
class SkillMetadata:
    """Metadata extracted from a SKILL.md file's YAML frontmatter.

    This is a lightweight representation used during the *discovery* phase.
    The store keeps one ``SkillMetadata`` per skill so it can generate
    the skill catalog (for the system prompt) without loading the full
    markdown body of every skill.

    Attributes:
        name: Human-readable skill identifier (e.g. "n8n-code-javascript").
              Used as the key for ``load_skill(skill_name)``.
        description: One-line summary shown to the LLM in the skill catalog.
        version: Semantic version string, defaults to "1.0".
        tags: Category labels for filtering/grouping (included in catalog).
        dependencies: Other skill names that should be loaded alongside this
                      skill.  Currently declared in frontmatter but dependency
                      auto-loading is not implemented in the ReAct loop — the
                      LLM is expected to load dependencies manually.
        path: Filesystem path to the SKILL.md file.  Set by the store
              during scanning so that ``load()`` knows where to read from.
    """

    name: str
    description: str
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> SkillMetadata:
        """Create a ``SkillMetadata`` from a parsed YAML frontmatter dict."""
        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            version=str(data.get("version", "1.0")),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            path=path,
        )


@dataclass
class ParsedSkill:
    """A fully parsed skill — metadata plus the markdown body.

    Created by ``parse_skill_file()`` when the LLM requests a skill via
    ``load_skill()``.  The ``content`` field contains the markdown body
    *without* the YAML frontmatter delimiters, ready to be sent to the LLM
    as tool output.
    """

    metadata: SkillMetadata
    content: str  # Markdown body (everything after the closing ``---``)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_skill_file(file_path: Path) -> ParsedSkill:
    """Parse a SKILL.md file and extract metadata and content."""
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return _parse_skill_content(text, file_path)


def parse_metadata_only(file_path: Path) -> SkillMetadata:
    """Parse only the metadata from a SKILL.md file (faster for scanning)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Invalid skill file format - missing YAML frontmatter: {file_path}")
    try:
        frontmatter_data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e
    return SkillMetadata.from_dict(frontmatter_data, path=file_path)


def read_skill_supporting_file(skill_dir: Path, filename: str) -> str:
    """Read a supporting file from a skill folder.

    **Security**: The filename is validated to prevent directory traversal.
    Without this check, a malicious tool call like
    ``read_skill_file("my-skill", "../../etc/passwd")`` could read arbitrary
    files on the host.

    The filename can include subdirectory paths (e.g. "resources/guides/usage.md")
    but cannot escape the skill directory.
    """
    if ".." in filename or filename.startswith("/"):
        raise ValueError(f"Invalid filename: {filename}")
    file_path = (skill_dir / filename).resolve()
    # Ensure the resolved path is still within the skill directory
    if not str(file_path).startswith(str(skill_dir.resolve())):
        raise ValueError(f"Invalid filename: {filename}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    return file_path.read_text(encoding="utf-8")


def _parse_skill_content(text: str, file_path: Path | None = None) -> ParsedSkill:
    """Parse skill content string and extract metadata and body."""
    match = FRONTMATTER_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Invalid skill file format - missing YAML frontmatter: {file_path}")
    frontmatter_yaml, body = match.groups()
    try:
        frontmatter_data = yaml.safe_load(frontmatter_yaml) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e
    metadata = SkillMetadata.from_dict(frontmatter_data, path=file_path)
    return ParsedSkill(metadata=metadata, content=body.strip())


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SkillStore:
    """Store for managing skills with lazy loading and caching.

    The store implements a two-level cache:

    - ``_metadata_cache``: Populated by ``scan()``.  Contains lightweight
      ``SkillMetadata`` for every discovered skill.  Used to generate the
      skill catalog that goes into the system prompt.
    - ``_content_cache``: Populated lazily by ``load()``.  Contains the full
      ``ParsedSkill`` (metadata + markdown body) only for skills that the LLM
      has actually requested.

    This design means the system prompt can list all available skills (cheap)
    without paying the cost of reading every SKILL.md file up front.

    Thread safety: The store is designed for single-threaded async use.
    The ``scan()`` and ``load()`` methods are synchronous (filesystem I/O)
    and are called from async graph nodes.  For production use with multiple
    workers, each worker gets its own Context and store instance.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._metadata_cache: dict[str, SkillMetadata] = {}
        self._content_cache: dict[str, ParsedSkill] = {}
        self._scanned = False

    def scan(self) -> dict[str, SkillMetadata]:
        """Recursively discover all skills and cache their metadata.

        Walks ``skills_dir`` looking for ``SKILL.md`` files.  For each one,
        only the YAML frontmatter is parsed (via ``parse_metadata_only``),
        keeping this operation fast even with many skills.

        The skill name is determined by the ``name`` field in the frontmatter.
        If the frontmatter doesn't specify a name (or it's "unknown"), the
        parent directory name is used as a fallback.

        Invalid SKILL.md files (bad YAML, missing frontmatter) are logged as
        warnings and skipped — they don't crash the entire scan.
        """
        if self._scanned and self._metadata_cache:
            log_progressive({
                "tier": 1,
                "event": "catalog.scan.cache_hit",
                "summary": f"skills_count={len(self._metadata_cache)}",
            })
            return self._metadata_cache

        scan_start = time.perf_counter()
        self._metadata_cache.clear()
        self._content_cache.clear()

        if not self.skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {self.skills_dir}")
            log_progressive({
                "tier": 1,
                "event": "catalog.scan.end",
                "duration_ms": int((time.perf_counter() - scan_start) * 1000),
                "summary": "skills_count=0",
            })
            self._scanned = True
            return self._metadata_cache

        for skill_file in self.skills_dir.rglob("SKILL.md"):
            try:
                metadata = parse_metadata_only(skill_file)
                skill_name = skill_file.parent.name
                if metadata.name and metadata.name != "unknown":
                    skill_name = metadata.name
                metadata.path = skill_file
                self._metadata_cache[skill_name] = metadata
                logger.debug(f"Registered skill: {skill_name}")
            except Exception as e:
                logger.warning(f"Failed to parse skill at {skill_file}: {e}")

        self._scanned = True
        logger.info(f"Scanned {len(self._metadata_cache)} skills from {self.skills_dir}")
        log_progressive({
            "tier": 1,
            "event": "catalog.scan.end",
            "duration_ms": int((time.perf_counter() - scan_start) * 1000),
            "summary": f"skills_count={len(self._metadata_cache)}",
        })
        return self._metadata_cache

    def load(self, skill_name: str) -> ParsedSkill | None:
        """Lazy-load the full SKILL.md content for a skill.

        Returns a cached ``ParsedSkill`` if available, otherwise reads and
        parses the file.  Returns ``None`` if the skill doesn't exist in the
        metadata cache (i.e. was not found during ``scan()``).

        **Progressive disclosure**: This only loads the SKILL.md body —
        supporting files are loaded separately via ``read_supporting_file()``.
        """
        load_start = time.perf_counter()
        if skill_name in self._content_cache:
            log_progressive({
                "tier": 2,
                "event": "skill.load.cache_hit",
                "skill": skill_name,
                "cache": "hit",
                "duration_ms": int((time.perf_counter() - load_start) * 1000),
            })
            return self._content_cache[skill_name]
        if not self._scanned:
            self.scan()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            logger.warning(f"Skill not found: {skill_name}")
            log_progressive({
                "tier": 2,
                "event": "skill.load.error",
                "skill": skill_name,
                "summary": "Skill not found",
            })
            return None
        try:
            parsed = parse_skill_file(metadata.path)
            self._content_cache[skill_name] = parsed
            logger.debug(f"Loaded skill content: {skill_name}")
            log_progressive({
                "tier": 2,
                "event": "skill.load.cache_miss",
                "skill": skill_name,
                "cache": "miss",
                "duration_ms": int((time.perf_counter() - load_start) * 1000),
                "summary": f"chars={len(parsed.content)}",
            })
            return parsed
        except Exception as e:
            logger.error(f"Failed to load skill {skill_name}: {e}")
            log_progressive({
                "tier": 2,
                "event": "skill.load.error",
                "skill": skill_name,
                "summary": str(e),
            })
            return None

    def read_supporting_file(self, skill_name: str, filename: str) -> str:
        """Read a supporting file from a skill folder."""
        read_start = time.perf_counter()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            log_progressive({
                "tier": 3,
                "event": "supporting_file.error",
                "skill": skill_name,
                "file": filename,
                "summary": "Skill not found",
            })
            raise ValueError(f"Skill not found: {skill_name}")
        skill_dir = metadata.path.parent
        try:
            content = read_skill_supporting_file(skill_dir, filename)
            log_progressive({
                "tier": 3,
                "event": "supporting_file.read",
                "skill": skill_name,
                "file": filename,
                "duration_ms": int((time.perf_counter() - read_start) * 1000),
                "summary": f"chars={len(content)}",
            })
            return content
        except Exception as e:
            log_progressive({
                "tier": 3,
                "event": "supporting_file.error",
                "skill": skill_name,
                "file": filename,
                "summary": str(e),
            })
            raise

    def list_supporting_files(self, skill_name: str) -> list[str]:
        """List supporting files (non-SKILL.md) in a skill's directory recursively.

        Returns relative paths from the skill directory so the LLM can use them
        with ``read_skill_file(skill_name, relative_path)``.
        """
        metadata = self._metadata_cache.get(skill_name)
        if not metadata or not metadata.path:
            return []
        skill_dir = metadata.path.parent
        return sorted(
            str(f.relative_to(skill_dir))
            for f in skill_dir.rglob("*.md")
            if f.is_file() and f.name != "SKILL.md"
        )

    def get_skill_catalog(self) -> str:
        """Generate an XML-structured catalog of all available skills.

        This catalog is inserted into the system prompt via the
        ``{skill_catalog}`` placeholder.  It uses XML tags (``<skill>``,
        ``<name>``, ``<description>``, etc.) because structured formats help
        LLMs parse and reason about the available options more reliably than
        plain text.

        The catalog includes supporting file names so the LLM knows what
        reference documents are available *before* loading the skill.  This
        enables the LLM to make informed decisions about which skills to load
        and which supporting files to request.
        """
        if not self._scanned:
            self.scan()
        if not self._metadata_cache:
            return "No skills available."
        render_start = time.perf_counter()
        lines: list[str] = []
        for name, metadata in sorted(self._metadata_cache.items()):
            lines.append("<skill>")
            lines.append(f"  <name>{name}</name>")
            lines.append(f"  <description>{metadata.description}</description>")
            if metadata.tags:
                lines.append(f"  <tags>{', '.join(metadata.tags)}</tags>")
            files = self.list_supporting_files(name)
            if files:
                lines.append(f"  <supporting_files>{', '.join(files)}</supporting_files>")
            lines.append("</skill>")
        catalog = "\n".join(lines)
        log_progressive({
            "tier": 1,
            "event": "catalog.render",
            "summary": f"skills_count={len(self._metadata_cache)} chars={len(catalog)}",
            "duration_ms": int((time.perf_counter() - render_start) * 1000),
        })
        return catalog

    def get_skill_names(self) -> list[str]:
        """Get list of all registered skill names."""
        if not self._scanned:
            self.scan()
        return list(self._metadata_cache.keys())

    def invalidate(self) -> None:
        """Clear all caches and force a rescan on next access.

        Useful during development when skills are being edited.  After
        calling this, the next ``scan()`` or ``load()`` will re-read from
        disk, picking up any changes to SKILL.md files or new skills.
        """
        self._metadata_cache.clear()
        self._content_cache.clear()
        self._scanned = False
        logger.info("Skill store cache invalidated")

# Skills Directory

This directory contains skills that extend the agent's capabilities with domain expertise.

## Skill Format

Each skill is a subdirectory containing:

1. **SKILL.md** (required) — The main skill file with YAML frontmatter and markdown content

2. **Supporting files** (optional) — Additional reference documents, cheatsheets, examples

### SKILL.md Format

```markdown
---
name: my-skill-name
description: A brief description of what this skill provides
tags:
  - relevant
  - tags
---

# Skill Title

Detailed instructions for when and how to use this skill.

## When to Use

Describe the scenarios where this skill applies.

## Instructions

Step-by-step guidance for applying the skill.

## Examples

Show examples of the skill in action.
```

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier (use kebab-case) |
| `description` | Yes | Brief summary shown in the skill catalog |
| `tags` | No | Keywords for categorization |

## Example Skill Structure

```
skills/
├── README.md
├── code-review/
│   ├── SKILL.md
│   ├── checklist.md
│   └── examples.md
└── api-design/
    ├── SKILL.md
    └── best-practices.md
```

## How Skills Are Used

1. The agent sees skill summaries in its system prompt
2. When a task matches a skill's description, the agent calls `load_skill(skill_name)`
3. The agent receives the full SKILL.md content and a list of available supporting files
4. If needed, the agent calls `read_skill_file(skill_name, filename)` for additional context
5. The agent applies the skill's guidance to complete the task

## Adding a New Skill

1. Create a new directory under `skills/` with your skill name
2. Add a `SKILL.md` file with the required frontmatter
3. Optionally add supporting files for reference documentation
4. Restart the agent to pick up the new skill

The skill will automatically appear in the agent's catalog.

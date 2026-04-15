"""Assemble system/user prompt blocks for Anthropic Messages API."""

from __future__ import annotations

import json
from typing import Any

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings", "summary"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {"$ref": "#/$defs/Finding"},
        },
        "summary": {"type": "string"},
    },
    "$defs": {
        "Finding": {
            "type": "object",
            "required": [
                "file_path",
                "line_start",
                "severity",
                "category",
                "title",
                "description",
                "confidence",
            ],
            "properties": {
                "file_path": {"type": "string"},
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": ["integer", "null"], "minimum": 1},
                "severity": {"enum": ["critical", "warning", "suggestion", "nitpick"]},
                "category": {
                    "enum": [
                        "security",
                        "performance",
                        "logic",
                        "style",
                        "architecture",
                        "testing",
                        "documentation",
                    ],
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "suggested_fix": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    },
}


def build_system_blocks(
    agent_role: str,
    convention_texts: dict[str, str],
    repo_map: str,
) -> list[dict[str, Any]]:
    """Return system prompt blocks in deterministic order.

    Order matters for caching: later blocks are the ones marked
    cache_control by the client.
    """
    role_block = {
        "type": "text",
        "text": f"{agent_role.strip()}\n\n"
                "Respond only in the JSON format described by the schema.",
    }
    schema_block = {
        "type": "text",
        "text": "## Output schema (enforced)\n\n```json\n"
                + json.dumps(FINDINGS_SCHEMA, indent=2)
                + "\n```",
    }
    convention_parts = []
    for name, text in convention_texts.items():
        convention_parts.append(f"### {name}\n\n{text.strip()}")
    if convention_parts:
        convention_block_text = "## Project conventions\n\n" + "\n\n".join(convention_parts)
    else:
        convention_block_text = "## Project conventions\n\n(none available)"
    convention_block = {"type": "text", "text": convention_block_text}
    map_block = {
        "type": "text",
        "text": f"## Repository map\n\n{repo_map.strip()}",
    }
    return [role_block, schema_block, convention_block, map_block]


def _files_block(heading: str, files: dict[str, str]) -> str:
    if not files:
        return f"## {heading}\n\n(none)"
    parts = [f"## {heading}\n"]
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```\n")
    return "\n".join(parts)


def build_user_blocks(
    pr_title: str,
    pr_body: str,
    diff: str,
    changed_files: dict[str, str],
    neighbor_files: dict[str, str],
    max_total_chars: int = 600_000,
) -> list[dict[str, Any]]:
    """Assemble the user message for review.

    Truncation priority (lowest first): neighbors, changed files.
    The diff is never truncated.
    """
    pr_meta = f"## PR metadata\n\n**Title:** {pr_title}\n\n**Description:**\n\n{pr_body or '(empty)'}"
    diff_block = f"## Diff\n\n```diff\n{diff}\n```"
    changed_block = _files_block("Changed files (full contents)", changed_files)
    neighbor_block = _files_block("Neighbor files (context)", neighbor_files)

    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    if len(assembled) <= max_total_chars:
        return [{"type": "text", "text": assembled}]

    neighbor_block = _files_block("Neighbor files (context)", {}) + "\n[... neighbors truncated ...]"
    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    if len(assembled) <= max_total_chars:
        return [{"type": "text", "text": assembled}]

    truncated: dict[str, str] = {}
    budget = max_total_chars - len(pr_meta) - len(diff_block) - len(neighbor_block) - 1000
    for path, content in changed_files.items():
        if budget <= 0:
            truncated[path] = "[... file omitted due to budget ...]"
            continue
        if len(content) > budget:
            truncated[path] = content[:budget] + "\n[... file truncated ...]"
            budget = 0
        else:
            truncated[path] = content
            budget -= len(content)
    changed_block = _files_block("Changed files (full contents)", truncated)
    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    return [{"type": "text", "text": assembled}]

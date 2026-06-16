"""Assemble system/user prompt blocks for Anthropic Messages API."""

from __future__ import annotations

import json
from typing import Any

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings", "summary"],
    "additionalProperties": False,
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
            "additionalProperties": False,
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
                "line_start": {"type": "integer"},
                "line_end": {"type": ["integer", "null"]},
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
                "confidence": {"type": "number"},
            },
        },
    },
}


# Shared review standard + severity rubric — every agent sees this so severity
# is calibrated consistently rather than decided per agent.
REVIEW_STANDARD_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": (
        "## Review standard\n\n"
        "Favor approving when the change improves overall code health, even if "
        "imperfect — there is no perfect code, only better code. Do not block on "
        "minor polish. Comment on the code, not the author, and explain *why* "
        "when you ask for a change.\n\n"
        "**Severity:**\n"
        "- `critical` — must fix: security bugs or data-corruption risks only.\n"
        "- `warning` — should fix: other serious correctness or maintainability issues.\n"
        "- `suggestion` — consider; optional improvement.\n"
        '- `nitpick` — optional polish; prefix the title with "Nit: ".\n\n'
        "**Grounding:** Only report issues on lines changed in this PR. Cite the "
        "file and line. Do not speculate about code outside the diff."
    ),
}

# Few-shot anchors: one specific/actionable finding and one vague one to avoid.
FEW_SHOT_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": (
        "## Finding quality\n\n"
        "GOOD (specific, actionable):\n"
        '{"file_path": "auth.py", "line_start": 45, "severity": "critical", '
        '"category": "security", "title": "SQL injection via string interpolation", '
        '"description": "User input is interpolated directly into the query without '
        'parameterization.", "suggested_fix": "Use a parameterized query: '
        'cursor.execute(\'… WHERE id = ?\', (user_id,))", "confidence": 0.95}\n\n'
        "BAD (vague — DO NOT produce these):\n"
        '{"file_path": "utils.py", "line_start": 1, "severity": "suggestion", '
        '"category": "testing", "title": "Consider adding more tests", '
        '"description": "The code could benefit from additional test coverage.", '
        '"confidence": 0.5}'
    ),
}


def _pr_tuning_block(pr_type: str | None, pr_size: str | None) -> dict[str, Any] | None:
    """Context-tuning guidance derived from PR type/size, or None if none applies.

    Type and size guidance may both apply and are concatenated.
    """
    parts: list[str] = []
    if pr_type == "docs":
        parts.append(
            "This PR is docs-only: report only factual errors, broken links, or "
            "security-sensitive content. Do not raise code style, tests, or nitpicks."
        )
    elif pr_type == "ci":
        parts.append(
            "This PR is CI/workflow-only: focus on workflow correctness (paths, "
            "steps, secrets). Do not raise code style or nitpicks."
        )
    if pr_size in ("trivial", "small"):
        parts.append(
            "Small change — prioritize precision: report only findings you are "
            "confident about, and do not pad the review with low-value suggestions."
        )
    elif pr_size == "large":
        parts.append(
            "Large change — prioritize high-severity issues (architecture, "
            "correctness, security) over minor style or nitpicks."
        )
    if not parts:
        return None
    return {"type": "text", "text": "## Review focus for this PR\n\n" + "\n\n".join(parts)}


def build_system_blocks(
    agent_role: str,
    convention_texts: dict[str, str],
    repo_map: str,
    pr_type: str | None = None,
    pr_size: str | None = None,
) -> list[dict[str, Any]]:
    """Return system prompt blocks in deterministic order.

    Order matters for caching: later blocks are the ones marked
    cache_control by the client.
    """
    role_block = {
        "type": "text",
        "text": f"{agent_role.strip()}\n\nRespond only in the JSON format described by the schema.",
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
    tuning_block = _pr_tuning_block(pr_type, pr_size)
    blocks = [role_block, REVIEW_STANDARD_BLOCK, FEW_SHOT_BLOCK]
    if tuning_block is not None:
        blocks.append(tuning_block)
    blocks.extend([schema_block, convention_block, map_block])
    return blocks


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
    pr_meta = (
        f"## PR metadata\n\n**Title:** {pr_title}\n\n**Description:**\n\n{pr_body or '(empty)'}"
    )
    diff_block = f"## Diff\n\n```diff\n{diff}\n```"
    changed_block = _files_block("Changed files (full contents)", changed_files)
    neighbor_block = _files_block("Neighbor files (context)", neighbor_files)

    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    if len(assembled) <= max_total_chars:
        return [{"type": "text", "text": assembled}]

    neighbor_block = (
        _files_block("Neighbor files (context)", {}) + "\n[... neighbors truncated ...]"
    )
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

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
#
# Coverage-first by design: the pipeline has downstream filters (per-severity
# confidence thresholds + a cross-review validation round), so the finder's job
# is coverage, not self-censorship. Telling Sonnet-5-era models "omit unless
# confident" measurably suppresses recall — they obey it literally.
REVIEW_STANDARD_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": (
        "## Review standard\n\n"
        "Favor approving when the change improves overall code health, even if "
        "imperfect — there is no perfect code, only better code. Do not block on "
        "minor polish. Technical facts and engineering principles outweigh personal "
        "preference: if the author's approach is a valid alternative, defer to it. "
        "Comment on the code, not the author, and explain *why* you ask for a change.\n\n"
        "Report every issue you find, including ones you are uncertain about — do "
        "not self-filter for importance or confidence. A separate validation step "
        "filters and ranks findings; your job at this stage is coverage. Signal "
        "certainty honestly through the `confidence` field (0.0-1.0) instead of "
        "omitting doubtful findings: a real-but-unproven concern belongs in the "
        "report at low confidence. Report each distinct issue exactly once, and do "
        "not flag mechanical formatting or import ordering that an "
        "autoformatter/linter already handles.\n\n"
        "Every finding must point to a specific changed line AND give a concrete fix "
        "or the precise reason the code is wrong. When a finding depends on code you "
        "cannot see in the diff (callers, definitions, configuration), use the "
        "provided repository tools (read_file / grep / glob) to check the actual "
        "code rather than guessing.\n\n"
        "**Severity:**\n"
        "- `critical` — must fix: security vulnerabilities or data-corruption/loss risks only.\n"
        "- `warning` — should fix: other correctness, concurrency, or serious maintainability issues.\n"
        "- `suggestion` — consider; an optional improvement.\n"
        '- `nitpick` — optional polish; prefix the title with "Nit: " (never blocking).\n\n'
        "**Grounding:** Only report issues on lines changed in this PR. Cite the "
        "file and line. Do not report issues in code outside the diff — but do use "
        "the tools to read surrounding code when it determines whether a changed "
        "line is correct."
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
            "Small change — the full diff fits comfortably in context, so verify it "
            "exhaustively. Do not pad the review with generic advice; every finding "
            "must still cite a specific changed line."
        )
    elif pr_size == "large":
        parts.append(
            "Large change — lead with high-severity issues (architecture, "
            "correctness, security), but still report lower-severity issues you "
            "notice with honest severity and confidence rather than omitting them."
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
    language_rules: str = "",
) -> list[dict[str, Any]]:
    """Return system prompt blocks in deterministic order.

    Order matters for caching: later blocks are the ones marked
    cache_control by the client. ``language_rules`` is the (already-rendered)
    language-specific high-severity guidance for the repo's languages; the
    caller computes it so this module stays language-agnostic.
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
    # Copy the shared constant blocks so a downstream in-place mutation (e.g. the
    # client adding cache_control to a block) can never clobber the module-level
    # originals across reviews.
    blocks = [role_block, dict(REVIEW_STANDARD_BLOCK), dict(FEW_SHOT_BLOCK)]
    if tuning_block is not None:
        blocks.append(tuning_block)
    if language_rules.strip():
        blocks.append(
            {
                "type": "text",
                "text": "## Language-specific priorities\n\n"
                + language_rules.strip()
                + "\n\nWeight the issues above as high severity for this repo.",
            }
        )
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

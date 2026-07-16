"""Assemble system/user prompt blocks for Anthropic Messages API."""

from __future__ import annotations

import json
import re
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
                "suggested_replacement": {"type": ["string", "null"]},
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
        "report at low confidence. Only conclude with zero findings if, after "
        "actually investigating the changed code, you genuinely found nothing — "
        '"I could not fully confirm it" is a reason to report at low confidence, '
        "not a reason to stay silent. Report each distinct issue exactly once, and do "
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
        'parameterization.", "suggested_fix": "Use a parameterized query.", '
        '"suggested_replacement": "cursor.execute(\'SELECT * FROM users WHERE id = ?\', '
        '(user_id,))", "confidence": 0.95}\n\n'
        "When the fix is a local replacement of the lines you flagged "
        "(line_start..line_end), set `suggested_replacement` to the EXACT new source "
        "for those whole lines - no diff markers, no `+`/`-` prefixes, no fences, "
        "correctly indented so it can be applied verbatim. Keep `suggested_fix` as the "
        "prose explanation. When the fix is non-local (spans multiple files, needs "
        "structural changes, or is not a clean line swap), leave `suggested_replacement` "
        "null and describe it in `suggested_fix` only.\n\n"
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
    conventions_max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Return system prompt blocks in deterministic order.

    Order matters for caching: later blocks are the ones marked
    cache_control by the client. ``language_rules`` is the (already-rendered)
    language-specific high-severity guidance for the repo's languages; the
    caller computes it so this module stays language-agnostic.
    ``conventions_max_chars`` caps the aggregate conventions block; None leaves
    it unbounded.
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
        body = "\n\n".join(convention_parts)
        if conventions_max_chars is not None and len(body) > conventions_max_chars:
            body = _truncate_on_line_boundary(body, conventions_max_chars) + (
                f"\n[conventions truncated at {conventions_max_chars} chars]"
            )
        convention_block_text = "## Project conventions\n\n" + body
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


def _truncate_on_line_boundary(text: str, max_chars: int) -> str:
    """Truncate to at most max_chars, backing up to the last newline when possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    return cut[:nl] if nl > 0 else cut


# New-side of a unified-diff hunk header: ``@@ -a,b +c,d @@`` (d defaults to 1).
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$")


def _hunk_windows_for_file(diff: str, target_path: str) -> list[tuple[int, int]]:
    """New-side (start_line, line_count) for each hunk of *target_path* in *diff*."""
    windows: list[tuple[int, int]] = []
    current: str | None = None
    for line in diff.splitlines():
        header = _DIFF_FILE_HEADER_RE.match(line)
        if header:
            current = header.group(1)
            continue
        if current != target_path:
            continue
        h = _HUNK_HEADER_RE.match(line)
        if h:
            start = int(h.group(1))
            count = int(h.group(2)) if h.group(2) is not None else 1
            windows.append((start, count))
    return windows


def _hunk_excerpt(content: str, hunks: list[tuple[int, int]], context: int) -> str:
    """Excerpt file lines around each hunk (+/- *context*), merging overlaps.

    Windows are joined by a ``...`` separator line and prefixed with a note
    stating kept/total line counts.
    """
    lines = content.splitlines()
    total = len(lines)
    ranges = [
        (max(1, start - context), min(total, start + count + context)) for start, count in hunks
    ]
    ranges.sort()
    merged: list[list[int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    kept = sum(hi - lo + 1 for lo, hi in merged)
    note = (
        f"[excerpt: {kept} of {total} lines - hunks +/-{context} context; "
        "use read_file for the full file]"
    )
    segments = ["\n".join(lines[lo - 1 : hi]) for lo, hi in merged]
    return note + "\n" + "\n...\n".join(segments)


def _cap_lines(content: str, max_lines: int) -> str:
    """Keep the first *max_lines* lines, noting the truncation when it happens."""
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    return "\n".join(lines[:max_lines]) + (
        f"\n[... truncated to first {max_lines} lines - use read_file for the full file]"
    )


def _files_block(heading: str, files: dict[str, str]) -> str:
    if not files:
        return f"## {heading}\n\n(none)"
    parts = [f"## {heading}\n"]
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```\n")
    return "\n".join(parts)


# Neighbor cap (lines) in hunk mode: imports/signatures live near the top.
_NEIGHBOR_HUNK_MODE_MAX_LINES = 40


def build_user_blocks(
    pr_title: str,
    pr_body: str,
    diff: str,
    changed_files: dict[str, str],
    neighbor_files: dict[str, str],
    max_total_chars: int = 600_000,
    full_file_max_lines: int | None = None,
    hunk_context_lines: int = 60,
    trimmed_paths_out: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the user message for review.

    Truncation priority (lowest first): neighbors, changed files.
    The diff is never truncated.

    When ``full_file_max_lines`` is set, changed files longer than it are sent as
    hunk excerpts (+/- ``hunk_context_lines``) rather than full contents, and
    neighbors are capped to their first lines - agents pull the rest via tools.
    ``full_file_max_lines=None`` preserves the full-content behavior. Excerpted
    paths are added to ``trimmed_paths_out`` when provided.
    """
    pr_meta = (
        f"## PR metadata\n\n**Title:** {pr_title}\n\n**Description:**\n\n{pr_body or '(empty)'}"
    )
    diff_block = f"## Diff\n\n```diff\n{diff}\n```"

    if full_file_max_lines is None:
        changed_for_block = changed_files
        neighbor_for_block = neighbor_files
    else:
        changed_for_block = {}
        for path, content in changed_files.items():
            if len(content.splitlines()) > full_file_max_lines:
                hunks = _hunk_windows_for_file(diff, path)
                # No discoverable hunks for an over-limit file (binary diff, path
                # mismatch, or a diff slice that omits this file's hunks) must still
                # be trimmed, or the per-file token cap silently leaks a full file.
                changed_for_block[path] = (
                    _hunk_excerpt(content, hunks, hunk_context_lines)
                    if hunks
                    else _cap_lines(content, full_file_max_lines)
                )
                if trimmed_paths_out is not None:
                    trimmed_paths_out.add(path)
                continue
            changed_for_block[path] = content
        neighbor_for_block = {
            path: _cap_lines(content, _NEIGHBOR_HUNK_MODE_MAX_LINES)
            for path, content in neighbor_files.items()
        }

    changed_block = _files_block("Changed files (full contents)", changed_for_block)
    neighbor_block = _files_block("Neighbor files (context)", neighbor_for_block)

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
    for path, content in changed_for_block.items():
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


# PR map: a deterministic, no-LLM digest of the whole PR's shape. Included in
# every shard's context so a shard reviewing one slice still sees the full PR.
_PR_MAP_MAX_FILES = 150
_PR_MAP_MAX_BYTES = 4096
_PR_MAP_FILE_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$")
# Grab the enclosing symbol name a diff hunk header carries after the second @@.
_PR_MAP_SYMBOL_RE = re.compile(
    r"\b(?:fn|def|func|class|impl|struct|trait|enum|interface|type)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


def build_pr_map_block(files: dict[str, str], diff: str) -> str:
    """Compact whole-PR digest: totals plus one line per file with hunk symbols.

    Each file line is ``path (+A/-D)`` optionally followed by the function/impl
    names scraped from that file's ``@@ ... @@ <context>`` hunk headers. Capped
    at ``_PR_MAP_MAX_FILES`` files and ``_PR_MAP_MAX_BYTES`` bytes.
    """
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    total_adds = total_dels = 0

    for line in diff.splitlines():
        header = _PR_MAP_FILE_HEADER_RE.match(line)
        if header:
            current = {"path": header.group(1), "adds": 0, "dels": 0, "symbols": []}
            entries.append(current)
            continue
        if current is None:
            continue
        if line.startswith("@@"):
            ctx = line.split("@@")[-1].strip() if "@@" in line[2:] else ""
            m = _PR_MAP_SYMBOL_RE.search(ctx)
            if m and m.group(1) not in current["symbols"]:
                current["symbols"].append(m.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            current["adds"] += 1
            total_adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            current["dels"] += 1
            total_dels += 1

    file_count = len(files) or len(entries)
    lines = [f"## PR map\n\nTotal: +{total_adds}/-{total_dels} across {file_count} file(s)\n"]
    for e in entries[:_PR_MAP_MAX_FILES]:
        row = f"- {e['path']} (+{e['adds']}/-{e['dels']})"
        if e["symbols"]:
            row += ": " + ", ".join(e["symbols"])
        lines.append(row)
    if len(entries) > _PR_MAP_MAX_FILES:
        lines.append(f"- ... and {len(entries) - _PR_MAP_MAX_FILES} more file(s)")

    block = "\n".join(lines)
    if len(block.encode("utf-8")) > _PR_MAP_MAX_BYTES:
        block = block.encode("utf-8")[:_PR_MAP_MAX_BYTES].decode("utf-8", errors="ignore")
        block += "\n[... PR map truncated ...]"
    return block

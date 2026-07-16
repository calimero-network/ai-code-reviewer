"""Validate structured suggested replacements before they become GitHub suggestions.

A suggestion GitHub renders as one-click "apply" is only trustworthy if it lands
on a valid line range and leaves the file parseable. This module applies the
replacement in memory and re-checks the patched file, so a broken fix is demoted
to prose instead of being offered as a blind-apply block.
"""

from __future__ import annotations

import ast
import json
import logging
import tomllib
from collections.abc import Callable
from pathlib import PurePosixPath

import yaml

from ai_reviewer.models.findings import ConsolidatedFinding, Severity

logger = logging.getLogger(__name__)

_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _bracket_balance(text: str) -> dict[str, int]:
    """Net open-minus-close count per bracket type (no nesting/order check).

    A crude structural signal for languages we cannot parse: if the patched file
    keeps the same per-type balance as the original, the edit did not orphan a
    delimiter. This is intentionally conservative, not a real parser.
    """
    counts = {"()": 0, "[]": 0, "{}": 0}
    for ch in text:
        if ch == "(":
            counts["()"] += 1
        elif ch == ")":
            counts["()"] -= 1
        elif ch == "[":
            counts["[]"] += 1
        elif ch == "]":
            counts["[]"] -= 1
        elif ch == "{":
            counts["{}"] += 1
        elif ch == "}":
            counts["{}"] -= 1
    return counts


def _syntactic_ok(patched: str, original: str, file_path: str) -> bool:
    """Extension-driven syntactic check on the patched file. Any error -> False."""
    ext = PurePosixPath(file_path).suffix.lower()
    try:
        if ext == ".py":
            ast.parse(patched)
        elif ext == ".json":
            json.loads(patched)
        elif ext in (".yaml", ".yml"):
            yaml.safe_load(patched)
        elif ext == ".toml":
            tomllib.loads(patched)
        else:
            # Conservative fallback: the patched file must keep the same bracket
            # balance the original had (compare deltas, not absolute balance, so
            # an already-unbalanced fragment is judged only on what the edit changed).
            return _bracket_balance(patched) == _bracket_balance(original)
    except Exception as e:  # noqa: BLE001
        logger.debug("Syntactic check failed for %s: %s", file_path, e)
        return False
    return True


def validate_replacement(
    original_content: str,
    line_start: int,
    line_end: int | None,
    replacement: str,
    file_path: str,
) -> bool:
    """Return True if *replacement* applies cleanly to line_start..line_end.

    Anchors the 1-indexed range within the file, splices the replacement in
    memory, and re-checks the patched file's syntax by extension. Returns False
    on any anchor violation, parse error, or unexpected exception (fail closed).
    """
    try:
        lines = original_content.splitlines()
        end = line_end if line_end is not None else line_start
        if line_start < 1 or end < line_start or end > len(lines):
            return False
        patched_lines = lines[: line_start - 1] + replacement.splitlines() + lines[end:]
        patched = "\n".join(patched_lines)
        return _syntactic_ok(patched, original_content, file_path)
    except Exception as e:  # noqa: BLE001
        logger.debug("validate_replacement error for %s: %s", file_path, e)
        return False


_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.SUGGESTION: 2,
    Severity.NITPICK: 3,
}


def validate_finding_fixes(
    findings: list[ConsolidatedFinding],
    get_file_content: Callable[[str], str | None],
    max_checks: int = 5,
) -> None:
    """Validate up to *max_checks* structured replacements in place (severity first).

    For each highest-severity finding carrying a ``suggested_replacement``: fetch
    the file, validate the replacement, and either set ``fix_validated = True`` or
    demote the fix to prose (null the replacement). A None from *get_file_content*
    fails closed to prose. Mutates the findings; returns nothing.
    """
    candidates = [f for f in findings if f.suggested_replacement]
    candidates.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 4))

    for finding in candidates[:max_checks]:
        replacement = finding.suggested_replacement
        if not replacement:  # candidates filter guarantees this; keep for narrowing + fail-safe
            continue
        content = get_file_content(finding.file_path)
        if content is None:
            logger.info(
                "fix-check: no content for %s; demoting fix on %r to prose",
                finding.file_path,
                finding.title,
            )
            finding.suggested_replacement = None
            continue
        if validate_replacement(
            content,
            finding.line_start,
            finding.line_end,
            replacement,
            finding.file_path,
        ):
            finding.fix_validated = True
        else:
            logger.info(
                "fix-check: replacement invalid for %s:%d %r; demoting to prose",
                finding.file_path,
                finding.line_start,
                finding.title,
            )
            finding.suggested_replacement = None

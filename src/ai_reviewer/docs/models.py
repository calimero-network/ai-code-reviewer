"""Shared data contracts for the doc-update pipeline (Understand → Route → Apply → Verify)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# FIND/REPLACE delimiters the HTML patcher uses to frame model edits. They are
# structural tokens, never document text; if one survives into finalized content
# a patch leaked its raw markers, so that file must not be committed.
DOC_PATCH_MARKERS: tuple[str, ...] = ("<<<FIND", "FIND>>>", "<<<REPLACE", "REPLACE>>>")


def contains_patch_marker(content: str) -> bool:
    return any(marker in content for marker in DOC_PATCH_MARKERS)


def meets_threshold(confidence: str, threshold: str) -> bool:
    """True if *confidence* is at least *threshold* on the low<medium<high scale."""
    # Unknown confidence -> 0 (low) is intentional: an unrecognized verifier
    # confidence must fail toward flagging, never silently meet the bar.
    return CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK.get(threshold, 1)


@dataclass(frozen=True)
class Change:
    kind: str  # "new_feature" | "fix" | "rename" | "removal" | "behavior_change"
    title: str
    what_changed: str
    why: str
    symbols: list[str]
    files: list[str]
    doc_impact: str


@dataclass
class ChangeSummary:
    pr_intent: str
    changes: list[Change]


@dataclass
class DocAction:
    change: Change
    action: str  # "update_section" | "add_section" | "create_page"
    target_path: str
    anchor: str | None = None
    best_fit_reason: str = (
        ""  # required in practice; default only due to field ordering after `anchor`
    )


@dataclass(frozen=True)
class FileWrite:
    path: str
    content: str


@dataclass
class DocDraft:
    """A concrete doc edit ready to commit (or flagged/failed)."""

    action: str
    target_path: str
    updated_content: str
    before_content: str | None = None
    aux_edits: list[FileWrite] = field(default_factory=list)
    change: Change | None = None
    error: str | None = None
    flagged_reason: str | None = None
    aux_meta: dict | None = None


@dataclass
class Verdict:
    reflects_change: bool
    confidence: str  # "low" | "medium" | "high"
    notes: str


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating fences/prose.

    Raises ValueError if no JSON *object* can be parsed (a bare array or
    scalar is rejected — every pipeline stage expects an object).
    """
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    obj = re.search(r"\{[\s\S]*\}", s)
    if obj:
        try:
            parsed = json.loads(obj.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse JSON object: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found in response")

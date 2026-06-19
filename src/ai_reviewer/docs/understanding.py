"""Stage 1 — read the full PR once into a structured ChangeSummary."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import Change, ChangeSummary, extract_json

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a software historian. Given a pull request (title, body, commit messages, and a
unified diff), describe the SUBSTANTIVE changes — what behavior/architecture changed and why —
NOT a line-by-line restatement. A function rename that reflects a behavioral change must be
reported as the behavioral change, with the rename mentioned as a detail.

Return ONLY a JSON object, no prose, with this exact shape:
{
  "pr_intent": "one or two sentences",
  "changes": [
    {
      "kind": "new_feature | fix | rename | removal | behavior_change",
      "title": "short imperative title",
      "what_changed": "the substance, in prose",
      "why": "rationale if known, else empty string",
      "symbols": ["affected function/type names"],
      "files": ["paths"],
      "doc_impact": "what documentation should now say (or empty if none)"
    }
  ]
}
If nothing is documentation-relevant, return {"pr_intent": "...", "changes": []}."""

_MERGE_SYSTEM = """\
You are merging several partial change-lists from one pull request into a single coherent
summary. Deduplicate, group related edits, and drop noise. Return ONLY the same JSON object
shape: {"pr_intent": "...", "changes": [...]} as previously specified."""


def _change_from_dict(d: dict) -> Change:
    return Change(
        kind=str(d.get("kind", "fix")),
        title=str(d.get("title", "")),
        what_changed=str(d.get("what_changed", "")),
        why=str(d.get("why", "")),
        symbols=[str(s) for s in d.get("symbols", [])],
        files=[str(f) for f in d.get("files", [])],
        doc_impact=str(d.get("doc_impact", "")),
    )


def _summary_from_dict(d: dict) -> ChangeSummary:
    return ChangeSummary(
        pr_intent=str(d.get("pr_intent", "")),
        changes=[_change_from_dict(c) for c in d.get("changes", [])],
    )


def _user_prompt(pr_title: str, pr_body: str, commit_messages: list[str], diff: str) -> str:
    commits = "\n".join(f"- {m}" for m in commit_messages)
    return (
        f"## PR Title\n{pr_title}\n\n"
        f"## PR Body\n{pr_body}\n\n"
        f"## Commit Messages\n{commits}\n\n"
        f"## Diff\n{diff}\n"
    )


def _split_diff_by_file(diff: str) -> list[str]:
    """Split a unified diff into per-file chunks on `diff --git` boundaries."""
    parts = diff.split("\ndiff --git ")
    if len(parts) == 1:
        return [diff]
    chunks = [parts[0]] if parts[0].startswith("diff --git ") else []
    chunks.extend("diff --git " + p for p in parts[1:])
    return [c for c in chunks if c.strip()]


# Summary length scales with the diff (more changes → more JSON), so the output
# cap scales too: ~1 token per 16 diff chars, clamped to [_MIN, _MAX]. max_tokens
# is a ceiling not a target, so small PRs still bill only their actual output.
_MIN_SUMMARY_TOKENS = 2048
_MAX_SUMMARY_TOKENS = 16384
_DIFF_CHARS_PER_SUMMARY_TOKEN = 16


def _summary_max_tokens(diff_chars: int) -> int:
    est = diff_chars // _DIFF_CHARS_PER_SUMMARY_TOKEN
    return max(_MIN_SUMMARY_TOKENS, min(_MAX_SUMMARY_TOKENS, est))


def _safe_summary(raw: str, pr_intent_fallback: str) -> ChangeSummary:
    """Parse a model summary; an unparseable response yields an empty summary
    (the run skips) rather than aborting with an unhandled error."""
    try:
        return _summary_from_dict(extract_json(raw))
    except ValueError as exc:
        logger.warning("Stage-1 summary JSON unparseable (%s); returning empty summary", exc)
        return ChangeSummary(pr_intent=pr_intent_fallback, changes=[])


async def summarize_pr_changes(
    *,
    pr_title: str,
    pr_body: str,
    commit_messages: list[str],
    diff: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    max_diff_chars: int = 250_000,
) -> ChangeSummary:
    """Produce a ChangeSummary from the full PR. Map-reduces when the diff exceeds the cap."""
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local: avoid circular

    async with AnthropicClient(anthropic_cfg) as client:
        if len(diff) <= max_diff_chars:
            raw = await client.run_completion(
                model=model,
                system=_SYSTEM,
                user=_user_prompt(pr_title, pr_body, commit_messages, diff),
                max_tokens=_summary_max_tokens(len(diff)),
            )
            return _safe_summary(raw, pr_title)

        # Map-reduce: summarize each file chunk (truncated to the cap), then merge.
        logger.info("Diff %d chars exceeds cap %d — map-reduce", len(diff), max_diff_chars)
        partials: list[dict] = []
        for chunk in _split_diff_by_file(diff):
            raw = await client.run_completion(
                model=model,
                system=_SYSTEM,
                user=_user_prompt(pr_title, pr_body, commit_messages, chunk[:max_diff_chars]),
                max_tokens=_summary_max_tokens(len(chunk)),
            )
            try:
                partials.append(extract_json(raw))
            except ValueError:
                logger.warning("Skipping unparseable partial summary for a diff chunk")
        if not partials:
            logger.warning("Map-reduce produced no parseable partials; returning empty summary")
            return ChangeSummary(pr_intent=pr_title, changes=[])
        merged_input = f"## PR Title\n{pr_title}\n\n## Partial change-lists (JSON)\n" + "\n".join(
            json.dumps(p) for p in partials
        )
        raw = await client.run_completion(
            model=model,
            system=_MERGE_SYSTEM,
            user=merged_input,
            max_tokens=_summary_max_tokens(len(diff)),
        )
        return _safe_summary(raw, pr_title)

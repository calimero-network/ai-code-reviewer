"""Stage 4 — confidence gate: flag (don't ship) drafts that don't reflect their change."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import DocDraft, extract_json, meets_threshold

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM = """\
You check whether an edited documentation page now reflects a specific code change.
You are given the change, the BEFORE content (or a note that the page is new), and the AFTER
content. Judge ONLY whether the AFTER faithfully conveys the change's substance — not style.
Return ONLY JSON: {"reflects_change": true|false, "confidence": "low|medium|high",
"notes": "one sentence; if false, say what is missing"}"""


async def verify_draft(
    *,
    draft: DocDraft,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    threshold: str,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    if draft.error or not draft.updated_content:
        return draft  # already failed upstream; nothing to verify

    change = draft.change
    user = (
        f"## Change\ntitle: {getattr(change, 'title', '')}\n"
        f"what_changed: {getattr(change, 'what_changed', '')}\n"
        f"doc_impact: {getattr(change, 'doc_impact', '')}\n\n"
        f"## BEFORE\n{draft.before_content or '(new page — no prior content)'}\n\n"
        f"## AFTER\n{draft.updated_content}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            raw = await client.run_completion(
                model=model, system=_VERIFY_SYSTEM, user=user, max_tokens=512
            )
            v = extract_json(raw)
        except Exception as exc:  # noqa: BLE001 — extract_json ValueError or any client error
            logger.warning("Verify failed for %s: %s — flagging", draft.target_path, exc)
            return replace(draft, updated_content="", flagged_reason=f"verification error: {exc}")

    # Parse robustly: a JSON string like "false" is truthy under bool(); only an
    # explicit true (bool or "true") counts as reflecting — everything else flags.
    _rc = v.get("reflects_change", False)
    reflects = _rc is True or (isinstance(_rc, str) and _rc.strip().lower() == "true")
    confidence = str(v.get("confidence", "low"))
    notes = str(v.get("notes", ""))
    if reflects and meets_threshold(confidence, threshold):
        return draft
    if not reflects:
        reason = f"does not reflect the change ({confidence} confidence): {notes}"
    else:
        reason = f"low-confidence doc update ({confidence}): {notes}"
    return replace(draft, updated_content="", flagged_reason=reason)

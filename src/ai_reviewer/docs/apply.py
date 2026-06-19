"""Stage 3 (existing pages) — surgical update_section and additive add_section edits."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from ai_reviewer.docs.analyzer import _apply_html_patches, _is_no_update_response
from ai_reviewer.docs.models import Change, DocAction, DocDraft

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_CARD_CYCLE = ["ga", "gb", "gc", "gd"]

# Insert a new section just before the .content/.main close that precedes the nav script.
_CONTENT_CLOSE_RE = re.compile(r"(\n</div>\s*\n</div>\s*\n<script src=\"nav\.js\")")

_UPDATE_SYSTEM = """\
You are updating an existing documentation file (HTML or Markdown) after a code change.
Output ONLY FIND/REPLACE blocks — do not return the whole file. Make the file reflect the
change described, including ADDING a sentence/bullet where the change introduces something new.
Format (repeatable):
<<<FIND
exact text copied verbatim from the file (enough to be unique)
FIND>>>
<<<REPLACE
replacement text
REPLACE>>>
If no change is needed, output exactly: NO_UPDATE_NEEDED"""

_REPAIR_NOTE = (
    "\n\n## RETRY — your previous FIND/REPLACE did not apply\n"
    "At least one FIND block was not found in the page. Each FIND must be copied "
    "CHARACTER-FOR-CHARACTER from the page above (exact whitespace, tags, punctuation). "
    "Re-emit corrected FIND/REPLACE blocks, or NO_UPDATE_NEEDED if nothing applies."
)

_ADD_SECTION_SYSTEM = """\
You are adding ONE new section to an existing HTML documentation page.
Output ONLY a single HTML block of the form:
<div class="card {card_class}"><h2>Title</h2> ...content... </div>
Use ONLY these constructs: <h2>/<h3>, <p>, <ul>/<ol>/<li>, <code>, <pre class="code">,
<strong>, <em>. Do NOT invent CSS classes. No commentary before or after the block."""


def next_card_class(html: str) -> str:
    matches = re.findall(r'class="card (g[abcd])"', html)
    if not matches:
        return "ga"
    last = matches[-1]
    return _CARD_CYCLE[(_CARD_CYCLE.index(last) + 1) % len(_CARD_CYCLE)]


def insert_section(html: str, section_html: str) -> str | None:
    """Insert *section_html* just before the .content wrapper closes. None if no anchor."""
    m = _CONTENT_CLOSE_RE.search(html)
    if not m:
        return None
    insert_at = m.start(1)
    return html[:insert_at] + "\n" + section_html + html[insert_at:]


async def apply_update_section(
    action: DocAction,
    current_content: str,
    change: Change,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    allow_new_sections: bool = True,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    user = (
        f"## Page: {action.target_path}\n\n{current_content}\n\n"
        f"## Change to reflect\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"doc_impact: {change.doc_impact}\nsymbols: {', '.join(change.symbols)}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            raw = (
                await client.run_completion(
                    model=model, system=_UPDATE_SYSTEM, user=user, max_tokens=8192
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001
            return DocDraft(
                action="update_section",
                target_path=action.target_path,
                updated_content="",
                change=change,
                before_content=current_content,
                error=str(exc),
            )

        if _is_no_update_response(raw):
            return DocDraft(
                action="update_section",
                target_path=action.target_path,
                updated_content="",
                before_content=current_content,
                change=change,
            )

        patched = _apply_html_patches(current_content, raw)
        if patched is None:
            # Most common apply failure: the FIND text wasn't copied verbatim.
            # Retry once, insisting on character-for-character FIND blocks.
            try:
                retry = (
                    await client.run_completion(
                        model=model,
                        system=_UPDATE_SYSTEM,
                        user=user + _REPAIR_NOTE,
                        max_tokens=8192,
                    )
                ).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Repair retry failed for %s: %s", action.target_path, exc)
                retry = ""
            if _is_no_update_response(retry):
                # Model now says nothing applies — respect it as a no-op rather
                # than forcing the add_section fallback below.
                return DocDraft(
                    action="update_section",
                    target_path=action.target_path,
                    updated_content="",
                    before_content=current_content,
                    change=change,
                )
            if retry:
                patched = _apply_html_patches(current_content, retry)

    if patched is not None:
        return DocDraft(
            action="update_section",
            target_path=action.target_path,
            updated_content=patched,
            before_content=current_content,
            change=change,
        )
    # Still unpatched after the retry. Rather than silently drop the page, fall
    # back to an additive section so the change is still documented (draft PR;
    # the verify stage still gates it).
    if allow_new_sections:
        return await apply_add_section(action, current_content, change, anthropic_cfg, model)
    return DocDraft(
        action="update_section",
        target_path=action.target_path,
        updated_content="",
        change=change,
        before_content=current_content,
        error="could not apply HTML patches (FIND not found or malformed)",
    )


async def apply_add_section(
    action: DocAction,
    current_content: str,
    change: Change,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    card_class = next_card_class(current_content)
    system = _ADD_SECTION_SYSTEM.replace("{card_class}", card_class)
    user = (
        f"## Page: {action.target_path}\n\n{current_content}\n\n"
        f"## New thing to document\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"why: {change.why}\ndoc_impact: {change.doc_impact}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            section = (
                await client.run_completion(model=model, system=system, user=user, max_tokens=4096)
            ).strip()
        except Exception as exc:  # noqa: BLE001
            return DocDraft(
                action="add_section",
                target_path=action.target_path,
                updated_content="",
                change=change,
                before_content=current_content,
                error=str(exc),
            )

    merged = insert_section(current_content, section)
    if merged is None:
        return DocDraft(
            action="add_section",
            target_path=action.target_path,
            updated_content="",
            change=change,
            before_content=current_content,
            error="could not locate content-wrapper anchor for section insertion",
        )
    return DocDraft(
        action="add_section",
        target_path=action.target_path,
        updated_content=merged,
        before_content=current_content,
        change=change,
    )

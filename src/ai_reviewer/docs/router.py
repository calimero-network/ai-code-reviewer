"""Stage 2 — map each change to a doc action (update_section | add_section | create_page)."""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import Change, ChangeSummary, DocAction, extract_json

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_DOC_DIR_PREFIXES = ("architecture/", "docs/", "docs-static/", "doc/")

_ROUTE_SYSTEM = """\
You route a code change to the single best documentation action.
You are given the change and a list of existing doc pages (with titles inferred from paths).
Decide ONE of:
- "update_section": an existing page already documents this; we'll edit it. (give target_path from the list)
- "add_section": an existing page is the right home but lacks a section for this. (give target_path from the list)
- "create_page": nothing fits; a new page is warranted. (give a new target_path like "architecture/<slug>.html")
Always also include "best_fit_existing": the closest existing page path from the list (or "").
Return ONLY JSON: {"action": "...", "target_path": "...", "anchor": null,
"best_fit_reason": "...", "best_fit_existing": "..."}"""


def build_doc_index(existing_paths: list[str]) -> list[str]:
    """Existing doc pages only (under known doc dirs)."""
    return [p for p in existing_paths if p.startswith(_DOC_DIR_PREFIXES) and p.endswith(".html")]


def _mapping_target(
    mapping: dict[str, list[str]],
    changed_paths: list[str],
    doc_index: list[str],
) -> str | None:
    for glob_pattern, targets in mapping.items():
        if any(fnmatch.fnmatch(p, glob_pattern) for p in changed_paths):
            for t in targets:
                if t in doc_index:
                    return t
    return None


async def route_changes(
    *,
    summary: ChangeSummary,
    source_to_docs_mapping: dict[str, list[str]],
    changed_paths: list[str],
    doc_index: list[str],
    allow_new_pages: bool,
    allow_new_sections: bool,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> list[DocAction]:
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local: avoid circular

    actions: list[DocAction] = []
    needs_model: list[Change] = []

    for change in summary.changes:
        target = _mapping_target(source_to_docs_mapping, changed_paths, doc_index)
        if target is not None:
            actions.append(
                DocAction(
                    change=change,
                    action="update_section",
                    target_path=target,
                    best_fit_reason="source_to_docs_mapping",
                )
            )
        else:
            needs_model.append(change)

    if not needs_model:
        return actions

    index_listing = "\n".join(f"- {p}" for p in doc_index) or "(no existing doc pages)"
    async with AnthropicClient(anthropic_cfg) as client:
        for change in needs_model:
            user = (
                f"## Change\nkind: {change.kind}\ntitle: {change.title}\n"
                f"what_changed: {change.what_changed}\ndoc_impact: {change.doc_impact}\n\n"
                f"## Existing doc pages\n{index_listing}\n"
            )
            try:
                raw = await client.run_completion(
                    model=model, system=_ROUTE_SYSTEM, user=user, max_tokens=512
                )
                d = extract_json(raw)
            except Exception as exc:  # noqa: BLE001 — extract_json ValueError or any client error
                logger.warning("Routing failed for %r: %s", change.title, exc)
                continue

            action = str(d.get("action", "add_section"))
            target_path = str(d.get("target_path", ""))
            best_fit = str(d.get("best_fit_existing", ""))

            if action == "create_page" and not allow_new_pages:
                if allow_new_sections and best_fit:
                    action, target_path = "add_section", best_fit
                else:
                    logger.info("Dropping change %r: new pages disabled, no best-fit", change.title)
                    continue
            if action == "add_section" and not allow_new_sections:
                action = "update_section"
            if action in ("update_section", "add_section") and not target_path:
                target_path = best_fit
                if not target_path:
                    continue

            actions.append(
                DocAction(
                    change=change,
                    action=action,
                    target_path=target_path,
                    anchor=d.get("anchor"),
                    best_fit_reason=str(d.get("best_fit_reason", "")),
                )
            )
    return actions

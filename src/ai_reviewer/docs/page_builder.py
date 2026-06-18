# src/ai_reviewer/docs/page_builder.py
"""Stage 3 (new pages) — build a new page and wire it into nav.js (+ best-effort index.html)."""

from __future__ import annotations

import html
import os
import re
from typing import TYPE_CHECKING

from ai_reviewer.docs.apply import apply_add_section
from ai_reviewer.docs.models import Change, DocAction, DocDraft, FileWrite

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

_NEW_PAGE_SYSTEM = """\
You are writing a new HTML documentation page for the Calimero Core architecture site.
You are given a SIBLING page to copy the exact skeleton from (doctype, <head>, <title> suffix
"— Calimero Core Architecture", stylesheet link, .main/.content wrappers, breadcrumb, and the
trailing <script src="nav.js"></script>). Produce a COMPLETE page that:
- keeps that skeleton EXACTLY (only change the <title>, breadcrumb label, and <h1>),
- expresses the content as a sequence of <div class="card ga|gb|gc|gd"> blocks (cycle the class),
- uses ONLY these constructs inside cards: <h2>/<h3>, <p>, <ul>/<ol>/<li>, <code>,
  <pre class="code">, <strong>, <em>. Do NOT invent CSS classes or inline styles.
Output ONLY the page HTML, nothing else."""


def insert_nav_entry(nav_js: str, label: str, href: str, dot: str, section: str) -> str | None:
    """Insert a NAV entry right after the `{ section: '<section>' }` marker. None if absent."""
    pattern = re.compile(r"(\{\s*section:\s*['\"]" + re.escape(section) + r"['\"]\s*\},?[ \t]*\n)")
    m = pattern.search(nav_js)
    if not m:
        return None
    indent_m = re.match(r"([ \t]*)", m.group(1))
    indent = indent_m.group(1) if indent_m else "    "
    entry = f"{indent}{{ label: '{label}', href: '{href}', dot: '{dot}' }},\n"
    return nav_js[: m.end(1)] + entry + nav_js[m.end(1) :]


def insert_index_link(index_html: str, href: str, title: str, blurb: str) -> str | None:
    """Best-effort: add a hero-card to the first g3 grid after 'Crate Index'. None if not found."""
    idx = index_html.find("Crate Index")
    if idx == -1:
        return None
    grid = index_html.find('<div class="g3">', idx)
    if grid == -1:
        return None
    insert_at = grid + len('<div class="g3">')
    safe_title = html.escape(title)
    safe_blurb = html.escape(blurb)
    card = (
        f'\n  <a href="{href}" class="hero-card">'
        f'<span class="card-icon">&#9670;</span><h3>{safe_title}</h3><p>{safe_blurb}</p></a>'
    )
    return index_html[:insert_at] + card + index_html[insert_at:]


async def build_new_page(
    *,
    action: DocAction,
    change: Change,
    sibling_html: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> str:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    user = (
        f"## Sibling page (copy the skeleton)\n{sibling_html}\n\n"
        f"## New page path\n{action.target_path}\n\n"
        f"## Feature to document\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"why: {change.why}\ndoc_impact: {change.doc_impact}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        return (
            await client.run_completion(
                model=model, system=_NEW_PAGE_SYSTEM, user=user, max_tokens=8192
            )
        ).strip()


async def apply_create_page(
    *,
    action: DocAction,
    sibling_html: str,
    nav_js: str,
    index_html: str,
    change: Change,
    section_group: str,
    dot: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    allow_new_sections: bool,
    best_fit_for_downgrade: str,
    best_fit_html: str,
) -> DocDraft:
    """Build a new page and register it. Orphan guard: must wire nav.js or we downgrade."""
    label = change.title
    href = action.target_path.split("/")[-1]

    # Orphan guard FIRST: if we can't register the page in nav, never create it.
    new_nav = insert_nav_entry(nav_js, label, href, dot, section_group)
    if new_nav is None:
        if allow_new_sections and best_fit_for_downgrade and best_fit_html:
            downgrade = DocAction(
                change=change, action="add_section", target_path=best_fit_for_downgrade
            )
            return await apply_add_section(downgrade, best_fit_html, change, anthropic_cfg, model)
        return DocDraft(
            action="create_page",
            target_path=action.target_path,
            updated_content="",
            change=change,
            error="orphan guard: nav.js section anchor not found",
        )

    page_html = await build_new_page(
        action=action,
        change=change,
        sibling_html=sibling_html,
        anthropic_cfg=anthropic_cfg,
        model=model,
    )
    doc_dir = os.path.dirname(action.target_path)
    nav_path = f"{doc_dir}/nav.js" if doc_dir else "nav.js"
    index_path = f"{doc_dir}/index.html" if doc_dir else "index.html"
    aux = [FileWrite(path=nav_path, content=new_nav)]
    new_index = insert_index_link(index_html, href, change.title, change.what_changed[:120])
    if new_index is not None:
        aux.append(FileWrite(path=index_path, content=new_index))
    return DocDraft(
        action="create_page",
        target_path=action.target_path,
        updated_content=page_html,
        change=change,
        aux_edits=aux,
    )

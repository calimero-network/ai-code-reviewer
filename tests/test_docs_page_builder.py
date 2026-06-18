# tests/test_docs_page_builder.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, DocAction
from ai_reviewer.docs.page_builder import (
    apply_create_page,
    insert_index_link,
    insert_nav_entry,
)

_NAV = (
    "  const NAV = [\n"
    "    { label: 'Home', href: 'index.html', dot: '#f59e0b' },\n"
    "    { section: 'Architecture Deep-Dive' },\n"
    "    { label: 'Auto-Follow', href: 'auto-follow.html', dot: '#10b981' },\n"
    "  ];\n"
)

_SIBLING = (
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
    "<title>Auto-Follow — Calimero Core Architecture</title>\n"
    '<link rel="stylesheet" href="styles.css"></head>\n<body>\n'
    '<div class="main"><div class="content">\n'
    '<div class="breadcrumb"><a href="index.html">Home</a><span class="sep">/</span><span>Auto-Follow</span></div>\n'
    "<h1>Auto-Follow</h1>\n"
    '</div></div>\n<script src="nav.js"></script>\n</body></html>'
)


def test_insert_nav_entry_after_section():
    out = insert_nav_entry(_NAV, "Widgets", "widgets.html", "#10b981", "Architecture Deep-Dive")
    assert out is not None
    assert "widgets.html" in out
    # Entry sits right after the section marker.
    assert out.index("Architecture Deep-Dive") < out.index("widgets.html")
    assert out.index("widgets.html") < out.index("Auto-Follow")
    # Still a single NAV array close.
    assert out.count("];") == 1


def test_insert_nav_entry_missing_section_returns_none():
    assert insert_nav_entry(_NAV, "X", "x.html", "#fff", "Nonexistent Section") is None


def test_insert_index_link_missing_returns_none():
    assert insert_index_link("<html>no crate index</html>", "x.html", "X", "b") is None


@pytest.mark.asyncio
async def test_create_page_wires_nav_and_emits_page_filewrite():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    page_body = (
        '<!DOCTYPE html>\n<html lang="en"><head>'
        "<title>Widgets — Calimero Core Architecture</title>"
        '<link rel="stylesheet" href="styles.css"></head><body>'
        '<div class="main"><div class="content"><h1>Widgets</h1>'
        '<div class="card ga"><h2>Overview</h2></div>'
        '</div></div><script src="nav.js"></script></body></html>'
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=page_body)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_create_page(
            action=action,
            sibling_html=_SIBLING,
            nav_js=_NAV,
            index_html="<html></html>",
            change=change,
            section_group="Architecture Deep-Dive",
            dot="#10b981",
            anthropic_cfg=cfg,
            model="m",
            allow_new_sections=True,
            best_fit_for_downgrade="architecture/auto-follow.html",
            best_fit_html=_SIBLING,
        )
    assert draft.error is None
    assert draft.action == "create_page"
    assert draft.target_path == "architecture/widgets.html"
    # Page content + a nav.js aux edit are present; nav.js edit registered the page.
    aux_paths = {fw.path for fw in draft.aux_edits}
    assert any(p.endswith("nav.js") for p in aux_paths)
    nav_edit = next(fw for fw in draft.aux_edits if fw.path.endswith("nav.js"))
    assert "widgets.html" in nav_edit.content


@pytest.mark.asyncio
async def test_create_page_orphan_guard_downgrades_when_nav_anchor_missing():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    section_block = '<div class="card gb"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        # First call = page body (unused after downgrade), then add_section block.
        inst.run_completion = AsyncMock(side_effect=["<html>page</html>", section_block])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_create_page(
            action=action,
            sibling_html=_SIBLING,
            nav_js="const NAV = [];",  # no section anchors
            index_html="<html></html>",
            change=change,
            section_group="Nonexistent Section",
            dot="#10b981",
            anthropic_cfg=cfg,
            model="m",
            allow_new_sections=True,
            best_fit_for_downgrade="architecture/auto-follow.html",
            best_fit_html=_SIBLING,
        )
    # Downgraded to an add_section on the best-fit page; no orphan page emitted.
    assert draft.action == "add_section"
    assert draft.target_path == "architecture/auto-follow.html"
    assert all(not fw.path.endswith("widgets.html") for fw in draft.aux_edits)

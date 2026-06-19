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
    wire_new_pages,
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
    # aux_edits is now empty for create_page; wiring info lives in aux_meta.
    assert draft.aux_edits == []
    # aux_meta carries the nav/index wiring data for orchestrator-level accumulation.
    assert draft.aux_meta is not None
    assert draft.aux_meta["nav"]["href"] == "widgets.html"
    assert draft.aux_meta["index"]["href"] == "widgets.html"


@pytest.mark.asyncio
async def test_create_page_orphan_guard_downgrades_when_nav_anchor_missing():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    section_block = '<div class="card gb"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        # nav anchor missing -> guard downgrades to apply_add_section, which makes ONE
        # run_completion call for the section; build_new_page is never reached.
        inst.run_completion = AsyncMock(return_value=section_block)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_create_page(
            action=action,
            sibling_html=_SIBLING,
            nav_js="const NAV = [];",  # no section anchors
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


@pytest.mark.asyncio
async def test_create_page_orphan_guard_errors_when_no_best_fit():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        draft = await apply_create_page(
            action=action,
            sibling_html=_SIBLING,
            nav_js="const NAV = [];",
            change=change,
            section_group="Nonexistent Section",
            dot="#10b981",
            anthropic_cfg=cfg,
            model="m",
            allow_new_sections=True,
            best_fit_for_downgrade="",
            best_fit_html="",
        )
    assert draft.error is not None
    assert "orphan guard" in draft.error
    assert draft.updated_content == ""
    inst.run_completion.assert_not_called()


def test_wire_new_pages_folds_multiple_pages_into_one_nav():
    """wire_new_pages accumulates two pages' nav entries into a single nav.js."""
    metas = [
        {
            "nav": {
                "label": "Widgets",
                "href": "widgets.html",
                "dot": "#10b981",
                "section": "Architecture Deep-Dive",
            },
            "index": {"href": "widgets.html", "title": "Widgets", "blurb": "widget stuff"},
        },
        {
            "nav": {
                "label": "Governance",
                "href": "governance.html",
                "dot": "#10b981",
                "section": "Architecture Deep-Dive",
            },
            "index": {
                "href": "governance.html",
                "title": "Governance",
                "blurb": "governance stuff",
            },
        },
    ]
    nav_out, index_out, wired = wire_new_pages(
        _NAV, '<html>Crate Index<div class="g3"></div></html>', metas
    )
    assert nav_out is not None
    assert wired == {"widgets.html", "governance.html"}
    assert "widgets.html" in nav_out
    assert "governance.html" in nav_out
    # Both entries land after the section marker (only one exists in _NAV).
    assert nav_out.index("Architecture Deep-Dive") < nav_out.index("widgets.html")
    assert nav_out.index("Architecture Deep-Dive") < nav_out.index("governance.html")


def test_wire_new_pages_returns_none_when_no_section_anchor():
    """If the section anchor is missing, nav returns None (nothing wired)."""
    metas = [
        {
            "nav": {
                "label": "X",
                "href": "x.html",
                "dot": "#fff",
                "section": "Nonexistent",
            }
        }
    ]
    nav_out, index_out, wired = wire_new_pages(_NAV, "<html></html>", metas)
    assert nav_out is None
    assert index_out is None
    assert wired == set()


def test_insert_nav_entry_escapes_quotes_in_label():
    """A label with an apostrophe must not break the NAV array's JavaScript."""
    out = insert_nav_entry(_NAV, "Owner's Role", "owner.html", "#10b981", "Architecture Deep-Dive")
    assert out is not None
    assert "\\'" in out  # apostrophe escaped for the single-quoted JS string
    assert "owner.html" in out
    assert out.count("];") == 1

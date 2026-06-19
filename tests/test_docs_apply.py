from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.apply import (
    apply_add_section,
    apply_update_section,
    insert_section,
    next_card_class,
)
from ai_reviewer.docs.models import Change, DocAction

_PAGE = (
    '<body>\n<div class="main">\n<div class="content">\n'
    '<div class="card ga"><h2>One</h2></div>\n'
    '<div class="card gb"><h2>Two</h2></div>\n'
    "</div>\n</div>\n"
    '<script src="nav.js"></script>\n</body>'
)


def test_next_card_class_cycles():
    assert next_card_class('<div class="card gb">') == "gc"
    assert next_card_class('<div class="card gd">') == "ga"
    assert next_card_class("<div>no cards</div>") == "ga"


def test_insert_section_places_before_content_close():
    out = insert_section(_PAGE, '<div class="card gc"><h2>Three</h2></div>')
    assert out is not None
    assert "Three" in out
    # Inserted inside .content, before its close and the nav script.
    assert out.index("Three") < out.index('<script src="nav.js">')
    assert out.index("Two") < out.index("Three")


def test_insert_section_anchor_missing_returns_none():
    assert insert_section("<html>no wrapper</html>", "<div>x</div>") is None


@pytest.mark.asyncio
async def test_apply_update_section_uses_find_replace():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("rename", "t", "w", "y", ["build_x"], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    current = "<p>old emit_x text</p>"
    patch_resp = (
        "<<<FIND\n<p>old emit_x text</p>\nFIND>>>\n<<<REPLACE\n<p>new build_x text</p>\nREPLACE>>>"
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=patch_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, current, change, cfg, "m")
    assert draft.error is None
    assert draft.updated_content == "<p>new build_x text</p>"
    assert draft.before_content == current


@pytest.mark.asyncio
async def test_apply_update_section_bad_patch_flags_when_no_fallback():
    """A patch that never applies errors out when the add_section fallback is disabled.
    Two model calls: the initial attempt + one verbatim-copy retry, both unmatched."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    bad = "<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>"
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=bad)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(
            action, "<p>actual</p>", change, cfg, "m", allow_new_sections=False
        )
    assert draft.updated_content == ""
    assert draft.error is not None
    assert inst.run_completion.call_count == 2  # initial + one repair retry


@pytest.mark.asyncio
async def test_apply_update_section_retries_then_succeeds():
    """A first patch whose FIND doesn't match is retried; the corrected retry applies."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    bad = "<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>"
    good = "<<<FIND\n<p>actual</p>\nFIND>>>\n<<<REPLACE\n<p>fixed</p>\nREPLACE>>>"
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=[bad, good])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, "<p>actual</p>", change, cfg, "m")
    assert draft.error is None
    assert draft.action == "update_section"
    assert draft.updated_content == "<p>fixed</p>"
    assert inst.run_completion.call_count == 2


@pytest.mark.asyncio
async def test_apply_update_section_falls_back_to_add_section():
    """When neither attempt applies, fall back to an additive section (still ships)."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    bad = "<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>"
    section = '<div class="card gc"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        # initial bad, retry bad, then the add_section fallback's section block.
        inst.run_completion = AsyncMock(side_effect=[bad, bad, section])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, _PAGE, change, cfg, "m")
    assert draft.error is None
    assert draft.action == "add_section"  # downgraded from update_section
    assert "Widgets" in draft.updated_content
    assert inst.run_completion.call_count == 3


@pytest.mark.asyncio
async def test_apply_update_section_retry_no_update_is_noop_not_fallback():
    """If the repair retry returns NO_UPDATE_NEEDED, treat it as a no-op — do NOT
    force the add_section fallback (the model said nothing applies)."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    bad = "<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>"
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=[bad, "NO_UPDATE_NEEDED"])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, _PAGE, change, cfg, "m")
    assert draft.error is None
    assert draft.action == "update_section"  # NOT downgraded to add_section
    assert draft.updated_content == ""  # no-op
    assert inst.run_completion.call_count == 2  # initial + retry, no fallback call


@pytest.mark.asyncio
async def test_apply_update_section_fallback_errors_when_no_anchor():
    """Both attempts miss AND the page has no content anchor → add_section fallback
    also fails, so the error draft propagates (not a silent empty success)."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    bad = "<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>"
    section = '<div class="card gc"><h2>X</h2></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=[bad, bad, section])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        # "<p>no anchor</p>" lacks the .content-wrapper anchor insert_section needs.
        draft = await apply_update_section(action, "<p>no anchor</p>", change, cfg, "m")
    assert draft.updated_content == ""
    assert draft.error is not None
    assert inst.run_completion.call_count == 3  # initial + retry + fallback attempt


@pytest.mark.asyncio
async def test_apply_add_section_inserts_card():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "document widgets")
    action = DocAction(change=change, action="add_section", target_path="architecture/x.html")
    section = '<div class="card gc"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=section)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_add_section(action, _PAGE, change, cfg, "m")
    assert draft.error is None
    assert "Widgets" in draft.updated_content
    assert draft.updated_content.index("Widgets") < draft.updated_content.index(
        '<script src="nav.js">'
    )
    assert draft.before_content == _PAGE


@pytest.mark.asyncio
async def test_apply_update_section_no_update_needed_is_not_an_error():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value="NO_UPDATE_NEEDED")
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        draft = await apply_update_section(action, "<p>already fine</p>", change, cfg, "m")
    assert draft.error is None  # NOT a failure
    assert draft.updated_content == ""  # nothing to ship -> orchestrator drops it

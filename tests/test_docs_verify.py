from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, DocDraft
from ai_reviewer.docs.verify import verify_draft


def _draft() -> DocDraft:
    c = Change("behavior_change", "t", "events now flushed after persist", "y", [], [], "i")
    return DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        updated_content="<p>events flushed after persist</p>",
        before_content="<p>old</p>",
        change=c,
    )


@pytest.mark.asyncio
async def test_passing_verdict_keeps_draft():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps({"reflects_change": True, "confidence": "high", "notes": "ok"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.flagged_reason is None
    assert out.updated_content


@pytest.mark.asyncio
async def test_failing_verdict_flags_and_clears_content():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps(
        {
            "reflects_change": False,
            "confidence": "low",
            "notes": "only renamed; missed the invariant",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.updated_content == ""
    assert out.flagged_reason is not None
    assert "missed the invariant" in out.flagged_reason


@pytest.mark.asyncio
async def test_low_confidence_below_threshold_flags():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps({"reflects_change": True, "confidence": "low", "notes": "unsure"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.flagged_reason is not None


@pytest.mark.asyncio
async def test_errored_draft_passes_through_untouched():
    cfg = AnthropicApiConfig(api_key="sk-test")
    c = Change("fix", "t", "w", "y", [], [], "i")
    errored = DocDraft(
        action="update_section",
        target_path="x.html",
        updated_content="",
        change=c,
        error="bad patch",
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=errored, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out is errored
    inst.run_completion.assert_not_called()


@pytest.mark.asyncio
async def test_high_confidence_rejection_not_labeled_low_confidence():
    """reflects_change=False at HIGH confidence is flagged as 'does not reflect', not low-confidence."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    c = Change("behavior_change", "t", "w", "y", [], [], "i")
    draft = DocDraft(
        action="update_section",
        target_path="x.html",
        updated_content="<p>x</p>",
        before_content="<p>old</p>",
        change=c,
    )
    verdict = json.dumps(
        {"reflects_change": False, "confidence": "high", "notes": "missed the invariant"}
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        out = await verify_draft(draft=draft, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.flagged_reason is not None
    assert "does not reflect" in out.flagged_reason
    assert "low-confidence" not in out.flagged_reason


@pytest.mark.asyncio
async def test_string_false_reflects_change_is_flagged():
    """A JSON string 'false' for reflects_change must NOT pass the gate (bool('false') is truthy)."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    c = Change("fix", "t", "w", "y", [], [], "i")
    draft = DocDraft(
        action="update_section",
        target_path="x.html",
        updated_content="<p>x</p>",
        before_content="<p>old</p>",
        change=c,
    )
    verdict = json.dumps({"reflects_change": "false", "confidence": "high", "notes": "no"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        out = await verify_draft(draft=draft, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.updated_content == ""  # flagged, not shipped
    assert out.flagged_reason is not None

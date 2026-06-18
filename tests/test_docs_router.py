from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, ChangeSummary
from ai_reviewer.docs.router import build_doc_index, route_changes


def _summary(kind: str, files: list[str]) -> ChangeSummary:
    return ChangeSummary(
        pr_intent="x",
        changes=[Change(kind, "t", "w", "y", [], files, "impact")],
    )


def test_build_doc_index_filters_to_doc_dirs():
    idx = build_doc_index(["architecture/auto-follow.html", "src/lib.rs", "README.md"])
    assert "architecture/auto-follow.html" in idx
    assert "src/lib.rs" not in idx


@pytest.mark.asyncio
async def test_mapping_hit_routes_update_section_no_model_call():
    cfg = AnthropicApiConfig(api_key="sk-test")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("fix", ["crates/governance-store/src/lib.rs"]),
            source_to_docs_mapping={
                "crates/governance-store/**": ["architecture/auto-follow.html"]
            },
            changed_paths=["crates/governance-store/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert len(actions) == 1
    assert actions[0].action == "update_section"
    assert actions[0].target_path == "architecture/auto-follow.html"
    inst.run_completion.assert_not_called()


@pytest.mark.asyncio
async def test_new_feature_routes_create_page_when_allowed():
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps(
        {
            "action": "create_page",
            "target_path": "architecture/widgets.html",
            "anchor": None,
            "best_fit_reason": "no existing widget page",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("new_feature", ["crates/widgets/src/lib.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/widgets/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions[0].action == "create_page"
    assert actions[0].target_path == "architecture/widgets.html"


@pytest.mark.asyncio
async def test_create_page_downgrades_to_add_section_when_pages_disabled():
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps(
        {
            "action": "create_page",
            "target_path": "architecture/widgets.html",
            "anchor": None,
            "best_fit_reason": "x",
            "best_fit_existing": "architecture/auto-follow.html",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("new_feature", ["crates/widgets/src/lib.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/widgets/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=False,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions[0].action == "add_section"
    assert actions[0].target_path == "architecture/auto-follow.html"

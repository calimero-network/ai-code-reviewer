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
    MockClient.assert_not_called()  # client never even constructed on the mapping-hit path


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


@pytest.mark.asyncio
async def test_multiple_changes_same_page_coalesce_to_one_action():
    from ai_reviewer.docs.models import ChangeSummary

    cfg = AnthropicApiConfig(api_key="sk-test")
    summary = ChangeSummary(
        pr_intent="x",
        changes=[
            Change("fix", "A", "change A", "y", [], ["crates/gov/src/a.rs"], "doc A"),
            Change("fix", "B", "change B", "y", [], ["crates/gov/src/b.rs"], "doc B"),
        ],
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        actions = await route_changes(
            summary=summary,
            source_to_docs_mapping={"crates/gov/**": ["architecture/auto-follow.html"]},
            changed_paths=["crates/gov/src/a.rs", "crates/gov/src/b.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert len(actions) == 1
    assert actions[0].target_path == "architecture/auto-follow.html"
    assert "change A" in actions[0].change.what_changed
    assert "change B" in actions[0].change.what_changed
    inst.run_completion.assert_not_called()


@pytest.mark.asyncio
async def test_add_section_downgrades_to_update_section_when_sections_disabled():
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps(
        {
            "action": "add_section",
            "target_path": "architecture/auto-follow.html",
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
            summary=_summary("fix", ["crates/x/src/lib.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/x/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=False,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions[0].action == "update_section"
    assert actions[0].target_path == "architecture/auto-follow.html"


def test_mapping_targets_prefer_change_files():
    """A change routes by its OWN files; unrelated PR files don't pull it to a mapped doc."""
    from ai_reviewer.docs.models import Change
    from ai_reviewer.docs.router import _mapping_targets

    mapping = {"crates/gov/**": ["architecture/gov.html"]}
    changed = ["crates/gov/a.rs", "crates/widgets/b.rs"]
    gov = Change("fix", "t", "w", "y", [], ["crates/gov/a.rs"], "i")
    widget = Change("fix", "t", "w", "y", [], ["crates/widgets/b.rs"], "i")
    nofiles = Change("fix", "t", "w", "y", [], [], "i")
    assert _mapping_targets(gov, mapping, changed) == ["architecture/gov.html"]
    # widget's own files don't match the gov glob -> no targets
    assert _mapping_targets(widget, mapping, changed) == []
    # empty files -> fall back to PR-level paths (gov file present -> matches)
    assert _mapping_targets(nofiles, mapping, changed) == ["architecture/gov.html"]
    # Multi-target + Markdown: BOTH targets returned, any extension, not gated on the HTML index.
    md_map = {"src/**": ["docs/api.md", "README.md"]}
    md_change = Change("fix", "t", "w", "y", [], ["src/x.py"], "i")
    assert _mapping_targets(md_change, md_map, ["src/x.py"]) == ["docs/api.md", "README.md"]


@pytest.mark.asyncio
async def test_duplicate_create_page_targets_coalesce():
    """Two changes routed to the same NEW page path collapse to one create_page action."""
    from ai_reviewer.docs.models import ChangeSummary

    cfg = AnthropicApiConfig(api_key="sk-test")
    summary = ChangeSummary(
        pr_intent="x",
        changes=[
            Change("new_feature", "A", "wa", "y", [], ["crates/w/a.rs"], "i"),
            Change("new_feature", "B", "wb", "y", [], ["crates/w/b.rs"], "i"),
        ],
    )
    decision = json.dumps(
        {
            "action": "create_page",
            "target_path": "architecture/widgets.html",
            "anchor": None,
            "best_fit_reason": "x",
            "best_fit_existing": "",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        actions = await route_changes(
            summary=summary,
            source_to_docs_mapping={},
            changed_paths=["crates/w/a.rs", "crates/w/b.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    create = [a for a in actions if a.target_path == "architecture/widgets.html"]
    assert len(create) == 1  # deduped, not two
    assert "wa" in create[0].change.what_changed and "wb" in create[0].change.what_changed


@pytest.mark.asyncio
async def test_empty_target_path_dropped():
    """A routed action with an empty target_path (e.g. create_page) is dropped, not appended."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps(
        {
            "action": "create_page",
            "target_path": "",
            "anchor": None,
            "best_fit_reason": "x",
            "best_fit_existing": "",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        actions = await route_changes(
            summary=_summary("new_feature", ["crates/w/a.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/w/a.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions == []

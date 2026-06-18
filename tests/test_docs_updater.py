from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig, DocGenerationSettings
from ai_reviewer.docs.models import Change, ChangeSummary, DocAction, DocDraft
from ai_reviewer.docs.updater import run_doc_update


def _gh_for(diff: str, html_files: list[str]):
    gh = MagicMock()
    pr = MagicMock()
    pr.base.ref = "master"
    pr.merge_commit_sha = "deadbeefcafe1234"
    pr.head.sha = "headsha"
    pr.title = "fix: x"
    pr.body = "body"
    pr.html_url = "https://example/pr/1"
    pr.user.login = "alice"
    pr.get_files.return_value = [MagicMock(filename="crates/gov/src/lib.rs", status="modified")]
    pr.get_commits.return_value = [MagicMock(commit=MagicMock(message="m"))]
    gh.get_pull_request.return_value = pr
    gh.has_open_doc_update_pr.return_value = False
    gh.load_repo_config.return_value = {
        "doc_generation": {"enabled": True},
        "documentation": {"static_docs_dirs": ["architecture/"]},
    }
    gh.get_pr_diff.return_value = diff
    gh.get_html_files_in_dirs.return_value = html_files
    gh.create_doc_update_pr.return_value = "https://example/pr/2"
    return gh, pr


@pytest.mark.asyncio
async def test_successful_update_opens_pr():
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)

    change = Change("behavior_change", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    good_draft = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<html>updated</html>",
        change=change,
    )

    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=good_draft)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=good_draft)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )

    assert result.pr_url == "https://example/pr/2"
    assert len(result.successful) == 1
    assert gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_all_flagged_posts_comment_no_pr():
    gh, pr = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)

    change = Change("behavior_change", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    flagged = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="",
        change=change,
        flagged_reason="low-confidence (low): missed the invariant",
    )

    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=flagged)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=flagged)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )

    assert result.pr_url is None
    assert len(result.flagged) == 1
    assert not gh.create_doc_update_pr.called
    assert gh.post_or_update_doc_comment.called


@pytest.mark.asyncio
async def test_no_changes_skips():
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    empty = ChangeSummary(pr_intent="nothing doc-relevant", changes=[])
    with patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=empty)):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert result.skipped
    assert not gh.create_doc_update_pr.called

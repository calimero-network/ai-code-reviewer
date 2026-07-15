from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig, DocGenerationSettings
from ai_reviewer.docs.models import Change, ChangeSummary, DocAction, DocDraft
from ai_reviewer.docs.updater import (
    _build_pr_body,
    _plain_lines,
    _rendered_change,
    run_doc_update,
)


def test_rendered_change_is_github_style_diff():
    """update_section preview is a ```diff block: added prose on + lines, with the
    surrounding unchanged line kept as context."""
    draft = DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        before_content="<p>Old behavior: drop the delta.</p>",
        updated_content="<p>Old behavior: drop the delta.</p><p>New: re-check the projection.</p>",
        change=Change("fix", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert out.startswith("```diff")
    assert out.rstrip().endswith("```")
    assert "+New: re-check the projection." in out  # added line, + prefix
    assert "Old behavior: drop the delta." in out  # unchanged context shown


def test_rendered_change_diff_shows_removed_lines():
    """A deletion is visible as a - line (the old added-only preview hid removals)."""
    draft = DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        before_content="<p>Keep this.</p><p>Delete this stale note.</p>",
        updated_content="<p>Keep this.</p>",
        change=Change("removal", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert "-Delete this stale note." in out


def test_rendered_change_diff_wraps_long_lines():
    """Diff blocks don't wrap on GitHub, so long paragraphs are pre-wrapped."""
    long = "word " * 60  # ~300 chars on one logical line
    draft = DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        before_content="<p>x</p>",
        updated_content=f"<p>x</p><p>{long.strip()}</p>",
        change=Change("fix", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert all(len(ln) <= 84 for ln in out.splitlines())  # wrapped to ~80 + prefix


def test_rendered_change_diff_keeps_code_lines():
    """A <div class="typedef">/<pre> code block shows as + lines with line breaks
    and angle brackets preserved (Option<bool> not stripped as a tag)."""
    draft = DocDraft(
        action="update_section",
        target_path="architecture/sync.html",
        before_content="<p>old</p>",
        updated_content=(
            "<p>old</p>"
            '<div class="typedef">'
            '<span class="kw">pub fn</span> foo(<br>'
            "&nbsp;&nbsp;&nbsp;&nbsp;ctx: &amp;Ctx,<br>"
            ") -&gt; Option&lt;bool&gt;"
            "</div>"
        ),
        change=Change("fix", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert "+pub fn foo(" in out  # syntax spans flattened to text on its own + line
    assert "Option<bool>" in out  # angle brackets preserved, not stripped as a tag


def test_rendered_change_create_page_is_all_additions():
    """A brand-new page shows entirely as + lines."""
    draft = DocDraft(
        action="create_page",
        target_path="architecture/widgets.html",
        updated_content="<h1>Widgets</h1><p>intro</p><h2>Overview</h2>",
        change=Change("new_feature", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert out.startswith("```diff")
    assert "+Widgets" in out
    assert "+intro" in out


def test_rendered_change_no_change_has_fallback():
    """Identical before/after yields a fallback line, never an empty diff block."""
    draft = DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        before_content="<p>same</p>",
        updated_content="<p>same</p>",
        change=Change("fix", "t", "w", "y", [], [], "i"),
    )
    out = _rendered_change(draft)
    assert "```diff" not in out
    assert out.strip()  # non-empty fallback


def test_plain_lines_collapses_source_formatting_newlines():
    """A paragraph hard-wrapped across source lines is one logical line, so it
    re-wraps cleanly instead of fragmenting."""
    lines = _plain_lines("<p>This sentence is\nsplit across\nsource lines.</p>")
    assert lines == ["This sentence is split across source lines."]


def test_build_pr_body_diff_and_collapses_rationale():
    d = DocDraft(
        action="update_section",
        target_path="architecture/x.html",
        before_content="<p>old</p>",
        updated_content="<p>old</p><p>added doc sentence</p>",
        change=Change("fix", "Grant on projection", "LONG SOURCE RATIONALE", "y", [], [], "i"),
    )
    body = _build_pr_body(1, "https://example/pr/1", [d], [])
    assert "#### `architecture/x.html` — Grant on projection" in body
    assert "```diff" in body  # rendered as a diff block
    assert "+added doc sentence" in body  # added doc text on a + line
    assert "<details><summary>Why this changed (source: PR #1)" in body
    assert "LONG SOURCE RATIONALE" in body  # rationale present but collapsed


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


@pytest.mark.asyncio
async def test_open_doc_pr_dedupe_guard_skips():
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    gh.has_open_doc_update_pr.return_value = True
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    with patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock()) as mock_sum:
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert result.skipped
    assert "open doc-update PR" in (result.skip_reason or "")
    mock_sum.assert_not_called()  # guard fires before the Understand stage
    assert not gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_two_create_page_drafts_produce_one_nav_js():
    """Two create_page drafts must result in exactly ONE nav.js FileWrite containing both hrefs."""
    _NAV = "  const NAV = [\n    { section: 'Architecture Deep-Dive' },\n  ];\n"
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])

    # Override get_file_contents so _read_file returns the baseline nav.js/index.html.
    def _file_contents(_repo, path, _ref):
        m = MagicMock()
        if path.endswith("nav.js"):
            m.decoded_content = _NAV.encode()
        else:
            m.decoded_content = b'<html>Crate Index<div class="g3"></div></html>'
        return m

    gh.get_file_contents.side_effect = _file_contents

    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)

    change_a = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    change_b = Change("new_feature", "Governance", "adds governance", "y", [], [], "doc governance")
    summary = ChangeSummary(pr_intent="i", changes=[change_a, change_b])
    action_a = DocAction(
        change=change_a, action="create_page", target_path="architecture/widgets.html"
    )
    action_b = DocAction(
        change=change_b, action="create_page", target_path="architecture/governance.html"
    )

    draft_a = DocDraft(
        action="create_page",
        target_path="architecture/widgets.html",
        updated_content="<html>widgets</html>",
        change=change_a,
        aux_edits=[],
        aux_meta={
            "nav": {
                "label": "Widgets",
                "href": "widgets.html",
                "dot": "#10b981",
                "section": "Architecture Deep-Dive",
            },
            "index": {"href": "widgets.html", "title": "Widgets", "blurb": "adds widgets"},
        },
    )
    draft_b = DocDraft(
        action="create_page",
        target_path="architecture/governance.html",
        updated_content="<html>governance</html>",
        change=change_b,
        aux_edits=[],
        aux_meta={
            "nav": {
                "label": "Governance",
                "href": "governance.html",
                "dot": "#10b981",
                "section": "Architecture Deep-Dive",
            },
            "index": {"href": "governance.html", "title": "Governance", "blurb": "adds governance"},
        },
    )

    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch(
            "ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action_a, action_b])
        ),
        patch("ai_reviewer.docs.updater._apply_one", AsyncMock(side_effect=[draft_a, draft_b])),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=[draft_a, draft_b])),
    ):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )

    assert result.pr_url == "https://example/pr/2"
    call_kwargs = gh.create_doc_update_pr.call_args
    file_writes = call_kwargs[1]["file_writes"] if call_kwargs[1] else call_kwargs[0][3]
    nav_writes = [fw for fw in file_writes if fw.path.endswith("nav.js")]
    # Exactly ONE nav.js write (not two clobbering each other).
    assert len(nav_writes) == 1
    assert "widgets.html" in nav_writes[0].content
    assert "governance.html" in nav_writes[0].content
    # Both page files are also present.
    page_paths = {fw.path for fw in file_writes}
    assert "architecture/widgets.html" in page_paths
    assert "architecture/governance.html" in page_paths


@pytest.mark.asyncio
async def test_repo_doc_generation_overrides_honored():
    """Per-repo .ai-reviewer.yaml doc_generation overrides (model/labels/draft) win over server defaults."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    gh.load_repo_config.return_value = {
        "doc_generation": {
            "enabled": True,
            "model": "repo-apply",
            "pr_labels": ["L"],
            "pr_draft": False,
        },
        "documentation": {"static_docs_dirs": ["architecture/"]},
    }
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("fix", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    good = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<html>updated</html>",
        change=change,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch(
            "ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=good)
        ) as mock_apply,
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=good)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg)
    # repo `model` override flows to the apply stage (5th positional arg of apply_update_section)
    assert mock_apply.await_args.args[4] == "repo-apply"
    _, kwargs = gh.create_doc_update_pr.call_args
    assert kwargs["labels"] == ["L"]
    assert kwargs["draft"] is False


@pytest.mark.asyncio
async def test_skip_reason_when_failures_without_flags():
    """All drafts failed (no flags) -> accurate skip_reason, no doc comment posted."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("fix", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    errored = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="",
        change=change,
        error="patch failed",
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=errored)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=errored)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert result.skipped
    assert "no doc updates produced" in (result.skip_reason or "")
    assert not gh.post_or_update_doc_comment.called
    assert not gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_max_files_caps_number_of_actions():
    """max_files bounds how many doc targets are processed in one run."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html", "architecture/concepts.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True, max_files=1)
    c1 = Change("fix", "A", "wa", "y", [], ["crates/gov/a.rs"], "i")
    c2 = Change("fix", "B", "wb", "y", [], ["crates/gov/b.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[c1, c2])
    a1 = DocAction(change=c1, action="update_section", target_path="architecture/auto-follow.html")
    a2 = DocAction(change=c2, action="update_section", target_path="architecture/concepts.html")
    d1 = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<html>1</html>",
        change=c1,
    )
    d2 = DocDraft(
        action="update_section",
        target_path="architecture/concepts.html",
        updated_content="<html>2</html>",
        change=c2,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[a1, a2])),
        patch("ai_reviewer.docs.updater._apply_one", AsyncMock(side_effect=[d1, d2])) as mock_apply,
        patch(
            "ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=lambda **kw: kw["draft"])
        ),
    ):
        await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg)
    assert mock_apply.await_count == 1  # capped at max_files=1


@pytest.mark.asyncio
async def test_create_page_does_not_clone_itself():
    """The sibling template for a new page must never be the page being created."""
    from ai_reviewer.docs.updater import _apply_one

    gh = MagicMock()
    gh.get_html_files_in_dirs.return_value = [
        "architecture/widgets.html",  # the target itself
        "architecture/auto-follow.html",  # a real sibling
    ]

    def _fc(_repo, path, _ref):
        m = MagicMock()
        m.decoded_content = f"<html>{path}</html>".encode()
        return m

    gh.get_file_contents.side_effect = _fc
    change = Change("new_feature", "Widgets", "w", "y", [], [], "i")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    cfg = AnthropicApiConfig(api_key="sk-test")
    captured = {}

    async def _fake_create(**kw):
        captured.update(kw)
        return DocDraft(
            action="create_page", target_path=action.target_path, updated_content="x", change=change
        )

    with patch("ai_reviewer.docs.updater.apply_create_page", _fake_create):
        await _apply_one(action, gh, "o/r", "ref", "architecture/", cfg, "m", True)
    assert "widgets.html" not in captured["sibling_html"]
    assert "auto-follow.html" in captured["sibling_html"]


@pytest.mark.asyncio
async def test_one_action_exception_does_not_abort_batch():
    """A single action raising (e.g. a create_page model error) must not abort the run."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html", "architecture/concepts.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    c1 = Change("new_feature", "A", "wa", "y", [], ["crates/gov/a.rs"], "i")
    c2 = Change("fix", "B", "wb", "y", [], ["crates/gov/b.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[c1, c2])
    a1 = DocAction(change=c1, action="create_page", target_path="architecture/widgets.html")
    a2 = DocAction(change=c2, action="update_section", target_path="architecture/concepts.html")
    good = DocDraft(
        action="update_section",
        target_path="architecture/concepts.html",
        updated_content="<html>ok</html>",
        change=c2,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[a1, a2])),
        patch(
            "ai_reviewer.docs.updater._apply_one",
            AsyncMock(side_effect=[RuntimeError("model boom"), good]),
        ),
        patch(
            "ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=lambda **kw: kw["draft"])
        ),
    ):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    # The run completed despite the exception; the failure is isolated to its target.
    assert any(d.target_path == "architecture/widgets.html" and d.error for d in result.failed)
    assert len(result.successful) == 1
    assert result.successful[0].target_path == "architecture/concepts.html"
    assert gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_create_page_not_committed_when_nav_unwired():
    """A new page whose nav entry can't wire is dropped (no orphan page), not shipped."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])

    def _fc(_repo, path, _ref):
        m = MagicMock()
        m.decoded_content = b"const NAV = [];" if path.endswith("nav.js") else b"<html></html>"
        return m

    gh.get_file_contents.side_effect = _fc
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("new_feature", "Widgets", "w", "y", [], [], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    draft = DocDraft(
        action="create_page",
        target_path="architecture/widgets.html",
        updated_content="<html>widgets</html>",
        change=change,
        aux_edits=[],
        aux_meta={
            "nav": {
                "label": "Widgets",
                "href": "widgets.html",
                "dot": "#10b981",
                "section": "Nonexistent Section",
            },
            "index": {"href": "widgets.html", "title": "Widgets", "blurb": "w"},
        },
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater._apply_one", AsyncMock(return_value=draft)),
        patch(
            "ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=lambda **kw: kw["draft"])
        ),
    ):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert result.skipped
    assert not gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_repo_model_override_also_drives_verify():
    """Legacy repo `model` override flows to the verify stage too (not just apply)."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    gh.load_repo_config.return_value = {
        "doc_generation": {"enabled": True, "model": "repo-model"},
        "documentation": {"static_docs_dirs": ["architecture/"]},
    }
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("fix", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    good = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<html>x</html>",
        change=change,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=good)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=good)) as mock_verify,
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg)
    assert mock_verify.await_args.kwargs["model"] == "repo-model"


@pytest.mark.asyncio
async def test_pr_body_excludes_orphan_skipped_page():
    """PR body lists only committed docs; an orphan-skipped new page is not listed."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])

    def _fc(_repo, path, _ref):
        m = MagicMock()
        m.decoded_content = b"const NAV = [];" if path.endswith("nav.js") else b"<html>old</html>"
        return m

    gh.get_file_contents.side_effect = _fc
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    c1 = Change("fix", "A", "wa", "y", [], ["crates/gov/a.rs"], "i")
    c2 = Change("new_feature", "Widgets", "wb", "y", [], [], "i")
    summary = ChangeSummary(pr_intent="i", changes=[c1, c2])
    a1 = DocAction(change=c1, action="update_section", target_path="architecture/auto-follow.html")
    a2 = DocAction(change=c2, action="create_page", target_path="architecture/widgets.html")
    d1 = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<html>up</html>",
        change=c1,
    )
    d2 = DocDraft(
        action="create_page",
        target_path="architecture/widgets.html",
        updated_content="<html>w</html>",
        change=c2,
        aux_edits=[],
        aux_meta={
            "nav": {
                "label": "W",
                "href": "widgets.html",
                "dot": "#10b981",
                "section": "Nonexistent",
            },
            "index": None,
        },
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[a1, a2])),
        patch("ai_reviewer.docs.updater._apply_one", AsyncMock(side_effect=[d1, d2])),
        patch(
            "ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=lambda **kw: kw["draft"])
        ),
    ):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert gh.create_doc_update_pr.called
    body = gh.create_doc_update_pr.call_args.kwargs["pr_body"]
    assert "auto-follow.html" in body
    assert "widgets.html" not in body  # orphan-skipped -> not listed as updated
    assert all(d.target_path != "architecture/widgets.html" for d in result.successful)


@pytest.mark.asyncio
async def test_marker_leak_content_dropped_clean_content_ships():
    """A draft whose finalized content still contains a FIND/REPLACE marker is
    dropped from the PR; a clean sibling draft still ships."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html", "architecture/concepts.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    c1 = Change("fix", "A", "wa", "y", [], ["crates/gov/a.rs"], "i")
    c2 = Change("fix", "B", "wb", "y", [], ["crates/gov/b.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[c1, c2])
    a1 = DocAction(change=c1, action="update_section", target_path="architecture/auto-follow.html")
    a2 = DocAction(change=c2, action="update_section", target_path="architecture/concepts.html")
    leaked = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<p>oops FIND>>> leaked</p>",
        change=c1,
    )
    clean = DocDraft(
        action="update_section",
        target_path="architecture/concepts.html",
        updated_content="<p>clean update</p>",
        change=c2,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[a1, a2])),
        patch("ai_reviewer.docs.updater._apply_one", AsyncMock(side_effect=[leaked, clean])),
        patch(
            "ai_reviewer.docs.updater.verify_draft", AsyncMock(side_effect=lambda **kw: kw["draft"])
        ),
    ):
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert gh.create_doc_update_pr.called
    written = {fw.path for fw in gh.create_doc_update_pr.call_args.kwargs["file_writes"]}
    assert "architecture/concepts.html" in written  # clean content ships
    assert "architecture/auto-follow.html" not in written  # marker-leaking file dropped
    shipped = {d.target_path for d in result.successful}
    assert "architecture/auto-follow.html" not in shipped


@pytest.mark.asyncio
async def test_marker_leak_only_draft_skips_pr():
    """If the only doc update leaks a marker, no PR is opened."""
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("fix", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    leaked = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<p>uses <<<FIND ... FIND>>> markers</p>",
        change=change,
    )
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=leaked)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=leaked)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(
            repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg
        )
    assert result.skipped
    assert not gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_empty_static_docs_dirs_override_honored():
    """An explicit static_docs_dirs: [] disables the HTML scan (not overridden by defaults)."""
    gh, _ = _gh_for("diff", [])
    gh.load_repo_config.return_value = {
        "doc_generation": {"enabled": True},
        "documentation": {"static_docs_dirs": []},
    }
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    change = Change("fix", "t", "w", "y", [], ["crates/gov/a.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[])),
    ):
        await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg, doc_generation=dg)
    # The HTML scan used the empty override, not the server defaults.
    assert any(call.args[2] == [] for call in gh.get_html_files_in_dirs.call_args_list)

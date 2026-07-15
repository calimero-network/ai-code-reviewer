"""Doc-update orchestration: Understand -> Route -> Apply -> Verify -> open PR (or flag)."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import html
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_reviewer.docs.apply import apply_add_section, apply_update_section
from ai_reviewer.docs.models import DocDraft, FileWrite, contains_patch_marker
from ai_reviewer.docs.page_builder import apply_create_page, wire_new_pages
from ai_reviewer.docs.router import build_doc_index, route_changes
from ai_reviewer.docs.understanding import summarize_pr_changes
from ai_reviewer.docs.verify import verify_draft

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig, DocGenerationSettings
    from ai_reviewer.github.client import GitHubClient

logger = logging.getLogger(__name__)

_DOC_COMMENT_MARKER = "<!-- AI-CODE-REVIEWER-DOC-BOT -->"
_DEFAULT_SECTION_GROUP = "Architecture Deep-Dive"
_DEFAULT_DOT = "#10b981"


@dataclass
class DocUpdateResult:
    """Outcome of a doc-update run."""

    successful: list[DocDraft] = field(default_factory=list)
    failed: list[DocDraft] = field(default_factory=list)
    flagged: list[DocDraft] = field(default_factory=list)
    pr_url: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _read_file(gh: GitHubClient, repo: str, path: str, ref: str) -> str | None:
    try:
        raw = gh.get_file_contents(repo, path, ref)
        if isinstance(raw, list):
            return None
        return raw.decoded_content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s@%s: %s", path, ref, exc)
        return None


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_WS_RE = re.compile(r"[ \t]+")
_PRE_RE = re.compile(r"(?is)<pre[^>]*>(.*?)</pre>")
_BR_RE = re.compile(r"(?is)<br\s*/?>")
# Block-level elements (plus headings, list items, pre) are logical line breaks.
_BLOCK_RE = re.compile(r"(?is)</?(?:p|div|ul|ol|section|tr|li|h[1-6]|pre)[^>]*>|<br\s*/?>")
_DIFF_WRAP = 80  # GitHub diff blocks don't wrap; pre-wrap long prose to this width
_DIFF_MAX_LINES = 40  # cap the preview; full content is always in the file diff


def _plain_lines(fragment: str) -> list[str]:
    """Flatten an HTML fragment to plain-text logical lines — one per paragraph,
    list item, heading, or code line. Source-formatting newlines are collapsed so a
    hard-wrapped paragraph stays a single line (and re-wraps cleanly); <br> and block
    elements are the real breaks. <pre> line breaks are preserved, and angle brackets
    (e.g. Option<bool>) survive because entities are decoded after tags are stripped."""
    s = _SCRIPT_STYLE_RE.sub(" ", fragment)
    s = _PRE_RE.sub(lambda m: "<pre>" + m.group(1).replace("\n", "<br>") + "</pre>", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _BR_RE.sub("\n", s)
    s = _BLOCK_RE.sub("\n", s)
    return [
        n
        for x in s.splitlines()
        if (n := _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", x)).replace("\xa0", " ")).strip())
    ]


def _wrap(prefix: str, line: str) -> list[str]:
    """Word-wrap a logical line for a non-wrapping diff block, keeping its prefix."""
    return [f"{prefix}{piece}" for piece in (textwrap.wrap(line, _DIFF_WRAP) or [""])]


def _rendered_change(draft: DocDraft) -> str:
    """Reviewer-facing preview of the documentation change as a GitHub-style diff
    (a ```diff block): added lines render green, removed red, with a little context.
    A brand-new page (no before-content) shows entirely as additions."""
    before = _plain_lines(draft.before_content or "")
    after = _plain_lines(draft.updated_content)
    out: list[str] = []
    for ln in difflib.unified_diff(before, after, n=2, lineterm=""):
        if ln.startswith(("---", "+++")):
            continue
        if ln.startswith("@@"):
            if out:
                out.append("")  # blank separator between hunks
            continue
        text = ln[1:].strip()
        if ln[:1] in ("+", "-"):
            out += _wrap(ln[0], text)  # changed lines shown in full (wrapped)
        else:  # context: trim to one row so long prose paragraphs don't dominate
            out.append(" " + (text if len(text) <= _DIFF_WRAP else text[: _DIFF_WRAP - 1] + "…"))
    while out and not out[-1].strip():
        out.pop()
    if not out:
        return "_(no rendered-text change — see file diff)_"
    if len(out) > _DIFF_MAX_LINES:
        out = out[:_DIFF_MAX_LINES] + ["…"]
    return "```diff\n" + "\n".join(out) + "\n```"


def _build_pr_body(
    pr_number: int, pr_html_url: str, successful: list[DocDraft], flagged: list[DocDraft]
) -> str:
    blocks = []
    for d in successful:
        title = (getattr(d.change, "title", "") or d.action).strip()
        block = f"#### `{d.target_path}` — {title}\n\n{_rendered_change(d)}\n"
        rationale = (getattr(d.change, "what_changed", "") or "").strip()
        if rationale:
            block += (
                f"\n<details><summary>Why this changed (source: PR #{pr_number})"
                f"</summary>\n\n{rationale}\n\n</details>\n"
            )
        blocks.append(block)
    blocks_str = "\n".join(blocks)
    body = (
        f"## Automatic Documentation Update\n\n"
        f"Opened automatically after [PR #{pr_number}]({pr_html_url}) merged.\n\n"
        f"Each block shows the documentation change as a diff (added lines in green, "
        f'removed in red); expand "Why this changed" for the source rationale.\n\n'
        f"### Documentation changes\n\n{blocks_str}"
    )
    if flagged:
        flags = "\n".join(f"- `{d.target_path}` — {d.flagged_reason}" for d in flagged)
        body += (
            "\n### Flagged for human review\n\n"
            "These docs look stale but the bot could not confidently update them:\n\n"
            f"{flags}\n"
        )
    body += "\n---\n*Generated by `ai-reviewer update-docs`. Nothing was auto-merged.*"
    return body


def _flagged_comment(pr_number: int, flagged: list[DocDraft]) -> str:
    flags = "\n".join(f"- `{d.target_path}` — {d.flagged_reason}" for d in flagged)
    return (
        f"{_DOC_COMMENT_MARKER}\n\n## Documentation Review\n\n"
        f"PR #{pr_number} likely makes these docs stale, but the bot could not confidently "
        f"update them. Please review:\n\n{flags}\n"
    )


async def _apply_one(
    action, gh, repo, ref, doc_dir, anthropic_cfg, apply_model, allow_new_sections
) -> DocDraft:
    current = _read_file(gh, repo, action.target_path, ref) or ""
    if action.action == "update_section":
        return await apply_update_section(
            action, current, action.change, anthropic_cfg, apply_model, allow_new_sections
        )
    if action.action == "add_section":
        return await apply_add_section(action, current, action.change, anthropic_cfg, apply_model)
    # create_page: read sibling/nav/index from the TARGET's own directory so the
    # read side matches where apply_create_page writes the aux edits.
    target_dir = os.path.dirname(action.target_path)
    prefix = f"{target_dir}/" if target_dir else ""
    scan_dir = prefix or doc_dir
    nav_js = _read_file(gh, repo, f"{prefix}nav.js", ref) or ""
    siblings = [
        p
        for p in gh.get_html_files_in_dirs(repo, ref, [scan_dir])
        if not p.endswith("index.html") and p != action.target_path
    ]
    sibling_html = _read_file(gh, repo, siblings[0], ref) if siblings else ""
    best_fit = siblings[0] if siblings else ""
    best_fit_html = sibling_html or ""
    return await apply_create_page(
        action=action,
        sibling_html=sibling_html or "",
        nav_js=nav_js,
        change=action.change,
        section_group=_DEFAULT_SECTION_GROUP,
        dot=_DEFAULT_DOT,
        anthropic_cfg=anthropic_cfg,
        model=apply_model,
        allow_new_sections=allow_new_sections,
        best_fit_for_downgrade=best_fit,
        best_fit_html=best_fit_html,
    )


async def run_doc_update(
    *,
    repo: str,
    pr_number: int,
    gh: GitHubClient,
    anthropic_cfg: AnthropicApiConfig,
    doc_generation: DocGenerationSettings,
    base: str | None = None,
    dry_run: bool = False,
) -> DocUpdateResult:
    """Detect stale docs, generate AI updates, and optionally open a PR.

    Returns a :class:`DocUpdateResult`.  When *dry_run* is ``True`` the PR
    creation step is skipped — ``result.pr_url`` will be ``None`` and
    ``result.successful`` contains the generated drafts for the caller to
    display.
    """
    pr = gh.get_pull_request(repo, pr_number)
    base_branch = base or pr.base.ref

    if not dry_run and gh.has_open_doc_update_pr(repo, base_branch, pr_number):
        return DocUpdateResult(
            skipped=True,
            skip_reason=f"open doc-update PR already exists for PR #{pr_number}",
        )

    ref = pr.merge_commit_sha or pr.head.sha
    repo_config = gh.load_repo_config(repo, ref) or {}
    repo_docgen = repo_config.get("doc_generation", {})
    if repo_docgen.get("enabled") is False:
        return DocUpdateResult(skipped=True, skip_reason="doc_generation disabled in repo config")
    if not (repo_docgen.get("enabled") or doc_generation.enabled):
        return DocUpdateResult(skipped=True, skip_reason="doc_generation not enabled")

    doc_config = repo_config.get("documentation") or {}
    # Explicit None-checks (not `or`) so an empty-list override (`[]`, "no HTML scan")
    # is honored instead of falling through to defaults.
    static_dirs = repo_docgen.get("static_docs_dirs")
    if static_dirs is None:
        static_dirs = doc_config.get("static_docs_dirs")
    if static_dirs is None:
        static_dirs = doc_generation.static_docs_dirs
    doc_dir = static_dirs[0] if static_dirs else "architecture/"
    mapping = doc_config.get("source_to_docs_mapping", {})

    # Per-repo overrides: prefer repo value then fall back to server default.
    eff_apply_model = (
        repo_docgen.get("model") or repo_docgen.get("apply_model") or doc_generation.apply_model
    )
    eff_understanding_model = (
        repo_docgen.get("understanding_model") or doc_generation.understanding_model
    )
    eff_verify_model = (
        repo_docgen.get("verify_model") or repo_docgen.get("model") or doc_generation.verify_model
    )
    eff_pr_labels = repo_docgen.get("pr_labels", doc_generation.pr_labels)
    eff_pr_draft = repo_docgen.get("pr_draft", doc_generation.pr_draft)
    eff_max_diff = int(
        repo_docgen.get("max_understanding_diff_chars", doc_generation.max_understanding_diff_chars)
    )
    eff_max_files = int(repo_docgen.get("max_files", doc_generation.max_files))
    eff_allow_pages = repo_docgen.get("allow_new_pages", doc_generation.allow_new_pages)
    eff_allow_sections = repo_docgen.get("allow_new_sections", doc_generation.allow_new_sections)
    eff_threshold = repo_docgen.get(
        "verify_confidence_threshold", doc_generation.verify_confidence_threshold
    )

    pr_files = list(pr.get_files())
    changed_paths = [f.filename for f in pr_files]
    commit_messages = [c.commit.message for c in pr.get_commits()]
    diff = gh.get_pr_diff(pr)

    summary = await summarize_pr_changes(
        pr_title=pr.title,
        pr_body=pr.body or "",
        commit_messages=commit_messages,
        diff=diff,
        anthropic_cfg=anthropic_cfg,
        model=eff_understanding_model,
        max_diff_chars=eff_max_diff,
    )
    if not summary.changes:
        return DocUpdateResult(skipped=True, skip_reason="no doc-relevant changes detected")

    html_files = gh.get_html_files_in_dirs(repo, ref, static_dirs)
    doc_index = build_doc_index(html_files)

    actions = await route_changes(
        summary=summary,
        source_to_docs_mapping=mapping,
        changed_paths=changed_paths,
        doc_index=doc_index,
        allow_new_pages=eff_allow_pages,
        allow_new_sections=eff_allow_sections,
        anthropic_cfg=anthropic_cfg,
        model=eff_understanding_model,
    )
    if not actions:
        return DocUpdateResult(skipped=True, skip_reason="no documentation targets routed")

    if eff_max_files >= 0 and len(actions) > eff_max_files:
        logger.warning(
            "Capping doc updates from %d to max_files=%d (rest deferred to a future run)",
            len(actions),
            eff_max_files,
        )
        actions = actions[:eff_max_files]

    async def _pipeline(action) -> DocDraft:
        draft = await _apply_one(
            action, gh, repo, ref, doc_dir, anthropic_cfg, eff_apply_model, eff_allow_sections
        )
        draft = await verify_draft(
            draft=draft,
            anthropic_cfg=anthropic_cfg,
            model=eff_verify_model,
            threshold=eff_threshold,
        )
        return draft

    # return_exceptions isolates one failing action instead of aborting the batch.
    results = await asyncio.gather(*[_pipeline(a) for a in actions], return_exceptions=True)
    drafts: list[DocDraft] = []
    for action, res in zip(actions, results, strict=True):
        if isinstance(res, BaseException):
            logger.warning("Doc action for %s failed: %s", action.target_path, res)
            drafts.append(
                DocDraft(
                    action=action.action,
                    target_path=action.target_path,
                    updated_content="",
                    change=action.change,
                    error=str(res),
                )
            )
        else:
            drafts.append(res)

    successful = [d for d in drafts if d.updated_content and not d.error and not d.flagged_reason]
    flagged = [d for d in drafts if d.flagged_reason]
    failed = [d for d in drafts if d.error and not d.flagged_reason]
    noop = [d for d in drafts if not d.updated_content and not d.error and not d.flagged_reason]
    if noop:
        logger.info("Doc update: %d routed page(s) needed no change", len(noop))

    if dry_run:
        return DocUpdateResult(successful=successful, failed=failed, flagged=flagged)

    if not successful:
        if flagged:
            with contextlib.suppress(Exception):
                gh.post_or_update_doc_comment(
                    pr, _flagged_comment(pr_number, flagged), _DOC_COMMENT_MARKER
                )
            reason = "no confident doc updates; flagged for humans"
        else:
            reason = "no doc updates produced (nothing stale or all no-op)"
        return DocUpdateResult(
            failed=failed,
            flagged=flagged,
            skipped=True,
            skip_reason=reason,
        )

    # create_page drafts are committed only after their nav entry wires (below);
    # everything else commits its content + aux edits directly.
    file_writes: list[FileWrite] = []
    create_by_dir: dict[str, list[DocDraft]] = {}
    for d in successful:
        if d.action == "create_page" and d.aux_meta:
            create_by_dir.setdefault(os.path.dirname(d.target_path), []).append(d)
        else:
            file_writes.append(FileWrite(path=d.target_path, content=d.updated_content))
            file_writes.extend(d.aux_edits)

    # Wire new pages per-directory so multiple new pages share one nav.js/index.html
    # (no clobber); a page ships ONLY if its nav entry wired (orphan guard).
    for target_dir, group in create_by_dir.items():
        prefix = f"{target_dir}/" if target_dir else ""
        baseline_nav = _read_file(gh, repo, f"{prefix}nav.js", ref) or ""
        baseline_index = _read_file(gh, repo, f"{prefix}index.html", ref) or ""
        nav_content, index_content, wired = wire_new_pages(
            baseline_nav, baseline_index, [d.aux_meta for d in group if d.aux_meta]
        )
        for d in group:
            href = (d.aux_meta or {}).get("nav", {}).get("href")
            if href in wired:
                file_writes.append(FileWrite(path=d.target_path, content=d.updated_content))
            else:
                logger.warning(
                    "Skipping new page %s: nav wiring failed (orphan guard)", d.target_path
                )
        if nav_content is not None:
            file_writes.append(FileWrite(path=f"{prefix}nav.js", content=nav_content))
        if index_content is not None:
            file_writes.append(FileWrite(path=f"{prefix}index.html", content=index_content))

    # Marker leak guard: never commit doc content that still carries raw FIND/REPLACE
    # delimiters (a patch that spliced its own markers into the page). Drop those files.
    clean_writes: list[FileWrite] = []
    for fw in file_writes:
        if contains_patch_marker(fw.content):
            logger.warning(
                "Dropping %s from doc-update PR: content still contains a FIND/REPLACE marker",
                fw.path,
            )
            continue
        clean_writes.append(fw)
    file_writes = clean_writes

    if not file_writes:
        return DocUpdateResult(
            failed=failed,
            flagged=flagged,
            skipped=True,
            skip_reason="no committable doc updates (new pages could not be wired)",
        )

    # Report only drafts whose content was actually committed — an orphan-skipped
    # create_page is in `successful` but was never written to the branch.
    shipped_paths = {fw.path for fw in file_writes}
    shipped = [d for d in successful if d.target_path in shipped_paths]

    pr_url = gh.create_doc_update_pr(
        repo_name=repo,
        base_branch=base_branch,
        base_sha=ref,
        file_writes=file_writes,
        pr_title=f"docs: auto-update for PR #{pr_number} — {pr.title}",
        pr_body=_build_pr_body(pr_number, pr.html_url, shipped, flagged),
        assignee=pr.user.login if pr.user else None,
        labels=eff_pr_labels,
        draft=eff_pr_draft,
        pr_number=pr_number,
    )
    return DocUpdateResult(successful=shipped, failed=failed, flagged=flagged, pr_url=pr_url)

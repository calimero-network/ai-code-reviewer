"""GitHub comment formatter for review output."""

from __future__ import annotations

from ai_reviewer.github.client import ReviewDelta, ReviewMeta
from ai_reviewer.models.findings import ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview


class GitHubFormatter:
    """Formats review output for GitHub comments."""

    SEVERITY_EMOJI = {
        Severity.CRITICAL: "🔴",
        Severity.WARNING: "🟡",
        Severity.SUGGESTION: "💡",
        Severity.NITPICK: "📝",
    }

    SEVERITY_LABEL = {
        Severity.CRITICAL: "Critical",
        Severity.WARNING: "Warning",
        Severity.SUGGESTION: "Suggestion",
        Severity.NITPICK: "Nitpick",
    }

    def __init__(self, reviewer_name: str = "AI Code Reviewer") -> None:
        """Initialize the formatter.

        Args:
            reviewer_name: Custom name to display in review header
        """
        self.reviewer_name = reviewer_name

    def format_review(self, review: ConsolidatedReview, meta: ReviewMeta | None = None) -> str:
        """Format a consolidated review as a GitHub comment.

        Args:
            review: Consolidated review to format
            meta: Optional review metadata to embed for cross-run tracking

        Returns:
            Markdown formatted comment
        """
        lines = [
            f"## 🤖 {self.reviewer_name}",
            "",
            self._format_header(review),
            "",
            "---",
            "",
        ]

        if not review.findings:
            if review.failed_agents:
                lines.extend(self._format_incomplete_block(review))
            else:
                lines.extend(
                    [
                        "### ✅ No Issues Found",
                        "",
                        "All agents reviewed the code and found no issues. LGTM! 🎉",
                        "",
                    ]
                )
        else:
            # Group by severity
            by_severity = self._group_findings_by_severity(review.findings)

            for severity in [
                Severity.CRITICAL,
                Severity.WARNING,
                Severity.SUGGESTION,
                Severity.NITPICK,
            ]:
                findings = by_severity.get(severity, [])
                if findings:
                    lines.extend(
                        self._format_severity_section(severity, findings, review.agent_count)
                    )
                    lines.append("")

        if review.findings and review.failed_agents:
            lines.extend(self._format_partial_trailer(review))

        lines.extend(
            [
                "---",
                "",
                self._format_footer(review, meta),
            ]
        )

        return "\n".join(lines)

    def format_review_compact(
        self,
        review: ConsolidatedReview,
        meta: ReviewMeta | None = None,
        inline_findings: list | None = None,
    ) -> str:
        """Format a minimal top-level body when inline comments are posted.

        Use this when posting findings as inline comments so the PR-level
        comment stays short; details live on the code.
        """
        header = self._format_header(review)
        findings_for_inline = review.findings if inline_findings is None else inline_findings
        if not findings_for_inline:
            if review.failed_agents:
                body = (
                    f"⚠️ Review incomplete: {', '.join(review.failed_agents)} did not finish "
                    "and no findings were produced. Treat this PR as **not yet reviewed**."
                )
            else:
                body = "✅ No issues found. LGTM!"
        else:
            by_sev = self._count_findings_by_severity(findings_for_inline)
            parts = []
            if by_sev.get(Severity.CRITICAL, 0) > 0:
                parts.append(f"🔴 {by_sev[Severity.CRITICAL]} critical")
            if by_sev.get(Severity.WARNING, 0) > 0:
                parts.append(f"🟡 {by_sev[Severity.WARNING]} warnings")
            if by_sev.get(Severity.SUGGESTION, 0) > 0:
                parts.append(f"💡 {by_sev[Severity.SUGGESTION]} suggestions")
            if by_sev.get(Severity.NITPICK, 0) > 0:
                parts.append(f"📝 {by_sev[Severity.NITPICK]} nitpicks")
            body = (
                (", ".join(parts) + ". See inline comments.") if parts else "See inline comments."
            )
            body += self._compact_partial_suffix(review)
        return "\n".join(
            [
                f"## 🤖 {self.reviewer_name}",
                "",
                header,
                "",
                body,
                "",
                "---",
                "",
                self._format_footer(review, meta),
            ]
        )

    def format_review_with_delta_compact(
        self,
        review: ConsolidatedReview,
        delta: ReviewDelta,
        meta: ReviewMeta | None = None,
        inline_new_findings: list | None = None,
    ) -> str:
        """Format a minimal top-level body when inline comments are posted (with delta)."""
        header = self._format_header(review)
        if delta.all_issues_resolved:
            if review.failed_agents:
                body = (
                    f"⚠️ Review incomplete: {', '.join(review.failed_agents)} did not finish — "
                    "treat this PR as **not yet reviewed**, not as approved."
                )
            else:
                body = "✅ All issues resolved. Ready to merge!"
        else:
            new_findings = (
                delta.new_findings if inline_new_findings is None else inline_new_findings
            )
            parts = []
            if delta.fixed_findings:
                parts.append(f"✅ {len(delta.fixed_findings)} fixed")
            if new_findings:
                parts.append(f"🆕 {len(new_findings)} new")
            if delta.open_findings:
                parts.append(f"⏳ {len(delta.open_findings)} open")
            body = (
                (" | ".join(parts) + ". See inline comments.") if parts else "See inline comments."
            )
            body += self._compact_partial_suffix(review)
        suppressed_line = self._format_suppressed_line(delta)
        content = [
            f"## 🤖 {self.reviewer_name}",
            "",
            header,
            "",
            body,
        ]
        if suppressed_line:
            content.append(suppressed_line)
        content.extend(
            [
                "",
                "---",
                "",
                self._format_footer(review, meta),
            ]
        )
        return "\n".join(content)

    def _format_header(self, review: ConsolidatedReview) -> str:
        """Format the review header.

        The composite quality score is deliberately not shown: it folds review
        setup (agent count) into a code-quality-looking percentage that readers
        cannot act on. It remains available in the JSON export.
        """
        time_sec = review.total_review_time_ms / 1000

        if review.id == "lgtm-fast-path":
            return "**All previous comments resolved**"

        return f"**Reviewed by {review.agent_count} agents** | Review time: {time_sec:.1f}s"

    def _format_severity_section(
        self, severity: Severity, findings: list, agent_count: int
    ) -> list[str]:
        """Format a section for a severity level."""
        emoji = self.SEVERITY_EMOJI[severity]
        label = self.SEVERITY_LABEL[severity]

        lines = [
            f"### {emoji} {label} ({len(findings)})",
            "",
        ]

        for i, finding in enumerate(findings, 1):
            # Consensus indicator
            consensus_count = len(finding.agreeing_agents)
            consensus_str = f"{consensus_count}/{agent_count} agents"
            if consensus_count == agent_count:
                consensus_str += " ✓"

            lines.extend(
                [
                    f"#### {i}. {finding.title}",
                    f"**File:** `{finding.file_path}` (line {finding.line_start}"
                    + (f"-{finding.line_end}" if finding.line_end else "")
                    + f") | **Consensus:** {consensus_str}",
                    "",
                    finding.description,
                    "",
                ]
            )

            if finding.suggested_fix:
                lines.extend(
                    [
                        "**Suggested fix:**",
                        "```",
                        finding.suggested_fix,
                        "```",
                        "",
                    ]
                )

            # Show which agents found this (for transparency)
            agents_str = ", ".join(finding.agreeing_agents[:3])
            if len(finding.agreeing_agents) > 3:
                agents_str += f" (+{len(finding.agreeing_agents) - 3} more)"
            lines.append(f"> *Found by: {agents_str}*")
            lines.append("")

        return lines

    def _format_footer(self, review: ConsolidatedReview, meta: ReviewMeta | None = None) -> str:
        """Format the review footer, optionally embedding review metadata."""
        footer = (
            f"<sub>🤖 Generated by "
            f"[{self.reviewer_name}](https://github.com/calimero-network/ai-code-reviewer) | "
            f"Review ID: `{review.id}`</sub>"
        )
        if meta is not None:
            footer += "\n" + meta.to_html_comment()
        return footer

    def format_review_with_delta(
        self,
        review: ConsolidatedReview,
        delta: ReviewDelta,
        meta: ReviewMeta | None = None,
    ) -> str:
        """Format a review showing changes from previous run.

        Args:
            review: Current consolidated review
            delta: Changes from previous review
            meta: Optional review metadata to embed for cross-run tracking

        Returns:
            Markdown formatted comment with status indicators
        """
        lines = [
            f"## 🤖 {self.reviewer_name}",
            "",
            self._format_header(review),
            "",
        ]

        # Add status summary banner — but never "Ready to Merge" when agents
        # failed: an all-resolved delta from a partial run proves nothing.
        review_incomplete = review.failed_agents and delta.all_issues_resolved
        if not review_incomplete:
            lines.extend(self._format_status_banner(delta))
            lines.extend(["", "---", ""])
        elif delta.fixed_findings:
            # Fixed issues below are real, but the empty-delta "Review
            # Incomplete" body block won't render here, so the banner carries it.
            lines.extend(
                [
                    "### ⚠️ Review Incomplete",
                    "",
                    f"{', '.join(review.failed_agents)} did not finish — previously reported "
                    "issues were fixed, but this run may have missed new ones. Treat this PR "
                    "as **not yet fully reviewed**.",
                    "",
                    "---",
                    "",
                ]
            )

        # Show FIXED issues first (good news!)
        if delta.fixed_findings:
            lines.extend(self._format_fixed_section(delta.fixed_findings))
            lines.append("")

        # Show NEW issues (need attention)
        if delta.new_findings:
            lines.extend(self._format_new_findings_section(delta.new_findings, review.agent_count))
            lines.append("")

        # Show OPEN issues (still pending)
        if delta.open_findings:
            lines.extend(
                self._format_open_findings_section(delta.open_findings, review.agent_count)
            )
            lines.append("")

        # If nothing to show
        if not delta.new_findings and not delta.open_findings and not delta.fixed_findings:
            if review.failed_agents:
                lines.extend(self._format_incomplete_block(review))
            else:
                lines.extend(
                    [
                        "### ✅ No Issues Found",
                        "",
                        "All agents reviewed the code and found no issues. LGTM! 🎉",
                        "",
                    ]
                )

        # Partial-review note when real findings coexist with failed agents.
        if (delta.new_findings or delta.open_findings) and review.failed_agents:
            lines.extend(self._format_partial_trailer(review))

        suppressed_line = self._format_suppressed_line(delta)
        if suppressed_line:
            lines.append(suppressed_line)
            lines.append("")

        lines.extend(
            [
                "---",
                "",
                self._format_footer(review, meta),
            ]
        )

        return "\n".join(lines)

    def format_all_clear(
        self,
        review: ConsolidatedReview,
        delta: ReviewDelta,
        meta: ReviewMeta | None = None,
    ) -> str:
        """Explicit convergence verdict: previous findings addressed, this pass clean.

        Distinct from the first-pass "No Issues Found" body - this states that a
        re-review converged. Callers gate this on ``is_convergence_all_clear``.
        """
        fixed_count = len(delta.fixed_findings)
        noun = "finding" if fixed_count == 1 else "findings"
        verdict = (
            f"All previous findings addressed - {fixed_count} {noun} fixed since the "
            "last review. LGTM."
        )
        return "\n".join(
            [
                f"## 🤖 {self.reviewer_name}",
                "",
                self._format_header(review),
                "",
                "### ✅ All clear",
                "",
                verdict,
                "",
                "---",
                "",
                self._format_footer(review, meta),
            ]
        )

    def format_all_agents_failed(self, review: ConsolidatedReview) -> str:
        """Body posted when EVERY agent failed (infrastructure error).

        Callers must never stay silent in this case - an author who sees no
        comment cannot tell the reviewer even ran. The action must be COMMENT
        (never APPROVE), which the get_review_action guards already enforce for
        a findings-empty, failed-agents review.
        """
        return "\n".join(
            [
                f"## 🤖 {self.reviewer_name}",
                "",
                "### ⚠️ Review could not complete",
                "",
                f"Review could not complete: all {review.agent_count} agent(s) failed "
                "(infrastructure error). No code has been reviewed. Re-trigger with "
                "`/ai-review` or by reopening the PR.",
                "",
            ]
        )

    @staticmethod
    def _format_incomplete_block(review: ConsolidatedReview) -> list[str]:
        """Render the "Review Incomplete" body used when no findings were produced."""
        return [
            "### ⚠️ Review Incomplete",
            "",
            f"{len(review.failed_agents)} of {review.agent_count} agent(s) did not "
            f"finish ({', '.join(review.failed_agents)}) and no findings were "
            "produced. Treat this PR as **not yet reviewed**, not as approved.",
            "",
        ]

    @staticmethod
    def _format_partial_trailer(review: ConsolidatedReview) -> list[str]:
        """Render the trailer warning findings may be incomplete due to failed agents."""
        return [
            f"> ⚠️ Partial review: {', '.join(review.failed_agents)} did not finish — "
            "findings above may be incomplete.",
            "",
        ]

    @staticmethod
    def _compact_partial_suffix(review: ConsolidatedReview) -> str:
        """One-line partial-review note appended to a compact body, or '' if complete.

        The compact body is what posts whenever inline comments exist, so a
        multi-agent run where one agent fails but others find issues must still
        signal incompleteness here — not only in the full/no-findings paths.
        """
        if not review.failed_agents:
            return ""
        return (
            f" ⚠️ Partial review: {', '.join(review.failed_agents)} did not finish — "
            "coverage may be incomplete."
        )

    @staticmethod
    def _format_suppressed_line(delta: ReviewDelta) -> str:
        """Return a short note about suppressed findings, or empty string if none."""
        n = len(delta.suppressed_findings)
        if n == 0:
            return ""
        noun = "finding" if n == 1 else "findings"
        return f"*{n} low-severity {noun} suppressed on recently-fixed code.*"

    def _format_status_banner(self, delta: ReviewDelta) -> list[str]:
        """Format the status summary banner."""
        if delta.all_issues_resolved:
            return [
                "### ✅ Ready to Merge",
                "",
                "All previously identified issues have been addressed!",
            ]

        new_count = len(delta.new_findings)
        fixed_count = len(delta.fixed_findings)
        open_count = len(delta.open_findings)

        # Status icons
        parts = []
        if fixed_count > 0:
            parts.append(f"✅ **{fixed_count} Fixed**")
        if new_count > 0:
            parts.append(f"🆕 **{new_count} New**")
        if open_count > 0:
            parts.append(f"⏳ **{open_count} Open**")

        status_line = " | ".join(parts)

        # Determine overall status
        has_critical = any(
            f.severity.value == "critical" for f in delta.new_findings + delta.open_findings
        )

        if has_critical:
            status_icon = "🔴"
            status_text = "Critical issues require attention"
        elif new_count > 0 or open_count > 0:
            status_icon = "🟡"
            status_text = "Issues pending resolution"
        else:
            status_icon = "✅"
            status_text = "Ready to merge"

        return [
            f"### {status_icon} {status_text}",
            "",
            status_line,
        ]

    def _format_fixed_section(self, fixed_findings: list) -> list[str]:
        """Format the section showing fixed issues."""
        lines = [
            "### ✅ Fixed Issues",
            "",
            "<details>",
            "<summary>The following issues from previous reviews have been addressed:</summary>",
            "",
        ]

        for i, finding in enumerate(fixed_findings, 1):
            lines.append(f"{i}. ~~{finding.title}~~ (`{finding.file_path}:{finding.line}`)")

        lines.extend(["", "</details>"])
        return lines

    def _format_new_findings_section(self, findings: list, agent_count: int) -> list[str]:
        """Format section for NEW findings."""
        lines = [
            "### 🆕 New Issues",
            "",
            "*These issues were found in the latest changes:*",
            "",
        ]

        # Group by severity
        by_severity = self._group_findings_by_severity(findings)

        for severity in [
            Severity.CRITICAL,
            Severity.WARNING,
            Severity.SUGGESTION,
            Severity.NITPICK,
        ]:
            sev_findings = by_severity.get(severity, [])
            if sev_findings:
                lines.extend(self._format_severity_section(severity, sev_findings, agent_count))

        return lines

    def _format_open_findings_section(self, findings: list, agent_count: int) -> list[str]:
        """Format section for OPEN (still unresolved) findings."""
        lines = [
            "### ⏳ Open Issues",
            "",
            "*These issues from previous reviews are still present:*",
            "",
        ]

        # Group by severity
        by_severity = self._group_findings_by_severity(findings)

        for severity in [
            Severity.CRITICAL,
            Severity.WARNING,
            Severity.SUGGESTION,
            Severity.NITPICK,
        ]:
            sev_findings = by_severity.get(severity, [])
            if sev_findings:
                lines.extend(self._format_severity_section(severity, sev_findings, agent_count))

        return lines

    def _group_findings_by_severity(self, findings: list) -> dict:
        """Group findings by severity."""
        groups: dict = {}
        for finding in findings:
            if finding.severity not in groups:
                groups[finding.severity] = []
            groups[finding.severity].append(finding)
        return groups

    def _count_findings_by_severity(self, findings: list) -> dict[Severity, int]:
        """Count findings by severity for compact summaries."""
        counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
        for finding in findings:
            counts[finding.severity] += 1
        return counts

    def get_review_action_with_delta(
        self,
        review: ConsolidatedReview,
        delta: ReviewDelta,
        allow_approve: bool = True,
    ) -> str:
        """Determine GitHub review action considering the delta.

        LGTM-with-comments: REQUEST_CHANGES only for critical findings. When there are
        only warnings, suggestions, or nitpicks, returns COMMENT so the author isn't blocked.

        Args:
            review: Consolidated review
            delta: Review delta
            allow_approve: Whether to allow APPROVE action (also controls REQUEST_CHANGES)

        Returns:
            GitHub review action
        """
        # Never APPROVE off an empty delta caused by agents not finishing.
        if delta.all_issues_resolved and allow_approve and not review.failed_agents:
            return "APPROVE"

        # Block merge only when there are critical findings (not warnings/suggestions/nitpicks)
        has_critical = any(
            f.severity.value == "critical" for f in delta.new_findings + delta.open_findings
        )
        if has_critical and allow_approve:
            return "REQUEST_CHANGES"

        # No critical: COMMENT (includes only nits/suggestions/warnings — don't block author)
        return "COMMENT"

    def get_review_action(self, review: ConsolidatedReview, allow_approve: bool = True) -> str:
        """Determine the GitHub review action based on findings.

        LGTM-with-comments: REQUEST_CHANGES only for critical findings. When there are
        only warnings, suggestions, or nitpicks, returns COMMENT so the author isn't blocked.

        Args:
            review: Consolidated review
            allow_approve: Whether to allow APPROVE/REQUEST_CHANGES actions
                          (False in GitHub Actions to avoid blocking merges)

        Returns:
            GitHub review action: "APPROVE", "REQUEST_CHANGES", or "COMMENT"
        """
        # Never APPROVE when no findings exist only because agents failed.
        if not review.findings and allow_approve and not review.failed_agents:
            return "APPROVE"
        # Block merge only on critical; warnings/suggestions/nitpicks → COMMENT
        if review.has_critical_issues and allow_approve:
            return "REQUEST_CHANGES"
        return "COMMENT"


def format_review_as_json(review: ConsolidatedReview) -> dict:
    """Format review as JSON-serializable dict."""
    return {
        "review_id": review.id,
        "created_at": review.created_at.isoformat(),
        "repo": review.repo,
        "pr_number": review.pr_number,
        "summary": review.summary,
        "quality_score": review.review_quality_score,
        "agent_count": review.agent_count,
        "total_time_ms": review.total_review_time_ms,
        "findings": [
            {
                "id": f.id,
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "severity": f.severity.value,
                "category": f.category.value,
                "title": f.title,
                "description": f.description,
                "suggested_fix": f.suggested_fix,
                # The local fix loop applies validated replacements directly; without
                # these it would have to re-derive a fix that was already verified.
                "suggested_replacement": f.suggested_replacement,
                "fix_validated": f.fix_validated,
                "consensus_score": f.consensus_score,
                "agreeing_agents": f.agreeing_agents,
                "confidence": f.confidence,
            }
            for f in review.findings
        ],
        "findings_by_severity": {
            k.value: v for k, v in review.findings_by_severity.items() if v > 0
        },
    }


# Local report: severities in descending order, with the low two collapsed by
# default so a terminal read starts with what blocks.
_LOCAL_SEVERITY_ORDER = [Severity.CRITICAL, Severity.WARNING, Severity.SUGGESTION, Severity.NITPICK]
_LOCAL_COLLAPSED = (Severity.SUGGESTION, Severity.NITPICK)


def format_local_report(
    review: ConsolidatedReview,
    scope: str,
    show_all: bool = False,
) -> str:
    """Render a consolidated review for a terminal, grouped by severity.

    Each finding shows where it is, how confident the agents were, how many of
    them agreed, and whether its fix is a validated replacement or only prose -
    so a reader can tell an applyable fix from a suggestion at a glance.
    """
    agents = review.agent_count
    lines = [f"Reviewed {scope} - {agents} agent(s), {review.total_review_time_ms}ms", ""]

    if not review.findings:
        lines.append("No findings.")
        return "\n".join(lines)

    by_severity: dict[Severity, list[ConsolidatedFinding]] = {}
    for finding in review.findings:
        by_severity.setdefault(finding.severity, []).append(finding)

    hidden = 0
    for severity in _LOCAL_SEVERITY_ORDER:
        found = by_severity.get(severity) or []
        if not found:
            continue
        if severity in _LOCAL_COLLAPSED and not show_all:
            hidden += len(found)
            continue
        lines.append(f"{severity.value.upper()} ({len(found)})")
        for finding in sorted(found, key=lambda f: (f.file_path, f.line_start)):
            lines.append(f"  {finding.file_path}:{finding.line_start}  {finding.title}")
            fix = "fix ready (validated)" if finding.fix_validated else "prose fix only"
            agreed = len(finding.agreeing_agents)
            lines.append(f"    conf {finding.confidence:.2f} - {agreed}/{agents} agents - {fix}")
        lines.append("")

    if hidden:
        lines.append(f"{hidden} lower-severity finding(s) collapsed - run with --all to expand")

    return "\n".join(lines).rstrip() + "\n"

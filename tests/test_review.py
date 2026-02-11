"""Tests for the review module, particularly aggregate_findings."""

from datetime import datetime

import pytest

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview
from ai_reviewer.review import (
    _cluster_raw_findings,
    _detect_pr_type,
    _raw_findings_similar,
    aggregate_findings,
    apply_cross_review,
    get_cross_review_prompt,
    parse_cross_review_response,
)


class TestDetectPrType:
    """Tests for _detect_pr_type."""

    def test_docs_only_markdown(self):
        assert _detect_pr_type(["README.md"]) == "docs"
        assert _detect_pr_type(["docs/a.md", "docs/b.mdx"]) == "docs"

    def test_ci_only_workflows(self):
        assert _detect_pr_type([".github/workflows/ci.yml"]) == "ci"
        assert _detect_pr_type([".github/dependabot.yaml"]) == "ci"

    def test_code_mixed_or_rust(self):
        assert _detect_pr_type(["src/lib.rs"]) == "code"
        assert _detect_pr_type(["README.md", "src/main.py"]) == "code"
        assert _detect_pr_type([]) == "code"


class TestRawFindingsSimilar:
    """Tests for _raw_findings_similar helper."""

    def test_same_file_same_lines_same_category_similar_text(self):
        """Findings in same file/line/category with similar text are similar."""
        # Use highly similar text to ensure clustering (threshold is 0.85)
        raw1 = {
            "file_path": "src/auth.py",
            "line_start": 10,
            "line_end": 15,
            "category": "security",
            "title": "SQL Injection vulnerability found",
            "description": "User input is directly interpolated into the SQL query",
        }
        raw2 = {
            "file_path": "src/auth.py",
            "line_start": 10,
            "line_end": 15,
            "category": "security",
            "title": "SQL Injection vulnerability detected",
            "description": "User input is directly interpolated into the SQL query string",
        }
        assert _raw_findings_similar(raw1, raw2) is True

    def test_different_files_not_similar(self):
        """Findings in different files are not similar."""
        raw1 = {
            "file_path": "src/auth.py",
            "line_start": 10,
            "category": "security",
            "title": "SQL Injection",
        }
        raw2 = {
            "file_path": "src/users.py",
            "line_start": 10,
            "category": "security",
            "title": "SQL Injection",
        }
        assert _raw_findings_similar(raw1, raw2) is False

    def test_different_categories_not_similar(self):
        """Findings with different categories are not similar."""
        raw1 = {
            "file_path": "src/auth.py",
            "line_start": 10,
            "category": "security",
            "title": "Issue found",
        }
        raw2 = {
            "file_path": "src/auth.py",
            "line_start": 10,
            "category": "performance",
            "title": "Issue found",
        }
        assert _raw_findings_similar(raw1, raw2) is False


class TestClusterRawFindings:
    """Tests for _cluster_raw_findings helper."""

    def test_clusters_similar_findings_from_different_agents(self):
        """Similar findings from different agents are clustered together."""
        # Use identical titles/descriptions to ensure clustering
        tagged = [
            (
                "agent-1",
                {
                    "file_path": "a.py",
                    "line_start": 10,
                    "category": "security",
                    "title": "SQL Injection vulnerability",
                    "description": "User input in query",
                },
            ),
            (
                "agent-2",
                {
                    "file_path": "a.py",
                    "line_start": 10,
                    "category": "security",
                    "title": "SQL Injection vulnerability",
                    "description": "User input in query",
                },
            ),
        ]
        clusters = _cluster_raw_findings(tagged)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_clusters_similar_findings_from_same_agent(self):
        """Multiple similar findings from same agent are also clustered."""
        # Use identical content with overlapping lines
        tagged = [
            (
                "agent-1",
                {
                    "file_path": "a.py",
                    "line_start": 10,
                    "category": "security",
                    "title": "SQL Injection vulnerability",
                    "description": "Dangerous query",
                },
            ),
            (
                "agent-1",
                {
                    "file_path": "a.py",
                    "line_start": 12,
                    "category": "security",
                    "title": "SQL Injection vulnerability",
                    "description": "Dangerous query",
                },
            ),
        ]
        clusters = _cluster_raw_findings(tagged)
        # Both findings are similar (same file, overlapping lines within tolerance, same category, same title)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_keeps_different_findings_separate(self):
        """Different findings are kept in separate clusters."""
        tagged = [
            (
                "agent-1",
                {
                    "file_path": "a.py",
                    "line_start": 10,
                    "category": "security",
                    "title": "SQL Injection",
                },
            ),
            (
                "agent-2",
                {
                    "file_path": "b.py",
                    "line_start": 50,
                    "category": "performance",
                    "title": "Slow loop",
                },
            ),
        ]
        clusters = _cluster_raw_findings(tagged)
        assert len(clusters) == 2


class TestAggregateFindingsConsensus:
    """Tests for consensus score calculation in aggregate_findings."""

    def test_consensus_score_uses_unique_agents(self):
        """
        Consensus score should count unique agents, not total findings.

        Bug scenario: Agent A reports 2 similar findings, clustered together.
        With 3 total agents, consensus should be 1/3, not 2/3.
        """
        # Use identical titles/descriptions so they cluster together
        all_findings = [
            # Agent A has 2 similar findings (same file, overlapping lines, same category, identical text)
            (
                "agent-A",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "line_end": 12,
                        "category": "security",
                        "severity": "warning",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                    {
                        "file_path": "auth.py",
                        "line_start": 11,
                        "line_end": 13,
                        "category": "security",
                        "severity": "warning",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "Agent A summary",
            ),
            # Agents B and C have no findings
            ("agent-B", [], "Agent B: no issues"),
            ("agent-C", [], "Agent C: no issues"),
        ]

        result = aggregate_findings(all_findings, "test/repo", 123)

        # Should have 1 consolidated finding (the two similar ones clustered)
        assert len(result.findings) == 1
        finding = result.findings[0]

        # Consensus should be 1/3 (only agent-A found it), NOT 2/3
        assert finding.consensus_score == pytest.approx(1 / 3, rel=0.01)

        # agreeing_agents should not have duplicates
        assert finding.agreeing_agents == ["agent-A"]

    def test_consensus_score_with_multiple_agents_agreeing(self):
        """
        When multiple agents find the same issue, consensus reflects unique count.
        """
        # Use identical text so they cluster together
        all_findings = [
            (
                "agent-A",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "critical",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "Agent A summary",
            ),
            (
                "agent-B",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "critical",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "Agent B summary",
            ),
            ("agent-C", [], "Agent C: no issues"),
        ]

        result = aggregate_findings(all_findings, "test/repo", 123)

        assert len(result.findings) == 1
        finding = result.findings[0]

        # 2 out of 3 agents found it
        assert finding.consensus_score == pytest.approx(2 / 3, rel=0.01)
        assert set(finding.agreeing_agents) == {"agent-A", "agent-B"}

    def test_agreeing_agents_deduplicated_with_mixed_scenario(self):
        """
        Mixed scenario: Agent A has 2 similar findings, Agent B has 1 similar.
        All should cluster together with unique agents.
        """
        # Use identical text so all three findings cluster together
        all_findings = [
            (
                "agent-A",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "warning",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                    {
                        "file_path": "auth.py",
                        "line_start": 11,
                        "category": "security",
                        "severity": "warning",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "Agent A summary",
            ),
            (
                "agent-B",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "warning",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "Agent B summary",
            ),
        ]

        result = aggregate_findings(all_findings, "test/repo", 123)

        assert len(result.findings) == 1
        finding = result.findings[0]

        # 2 unique agents, 2 total agents = 100% consensus
        assert finding.consensus_score == pytest.approx(1.0, rel=0.01)
        # Should have exactly 2 unique agents, not 3 (which would happen with duplicates)
        assert len(finding.agreeing_agents) == 2
        assert set(finding.agreeing_agents) == {"agent-A", "agent-B"}

    def test_full_consensus_all_agents_find_same_issue(self):
        """Full consensus when all agents find the same issue."""
        # Use identical text so all findings cluster together
        all_findings = [
            (
                "agent-A",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "critical",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "A",
            ),
            (
                "agent-B",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "critical",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "B",
            ),
            (
                "agent-C",
                [
                    {
                        "file_path": "auth.py",
                        "line_start": 10,
                        "category": "security",
                        "severity": "critical",
                        "title": "SQL Injection vulnerability",
                        "description": "User input in query",
                    },
                ],
                "C",
            ),
        ]

        result = aggregate_findings(all_findings, "test/repo", 123)

        assert len(result.findings) == 1
        assert result.findings[0].consensus_score == pytest.approx(1.0, rel=0.01)
        assert len(result.findings[0].agreeing_agents) == 3


def _make_finding(fid: str, severity: Severity = Severity.WARNING) -> ConsolidatedFinding:
    """Minimal ConsolidatedFinding for cross-review tests."""
    return ConsolidatedFinding(
        id=fid,
        file_path="src/foo.py",
        line_start=10,
        line_end=12,
        severity=severity,
        category=Category.LOGIC,
        title="Test finding",
        description="Description",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["agent-1"],
        confidence=0.9,
    )


def _make_review(findings: list[ConsolidatedFinding]) -> ConsolidatedReview:
    """Minimal ConsolidatedReview for cross-review tests."""
    return ConsolidatedReview(
        id="review-1",
        created_at=datetime.now(),
        repo="test/repo",
        pr_number=1,
        findings=findings,
        summary="Summary",
        agent_count=3,
        review_quality_score=0.9,
        total_review_time_ms=0,
        failed_agents=[],
    )


class TestParseCrossReviewResponse:
    """Tests for parse_cross_review_response."""

    def test_valid_json(self):
        content = '{"assessments": [{"id": "finding-1", "valid": true, "rank": 1}], "summary": "OK"}'
        assessments, summary = parse_cross_review_response(content)
        assert len(assessments) == 1
        assert assessments[0]["id"] == "finding-1"
        assert assessments[0]["valid"] is True
        assert assessments[0]["rank"] == 1
        assert summary == "OK"

    def test_markdown_fenced_json(self):
        content = """Some text
```json
{"assessments": [{"id": "f1", "valid": false, "rank": 2}], "summary": "Done"}
```
"""
        assessments, summary = parse_cross_review_response(content)
        assert len(assessments) == 1
        assert assessments[0]["id"] == "f1"
        assert assessments[0]["valid"] is False
        assert summary == "Done"

    def test_malformed_input_returns_empty(self):
        assessments, summary = parse_cross_review_response("not json at all")
        assert assessments == []
        assert summary == ""

    def test_invalid_json_returns_empty(self):
        content = '{"assessments": [invalid]}'
        assessments, summary = parse_cross_review_response(content)
        assert assessments == []
        assert summary == ""


class TestApplyCrossReview:
    """Tests for apply_cross_review."""

    def test_no_assessments_returns_unchanged(self):
        review = _make_review([_make_finding("f1")])
        result = apply_cross_review(review, [])
        assert result.findings == review.findings
        assert result.summary == review.summary

    def test_no_votes_for_finding_kept(self):
        """Findings with zero votes are kept (not counted as rejected)."""
        review = _make_review([_make_finding("f1"), _make_finding("f2")])
        # Only agent assesses f1; f2 gets no votes
        all_assessments = [
            ("agent-1", [{"id": "f1", "valid": True, "rank": 1}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        assert len(result.findings) == 2
        ids = [f.id for f in result.findings]
        assert "f1" in ids and "f2" in ids

    def test_finding_id_alias_accepted(self):
        """Assessments can use 'finding_id' instead of 'id' (alias)."""
        review = _make_review([_make_finding("f1")])
        all_assessments = [
            ("a1", [{"finding_id": "f1", "valid": True, "rank": 1}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert len(result.findings) == 1
        assert result.findings[0].id == "f1"

    def test_partial_votes_uses_len_votes_not_n_agents(self):
        """Valid ratio is over assessing agents, not total agents."""
        review = _make_review([_make_finding("f1")])
        # 1 valid out of 1 assessing agent -> ratio 1.0, kept
        all_assessments = [("agent-1", [{"id": "f1", "valid": True, "rank": 1}])]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        assert len(result.findings) == 1
        # 1 valid, 1 invalid from 2 agents that assessed it -> 0.5
        all_assessments = [
            ("agent-1", [{"id": "f1", "valid": True, "rank": 1}]),
            ("agent-2", [{"id": "f1", "valid": False, "rank": 5}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        assert len(result.findings) == 1
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.67)
        assert len(result.findings) == 0

    def test_threshold_drops_below_keeps_at_or_above(self):
        review = _make_review([_make_finding("f1")])
        # 2/3 valid
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 2}]),
            ("a3", [{"id": "f1", "valid": False, "rank": 3}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=2 / 3)
        assert len(result.findings) == 1
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.9)
        assert len(result.findings) == 0

    def test_reordered_by_avg_rank_then_severity(self):
        f1 = _make_finding("f1", Severity.WARNING)
        f2 = _make_finding("f2", Severity.CRITICAL)
        f3 = _make_finding("f3", Severity.SUGGESTION)
        review = _make_review([f1, f2, f3])
        # f1 rank 3, f2 rank 1, f3 rank 2 -> order f2, f3, f1
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 3}, {"id": "f2", "valid": True, "rank": 1}, {"id": "f3", "valid": True, "rank": 2}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert [x.id for x in result.findings] == ["f2", "f3", "f1"]

    def test_summary_unchanged_when_nothing_dropped_or_reordered(self):
        review = _make_review([_make_finding("f1")])
        all_assessments = [("a1", [{"id": "f1", "valid": True, "rank": 1}])]
        result = apply_cross_review(review, all_assessments)
        assert result.summary == review.summary

    def test_summary_appends_when_dropped(self):
        """When only dropping (no reorder), summary should not claim 're-ranked'."""
        review = _make_review([_make_finding("f1"), _make_finding("f2")])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": False, "rank": 2}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": False, "rank": 2}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=1.0)
        assert len(result.findings) == 1
        assert "1 finding(s) dropped" in result.summary
        assert "re-ranked" not in result.summary

    def test_quality_score_recalculated_after_cross_review(self):
        """Quality score is recomputed from cross-review valid_ratio, not copied unchanged."""
        review = _make_review([_make_finding("f1"), _make_finding("f2")])
        assert review.review_quality_score == 0.9
        # All agents say valid for both -> avg valid_ratio 1.0, agent_factor 1.0 -> score 1.0
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}]),
            ("a3", [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert result.review_quality_score == 1.0

    def test_valid_field_string_coerced_to_bool(self):
        """LLM may return valid as string 'false'; must be coerced so finding is dropped when appropriate."""
        review = _make_review([_make_finding("f1")])
        # One agent says valid True, one says "valid": "false" (string). Without coercion, "false" is truthy.
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1}]),
            ("a2", [{"id": "f1", "valid": "false", "rank": 2}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.6)
        assert len(result.findings) == 0


class TestGetCrossReviewPrompt:
    """Tests for get_cross_review_prompt."""

    def test_diff_truncated_at_newline(self):
        """Diff excerpt does not cut mid-line."""
        context = ReviewContext(
            repo_name="test/repo",
            pr_number=1,
            pr_title="Title",
            pr_description="",
            base_branch="main",
            head_branch="feature",
            author="dev",
            changed_files_count=1,
            additions=10,
            deletions=2,
        )
        review = _make_review([_make_finding("finding-1")])
        # Create diff that would be cut at 50 chars (mid-line)
        line = "a" * 30 + "\n" + "b" * 30
        diff = line + "\nlast"
        import ai_reviewer.review as review_mod

        old_max = review_mod._CROSS_REVIEW_DIFF_MAX_CHARS
        try:
            review_mod._CROSS_REVIEW_DIFF_MAX_CHARS = 50
            prompt = get_cross_review_prompt(context, review, diff)
            # Excerpt should end at newline, not mid "b"
            assert "```diff" in prompt
            excerpt = prompt.split("```diff")[1].split("```")[0].strip()
            assert excerpt.endswith("a" * 30)
            assert not excerpt.endswith("b")
        finally:
            review_mod._CROSS_REVIEW_DIFF_MAX_CHARS = old_max

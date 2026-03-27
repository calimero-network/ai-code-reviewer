"""Tests for the review module, particularly aggregate_findings."""

from datetime import datetime

import pytest

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview
from ai_reviewer.review import (
    CONFIDENCE_THRESHOLDS,
    _cluster_raw_findings,
    _detect_pr_type,
    _effective_agent_count,
    _raw_findings_similar,
    aggregate_findings,
    apply_cross_review,
    compute_quality_score,
    dedup_cross_file,
    get_cross_review_prompt,
    get_output_format,
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
        content = (
            '{"assessments": [{"id": "finding-1", "valid": true, "rank": 1}], "summary": "OK"}'
        )
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
            (
                "a1",
                [
                    {"id": "f1", "valid": True, "rank": 3},
                    {"id": "f2", "valid": True, "rank": 1},
                    {"id": "f3", "valid": True, "rank": 2},
                ],
            ),
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
            (
                "a1",
                [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": False, "rank": 2}],
            ),
            (
                "a2",
                [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": False, "rank": 2}],
            ),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=1.0)
        assert len(result.findings) == 1
        assert "1 finding(s) dropped" in result.summary
        assert "re-ranked" not in result.summary

    def test_quality_score_recalculated_after_cross_review(self):
        """Quality score is recomputed via compute_quality_score, not copied unchanged."""
        review = _make_review([_make_finding("f1"), _make_finding("f2")])
        assert review.review_quality_score == 0.9
        all_assessments = [
            (
                "a1",
                [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}],
            ),
            (
                "a2",
                [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}],
            ),
            (
                "a3",
                [{"id": "f1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}],
            ),
        ]
        result = apply_cross_review(review, all_assessments)
        expected_score, expected_breakdown = compute_quality_score(
            result.findings, review.agent_count, total_lines=0
        )
        assert result.review_quality_score == expected_score
        assert result.score_breakdown == expected_breakdown

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

    def test_critical_security_finding_survives_unanimous_rejection(self):
        """Critical security findings must never be dropped by cross-review, even if all agents mark them invalid."""
        critical_sec = ConsolidatedFinding(
            id="sec1",
            file_path="src/auth.py",
            line_start=10,
            line_end=12,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="SQL injection",
            description="User input interpolated into SQL",
            suggested_fix="Use parameterized queries",
            consensus_score=1.0,
            agreeing_agents=["agent-1"],
            confidence=0.95,
        )
        normal = _make_finding("f2")
        review = _make_review([critical_sec, normal])
        all_assessments = [
            (
                "a1",
                [{"id": "sec1", "valid": False, "rank": 5}, {"id": "f2", "valid": True, "rank": 1}],
            ),
            (
                "a2",
                [{"id": "sec1", "valid": False, "rank": 5}, {"id": "f2", "valid": True, "rank": 2}],
            ),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        result_ids = [f.id for f in result.findings]
        assert "sec1" in result_ids, "Critical security finding must survive cross-review rejection"
        assert result.findings[0].id == "sec1", "Critical security finding should be ranked first"

    def test_critical_nonsecurity_finding_can_be_dropped(self):
        """Critical findings that are NOT security-category follow normal cross-review rules."""
        critical_logic = ConsolidatedFinding(
            id="logic1",
            file_path="src/calc.py",
            line_start=10,
            line_end=12,
            severity=Severity.CRITICAL,
            category=Category.LOGIC,
            title="Off-by-one error",
            description="Loop bound is wrong",
            suggested_fix=None,
            consensus_score=1.0,
            agreeing_agents=["agent-1"],
            confidence=0.9,
        )
        review = _make_review([critical_logic])
        all_assessments = [
            ("a1", [{"id": "logic1", "valid": False, "rank": 5}]),
            ("a2", [{"id": "logic1", "valid": False, "rank": 5}]),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        assert len(result.findings) == 0, "Non-security critical finding should be droppable"


class TestGetCrossReviewPrompt:
    """Tests for get_cross_review_prompt."""

    def test_diff_truncated_at_newline(self, monkeypatch):
        """Diff excerpt does not cut mid-line."""
        import ai_reviewer.review as review_mod

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
        monkeypatch.setattr(review_mod, "_CROSS_REVIEW_DIFF_MAX_CHARS", 50)
        prompt = get_cross_review_prompt(context, review, diff)
        # Excerpt should end at newline, not mid "b"
        assert "```diff" in prompt
        excerpt = prompt.split("```diff")[1].split("```")[0].strip()
        assert excerpt.endswith("a" * 30)
        assert not excerpt.endswith("b")


class TestEffectiveAgentCount:
    """Tests for _effective_agent_count."""

    def test_tiny_pr_caps_at_one(self):
        assert _effective_agent_count(additions=50, deletions=20, changed_files=2, requested=3) == 1

    def test_small_pr_caps_at_two(self):
        assert (
            _effective_agent_count(additions=200, deletions=100, changed_files=5, requested=3) == 2
        )

    def test_large_pr_uses_requested(self):
        assert (
            _effective_agent_count(additions=400, deletions=200, changed_files=10, requested=3) == 3
        )

    def test_requested_one_always_one(self):
        assert (
            _effective_agent_count(additions=1000, deletions=500, changed_files=20, requested=1)
            == 1
        )

    def test_boundary_150_lines_3_files(self):
        assert (
            _effective_agent_count(additions=100, deletions=49, changed_files=3, requested=3) == 1
        )
        assert (
            _effective_agent_count(additions=100, deletions=50, changed_files=3, requested=3) == 2
        )

    def test_boundary_150_lines_4_files(self):
        assert (
            _effective_agent_count(additions=100, deletions=30, changed_files=4, requested=3) == 2
        )

    def test_boundary_500_lines(self):
        assert (
            _effective_agent_count(additions=300, deletions=199, changed_files=5, requested=3) == 2
        )
        assert (
            _effective_agent_count(additions=300, deletions=200, changed_files=5, requested=3) == 3
        )

    def test_requested_caps_result(self):
        assert (
            _effective_agent_count(additions=200, deletions=100, changed_files=5, requested=1) == 1
        )
        assert _effective_agent_count(additions=50, deletions=20, changed_files=2, requested=0) == 0


class TestGetOutputFormatFewShotExamples:
    """Tests for few-shot examples in get_output_format (Task 5)."""

    def test_good_example_present(self):
        output = get_output_format()
        assert "Example of a GOOD finding" in output
        assert "SQL injection via string interpolation" in output

    def test_bad_example_present(self):
        output = get_output_format()
        assert "Example of a BAD finding" in output
        assert "Consider adding more tests" in output
        assert "DO NOT produce these" in output

    def test_examples_appear_after_rules_before_analyze(self):
        output = get_output_format()
        rules_pos = output.index("**Rules**")
        good_pos = output.index("Example of a GOOD finding")
        bad_pos = output.index("Example of a BAD finding")
        analyze_pos = output.index("Analyze the PR")
        assert rules_pos < good_pos < bad_pos < analyze_pos


class TestGetOutputFormatAdaptiveFindings:
    """Tests for adaptive max findings in get_output_format (Task 7)."""

    def test_zero_lines_gives_minimum_3(self):
        output = get_output_format(total_lines=0)
        assert "Maximum 3 findings per agent" in output

    def test_500_lines_gives_8(self):
        output = get_output_format(total_lines=500)
        assert "Maximum 8 findings per agent" in output

    def test_2000_lines_capped_at_10(self):
        output = get_output_format(total_lines=2000)
        assert "Maximum 10 findings per agent" in output

    def test_100_lines_gives_4(self):
        output = get_output_format(total_lines=100)
        assert "Maximum 4 findings per agent" in output

    def test_default_no_lines_gives_3(self):
        output = get_output_format()
        assert "Maximum 3 findings per agent" in output

    def test_docs_type_still_has_adaptive_limit(self):
        output = get_output_format(pr_type="docs", total_lines=700)
        assert "Maximum 10 findings per agent" in output
        assert "Only factual errors or security" in output


def _make_raw_finding(
    severity: str = "warning",
    confidence: float = 0.8,
    file_path: str = "src/foo.py",
    title: str = "Test issue",
) -> dict:
    return {
        "file_path": file_path,
        "line_start": 10,
        "severity": severity,
        "category": "logic",
        "title": title,
        "description": "Description",
        "confidence": confidence,
    }


class TestConfidenceFiltering:
    """Tests for confidence-based filtering in aggregate_findings."""

    def test_default_thresholds_exist(self):
        assert Severity.CRITICAL in CONFIDENCE_THRESHOLDS
        assert Severity.WARNING in CONFIDENCE_THRESHOLDS
        assert Severity.SUGGESTION in CONFIDENCE_THRESHOLDS
        assert Severity.NITPICK in CONFIDENCE_THRESHOLDS

    def test_default_threshold_values(self):
        assert CONFIDENCE_THRESHOLDS[Severity.CRITICAL] == 0.5
        assert CONFIDENCE_THRESHOLDS[Severity.WARNING] == 0.6
        assert CONFIDENCE_THRESHOLDS[Severity.SUGGESTION] == 0.7
        assert CONFIDENCE_THRESHOLDS[Severity.NITPICK] == 0.8

    def test_high_confidence_findings_kept(self):
        """Findings at or above their severity threshold are kept."""
        all_findings = [
            ("agent-1", [_make_raw_finding("critical", 0.95)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 1

    def test_low_confidence_critical_dropped(self):
        """Critical finding below 0.5 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("critical", 0.4)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_warning_dropped(self):
        """Warning finding below 0.6 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("warning", 0.5)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_suggestion_dropped(self):
        """Suggestion finding below 0.7 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("suggestion", 0.6)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_nitpick_dropped(self):
        """Nitpick finding below 0.8 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("nitpick", 0.7)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_exact_threshold_kept(self):
        """Findings exactly at the threshold are kept (>= comparison)."""
        all_findings = [
            (
                "agent-1",
                [
                    _make_raw_finding("critical", 0.5),
                    _make_raw_finding("warning", 0.6, title="Warning issue"),
                    _make_raw_finding("suggestion", 0.7, title="Suggestion issue"),
                    _make_raw_finding("nitpick", 0.8, title="Nit: Style issue"),
                ],
                "summary",
            ),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 4

    def test_mixed_confidence_partial_filtering(self):
        """Only low-confidence findings are dropped; high-confidence ones survive."""
        all_findings = [
            (
                "agent-1",
                [
                    _make_raw_finding("critical", 0.95),
                    _make_raw_finding("nitpick", 0.5, title="Nit: Low confidence nit"),
                    _make_raw_finding("warning", 0.9, title="High conf warning"),
                ],
                "summary",
            ),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 2
        severities = {f.severity for f in result.findings}
        assert Severity.CRITICAL in severities
        assert Severity.WARNING in severities
        assert Severity.NITPICK not in severities

    def test_custom_thresholds_override_defaults(self):
        """Custom thresholds passed to aggregate_findings override defaults."""
        custom = {
            Severity.CRITICAL: 0.99,
            Severity.WARNING: 0.99,
            Severity.SUGGESTION: 0.99,
            Severity.NITPICK: 0.99,
        }
        all_findings = [
            ("agent-1", [_make_raw_finding("critical", 0.95)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1, confidence_thresholds=custom)
        assert len(result.findings) == 0

    def test_custom_thresholds_zero_keeps_all(self):
        """Setting all thresholds to 0 keeps every finding."""
        custom = {
            Severity.CRITICAL: 0.0,
            Severity.WARNING: 0.0,
            Severity.SUGGESTION: 0.0,
            Severity.NITPICK: 0.0,
        }
        all_findings = [
            (
                "agent-1",
                [
                    _make_raw_finding("critical", 0.1),
                    _make_raw_finding("nitpick", 0.01, title="Nit: tiny"),
                ],
                "summary",
            ),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1, confidence_thresholds=custom)
        assert len(result.findings) == 2

    def test_quality_score_computed_after_filtering(self):
        """Quality score should be based on the filtered set, not the pre-filter set."""
        all_findings = [
            (
                "agent-1",
                [
                    _make_raw_finding("critical", 0.95),
                    _make_raw_finding("nitpick", 0.1, title="Nit: low"),
                ],
                "summary",
            ),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 1
        assert result.review_quality_score > 0


def _make_consolidated(
    file_path: str = "src/foo.py",
    category: Category = Category.LOGIC,
    title: str = "Test issue",
    severity: Severity = Severity.WARNING,
    confidence: float = 0.9,
    consensus_score: float = 1.0,
) -> ConsolidatedFinding:
    """Factory for ConsolidatedFinding used in cross-file dedup tests."""
    return ConsolidatedFinding(
        id=f"finding-{id(object())}",
        file_path=file_path,
        line_start=10,
        line_end=12,
        severity=severity,
        category=category,
        title=title,
        description="Description of the issue.",
        suggested_fix=None,
        consensus_score=consensus_score,
        agreeing_agents=["agent-1"],
        confidence=confidence,
    )


class TestDedupCrossFile:
    """Tests for dedup_cross_file() cross-file deduplication."""

    def test_two_same_title_different_files_kept(self):
        """Two findings with the same (category, title) in different files stay separate."""
        findings = [
            _make_consolidated(file_path="src/a.py", title="Missing null check"),
            _make_consolidated(file_path="src/b.py", title="Missing null check"),
        ]
        result = dedup_cross_file(findings)
        assert len(result) == 2

    def test_three_same_title_different_files_collapse(self):
        """Three or more findings with the same (category, title) in different files collapse to one."""
        findings = [
            _make_consolidated(file_path="src/a.py", title="Missing null check"),
            _make_consolidated(file_path="src/b.py", title="Missing null check"),
            _make_consolidated(file_path="src/c.py", title="Missing null check"),
        ]
        result = dedup_cross_file(findings)
        assert len(result) == 1
        assert "Also found in:" in result[0].description

    def test_different_titles_not_collapsed(self):
        """Findings with different titles are never collapsed, even if category matches."""
        findings = [
            _make_consolidated(file_path="src/a.py", title="Missing null check"),
            _make_consolidated(file_path="src/b.py", title="Missing null check"),
            _make_consolidated(file_path="src/c.py", title="Unused import"),
        ]
        result = dedup_cross_file(findings)
        titles = [f.title for f in result]
        assert "Unused import" in titles
        assert titles.count("Missing null check") == 2

    def test_collapsed_group_keeps_highest_priority(self):
        """The representative of a collapsed group is the finding with the highest priority_score."""
        low = _make_consolidated(
            file_path="src/a.py",
            title="Missing null check",
            severity=Severity.SUGGESTION,
            confidence=0.7,
        )
        mid = _make_consolidated(
            file_path="src/b.py",
            title="Missing null check",
            severity=Severity.WARNING,
            confidence=0.8,
        )
        high = _make_consolidated(
            file_path="src/c.py",
            title="Missing null check",
            severity=Severity.CRITICAL,
            confidence=0.95,
        )
        findings = [low, mid, high]
        result = dedup_cross_file(findings)
        assert len(result) == 1
        assert result[0].file_path == "src/c.py"
        assert result[0].severity == Severity.CRITICAL

    def test_also_found_in_note_caps_paths(self):
        """The 'Also found in' note is capped to a readable subset of paths."""
        findings = [
            _make_consolidated(file_path=f"src/file_{i}.py", title="Repeated pattern")
            for i in range(10)
        ]
        result = dedup_cross_file(findings)
        assert len(result) == 1
        note = result[0].description
        assert "Also found in:" in note
        assert "and" in note or note.count("src/") <= 6

    def test_grouping_uses_normalized_category_and_title(self):
        """Grouping normalizes category and title (case-insensitive, stripped)."""
        findings = [
            _make_consolidated(
                file_path="src/a.py", category=Category.LOGIC, title="  Null Check "
            ),
            _make_consolidated(file_path="src/b.py", category=Category.LOGIC, title="null check"),
            _make_consolidated(file_path="src/c.py", category=Category.LOGIC, title="NULL CHECK"),
        ]
        result = dedup_cross_file(findings)
        assert len(result) == 1

    def test_different_categories_same_title_not_collapsed(self):
        """Same title but different categories should not be grouped together."""
        findings = [
            _make_consolidated(
                file_path="src/a.py", category=Category.LOGIC, title="Missing check"
            ),
            _make_consolidated(
                file_path="src/b.py", category=Category.SECURITY, title="Missing check"
            ),
            _make_consolidated(
                file_path="src/c.py", category=Category.LOGIC, title="Missing check"
            ),
            _make_consolidated(
                file_path="src/d.py", category=Category.SECURITY, title="Missing check"
            ),
            _make_consolidated(
                file_path="src/e.py", category=Category.SECURITY, title="Missing check"
            ),
        ]
        result = dedup_cross_file(findings)
        logic_findings = [f for f in result if f.category == Category.LOGIC]
        security_findings = [f for f in result if f.category == Category.SECURITY]
        assert len(logic_findings) == 2
        assert len(security_findings) == 1

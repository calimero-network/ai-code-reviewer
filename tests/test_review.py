"""Tests for the review module, particularly aggregate_findings."""

from datetime import datetime

import pytest

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview
from ai_reviewer.review import (
    _NO_CROSS_REVIEW_THRESHOLDS,
    CONFIDENCE_THRESHOLDS,
    _cap_findings,
    _cluster_raw_findings,
    _detect_pr_type,
    _effective_agent_count,
    _raw_findings_similar,
    _raw_text_similarity,
    _thresholds_for_run,
    _truncate_to_byte_limit,
    aggregate_findings,
    apply_cross_review,
    compute_quality_score,
    dedup_cross_file,
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

    def test_line_overlap_lowers_threshold_to_merge_reworded(self):
        """Same file/lines/category with moderately similar (0.6+) text now clusters."""
        raw1 = {
            "file_path": "src/agg.py",
            "line_start": 20,
            "line_end": 24,
            "category": "logic",
            "title": "Aggregation shows false green on partial failure",
            "description": "A failed shard is not reflected in the summary.",
        }
        raw2 = {
            "file_path": "src/agg.py",
            "line_start": 22,
            "line_end": 26,
            "category": "logic",
            "title": "Aggregation shows false green when a shard fails",
            "description": "One failing shard is hidden from the overall status.",
        }
        # Merges under the loosened (line-overlap) 0.6 bar.
        assert _raw_findings_similar(raw1, raw2) is True
        # The same moderately-similar pair would NOT merge at the strict 0.85 bar
        # if the lines did not overlap; here they overlap so 0.6 wins regardless.
        combined = _raw_text_similarity(raw1["title"], raw2["title"]) * 0.6 + (
            _raw_text_similarity(raw1["description"], raw2["description"]) * 0.4
        )
        assert 0.6 <= combined < 0.85

    def test_different_severities_still_cluster(self):
        """Severity is not part of the similarity gate - differing severities merge."""
        raw1 = {
            "file_path": "src/agg.py",
            "line_start": 20,
            "category": "logic",
            "severity": "critical",
            "title": "False green aggregation",
            "description": "Failed shard hidden.",
        }
        raw2 = {
            "file_path": "src/agg.py",
            "line_start": 21,
            "category": "logic",
            "severity": "warning",
            "title": "False green aggregation",
            "description": "Failed shard hidden.",
        }
        assert _raw_findings_similar(raw1, raw2) is True

    def test_unrelated_findings_still_separate(self):
        """A genuinely different issue in the same file/category does not merge."""
        raw1 = {
            "file_path": "src/agg.py",
            "line_start": 20,
            "category": "logic",
            "title": "False green aggregation on partial failure",
            "description": "Failed shard is hidden from the summary.",
        }
        raw2 = {
            "file_path": "src/agg.py",
            "line_start": 22,
            "category": "logic",
            "title": "Off-by-one in retry backoff counter",
            "description": "The exponential backoff overshoots by one attempt.",
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

    def test_adjusted_severity_and_fix_ok_accepted(self):
        """New optional keys pass through the parser."""
        content = (
            '{"assessments": [{"id": "f1", "valid": true, "rank": 1, '
            '"adjusted_severity": "warning", "fix_ok": false}], "summary": "s"}'
        )
        assessments, _ = parse_cross_review_response(content)
        assert assessments[0]["adjusted_severity"] == "warning"
        assert assessments[0]["fix_ok"] is False

    def test_missing_optional_keys_tolerated(self):
        """Absent adjusted_severity/fix_ok simply are not present (no error)."""
        content = '{"assessments": [{"id": "f1", "valid": true, "rank": 1}], "summary": "s"}'
        assessments, _ = parse_cross_review_response(content)
        assert "adjusted_severity" not in assessments[0]
        assert "fix_ok" not in assessments[0]


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

    def test_downgrade_on_majority(self):
        """Majority proposing a lower severity downgrades the finding (kept)."""
        review = _make_review([_make_finding("f1", Severity.CRITICAL)])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1, "adjusted_severity": "warning"}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1, "adjusted_severity": "warning"}]),
            ("a3", [{"id": "f1", "valid": True, "rank": 1}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert len(result.findings) == 1
        assert result.findings[0].severity == Severity.WARNING

    def test_never_upgrades(self):
        """A proposed higher severity is ignored (cross-review only downgrades)."""
        review = _make_review([_make_finding("f1", Severity.SUGGESTION)])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1, "adjusted_severity": "critical"}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1, "adjusted_severity": "critical"}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert result.findings[0].severity == Severity.SUGGESTION

    def test_no_downgrade_without_majority(self):
        """A single downgrade vote among several validators does not move severity."""
        review = _make_review([_make_finding("f1", Severity.CRITICAL)])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1, "adjusted_severity": "nitpick"}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1}]),
            ("a3", [{"id": "f1", "valid": True, "rank": 1}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert result.findings[0].severity == Severity.CRITICAL

    def test_fix_stripped_on_fix_ok_false_majority(self):
        """Majority fix_ok=false drops the suggested_fix (finding stays)."""
        f = _make_finding("f1", Severity.WARNING)
        f.suggested_fix = "anchor the regex with \\b"
        review = _make_review([f])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": False}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": False}]),
            ("a3", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": True}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert len(result.findings) == 1
        assert result.findings[0].suggested_fix is None

    def test_fix_kept_when_no_false_majority(self):
        """A minority fix_ok=false keeps the suggested_fix."""
        f = _make_finding("f1", Severity.WARNING)
        f.suggested_fix = "use parameterized query"
        review = _make_review([f])
        all_assessments = [
            ("a1", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": False}]),
            ("a2", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": True}]),
            ("a3", [{"id": "f1", "valid": True, "rank": 1, "fix_ok": True}]),
        ]
        result = apply_cross_review(review, all_assessments)
        assert result.findings[0].suggested_fix == "use parameterized query"

    def test_critical_security_bypass_not_downgraded(self):
        """CRITICAL+SECURITY findings keep their severity and fix despite adjustment votes."""
        f = _make_finding("f1", Severity.CRITICAL)
        f.category = Category.SECURITY
        f.suggested_fix = "sanitize input"
        review = _make_review([f])
        all_assessments = [
            (
                "a1",
                [
                    {
                        "id": "f1",
                        "valid": True,
                        "rank": 1,
                        "adjusted_severity": "nitpick",
                        "fix_ok": False,
                    }
                ],
            ),
            (
                "a2",
                [
                    {
                        "id": "f1",
                        "valid": True,
                        "rank": 1,
                        "adjusted_severity": "nitpick",
                        "fix_ok": False,
                    }
                ],
            ),
        ]
        result = apply_cross_review(review, all_assessments)
        assert len(result.findings) == 1
        assert result.findings[0].severity == Severity.CRITICAL
        assert result.findings[0].suggested_fix == "sanitize input"

    def test_legacy_plain_valid_rank_still_parses(self):
        """Old-style assessments without adjusted_severity/fix_ok still apply."""
        review = _make_review([_make_finding("f1", Severity.WARNING)])
        all_assessments = [("a1", [{"id": "f1", "valid": True, "rank": 1}])]
        result = apply_cross_review(review, all_assessments)
        assert len(result.findings) == 1
        assert result.findings[0].severity == Severity.WARNING

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

    def test_critical_security_finding_dropped_by_unanimous_rejection(self):
        """Critical security findings are dropped when ALL agents explicitly reject them."""
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
        assert "sec1" not in result_ids, (
            "Critical security finding must be dropped when all agents reject it"
        )

    def test_critical_security_finding_survives_with_one_valid_vote(self):
        """Critical security findings survive cross-review when at least one agent validates them."""
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
                [{"id": "sec1", "valid": True, "rank": 1}, {"id": "f2", "valid": True, "rank": 2}],
            ),
            (
                "a2",
                [{"id": "sec1", "valid": False, "rank": 5}, {"id": "f2", "valid": True, "rank": 1}],
            ),
        ]
        result = apply_cross_review(review, all_assessments, min_validation_agreement=0.5)
        result_ids = [f.id for f in result.findings]
        assert "sec1" in result_ids, (
            "Critical security finding must survive when at least one agent validates it"
        )
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

    def test_prompt_contains_refute_and_reachability_language(self):
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
        prompt = get_cross_review_prompt(context, review, "diff")
        lowered = prompt.lower()
        assert "refute" in lowered
        assert "reachable" in lowered
        assert "concrete" in lowered
        assert "adjusted_severity" in prompt
        assert "fix_ok" in prompt


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


def _make_raw_finding(
    severity: str = "warning",
    confidence: float = 0.8,
    file_path: str = "src/foo.py",
    title: str = "Test issue",
    line_start: int = 10,
) -> dict:
    return {
        "file_path": file_path,
        "line_start": line_start,
        "severity": severity,
        "category": "logic",
        "title": title,
        "description": "Description",
        "confidence": confidence,
    }


class TestThresholdsForRun:
    """Tests for _thresholds_for_run conservative-floor fallback."""

    def test_cross_review_active_returns_base_unchanged(self):
        base = {
            Severity.CRITICAL: 0.3,
            Severity.WARNING: 0.4,
            Severity.SUGGESTION: 0.5,
            Severity.NITPICK: 0.6,
        }
        assert _thresholds_for_run(base, cross_review_active=True) is base

    def test_cross_review_inactive_raises_to_conservative_floors(self):
        base = {
            Severity.CRITICAL: 0.3,
            Severity.WARNING: 0.4,
            Severity.SUGGESTION: 0.5,
            Severity.NITPICK: 0.6,
        }
        result = _thresholds_for_run(base, cross_review_active=False)
        assert result[Severity.CRITICAL] == _NO_CROSS_REVIEW_THRESHOLDS[Severity.CRITICAL]
        assert result[Severity.WARNING] == _NO_CROSS_REVIEW_THRESHOLDS[Severity.WARNING]
        assert result[Severity.SUGGESTION] == _NO_CROSS_REVIEW_THRESHOLDS[Severity.SUGGESTION]
        assert result[Severity.NITPICK] == _NO_CROSS_REVIEW_THRESHOLDS[Severity.NITPICK]

    def test_base_above_conservative_is_kept(self):
        base = {Severity.CRITICAL: 0.9}
        result = _thresholds_for_run(base, cross_review_active=False)
        assert result[Severity.CRITICAL] == 0.9


class TestConfidenceFiltering:
    """Tests for confidence-based filtering in aggregate_findings."""

    def test_default_thresholds_exist(self):
        assert Severity.CRITICAL in CONFIDENCE_THRESHOLDS
        assert Severity.WARNING in CONFIDENCE_THRESHOLDS
        assert Severity.SUGGESTION in CONFIDENCE_THRESHOLDS
        assert Severity.NITPICK in CONFIDENCE_THRESHOLDS

    def test_default_threshold_values(self):
        assert CONFIDENCE_THRESHOLDS[Severity.CRITICAL] == 0.3
        assert CONFIDENCE_THRESHOLDS[Severity.WARNING] == 0.4
        assert CONFIDENCE_THRESHOLDS[Severity.SUGGESTION] == 0.5
        assert CONFIDENCE_THRESHOLDS[Severity.NITPICK] == 0.6

    def test_high_confidence_findings_kept(self):
        """Findings at or above their severity threshold are kept."""
        all_findings = [
            ("agent-1", [_make_raw_finding("critical", 0.95)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 1

    def test_low_confidence_critical_dropped(self):
        """Critical finding below 0.3 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("critical", 0.2)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_warning_dropped(self):
        """Warning finding below 0.4 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("warning", 0.3)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_suggestion_dropped(self):
        """Suggestion finding below 0.5 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("suggestion", 0.4)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_low_confidence_nitpick_dropped(self):
        """Nitpick finding below 0.6 confidence is dropped."""
        all_findings = [
            ("agent-1", [_make_raw_finding("nitpick", 0.5)], "summary"),
        ]
        result = aggregate_findings(all_findings, "test/repo", 1)
        assert len(result.findings) == 0

    def test_exact_threshold_kept(self):
        """Findings exactly at the threshold are kept (>= comparison)."""
        all_findings = [
            (
                "agent-1",
                [
                    _make_raw_finding("critical", 0.3, line_start=10),
                    _make_raw_finding("warning", 0.4, title="Warning issue", line_start=40),
                    _make_raw_finding("suggestion", 0.5, title="Suggestion issue", line_start=70),
                    _make_raw_finding("nitpick", 0.6, title="Nit: Style issue", line_start=100),
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
                    _make_raw_finding("critical", 0.95, line_start=10),
                    _make_raw_finding(
                        "nitpick", 0.5, title="Nit: Low confidence nit", line_start=40
                    ),
                    _make_raw_finding("warning", 0.9, title="High conf warning", line_start=70),
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


class TestAdaptiveFindingCap:
    """Tests for _cap_findings — the adaptive, PR-size-scaled finding cap."""

    def test_noop_when_under_cap(self):
        findings = [_make_consolidated(severity=Severity.WARNING) for _ in range(3)]
        out = _cap_findings(findings, total_lines=0)  # N=5
        assert out == findings

    def test_caps_to_n_for_small_pr(self):
        findings = [_make_consolidated(severity=Severity.SUGGESTION) for _ in range(8)]
        out = _cap_findings(findings, total_lines=0)  # N=5
        assert len(out) == 5

    def test_n_scales_with_pr_size(self):
        many = [_make_consolidated(severity=Severity.SUGGESTION) for _ in range(30)]
        assert len(_cap_findings(many, total_lines=500)) == 10  # N = 500//100+5
        assert len(_cap_findings(many, total_lines=2000)) == 20  # N capped at 20

    def test_criticals_never_dropped_even_beyond_cap(self):
        criticals = [_make_consolidated(severity=Severity.CRITICAL) for _ in range(7)]
        warnings = [_make_consolidated(severity=Severity.WARNING) for _ in range(3)]
        out = _cap_findings(criticals + warnings, total_lines=0)  # N=5
        assert sum(1 for f in out if f.severity == Severity.CRITICAL) == 7
        assert all(f.severity == Severity.CRITICAL for f in out)
        assert len(out) == 7  # all criticals, zero non-criticals leaked through

    def test_keeps_highest_priority_non_criticals(self):
        critical = _make_consolidated(severity=Severity.CRITICAL, confidence=0.9)
        hi = _make_consolidated(severity=Severity.WARNING, confidence=0.95, title="hi")
        nits = [_make_consolidated(severity=Severity.NITPICK, confidence=0.5) for _ in range(6)]
        out = _cap_findings([critical, hi, *nits], total_lines=0)  # N=5
        assert len(out) == 5
        assert critical in out
        assert hi in out


class TestTruncateToByteLimit:
    """Regression for the second byte-vs-char truncation site (#56), the
    neighbor-file fetch in _prepare_shared_context."""

    def test_under_limit_unchanged(self):
        assert _truncate_to_byte_limit("hello", 100) == "hello"

    def test_ascii_capped_to_byte_count(self):
        assert _truncate_to_byte_limit("a" * 50, 10) == "a" * 10

    def test_multibyte_capped_on_byte_boundary(self):
        out = _truncate_to_byte_limit("😀" * 10, 10, marker="\n[truncated]")  # 40 bytes
        body = out[: -len("\n[truncated]")]
        assert len(body.encode("utf-8")) <= 10
        assert "�" not in body  # no dangling partial multi-byte char
        assert out.endswith("\n[truncated]")


class TestCrossAgentRoutesThroughClient:
    """#54: cross-review must go through AnthropicClient, not client._sdk."""

    @pytest.mark.asyncio
    async def test_uses_complete_simple_with_config_model(self):
        from unittest.mock import AsyncMock, MagicMock

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.review import _run_single_cross_agent

        client = MagicMock()
        client.config = AnthropicApiConfig(api_key="sk", default_model="claude-sonnet-4-6")
        client.complete_simple = AsyncMock(return_value="{}")

        name, _assessments = await _run_single_cross_agent(
            client, "the cross prompt", "security-reviewer", None
        )

        assert name == "security-reviewer"
        client.complete_simple.assert_awaited_once()
        kw = client.complete_simple.call_args.kwargs
        assert kw["model"] == "claude-sonnet-4-6"  # from config, not hardcoded
        assert "the cross prompt" in kw["user"]
        # The positive assertion above already proves routing through the wrapper;
        # a `not client._sdk...called` check on a bare MagicMock is vacuous (the
        # attribute auto-creates), so it's intentionally omitted. A reintroduced
        # _sdk bypass would call _sdk instead of complete_simple, failing
        # assert_awaited_once() above.


def _file_diff(path: str, adds: int, dels: int, symbol: str | None = None) -> str:
    """Synthesize a single-file unified-diff section with `adds`+`dels` changed lines."""
    ctx = f" fn {symbol}(&self)" if symbol else ""
    lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        f"@@ -1,{dels + 1} +1,{adds + 1} @@{ctx}",
    ]
    lines += [f"+added {i}" for i in range(adds)]
    lines += [f"-removed {i}" for i in range(dels)]
    return "\n".join(lines) + "\n"


class TestBuildShards:
    """Tests for _build_shards directory-grouped sharding."""

    def _big_diff(self) -> tuple[dict[str, str], str]:
        """33 files / 2583 changed lines across several dirs and crates."""
        specs = {
            "crates/node": 8,
            "crates/store": 8,
            "crates/network": 5,
            "src": 6,
            "server": 6,
        }
        files: dict[str, str] = {}
        parts: list[str] = []
        # 39 add / 39 del = 78 changed lines per file; 33 files = 2574 lines.
        for group, count in specs.items():
            for i in range(count):
                path = f"{group}/file_{i}.rs"
                files[path] = f"// contents of {path}\n"
                parts.append(_file_diff(path, 39, 39, symbol=f"do_{i}"))
        return files, "".join(parts)

    def test_deterministic(self):
        from ai_reviewer.review import _build_shards

        files, diff = self._big_diff()
        a = _build_shards(files, diff)
        b = _build_shards(files, diff)
        assert [s.label for s in a] == [s.label for s in b]
        assert [s.diff for s in a] == [s.diff for s in b]

    def test_big_pr_yields_four_to_six_shards(self):
        from ai_reviewer.review import _build_shards

        files, diff = self._big_diff()
        shards = _build_shards(files, diff)
        assert 4 <= len(shards) <= 6

    def test_respects_target_or_single_group(self):
        from ai_reviewer.review import _SHARD_TARGET_LINES, _build_shards, _changed_line_count

        files, diff = self._big_diff()
        for shard in _build_shards(files, diff):
            over_budget = _changed_line_count(shard.diff) > _SHARD_TARGET_LINES
            # Over-budget shards are only allowed when they hold a single group.
            assert not over_budget or "," not in shard.label

    def test_never_splits_a_group(self):
        from ai_reviewer.review import _build_shards

        files, diff = self._big_diff()
        # Every file of crates/node must land in exactly one shard.
        node_files = [p for p in files if p.startswith("crates/node/")]
        for path in node_files:
            containing = [s for s in _build_shards(files, diff) if path in s.files]
            assert len(containing) == 1
        # And all crates/node files share that one shard.
        node_shards = {
            id(s) for s in _build_shards(files, diff) for p in node_files if p in s.files
        }
        assert len(node_shards) == 1

    def test_caps_at_max_with_repack(self):
        from ai_reviewer.review import _SHARD_MAX, _build_shards

        # 64 groups of 100 changed lines each: budget 600 would yield 11 shards;
        # the re-pack at total/8 must bring it down to <= _SHARD_MAX.
        files: dict[str, str] = {}
        parts: list[str] = []
        for g in range(64):
            path = f"dir_{g:02d}/file.py"
            files[path] = "x\n"
            parts.append(_file_diff(path, 50, 50))
        shards = _build_shards(files, "".join(parts))
        assert len(shards) <= _SHARD_MAX


class TestBuildPrMapBlock:
    """Tests for build_pr_map_block."""

    def test_contains_counts_and_symbols(self):
        from ai_reviewer.context.builder import build_pr_map_block

        diff = _file_diff("src/auth.py", 5, 2, symbol="login") + _file_diff("src/util.py", 1, 0)
        files = {"src/auth.py": "", "src/util.py": ""}
        block = build_pr_map_block(files, diff)
        assert "src/auth.py (+5/-2)" in block
        assert "src/util.py (+1/-0)" in block
        assert "login" in block
        assert "2 file(s)" in block
        assert "+6/-2" in block  # totals across files

    def test_capped_size(self):
        from ai_reviewer.context.builder import _PR_MAP_MAX_BYTES, build_pr_map_block

        files: dict[str, str] = {}
        parts: list[str] = []
        for i in range(300):
            path = f"pkg/module_{i:03d}/handler.py"
            files[path] = ""
            parts.append(_file_diff(path, 3, 1, symbol=f"handle_{i}"))
        block = build_pr_map_block(files, "".join(parts))
        assert len(block.encode("utf-8")) <= _PR_MAP_MAX_BYTES + 40


async def _run_review_pr_mocked(
    additions: int, deletions: int, changed_files_count: int, num_agents: int = 3
):
    """Drive review_pr with all I/O + agent drivers mocked; return (safe, sharded) mocks."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import ai_reviewer.review as rev
    from ai_reviewer.config import AnthropicApiConfig
    from ai_reviewer.models.review import AgentReview

    gh = MagicMock()
    pr = MagicMock()
    pr.head.sha = "sha"
    pr.title = "t"
    pr.body = "b"
    gh.get_pull_request.return_value = pr
    gh.get_repo.return_value = MagicMock()
    gh.get_pr_diff.return_value = ""
    gh.get_changed_files.return_value = {}
    gh.load_repo_config.return_value = {}
    gh.load_repo_conventions.return_value = ""
    gh.build_review_context.return_value = ReviewContext(
        repo_name="o/r",
        pr_number=1,
        pr_title="t",
        pr_description="",
        base_branch="main",
        head_branch="f",
        author="a",
        changed_files_count=changed_files_count,
        additions=additions,
        deletions=deletions,
    )

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    fake = AgentReview(
        agent_id="a", agent_type="t", focus_areas=[], findings=[], summary="ok", review_time_ms=0
    )

    with (
        patch.object(rev, "GitHubClient", return_value=gh),
        patch.object(rev, "AnthropicClient", return_value=client),
        patch.object(rev, "_prepare_shared_context", new=AsyncMock(return_value=([], [], set()))),
        patch.object(rev, "_run_agent_safe", new=AsyncMock(return_value=fake)) as safe,
        patch.object(rev, "_run_agent_sharded", new=AsyncMock(return_value=fake)) as sharded,
    ):
        await rev.review_pr(
            repo="o/r",
            pr_number=1,
            anthropic_cfg=AnthropicApiConfig(api_key="sk"),
            github_token="tok",
            num_agents=num_agents,
        )
    return safe, sharded


@pytest.mark.asyncio
async def test_tool_registry_receives_trimmed_paths():
    """review_pr must pass the excerpted paths from _prepare_shared_context into ToolRegistry."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import ai_reviewer.review as rev
    from ai_reviewer.config import AnthropicApiConfig
    from ai_reviewer.models.review import AgentReview

    gh = MagicMock()
    pr = MagicMock()
    pr.head.sha = "sha"
    pr.title = "t"
    pr.body = "b"
    gh.get_pull_request.return_value = pr
    gh.get_repo.return_value = MagicMock()
    gh.get_pr_diff.return_value = ""
    gh.get_changed_files.return_value = {}
    gh.load_repo_config.return_value = {}
    gh.load_repo_conventions.return_value = ""
    gh.build_review_context.return_value = ReviewContext(
        repo_name="o/r",
        pr_number=1,
        pr_title="t",
        pr_description="",
        base_branch="main",
        head_branch="f",
        author="a",
        changed_files_count=1,
        additions=10,
        deletions=10,
    )

    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    fake = AgentReview(
        agent_id="a", agent_type="t", focus_areas=[], findings=[], summary="ok", review_time_ms=0
    )
    trimmed = {"big.py"}

    with (
        patch.object(rev, "GitHubClient", return_value=gh),
        patch.object(rev, "AnthropicClient", return_value=client),
        patch.object(rev, "_prepare_shared_context", new=AsyncMock(return_value=([], [], trimmed))),
        patch.object(rev, "_run_agent_safe", new=AsyncMock(return_value=fake)),
        patch.object(rev, "ToolRegistry") as registry_cls,
    ):
        await rev.review_pr(
            repo="o/r",
            pr_number=1,
            anthropic_cfg=AnthropicApiConfig(api_key="sk"),
            github_token="tok",
            num_agents=1,
        )

    registry_cls.assert_called()
    assert registry_cls.call_args.kwargs["trimmed_paths"] == trimmed


class TestShardGate:
    """Tests for the size gate that selects the sharded vs single-conversation path."""

    @pytest.mark.asyncio
    async def test_small_pr_uses_single_path(self):
        safe, sharded = await _run_review_pr_mocked(
            additions=500, deletions=499, changed_files_count=19
        )
        sharded.assert_not_called()
        safe.assert_called()

    @pytest.mark.asyncio
    async def test_over_line_gate_uses_sharded_path(self):
        safe, sharded = await _run_review_pr_mocked(
            additions=501, deletions=500, changed_files_count=19
        )
        sharded.assert_called()
        safe.assert_not_called()

    @pytest.mark.asyncio
    async def test_over_file_gate_uses_sharded_path(self):
        safe, sharded = await _run_review_pr_mocked(
            additions=10, deletions=10, changed_files_count=21
        )
        sharded.assert_called()
        safe.assert_not_called()


def _make_shards(n: int):
    from ai_reviewer.review import Shard

    return [
        Shard(files={f"g{k}/a.py": "x"}, diff=_file_diff(f"g{k}/a.py", 1, 0), label=f"g{k}")
        for k in range(n)
    ]


def _make_review_finding():
    from ai_reviewer.models.findings import Category, ReviewFinding, Severity

    return ReviewFinding(
        file_path="g0/a.py",
        line_start=1,
        line_end=None,
        severity=Severity.WARNING,
        category=Category.LOGIC,
        title="Issue",
        description="d",
        suggested_fix=None,
        confidence=0.8,
    )


def _sharded_kwargs(cls, shards, client):
    """Common keyword args for _run_agent_sharded in the failure-semantics tests."""
    from unittest.mock import MagicMock

    from ai_reviewer.config import AnthropicApiConfig

    return {
        "cls": cls,
        "agent_name": "logic-reviewer",
        "agent_index": 0,
        "client": client,
        "system_blocks": [],
        "shards": shards,
        "pr_map": "## PR map\n\nTotal: +1/-0 across 1 file(s)",
        "pr_title": "t",
        "pr_body": "b",
        "max_total_chars": 600_000,
        "allow_tools": False,
        "max_tokens": 8192,
        "temperature": 0.3,
        "thinking_enabled": None,
        "model": None,
        "session": MagicMock(),
        "gh": MagicMock(),
        "anthropic_cfg": AnthropicApiConfig(api_key="sk"),
        "context": MagicMock(),
        "on_status": None,
    }


def _fake_agent_cls(queue):
    """A ReviewAgent stand-in whose review() pops queued results/exceptions per shard."""

    class _FakeAgent:
        AGENT_TYPE = "logic-reviewer"
        FOCUS_AREAS = ["logic"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def review(self, diff, file_contents, context):
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    return _FakeAgent


class TestRawToReviewFinding:
    """Malformed cross-shard findings are dropped, not dressed up with defaults."""

    def test_missing_severity_dropped(self):
        from ai_reviewer.review import _raw_to_review_finding

        raw = {"file_path": "a.py", "line_start": 3, "category": "logic", "title": "t"}
        assert _raw_to_review_finding(raw) is None

    def test_missing_category_dropped(self):
        from ai_reviewer.review import _raw_to_review_finding

        raw = {"file_path": "a.py", "line_start": 3, "severity": "warning", "title": "t"}
        assert _raw_to_review_finding(raw) is None

    def test_missing_title_dropped(self):
        from ai_reviewer.review import _raw_to_review_finding

        raw = {"file_path": "a.py", "line_start": 3, "severity": "warning", "category": "logic"}
        assert _raw_to_review_finding(raw) is None

    def test_complete_finding_parses(self):
        from ai_reviewer.review import _raw_to_review_finding

        raw = {
            "file_path": "a.py",
            "line_start": 3,
            "severity": "warning",
            "category": "logic",
            "title": "stale caller",
            "description": "d",
        }
        f = _raw_to_review_finding(raw)
        assert f is not None and f.severity == Severity.WARNING


class TestRunAgentSharded:
    """Failure and coverage semantics of _run_agent_sharded."""

    def _client(self):
        from unittest.mock import AsyncMock, MagicMock

        client = MagicMock()
        client.config.default_model = "m"
        client.complete_simple = AsyncMock(return_value='{"findings": [], "summary": "none"}')
        return client

    @pytest.mark.asyncio
    async def test_partial_failure_still_succeeds_with_coverage_note(self):
        from ai_reviewer.agents.anthropic_client import INCOMPLETE_SUMMARY_MARKERS
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.review import _run_agent_sharded

        finding = _make_review_finding()
        queue = [
            AgentReview("a", "t", [], [finding], "shard0 ok", 0),
            RuntimeError("shard1 boom"),
            AgentReview("a", "t", [], [], "shard2 ok", 0),
        ]
        cls = _fake_agent_cls(queue)
        client = self._client()
        result = await _run_agent_sharded(**_sharded_kwargs(cls, _make_shards(3), client))

        assert isinstance(result, AgentReview)
        assert finding in result.findings
        assert "Coverage gap" in result.summary
        assert "g1" in result.summary
        # Partial-failure note must not read as an incomplete-review marker.
        assert not any(m in result.summary for m in INCOMPLETE_SUMMARY_MARKERS)
        # Merged findings were non-empty, so the cross-shard pass ran.
        client.complete_simple.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_incomplete_marker_shard_counts_as_failure_without_leaking_marker(self):
        from ai_reviewer.agents.anthropic_client import (
            INCOMPLETE_SUMMARY_MARKERS,
            TOOL_LOOP_CAP_MARKER,
        )
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.review import _run_agent_sharded

        queue = [
            AgentReview("a", "t", [], [_make_review_finding()], "ok", 0),
            AgentReview("a", "t", [], [], TOOL_LOOP_CAP_MARKER, 0),
        ]
        cls = _fake_agent_cls(queue)
        result = await _run_agent_sharded(**_sharded_kwargs(cls, _make_shards(2), self._client()))

        assert isinstance(result, AgentReview)
        assert "Coverage gap" in result.summary
        assert not any(m in result.summary for m in INCOMPLETE_SUMMARY_MARKERS)

    @pytest.mark.asyncio
    async def test_all_shards_fail_marks_agent_failed(self):
        from ai_reviewer.review import _run_agent_sharded

        queue = [RuntimeError("boom") for _ in range(3)]
        cls = _fake_agent_cls(queue)
        result = await _run_agent_sharded(**_sharded_kwargs(cls, _make_shards(3), self._client()))

        assert isinstance(result, Exception)


def test_aggregate_findings_marks_incomplete_agent_as_failed():
    from ai_reviewer.agents.anthropic_client import PARSE_ERROR_MARKER, TOOL_LOOP_CAP_MARKER
    from ai_reviewer.review import aggregate_findings

    review = aggregate_findings(
        [
            ("security-reviewer", [], TOOL_LOOP_CAP_MARKER),
            ("logic-reviewer", [], PARSE_ERROR_MARKER),
            ("patterns-reviewer", [], "Reviewed thoroughly, code looks good."),
        ],
        "o/r",
        1,
    )
    assert "security-reviewer" in review.failed_agents
    assert "logic-reviewer" in review.failed_agents
    assert "patterns-reviewer" not in review.failed_agents


def test_all_agents_failed_posts_honest_block_never_approve():
    """When every agent fails, the formatter must produce a visible 'could not
    complete' body and the review action must never be APPROVE."""
    from ai_reviewer.github.formatter import GitHubFormatter

    review = ConsolidatedReview(
        id="review-x",
        created_at=datetime.now(),
        repo="test/repo",
        pr_number=7,
        findings=[],
        summary="all failed",
        agent_count=3,
        review_quality_score=0.0,
        total_review_time_ms=0,
        failed_agents=["security-reviewer", "logic-reviewer", "patterns-reviewer"],
    )
    assert review.all_agents_failed

    formatter = GitHubFormatter("AI Code Reviewer")
    body = formatter.format_all_agents_failed(review)
    assert "Review could not complete" in body
    assert "all 3 agent(s) failed" in body

    assert formatter.get_review_action(review, allow_approve=True) != "APPROVE"


class TestExtraReviewerUsersWiring:
    """config.github.extra_reviewer_users must reach the GitHubClient (was dead config)."""

    @pytest.mark.asyncio
    async def test_review_pr_passes_extra_reviewer_users(self):
        from unittest.mock import MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.review import review_pr

        config = MagicMock()
        config.github.extra_reviewer_users = ["meroreviewer[bot]"]

        class _Stop(Exception):
            pass

        with patch("ai_reviewer.review.GitHubClient") as gh_cls:
            gh_cls.return_value.get_pull_request.side_effect = _Stop()
            with pytest.raises(_Stop):
                await review_pr(
                    repo="o/r",
                    pr_number=1,
                    anthropic_cfg=AnthropicApiConfig(api_key="sk-test"),
                    github_token="ghp_x",
                    config=config,
                )
            assert gh_cls.call_args.kwargs["extra_reviewer_users"] == ["meroreviewer[bot]"]

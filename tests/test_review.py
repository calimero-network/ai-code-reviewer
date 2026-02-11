"""Tests for the review module, particularly aggregate_findings."""

import pytest

from ai_reviewer.review import (
    _cluster_raw_findings,
    _detect_pr_type,
    _raw_findings_similar,
    aggregate_findings,
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

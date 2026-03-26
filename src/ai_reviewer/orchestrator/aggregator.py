"""Review aggregator for combining multiple agent reviews."""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

from ai_reviewer.models.findings import (
    ConsolidatedFinding,
    ReviewFinding,
    Severity,
)
from ai_reviewer.models.review import AgentReview, ConsolidatedReview
from ai_reviewer.review import compute_quality_score

logger = logging.getLogger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for the aggregator."""

    similarity_threshold: float = 0.85
    min_consensus_for_critical: float = 0.5
    use_embeddings: bool = False  # For future enhancement


class ReviewAggregator:
    """Combines multiple agent reviews into a unified review."""

    def __init__(self, config: AggregatorConfig | None = None) -> None:
        """Initialize the aggregator.

        Args:
            config: Optional configuration
        """
        self.config = config or AggregatorConfig()

    def aggregate(
        self,
        reviews: list[AgentReview],
        repo: str = "unknown",
        pr_number: int = 0,
    ) -> ConsolidatedReview:
        """Merge findings from multiple agents into a unified review.

        Algorithm:
        1. Extract all findings from all agents
        2. Cluster similar findings using text similarity
        3. For each cluster, compute consensus score
        4. Merge descriptions from agreeing agents
        5. Rank by severity × consensus score

        Args:
            reviews: List of agent reviews to aggregate
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Consolidated review with merged findings
        """
        if not reviews:
            return self._empty_review(repo, pr_number)

        # Extract and tag all findings with their agent
        tagged_findings = self._extract_tagged_findings(reviews)

        if not tagged_findings:
            return self._clean_review(reviews, repo, pr_number)

        # Cluster similar findings
        clusters = self._cluster_findings(tagged_findings)

        # Merge each cluster into a consolidated finding
        consolidated_findings = [self._merge_cluster(cluster, len(reviews)) for cluster in clusters]

        # Sort by priority (severity × consensus × confidence)
        consolidated_findings.sort(key=lambda f: f.priority_score, reverse=True)

        # Generate summary
        summary = self._generate_summary(consolidated_findings, len(reviews))

        quality_score = compute_quality_score(consolidated_findings, len(reviews), total_lines=0)

        # Calculate total review time
        total_time = sum(r.review_time_ms for r in reviews)

        return ConsolidatedReview(
            id=f"review-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(),
            repo=repo,
            pr_number=pr_number,
            findings=consolidated_findings,
            summary=summary,
            agent_count=len(reviews),
            review_quality_score=quality_score,
            total_review_time_ms=total_time,
            agent_reviews=reviews,
        )

    def _extract_tagged_findings(
        self, reviews: list[AgentReview]
    ) -> list[tuple[str, ReviewFinding]]:
        """Extract all findings tagged with their agent ID."""
        tagged = []
        for review in reviews:
            for finding in review.findings:
                tagged.append((review.agent_id, finding))
        return tagged

    def _cluster_findings(
        self, tagged_findings: list[tuple[str, ReviewFinding]]
    ) -> list[list[tuple[str, ReviewFinding]]]:
        """Cluster similar findings together."""
        if not tagged_findings:
            return []

        clusters: list[list[tuple[str, ReviewFinding]]] = []
        used = set()

        for i, (agent_i, finding_i) in enumerate(tagged_findings):
            if i in used:
                continue

            # Start new cluster
            cluster = [(agent_i, finding_i)]
            used.add(i)

            # Find similar findings
            for j, (agent_j, finding_j) in enumerate(tagged_findings):
                if j in used:
                    continue

                if self._are_similar(finding_i, finding_j):
                    cluster.append((agent_j, finding_j))
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _are_similar(self, f1: ReviewFinding, f2: ReviewFinding) -> bool:
        """Check if two findings are similar enough to merge."""
        # Must be same file
        if f1.file_path != f2.file_path:
            return False

        # Must be same category
        if f1.category != f2.category:
            return False

        # Lines must overlap or be close
        if not self._lines_overlap(f1, f2):
            return False

        # Title/description similarity
        title_sim = self._text_similarity(f1.title, f2.title)
        desc_sim = self._text_similarity(f1.description, f2.description)

        # Combined similarity
        combined = (title_sim * 0.6) + (desc_sim * 0.4)
        return combined >= self.config.similarity_threshold

    def _lines_overlap(self, f1: ReviewFinding, f2: ReviewFinding) -> bool:
        """Check if line ranges overlap or are close."""
        # Get ranges
        start1, end1 = f1.line_start, f1.line_end or f1.line_start
        start2, end2 = f2.line_start, f2.line_end or f2.line_start

        # Allow some tolerance (within 5 lines)
        tolerance = 5
        return not (end1 + tolerance < start2 or end2 + tolerance < start1)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _merge_cluster(
        self, cluster: list[tuple[str, ReviewFinding]], total_agents: int
    ) -> ConsolidatedFinding:
        """Merge a cluster of similar findings into one."""
        agents = [agent for agent, _ in cluster]
        findings = [finding for _, finding in cluster]

        # Use the finding with highest confidence as base
        base_finding = max(findings, key=lambda f: f.confidence)

        # Merge descriptions if different
        unique_descriptions = list({f.description for f in findings})
        if len(unique_descriptions) > 1:
            description = (
                base_finding.description
                + "\n\n**Also noted:**\n"
                + "\n".join(f"- {d}" for d in unique_descriptions if d != base_finding.description)
            )
        else:
            description = base_finding.description

        # Merge suggested fixes
        suggested_fix = base_finding.suggested_fix
        other_fixes = [
            f.suggested_fix
            for f in findings
            if f.suggested_fix and f.suggested_fix != suggested_fix
        ]
        if other_fixes:
            suggested_fix = (
                (suggested_fix or "")
                + "\n\n**Alternative suggestions:**\n"
                + "\n".join(
                    f"- {fix}"
                    for fix in other_fixes[:2]  # Limit to 2 alternatives
                )
            )

        # Use most severe rating
        severity = max(findings, key=lambda f: list(Severity).index(f.severity)).severity

        # Consensus score
        consensus = len(cluster) / total_agents

        # Average confidence
        avg_confidence = sum(f.confidence for f in findings) / len(findings)

        return ConsolidatedFinding(
            id=f"finding-{hashlib.md5(f'{base_finding.file_path}:{base_finding.line_start}:{base_finding.title}'.encode()).hexdigest()[:8]}",
            file_path=base_finding.file_path,
            line_start=base_finding.line_start,
            line_end=base_finding.line_end,
            severity=severity,
            category=base_finding.category,
            title=base_finding.title,
            description=description,
            suggested_fix=suggested_fix,
            consensus_score=consensus,
            agreeing_agents=agents,
            confidence=avg_confidence,
            original_findings=findings,
        )

    def _generate_summary(self, findings: list[ConsolidatedFinding], agent_count: int) -> str:
        """Generate a summary of the review."""
        if not findings:
            return f"✅ No issues found by {agent_count} agents."

        by_severity: dict[Severity, int] = {}
        for f in findings:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        parts = []
        if Severity.CRITICAL in by_severity:
            parts.append(f"🔴 {by_severity[Severity.CRITICAL]} critical")
        if Severity.WARNING in by_severity:
            parts.append(f"🟡 {by_severity[Severity.WARNING]} warnings")
        if Severity.SUGGESTION in by_severity:
            parts.append(f"💡 {by_severity[Severity.SUGGESTION]} suggestions")
        if Severity.NITPICK in by_severity:
            parts.append(f"📝 {by_severity[Severity.NITPICK]} nitpicks")

        return f"Found {', '.join(parts)} across {len(findings)} unique issues."

    def _empty_review(self, repo: str, pr_number: int) -> ConsolidatedReview:
        """Create an empty review (no agents)."""
        return ConsolidatedReview(
            id=f"review-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(),
            repo=repo,
            pr_number=pr_number,
            findings=[],
            summary="⚠️ No agents available for review.",
            agent_count=0,
            review_quality_score=0.0,
            total_review_time_ms=0,
        )

    def _clean_review(
        self, reviews: list[AgentReview], repo: str, pr_number: int
    ) -> ConsolidatedReview:
        """Create a clean review (no findings)."""
        total_time = sum(r.review_time_ms for r in reviews)
        return ConsolidatedReview(
            id=f"review-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now(),
            repo=repo,
            pr_number=pr_number,
            findings=[],
            summary=f"✅ No issues found by {len(reviews)} agents. LGTM!",
            agent_count=len(reviews),
            review_quality_score=compute_quality_score([], len(reviews), total_lines=0),
            total_review_time_ms=total_time,
            agent_reviews=reviews,
        )

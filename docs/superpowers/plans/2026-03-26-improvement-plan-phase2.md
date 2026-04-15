# Improvement Plan Phase 2 Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 8 medium-effort improvements from the AI Code Reviewer improvement plan (Phase 2): repo config loading, quality scoring overhaul, multi-tier finding hashes, cross-file dedup, convergence detection, severity stabilization, PR size adaptive prompts, and language-specific rules.

**Architecture:** All changes build on Phase 1's config-driven pipeline. New features integrate into existing functions (`aggregate_findings`, `compute_review_delta`, `get_base_prompt`) rather than creating new subsystems. State persistence uses GitHub PR comments as the storage layer (no external DB). Dependencies: P2-3 (hashes) before P2-5 (convergence) and P2-6 (stabilization); P2-1 (repo config) before P2-7 (adaptive prompts).

**Tech Stack:** Python 3.11, PyGithub, pytest, ruff, mypy. Line length 100. Async tests with `@pytest.mark.asyncio`. Coverage via pytest-cov.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/ai_reviewer/github/client.py` | Add `load_repo_config()`, `load_repo_conventions()`, multi-tier hash matching in `compute_review_delta()`, convergence check, review history parsing |
| Modify | `src/ai_reviewer/review.py` | Wire repo config into prompts, new scoring formula, cross-file dedup, PR size classification, language rules |
| Modify | `src/ai_reviewer/models/findings.py` | Add `finding_hash_fuzzy` property |
| Modify | `src/ai_reviewer/models/review.py` | Add `ScoreBreakdown` and `ReviewHistory` dataclasses |
| Modify | `src/ai_reviewer/models/context.py` | Add `repo_config` and `conventions` fields |
| Modify | `src/ai_reviewer/config.py` | Add scoring and convergence config fields |
| Modify | `src/ai_reviewer/github/formatter.py` | Display score breakdown, parse previous score |
| Modify | `src/ai_reviewer/cli.py` | Add `--force-review` flag, wire convergence skip |
| Modify | `src/ai_reviewer/orchestrator/aggregator.py` | Remove `_compute_quality_score()`, delegate to review.py |
| Create | `tests/test_scoring.py` | Tests for new quality scoring formula |
| Modify | `tests/test_models.py` | Fuzzy hash tests for ConsolidatedFinding |
| Modify | `tests/test_github.py` | Fuzzy hash + delta matching tests for PreviousComment and compute_review_delta |
| Modify | `tests/test_review.py` | Cross-file dedup tests |
| Create | `tests/test_convergence.py` | Tests for convergence detection and severity stabilization |
| Create | `tests/test_prompts.py` | Tests for repo config loading, PR classification, language rules |

---

### Task 1: Multi-tier finding hash (P2-3)

This is a dependency for P2-5 and P2-6, so it goes first.

**Files:**
- Modify: `src/ai_reviewer/models/findings.py:82-94`
- Modify: `src/ai_reviewer/github/client.py:498-515`
- Modify: `tests/test_models.py`, `tests/test_github.py`

- [ ] **Step 1: Write failing test for `finding_hash_fuzzy`**

```python
# Append to tests/test_models.py (fuzzy hash tests) and tests/test_github.py (delta tests)
import re

from ai_reviewer.models.findings import (
    Category,
    ConsolidatedFinding,
    Severity,
)


def _make_finding(
    file_path: str = "src/auth.py",
    line_start: int = 10,
    title: str = "SQL Injection Vulnerability",
    severity: Severity = Severity.WARNING,
    category: Category = Category.SECURITY,
) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="test-1",
        file_path=file_path,
        line_start=line_start,
        line_end=None,
        severity=severity,
        category=category,
        title=title,
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=0.9,
    )


class TestFindingHashFuzzy:
    """Tests for the fuzzy hash property on ConsolidatedFinding."""

    def test_fuzzy_hash_is_12_chars_hex(self):
        """Fuzzy hash returns a 12-character hex string."""
        f = _make_finding()
        assert re.fullmatch(r"[a-f0-9]{12}", f.finding_hash_fuzzy)

    def test_fuzzy_hash_ignores_line_number(self):
        """Same file+title at different lines produce same fuzzy hash."""
        f1 = _make_finding(line_start=10)
        f2 = _make_finding(line_start=50)
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_differs_from_primary(self):
        """Fuzzy and primary hashes are different values."""
        f = _make_finding()
        assert f.finding_hash != f.finding_hash_fuzzy

    def test_fuzzy_hash_ignores_minor_title_variation(self):
        """Fuzzy hash matches when titles share the same keywords."""
        f1 = _make_finding(title="SQL Injection Vulnerability Found")
        f2 = _make_finding(title="Found SQL Injection Vulnerability")
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_stable_across_category_changes(self):
        """Fuzzy hash is same regardless of category (not included)."""
        f1 = _make_finding(category=Category.SECURITY)
        f2 = _make_finding(category=Category.PERFORMANCE)
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_differs_for_different_file(self):
        """Different files produce different fuzzy hashes."""
        f1 = _make_finding(file_path="a.py")
        f2 = _make_finding(file_path="b.py")
        assert f1.finding_hash_fuzzy != f2.finding_hash_fuzzy

    def test_primary_hash_unchanged(self):
        """Existing finding_hash behavior is preserved."""
        f1 = _make_finding(line_start=10)
        f2 = _make_finding(line_start=50)
        assert f1.finding_hash != f2.finding_hash  # Primary IS line-sensitive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_models.py::TestFindingHashFuzzy -v --override-ini="addopts="`
Expected: FAIL with `AttributeError: 'ConsolidatedFinding' object has no attribute 'finding_hash_fuzzy'`

- [ ] **Step 3: Implement `finding_hash_fuzzy` property**

In `src/ai_reviewer/models/findings.py`, after the existing `finding_hash` property (line 94), add:

```python
    @property
    def finding_hash_fuzzy(self) -> str:
        """Fuzzy hash ignoring line number and category for cross-run matching.

        Uses file_path + sorted title keywords (4+ chars).
        """
        import hashlib
        import re as _re

        words = sorted(set(_re.findall(r"\b\w{4,}\b", self.title.lower())))
        key = f"{self.file_path or ''}:{':'.join(words[:5])}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_models.py::TestFindingHashFuzzy -v --override-ini="addopts="`
Expected: All 7 tests PASS

- [ ] **Step 5: Add `finding_hash_fuzzy` to `PreviousComment`**

In `src/ai_reviewer/github/client.py`, add a property to `PreviousComment` (around line 81-91):

```python
    @property
    def finding_hash_fuzzy(self) -> str | None:
        """Fuzzy hash for cross-run matching (ignores line, category)."""
        if not self.file_path or not self.title:
            return None
        import hashlib
        import re

        words = sorted(set(re.findall(r"\b\w{4,}\b", self.title.lower())))
        key = f"{self.file_path}:{':'.join(words[:5])}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]
```

- [ ] **Step 6: Wire multi-tier matching into `compute_review_delta()`**

In `compute_review_delta()` (around line 498-515), update the lookup building:

```python
        hash_lookup: dict[str, PreviousComment] = {}
        fuzzy_lookup: dict[str, PreviousComment] = {}
        title_lookup: dict[tuple[str, int, str], PreviousComment] = {}
        for comment in previous_comments:
            if comment.finding_hash:
                hash_lookup[comment.finding_hash] = comment
            fuzzy = comment.finding_hash_fuzzy
            if fuzzy:
                fuzzy_lookup[fuzzy] = comment
            key = (
                comment.file_path,
                comment.line,
                self._normalize_title(comment.title),
            )
            title_lookup[key] = comment
```

Update the matching block to try fuzzy between primary and title:

```python
            matched_comment = hash_lookup.get(finding.finding_hash)
            if matched_comment is None:
                matched_comment = fuzzy_lookup.get(finding.finding_hash_fuzzy)
            if matched_comment is None:
                key = (
                    finding.file_path,
                    finding.line_start,
                    self._normalize_title(finding.title),
                )
                matched_comment = title_lookup.get(key)
```

- [ ] **Step 7: Run all tests**

Run: `PYTHONPATH=src pytest tests/test_models.py tests/test_github.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 8: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/models/findings.py src/ai_reviewer/github/client.py tests/test_models.py tests/test_github.py && ruff format --check src/ai_reviewer/models/findings.py src/ai_reviewer/github/client.py tests/test_models.py tests/test_github.py`
Expected: All checks passed. If not, fix issues (`ruff format <file>` for formatting, manual fix for lint errors) and re-run.

- [ ] **Step 9: Commit**

```bash
git add src/ai_reviewer/models/findings.py src/ai_reviewer/github/client.py tests/test_models.py tests/test_github.py
git commit -m "feat(P2-3): add finding_hash_fuzzy and multi-tier delta matching"
```

---

### Task 2: New quality scoring formula (P2-2)

**Files:**
- Modify: `src/ai_reviewer/review.py:559-660` (aggregate_findings)
- Modify: `src/ai_reviewer/orchestrator/aggregator.py` (remove duplicate scoring)
- Modify: `src/ai_reviewer/models/review.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests for the new scoring formula**

```python
# tests/test_scoring.py
from ai_reviewer.models.findings import (
    Category,
    ConsolidatedFinding,
    Severity,
)
from ai_reviewer.review import compute_quality_score


def _make_finding(
    severity: Severity = Severity.WARNING,
    confidence: float = 0.9,
) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="test-1",
        file_path="src/auth.py",
        line_start=10,
        line_end=None,
        severity=severity,
        category=Category.SECURITY,
        title="Issue",
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=confidence,
    )


class TestComputeQualityScore:
    """Tests for the new composite quality scoring formula."""

    def test_clean_review_scores_high(self):
        """No findings should score 0.85-0.95."""
        score = compute_quality_score([], agent_count=3, total_lines=100)
        assert 0.85 <= score <= 0.95

    def test_clean_review_scales_with_agents(self):
        """More agents on clean review = higher confidence."""
        s1 = compute_quality_score([], agent_count=1, total_lines=100)
        s3 = compute_quality_score([], agent_count=3, total_lines=100)
        assert s3 > s1

    def test_critical_finding_penalizes_heavily(self):
        """A critical finding should significantly lower the score."""
        findings = [_make_finding(severity=Severity.CRITICAL)]
        score = compute_quality_score(findings, agent_count=3, total_lines=500)
        assert score < 0.85

    def test_nitpick_barely_penalizes(self):
        """A single nitpick should barely affect the score."""
        findings = [_make_finding(severity=Severity.NITPICK)]
        score = compute_quality_score(findings, agent_count=3, total_lines=500)
        assert score > 0.90

    def test_density_normalized_by_pr_size(self):
        """Same findings in a large PR should score higher than in a small PR."""
        findings = [_make_finding(severity=Severity.WARNING) for _ in range(3)]
        small_pr = compute_quality_score(findings, agent_count=3, total_lines=50)
        large_pr = compute_quality_score(findings, agent_count=3, total_lines=5000)
        assert large_pr > small_pr

    def test_score_never_negative(self):
        """Score should never go below 0.0 even with many critical findings."""
        findings = [_make_finding(severity=Severity.CRITICAL) for _ in range(10)]
        score = compute_quality_score(findings, agent_count=3, total_lines=50)
        assert score >= 0.0

    def test_score_capped_at_095(self):
        """Clean review score should never exceed 0.95."""
        score = compute_quality_score([], agent_count=10, total_lines=10)
        assert score <= 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_scoring.py -v --override-ini="addopts="`
Expected: FAIL with `ImportError: cannot import name 'compute_quality_score' from 'ai_reviewer.review'`

- [ ] **Step 3: Implement `compute_quality_score()` in review.py**

Add as a module-level function before `aggregate_findings()`:

```python
def compute_quality_score(
    findings: list[ConsolidatedFinding],
    agent_count: int,
    total_lines: int,
) -> float:
    """Composite quality score factoring severity, density, consensus, agents.

    Returns a float between 0.0 and 0.95.
    """
    if not findings:
        base = 0.85
        agent_bonus = min(0.10, (agent_count - 1) * 0.05)
        return min(0.95, base + agent_bonus)

    severity_penalty = {
        Severity.CRITICAL: 0.15,
        Severity.WARNING: 0.06,
        Severity.SUGGESTION: 0.02,
        Severity.NITPICK: 0.005,
    }
    total_penalty = sum(
        severity_penalty.get(f.severity, 0.02) * f.confidence
        for f in findings
    )

    density = len(findings) / max(total_lines / 100, 1)
    density_penalty = min(0.15, density * 0.03)

    avg_consensus = sum(f.consensus_score for f in findings) / len(findings)
    consensus_factor = 0.8 + (avg_consensus * 0.2)
    agent_factor = min(1.0, agent_count / 3)

    raw_score = max(0.0, 1.0 - total_penalty - density_penalty)
    return round(min(0.95, raw_score * consensus_factor * agent_factor), 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_scoring.py -v --override-ini="addopts="`
Expected: All 7 tests PASS

- [ ] **Step 5: Wire new scoring into `aggregate_findings()`**

Update `aggregate_findings()` signature to accept `total_lines: int = 0`. Replace the inline quality score calculation with:

```python
    quality_score = compute_quality_score(consolidated, total_agents, total_lines)
```

In `review_pr_with_cursor_agent()`, pass `total_lines` when calling `aggregate_findings()`:

```python
    total_lines = context.additions + context.deletions
    review = aggregate_findings(
        list(all_findings), repo, pr_number,
        confidence_thresholds=thresholds,
        total_lines=total_lines,
    )
```

- [ ] **Step 6: Add `ScoreBreakdown` to `models/review.py`**

```python
@dataclass
class ScoreBreakdown:
    """Transparent breakdown of quality score components."""

    severity_penalty: float
    density_penalty: float
    consensus_factor: float
    agent_factor: float
    raw_score: float
```

Add `score_breakdown: ScoreBreakdown | None = None` field to `ConsolidatedReview`.

- [ ] **Step 7: Remove duplicate `_compute_quality_score()` from aggregator.py**

In `src/ai_reviewer/orchestrator/aggregator.py`, remove the duplicate quality score method. If its `aggregate()` method uses it, import `compute_quality_score` from `review.py` instead.

- [ ] **Step 8: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v --override-ini="addopts="`
Expected: All tests PASS (fix any assertions on exact quality score values)

- [ ] **Step 9: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/review.py src/ai_reviewer/models/review.py src/ai_reviewer/orchestrator/aggregator.py tests/test_scoring.py && ruff format --check src/ai_reviewer/review.py src/ai_reviewer/models/review.py src/ai_reviewer/orchestrator/aggregator.py tests/test_scoring.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 10: Commit**

```bash
git add src/ai_reviewer/review.py src/ai_reviewer/models/review.py src/ai_reviewer/orchestrator/aggregator.py tests/test_scoring.py
git commit -m "feat(P2-2): new composite quality scoring with density normalization"
```

---

### Task 3: Cross-file deduplication (P2-4)

**Files:**
- Modify: `src/ai_reviewer/review.py`
- Add to: `tests/test_review.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_review.py`:

```python
from ai_reviewer.review import dedup_cross_file


class TestDedupCrossFile:
    """Tests for cross-file finding deduplication."""

    def test_two_same_findings_kept(self):
        """Two identical findings in different files are kept as-is."""
        f1 = _make_finding(file_path="a.py", title="Missing null check")
        f2 = _make_finding(file_path="b.py", title="Missing null check")
        result = dedup_cross_file([f1, f2])
        assert len(result) == 2

    def test_three_same_findings_collapsed(self):
        """Three+ identical findings collapsed to one with note."""
        f1 = _make_finding(file_path="a.py", title="Missing null check")
        f2 = _make_finding(file_path="b.py", title="Missing null check")
        f3 = _make_finding(file_path="c.py", title="Missing null check")
        result = dedup_cross_file([f1, f2, f3])
        assert len(result) == 1
        assert "Also found in" in result[0].description

    def test_different_titles_not_collapsed(self):
        """Findings with different titles are not collapsed."""
        f1 = _make_finding(file_path="a.py", title="Missing null check")
        f2 = _make_finding(file_path="b.py", title="SQL injection")
        result = dedup_cross_file([f1, f2])
        assert len(result) == 2

    def test_collapsed_keeps_highest_priority(self):
        """When collapsing, keep the finding with highest priority_score."""
        f_low = _make_finding(
            file_path="a.py",
            title="Missing null check",
            severity=Severity.NITPICK,
        )
        f_high = _make_finding(
            file_path="b.py",
            title="Missing null check",
            severity=Severity.CRITICAL,
        )
        f_med = _make_finding(
            file_path="c.py",
            title="Missing null check",
            severity=Severity.WARNING,
        )
        result = dedup_cross_file([f_low, f_high, f_med])
        assert len(result) == 1
        assert result[0].file_path == "b.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_review.py::TestDedupCrossFile -v --override-ini="addopts="`
Expected: FAIL with `ImportError: cannot import name 'dedup_cross_file'`

- [ ] **Step 3: Implement `dedup_cross_file()`**

Add to `src/ai_reviewer/review.py`:

```python
def dedup_cross_file(
    findings: list[ConsolidatedFinding],
) -> list[ConsolidatedFinding]:
    """Collapse findings with same title+category across 3+ files into one."""
    groups: dict[tuple[str, str], list[ConsolidatedFinding]] = defaultdict(list)
    for f in findings:
        key = (f.category.value, f.title.lower().strip())
        groups[key].append(f)

    result: list[ConsolidatedFinding] = []
    for group in groups.values():
        if len(group) <= 2:
            result.extend(group)
        else:
            group.sort(key=lambda f: f.priority_score, reverse=True)
            primary = group[0]
            others = [f.file_path for f in group[1:]]
            primary.description += (
                f"\n\nAlso found in: {', '.join(others[:5])}"
            )
            result.append(primary)
    return result
```

Add `from collections import defaultdict` at the top if not already present.

- [ ] **Step 4: Wire into `aggregate_findings()` after confidence filtering**

In `aggregate_findings()`, after the confidence filter and before building the final review:

```python
    consolidated = dedup_cross_file(consolidated)
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/test_review.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 6: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/review.py tests/test_review.py && ruff format --check src/ai_reviewer/review.py tests/test_review.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 7: Commit**

```bash
git add src/ai_reviewer/review.py tests/test_review.py
git commit -m "feat(P2-4): cross-file finding deduplication for repeated patterns"
```

---

### Task 4: Load `.ai-reviewer.yaml` from target repo (P2-1)

**Files:**
- Modify: `src/ai_reviewer/github/client.py`
- Modify: `src/ai_reviewer/models/context.py`
- Modify: `src/ai_reviewer/review.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompts.py
from unittest.mock import MagicMock, patch

from ai_reviewer.github.client import GitHubClient


class TestLoadRepoConfig:
    """Tests for loading .ai-reviewer.yaml from target repos."""

    def test_returns_parsed_yaml(self):
        """Valid YAML file returns parsed dict."""
        mock_content = MagicMock()
        mock_content.decoded_content = (
            b"custom_rules:\n  - No dangerous APIs\nignore:\n  - '*.md'"
        )

        with patch("ai_reviewer.github.client.Github") as mock_gh:
            mock_repo = mock_gh.return_value.get_repo.return_value
            mock_repo.get_contents.return_value = mock_content
            client = GitHubClient(token="test")
            config = client.load_repo_config("owner/repo", "main")
            assert config == {
                "custom_rules": ["No dangerous APIs"],
                "ignore": ["*.md"],
            }

    def test_returns_none_on_missing_file(self):
        """Missing file returns None without error."""
        from github.GithubException import UnknownObjectException

        with patch("ai_reviewer.github.client.Github") as mock_gh:
            mock_repo = mock_gh.return_value.get_repo.return_value
            mock_repo.get_contents.side_effect = UnknownObjectException(
                404, data={}, headers={}
            )
            client = GitHubClient(token="test")
            config = client.load_repo_config("owner/repo", "main")
            assert config is None

    def test_returns_none_on_invalid_yaml(self):
        """Malformed YAML returns None without error."""
        mock_content = MagicMock()
        mock_content.decoded_content = b": invalid: yaml: [["

        with patch("ai_reviewer.github.client.Github") as mock_gh:
            mock_repo = mock_gh.return_value.get_repo.return_value
            mock_repo.get_contents.return_value = mock_content
            client = GitHubClient(token="test")
            config = client.load_repo_config("owner/repo", "main")
            assert config is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_prompts.py::TestLoadRepoConfig -v --override-ini="addopts="`
Expected: FAIL with `AttributeError: 'GitHubClient' object has no attribute 'load_repo_config'`

- [ ] **Step 3: Implement `load_repo_config()` and `load_repo_conventions()`**

Add to `src/ai_reviewer/github/client.py`:

```python
import yaml


    def load_repo_config(self, repo_name: str, ref: str) -> dict | None:
        """Load .ai-reviewer.yaml from the target repo."""
        try:
            repo = self.get_repo(repo_name)
            content = repo.get_contents(".ai-reviewer.yaml", ref=ref)
            return yaml.safe_load(content.decoded_content.decode("utf-8"))
        except Exception:
            return None

    def load_repo_conventions(
        self, repo_name: str, ref: str
    ) -> str | None:
        """Best-effort load of AGENTS.md, CLAUDE.md, CONTRIBUTING.md. Capped at 3k chars."""
        context_files = [
            "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".cursor/rules/README.md",
        ]
        repo = self.get_repo(repo_name)
        parts: list[str] = []
        for path in context_files:
            try:
                content = repo.get_contents(path, ref=ref)
                decoded = content.decoded_content.decode("utf-8")[:1500]
                parts.append(f"### {path}\n{decoded}")
            except Exception:
                continue
        combined = "\n".join(parts)[:3000]
        return combined if combined else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_prompts.py::TestLoadRepoConfig -v --override-ini="addopts="`
Expected: All 3 tests PASS

- [ ] **Step 5: Add fields to `ReviewContext`**

In `src/ai_reviewer/models/context.py`, add to `ReviewContext`:

```python
    repo_config: dict | None = None
    conventions: str | None = None
```

- [ ] **Step 6: Wire into `get_base_prompt()` and `review_pr_with_cursor_agent()`**

In `review_pr_with_cursor_agent()`, after building context:

```python
    context.repo_config = gh.load_repo_config(repo, pr.base.ref)
    context.conventions = gh.load_repo_conventions(repo, pr.base.ref)
```

In `get_base_prompt()`, after the PR description section:

```python
    if context.conventions:
        prompt += f"\n## Repository Conventions\n{context.conventions}\n"

    if context.repo_config and context.repo_config.get("custom_rules"):
        rules = context.repo_config["custom_rules"]
        rules_text = "\n".join(f"- {r}" for r in rules)
        prompt += f"\n## Repository-Specific Rules\n{rules_text}\n"
```

Also apply ignore patterns to filter the diff (use `fnmatch` to exclude matched file paths from the diff before sending to agents).

- [ ] **Step 7: Run full tests**

Run: `PYTHONPATH=src pytest tests/test_prompts.py tests/test_review.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 8: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/github/client.py src/ai_reviewer/models/context.py src/ai_reviewer/review.py tests/test_prompts.py && ruff format --check src/ai_reviewer/github/client.py src/ai_reviewer/models/context.py src/ai_reviewer/review.py tests/test_prompts.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 9: Commit**

```bash
git add src/ai_reviewer/github/client.py src/ai_reviewer/models/context.py src/ai_reviewer/review.py tests/test_prompts.py
git commit -m "feat(P2-1): load .ai-reviewer.yaml and conventions from target repo"
```

---

### Task 5: PR size classification + adaptive prompts (P2-7)

Depends on P2-1 (repo config loaded).

**Files:**
- Modify: `src/ai_reviewer/review.py`
- Add to: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_prompts.py`:

```python
from ai_reviewer.review import classify_pr


class TestClassifyPR:
    """Tests for PR type + size classification."""

    def test_small_code_pr(self):
        """Small code PR classified correctly."""
        pr_type, size = classify_pr(["src/app.py"], additions=30, deletions=10)
        assert pr_type == "code"
        assert size == "trivial"

    def test_medium_code_pr(self):
        """Medium code PR classified correctly."""
        pr_type, size = classify_pr(["src/app.py"], additions=300, deletions=100)
        assert pr_type == "code"
        assert size == "medium"

    def test_large_code_pr(self):
        """Large code PR classified correctly."""
        pr_type, size = classify_pr(["src/app.py"], additions=3000, deletions=2000)
        assert pr_type == "code"
        assert size == "large"

    def test_docs_pr_type(self):
        """Docs-only PR detected."""
        pr_type, _size = classify_pr(
            ["README.md", "docs/guide.md"], additions=50, deletions=10,
        )
        assert pr_type == "docs"

    def test_ci_pr_type(self):
        """CI-only PR detected."""
        pr_type, _size = classify_pr(
            [".github/workflows/ci.yaml"], additions=10, deletions=5,
        )
        assert pr_type == "ci"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_prompts.py::TestClassifyPR -v --override-ini="addopts="`
Expected: FAIL with `ImportError: cannot import name 'classify_pr'`

- [ ] **Step 3: Implement `classify_pr()`**

Add to `src/ai_reviewer/review.py`:

```python
def classify_pr(
    changed_paths: list[str],
    additions: int = 0,
    deletions: int = 0,
) -> tuple[str, str]:
    """Classify PR by type (code/docs/ci) and size."""
    pr_type = _detect_pr_type(changed_paths)
    total_lines = additions + deletions
    if total_lines < 50:
        size = "trivial"
    elif total_lines < 200:
        size = "small"
    elif total_lines < 1000:
        size = "medium"
    else:
        size = "large"
    return pr_type, size
```

- [ ] **Step 4: Wire size-adaptive instructions into `get_base_prompt()`**

After the review standard section:

```python
    _pr_type, pr_size = classify_pr(
        changed_paths or [], context.additions, context.deletions,
    )
    if pr_size in ("trivial", "small"):
        prompt += (
            "\n**Note:** This is a small change. Be extra precise "
            "-- only flag genuine issues. Do not pad with low-value "
            "suggestions.\n"
        )
    elif pr_size == "large":
        prompt += (
            "\n**Note:** This is a large change. Focus on architectural "
            "concerns and high-severity issues first. Ignore minor style.\n"
        )
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/test_prompts.py tests/test_review.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 6: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/review.py tests/test_prompts.py && ruff format --check src/ai_reviewer/review.py tests/test_prompts.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 7: Commit**

```bash
git add src/ai_reviewer/review.py tests/test_prompts.py
git commit -m "feat(P2-7): PR size classification with adaptive prompt instructions"
```

---

### Task 6: Language-specific prompt rules (P2-8)

**Files:**
- Modify: `src/ai_reviewer/review.py`
- Add to: `tests/test_prompts.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_prompts.py`:

```python
from ai_reviewer.review import get_language_rules


class TestLanguageRules:
    """Tests for language-specific review rules."""

    def test_python_rules(self):
        """Python language returns Python-specific rules."""
        rules = get_language_rules(["Python"])
        assert "mutable default" in rules.lower()
        assert "type hint" in rules.lower()
        assert "__init__" in rules or "context manager" in rules.lower()

    def test_rust_rules(self):
        """Rust language returns Rust-specific rules."""
        rules = get_language_rules(["Rust"])
        assert "unwrap" in rules.lower()
        assert "unsafe" in rules.lower()
        assert "clone" in rules.lower() or "lifetime" in rules.lower()

    def test_javascript_rules(self):
        """JavaScript language returns JS-specific rules."""
        rules = get_language_rules(["JavaScript"])
        assert "prototype" in rules.lower()
        assert "===" in rules
        assert "async" in rules.lower() or "promise" in rules.lower()

    def test_typescript_rules(self):
        """TypeScript language returns TS-specific rules."""
        rules = get_language_rules(["TypeScript"])
        assert "any" in rules.lower()

    def test_unknown_language_returns_empty(self):
        """Unknown language returns empty string."""
        rules = get_language_rules(["BrainFuck"])
        assert rules == ""

    def test_multiple_languages(self):
        """Multiple languages returns combined rules."""
        rules = get_language_rules(["Python", "Rust", "JavaScript"])
        assert "mutable default" in rules.lower()
        assert "unwrap" in rules.lower()
        assert "prototype" in rules.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_prompts.py::TestLanguageRules -v --override-ini="addopts="`
Expected: FAIL

- [ ] **Step 3: Implement `get_language_rules()`**

Add to `src/ai_reviewer/review.py`:

```python
_LANGUAGE_RULES: dict[str, str] = {
    "python": (
        "For Python:\n"
        "- Mutable default arguments (e.g. `def f(x=[])`) — flag every occurrence.\n"
        "- Bare `except:` or `except Exception:` without re-raise — require specific exception types.\n"
        "- Missing type hints on public function signatures.\n"
        "- f-string injection in `logging.info(f\"...\")` — use `logging.info(\"...\", arg)` instead.\n"
        "- Missing context managers (`with`) for file handles, DB connections, locks.\n"
        "- `subprocess` calls with `shell=True` — flag as security risk.\n"
        "- Shadowing built-in names (`id`, `type`, `list`, `dict`, `input`, `hash`).\n"
        "- `import *` usage — require explicit imports.\n"
        "- Missing `__all__` in public-facing modules.\n"
        "- `os.path` usage where `pathlib.Path` is preferred in modern Python."
    ),
    "rust": (
        "For Rust:\n"
        "- `.unwrap()` / `.expect()` in non-test code — require proper error propagation with `?` or `match`.\n"
        "- `unsafe` blocks without a `// SAFETY:` comment justifying invariants.\n"
        "- Unnecessary `.clone()` — flag when borrowing or references would suffice.\n"
        "- Unbounded allocations: `Vec::new()` in loops without pre-allocated capacity, "
        "  or `collect()` on unbounded iterators without size hints.\n"
        "- Missing lifetime annotations where the compiler cannot elide them.\n"
        "- `panic!()` / `todo!()` / `unimplemented!()` in library code — should return `Result`.\n"
        "- Mutex poisoning: using `.lock().unwrap()` without handling `PoisonError`.\n"
        "- Large types on the stack — suggest `Box<T>` for types > ~1KB.\n"
        "- Missing `#[must_use]` on functions returning `Result` or important values.\n"
        "- `String` vs `&str` in function parameters — prefer `&str` / `impl AsRef<str>` for inputs."
    ),
    "javascript": (
        "For JavaScript:\n"
        "- Prototype pollution via unguarded `Object.assign` or bracket notation from user input.\n"
        "- `==` vs `===` — require strict equality everywhere.\n"
        "- Unhandled Promise rejections: missing `.catch()` or `try/catch` around `await`.\n"
        "- `eval()`, `new Function()`, `innerHTML` — flag as security risks.\n"
        "- Missing input sanitization on user-facing data (XSS vectors).\n"
        "- `var` usage — require `const`/`let`.\n"
        "- Callback hell — suggest async/await refactoring when nesting > 2 levels.\n"
        "- Missing `AbortController` for fetch calls that should be cancellable.\n"
        "- `JSON.parse()` without try/catch.\n"
        "- Regex denial-of-service (ReDoS) — flag catastrophic backtracking patterns."
    ),
    "typescript": (
        "For TypeScript:\n"
        "- `any` type escapes — flag every `any` that isn't explicitly justified.\n"
        "- Missing error boundaries in React components.\n"
        "- Unhandled Promise rejections: missing `.catch()` or `try/catch` around `await`.\n"
        "- Loose equality (`==` vs `===`).\n"
        "- Type assertions (`as T`) that bypass type safety — prefer type guards.\n"
        "- `@ts-ignore` / `@ts-expect-error` without justification comment."
    ),
    "go": (
        "For Go: watch for unchecked errors, SQL string concatenation, "
        "goroutine leaks, and missing `defer` for resource cleanup."
    ),
}


def get_language_rules(languages: list[str]) -> str:
    """Return language-specific review rules for the given languages."""
    rules = []
    for lang in languages:
        rule = _LANGUAGE_RULES.get(lang.lower(), "")
        if rule:
            rules.append(rule)
    return "\n\n".join(rules)
```

- [ ] **Step 4: Wire into `get_base_prompt()`**

After the size-adaptive instructions:

```python
    lang_rules = get_language_rules(context.repo_languages)
    if lang_rules:
        prompt += f"\n**Language-specific guidance:** {lang_rules}\n"
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/test_prompts.py tests/test_review.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 6: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/review.py tests/test_prompts.py && ruff format --check src/ai_reviewer/review.py tests/test_prompts.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 7: Commit**

```bash
git add src/ai_reviewer/review.py tests/test_prompts.py
git commit -m "feat(P2-8): language-specific review rules in prompts"
```

---

### Task 7: Convergence detection (P2-5)

Depends on P2-3 (multi-tier hashes).

**Files:**
- Modify: `src/ai_reviewer/github/client.py`
- Modify: `src/ai_reviewer/cli.py`
- Create: `tests/test_convergence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_convergence.py
from ai_reviewer.github.client import (
    PreviousComment,
    ReviewDelta,
    has_converged,
    should_skip_review,
)
from ai_reviewer.models.findings import (
    Category,
    ConsolidatedFinding,
    Severity,
)


def _make_finding(severity: Severity = Severity.WARNING) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="test-1",
        file_path="src/auth.py",
        line_start=10,
        line_end=None,
        severity=severity,
        category=Category.SECURITY,
        title="Issue",
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=0.9,
    )


def _make_comment() -> PreviousComment:
    return PreviousComment(
        id=1, file_path="src/auth.py", line=10,
        title="Issue", severity="warning", body="body",
    )


class TestHasConverged:
    """Tests for convergence detection."""

    def test_converged_when_no_changes(self):
        """No new and no fixed findings means converged."""
        delta = ReviewDelta()
        delta.open_findings = [_make_finding()]
        assert has_converged(delta) is True

    def test_not_converged_with_new_findings(self):
        """New findings means not converged."""
        delta = ReviewDelta()
        delta.new_findings = [_make_finding()]
        assert has_converged(delta) is False

    def test_not_converged_with_fixed_findings(self):
        """Fixed findings means something changed."""
        delta = ReviewDelta()
        delta.fixed_findings = [_make_comment()]
        assert has_converged(delta) is False


class TestShouldSkipReview:
    """Tests for review skip logic."""

    def test_never_skip_first_review(self):
        """First review should never be skipped."""
        delta = ReviewDelta()
        assert should_skip_review(review_count=0, delta=delta) is False
        assert should_skip_review(review_count=1, delta=delta) is False

    def test_skip_when_converged_on_second_review(self):
        """Skip on 2nd+ review when converged."""
        delta = ReviewDelta()
        delta.open_findings = [_make_finding()]
        assert should_skip_review(review_count=2, delta=delta) is True

    def test_dont_skip_with_new_warnings(self):
        """Don't skip when there are new non-nitpick findings."""
        delta = ReviewDelta()
        delta.new_findings = [_make_finding(severity=Severity.WARNING)]
        assert should_skip_review(review_count=3, delta=delta) is False

    def test_skip_on_third_review_with_only_new_nitpicks(self):
        """On 3rd+ review, tolerate new nitpicks only."""
        delta = ReviewDelta()
        delta.new_findings = [_make_finding(severity=Severity.NITPICK)]
        assert should_skip_review(review_count=3, delta=delta) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_convergence.py -v --override-ini="addopts="`
Expected: FAIL with `ImportError: cannot import name 'has_converged'`

- [ ] **Step 3: Implement `has_converged()` and `should_skip_review()`**

Add to `src/ai_reviewer/github/client.py` as module-level functions:

```python
def has_converged(delta: ReviewDelta) -> bool:
    """PR review has converged if no new findings and no fixed findings."""
    if delta.new_findings:
        return False
    if delta.fixed_findings:
        return False
    return True


def should_skip_review(review_count: int, delta: ReviewDelta) -> bool:
    """Decide whether to skip posting a review.

    Never skip first review. Skip if converged. On 3rd+ review,
    tolerate new nitpicks only.
    """
    if review_count <= 1:
        return False
    if has_converged(delta):
        return True
    if review_count >= 3:
        new_non_nit = [
            f for f in delta.new_findings
            if f.severity != Severity.NITPICK
        ]
        if not new_non_nit:
            return True
    return False
```

Add `from ai_reviewer.models.findings import Severity` import if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_convergence.py -v --override-ini="addopts="`
Expected: All 7 tests PASS

- [ ] **Step 5: Add `--force-review` flag and wire convergence into CLI**

In `src/ai_reviewer/cli.py`, add to `review_pr` command options:

```python
@click.option("--force-review", is_flag=True, default=False, help="Force review even if converged")
```

Add `force_review` parameter to both `review_pr()` and `review_pr_async()`.

After computing delta (around line 230), add:

```python
    from ai_reviewer.github.client import should_skip_review

    if delta.previous_comments and not force_review:
        review_count = 1
        if len(delta.previous_comments) > 10:
            review_count = 3
        elif len(delta.previous_comments) > 3:
            review_count = 2
        if should_skip_review(review_count, delta):
            console.print(
                "[dim]Review converged -- findings unchanged. "
                "Skipping. Use --force-review to override.[/dim]"
            )
            return
```

- [ ] **Step 6: Run full tests**

Run: `PYTHONPATH=src pytest tests/test_convergence.py tests/test_cli.py tests/test_github.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 7: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/github/client.py src/ai_reviewer/cli.py tests/test_convergence.py && ruff format --check src/ai_reviewer/github/client.py src/ai_reviewer/cli.py tests/test_convergence.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 8: Commit**

```bash
git add src/ai_reviewer/github/client.py src/ai_reviewer/cli.py tests/test_convergence.py
git commit -m "feat(P2-5): convergence detection to skip redundant reviews"
```

---

### Task 8: Severity stabilization across runs (P2-6)

Depends on P2-3 (multi-tier hashes).

**Files:**
- Modify: `src/ai_reviewer/github/client.py`
- Modify: `src/ai_reviewer/models/review.py`
- Add to: `tests/test_convergence.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_convergence.py`:

```python
from ai_reviewer.github.client import stabilize_severity


class TestSeverityStabilization:
    """Tests for severity stabilization across review runs."""

    def test_no_change_returns_current(self):
        """Same severity returns current."""
        result = stabilize_severity(
            current=Severity.WARNING,
            previous=Severity.WARNING,
            review_count=2,
        )
        assert result == Severity.WARNING

    def test_upgrade_always_allowed(self):
        """Upgrading severity (more severe) is always allowed."""
        result = stabilize_severity(
            current=Severity.CRITICAL,
            previous=Severity.WARNING,
            review_count=5,
        )
        assert result == Severity.CRITICAL

    def test_downgrade_blocked_after_two_runs(self):
        """Downgrading blocked after 2+ reviews at higher severity."""
        result = stabilize_severity(
            current=Severity.SUGGESTION,
            previous=Severity.WARNING,
            review_count=2,
        )
        assert result == Severity.WARNING

    def test_downgrade_allowed_on_first_review(self):
        """Downgrade allowed when review_count < 2."""
        result = stabilize_severity(
            current=Severity.SUGGESTION,
            previous=Severity.WARNING,
            review_count=1,
        )
        assert result == Severity.SUGGESTION

    def test_downgrade_from_critical_blocked(self):
        """Downgrading from critical is blocked after 2+ runs."""
        result = stabilize_severity(
            current=Severity.NITPICK,
            previous=Severity.CRITICAL,
            review_count=3,
        )
        assert result == Severity.CRITICAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_convergence.py::TestSeverityStabilization -v --override-ini="addopts="`
Expected: FAIL with `ImportError: cannot import name 'stabilize_severity'`

- [ ] **Step 3: Implement `stabilize_severity()`**

Add to `src/ai_reviewer/github/client.py`:

```python
_SEVERITY_ORDER = [
    Severity.CRITICAL, Severity.WARNING, Severity.SUGGESTION, Severity.NITPICK,
]


def stabilize_severity(
    current: Severity,
    previous: Severity,
    review_count: int,
) -> Severity:
    """Prevent severity flip-flopping across review runs.

    After 2+ reviews at a given severity, don't allow downgrade.
    Upgrades (more severe) are always allowed.
    """
    cur_idx = _SEVERITY_ORDER.index(current)
    prev_idx = _SEVERITY_ORDER.index(previous)

    if cur_idx < prev_idx:
        return current  # Upgrade always allowed
    if review_count >= 2 and cur_idx > prev_idx:
        return previous  # Block downgrade
    return current
```

- [ ] **Step 4: Add `ReviewHistory` dataclass**

Add to `src/ai_reviewer/models/review.py`:

```python
@dataclass
class ReviewHistory:
    """Parsed history from previous AI review comments on a PR."""

    review_count: int = 0
    previous_hashes: set[str] = field(default_factory=set)
    resolved_hashes: set[str] = field(default_factory=set)
    last_severity_map: dict[str, str] = field(default_factory=dict)
    last_quality_score: float | None = None
    last_review_sha: str | None = None
```

- [ ] **Step 5: Wire stabilization into `compute_review_delta()`**

Add a helper to parse severity strings:

```python
def _parse_severity(severity_str: str) -> Severity | None:
    """Parse severity string to enum, returning None for unknown."""
    try:
        return Severity(severity_str)
    except ValueError:
        return None
```

In `compute_review_delta()`, after matching a finding to a previous comment, apply stabilization:

```python
            if matched_comment is not None:
                prev_sev = _parse_severity(matched_comment.severity)
                if prev_sev is not None:
                    review_round = max(1, len(previous_comments) // 3)
                    stable_sev = stabilize_severity(
                        current=finding.severity,
                        previous=prev_sev,
                        review_count=review_round,
                    )
                    if stable_sev != finding.severity:
                        finding.severity = stable_sev
                delta.open_findings.append(finding)
                matched_previous.add(matched_comment.id)
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src pytest tests/test_convergence.py tests/test_github.py -v --override-ini="addopts="`
Expected: All PASS

- [ ] **Step 7: Ruff lint and format check**

Run: `ruff check src/ai_reviewer/github/client.py src/ai_reviewer/models/review.py tests/test_convergence.py && ruff format --check src/ai_reviewer/github/client.py src/ai_reviewer/models/review.py tests/test_convergence.py`
Expected: All checks passed. If not, fix issues and re-run.

- [ ] **Step 8: Commit**

```bash
git add src/ai_reviewer/github/client.py src/ai_reviewer/models/review.py tests/test_convergence.py
git commit -m "feat(P2-6): severity stabilization prevents flip-flopping across runs"
```

---

### Task 9: Final integration test and cleanup

**Files:**
- All modified files
- Existing test files

- [ ] **Step 1: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v --override-ini="addopts="`
Expected: All tests PASS

- [ ] **Step 2: Run ruff lint**

Run: `ruff check src/ tests/`
Expected: No errors (fix any that appear)

- [ ] **Step 3: Run ruff format**

Run: `ruff format --check src/ tests/`
Expected: No format issues (run `ruff format src/ tests/` to fix)

- [ ] **Step 4: Run mypy**

Run: `mypy src/ --ignore-missing-imports`
Expected: No errors

- [ ] **Step 5: Update docs/IMPROVEMENT-PLAN.md Phase 2 status**

Mark all Phase 2 items as completed with implementation locations.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: Phase 2 cleanup - lint, format, type checks"
```

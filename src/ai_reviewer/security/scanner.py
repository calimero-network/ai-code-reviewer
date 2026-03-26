"""Regex + entropy-based secret scanner that runs before LLM agents.

Scans unified diffs for potential secrets on added lines only. Produces
ConsolidatedFinding objects with severity=CRITICAL and category=SECURITY
that bypass aggregation and cross-review.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import re
from collections import Counter

from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

logger = logging.getLogger(__name__)

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "AWS Access Key ID",
    ),
    (
        re.compile(r"(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
        "AWS Secret Access Key",
    ),
    (
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
        "GitHub Personal Access Token",
    ),
    (
        re.compile(r"gho_[A-Za-z0-9]{36}"),
        "GitHub OAuth Token",
    ),
    (
        re.compile(r"ghs_[A-Za-z0-9]{36}"),
        "GitHub App Installation Token",
    ),
    (
        re.compile(r"github_pat_[A-Za-z0-9_]{82}"),
        "GitHub Fine-Grained PAT",
    ),
    (
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        "Private key",
    ),
    (
        re.compile(
            r"""(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key)\s*[=:]\s*['"][A-Za-z0-9/+=_\-]{20,}['"]""",
            re.IGNORECASE,
        ),
        "Generic secret/API key assignment",
    ),
    (
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "OpenAI / Stripe secret key",
    ),
    (
        re.compile(r"xox[bpras]-[A-Za-z0-9\-]{10,}"),
        "Slack token",
    ),
]

_ENTROPY_MIN_LENGTH = 20
_ENTROPY_THRESHOLD = 4.5

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DIFF_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$")


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (bits) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _extract_high_entropy_tokens(line: str) -> list[str]:
    """Extract tokens from a line that look like potential secrets (alphanumeric runs)."""
    return [
        tok
        for tok in re.findall(r"[A-Za-z0-9/+=_\-]{20,}", line)
        if _shannon_entropy(tok) > _ENTROPY_THRESHOLD
    ]


def _file_matches_exclude(file_path: str, exclude_patterns: list[str]) -> bool:
    """Check if file_path matches any of the exclude glob patterns."""
    return any(fnmatch.fnmatch(file_path, pat) for pat in exclude_patterns)


def scan_for_secrets(
    diff: str,
    exclude_patterns: list[str] | None = None,
) -> list[ConsolidatedFinding]:
    """Scan a unified diff for potential secrets on added lines.

    Args:
        diff: Full unified diff text.
        exclude_patterns: Glob patterns for file paths to skip.

    Returns:
        List of ConsolidatedFinding with severity=CRITICAL, category=SECURITY.
    """
    excludes = exclude_patterns or []
    findings: list[ConsolidatedFinding] = []
    seen_keys: set[str] = set()

    current_file: str | None = None
    current_line = 0
    finding_counter = 0

    for raw_line in diff.splitlines():
        file_match = _DIFF_FILE_RE.match(raw_line)
        if file_match:
            current_file = file_match.group(1)
            current_line = 0
            continue

        hunk_match = _HUNK_HEADER_RE.match(raw_line)
        if hunk_match:
            current_line = int(hunk_match.group(1)) - 1
            continue

        if raw_line.startswith("-"):
            continue

        if raw_line.startswith("+"):
            current_line += 1
        else:
            current_line += 1
            continue

        if current_file is None:
            continue

        if _file_matches_exclude(current_file, excludes):
            continue

        added_content = raw_line[1:]

        for pattern, description in SECRET_PATTERNS:
            if pattern.search(added_content):
                dedup_key = f"{current_file}:{current_line}:{description}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                finding_counter += 1
                findings.append(
                    ConsolidatedFinding(
                        id=f"secret-{finding_counter}",
                        file_path=current_file,
                        line_start=current_line,
                        line_end=None,
                        severity=Severity.CRITICAL,
                        category=Category.SECURITY,
                        title=f"Potential secret detected: {description}",
                        description=(
                            f"A pattern matching '{description}' was found on an added line. "
                            "Remove the secret and rotate it immediately."
                        ),
                        suggested_fix="Remove the hardcoded secret; use environment variables or a secrets manager.",
                        consensus_score=1.0,
                        agreeing_agents=["secret-scanner"],
                        confidence=0.95,
                    )
                )

        high_entropy_tokens = _extract_high_entropy_tokens(added_content)
        for token in high_entropy_tokens:
            dedup_key = f"{current_file}:{current_line}:entropy:{token[:12]}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            finding_counter += 1
            findings.append(
                ConsolidatedFinding(
                    id=f"secret-{finding_counter}",
                    file_path=current_file,
                    line_start=current_line,
                    line_end=None,
                    severity=Severity.CRITICAL,
                    category=Category.SECURITY,
                    title="High-entropy string detected (possible secret)",
                    description=(
                        f"A high-entropy string (Shannon entropy > {_ENTROPY_THRESHOLD}) "
                        "was found on an added line, which may be a hardcoded secret or key."
                    ),
                    suggested_fix="Remove the hardcoded secret; use environment variables or a secrets manager.",
                    consensus_score=1.0,
                    agreeing_agents=["secret-scanner"],
                    confidence=0.95,
                )
            )

    if findings:
        logger.info("Secret scanner found %d potential secret(s) in diff", len(findings))

    return findings

"""Tests for the security secret scanner."""

from ai_reviewer.models.findings import Category, Severity
from ai_reviewer.security.scanner import (
    scan_for_secrets,
)


class TestScanForSecretsPatterns:
    """Each regex pattern should be detected on added lines."""

    def _make_diff(self, file_path: str, added_line: str, line_number: int = 10) -> str:
        return (
            f"diff --git a/{file_path} b/{file_path}\n"
            f"--- a/{file_path}\n"
            f"+++ b/{file_path}\n"
            f"@@ -1,5 +1,6 @@\n"
            f" context line\n"
            f"+{added_line}\n"
            f" more context\n"
        )

    def test_aws_access_key(self):
        diff = self._make_diff("config.py", 'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"')
        findings = scan_for_secrets(diff)
        assert any("AWS Access Key ID" in f.title for f in findings)
        assert all(f.severity == Severity.CRITICAL for f in findings)
        assert all(f.category == Category.SECURITY for f in findings)

    def test_aws_secret_key(self):
        diff = self._make_diff(
            "config.py",
            'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
        )
        findings = scan_for_secrets(diff)
        assert any("AWS Secret Access Key" in f.title for f in findings)

    def test_github_pat(self):
        diff = self._make_diff("ci.py", 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"')
        findings = scan_for_secrets(diff)
        assert any("GitHub Personal Access Token" in f.title for f in findings)

    def test_github_oauth_token(self):
        diff = self._make_diff("auth.py", 'token = "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"')
        findings = scan_for_secrets(diff)
        assert any("GitHub OAuth Token" in f.title for f in findings)

    def test_github_app_installation_token(self):
        diff = self._make_diff("auth.py", 'token = "ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"')
        findings = scan_for_secrets(diff)
        assert any("GitHub App Installation Token" in f.title for f in findings)

    def test_github_fine_grained_pat(self):
        pat = "github_pat_" + "A" * 82
        diff = self._make_diff("auth.py", f'token = "{pat}"')
        findings = scan_for_secrets(diff)
        assert any("GitHub Fine-Grained PAT" in f.title for f in findings)

    def test_private_key(self):
        diff = self._make_diff("keys.py", "-----BEGIN RSA PRIVATE KEY-----")
        findings = scan_for_secrets(diff)
        assert any("Private key" in f.title for f in findings)

    def test_generic_api_key(self):
        # Use a generic alphanumeric value; some literals trigger GitHub push protection.
        diff = self._make_diff(
            "settings.py",
            'api_key = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdef"',
        )
        findings = scan_for_secrets(diff)
        assert any("Generic secret" in f.title or "API key" in f.title for f in findings)

    def test_openai_stripe_key(self):
        diff = self._make_diff("llm.py", 'key = "sk-abcdefghijklmnopqrstuvwx"')
        findings = scan_for_secrets(diff)
        assert any("OpenAI" in f.title or "Stripe" in f.title for f in findings)

    def test_slack_token(self):
        diff = self._make_diff("notify.py", 'SLACK = "xoxb-1234567890-abcdefghij"')
        findings = scan_for_secrets(diff)
        assert any("Slack" in f.title for f in findings)


class TestScanOnlyAddedLines:
    """Only lines starting with '+' should be scanned."""

    def test_removed_lines_not_scanned(self):
        diff = (
            "diff --git a/config.py b/config.py\n"
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,3 +1,3 @@\n"
            '-SECRET = "AKIAIOSFODNN7EXAMPLE"\n'
            "+# secret removed\n"
            " context\n"
        )
        findings = scan_for_secrets(diff)
        assert not any("AWS" in f.title for f in findings)

    def test_context_lines_not_scanned(self):
        diff = (
            "diff --git a/config.py b/config.py\n"
            "--- a/config.py\n"
            "+++ b/config.py\n"
            "@@ -1,3 +1,4 @@\n"
            ' SECRET = "AKIAIOSFODNN7EXAMPLE"\n'
            "+# new comment\n"
            " context\n"
        )
        findings = scan_for_secrets(diff)
        assert not any("AWS" in f.title for f in findings)


class TestScanExcludePatterns:
    """Files matching exclude patterns should be skipped."""

    def _make_diff(self, file_path: str) -> str:
        return (
            f"diff --git a/{file_path} b/{file_path}\n"
            f"--- a/{file_path}\n"
            f"+++ b/{file_path}\n"
            "@@ -1,3 +1,4 @@\n"
            " context\n"
            '+SECRET = "AKIAIOSFODNN7EXAMPLE"\n'
            " more context\n"
        )

    def test_excluded_file_skipped(self):
        diff = self._make_diff("tests/fixtures/secrets.py")
        findings = scan_for_secrets(diff, exclude_patterns=["tests/fixtures/*"])
        assert len(findings) == 0

    def test_non_excluded_file_scanned(self):
        diff = self._make_diff("src/config.py")
        findings = scan_for_secrets(diff, exclude_patterns=["tests/fixtures/*"])
        assert any("AWS" in f.title for f in findings)

    def test_multiple_exclude_patterns(self):
        diff = self._make_diff("vendor/lib.py")
        findings = scan_for_secrets(diff, exclude_patterns=["tests/*", "vendor/*"])
        assert len(findings) == 0


class TestScanLineTracking:
    """Line numbers should be correctly tracked from hunk headers."""

    def test_line_number_from_hunk_header(self):
        diff = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -10,3 +20,4 @@\n"
            " context\n"
            '+KEY = "AKIAIOSFODNN7EXAMPLE"\n'
            " more context\n"
        )
        findings = scan_for_secrets(diff)
        aws_findings = [f for f in findings if "AWS" in f.title]
        assert len(aws_findings) >= 1
        assert aws_findings[0].line_start == 21

    def test_multiple_hunks(self):
        diff = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,3 +1,4 @@\n"
            " context\n"
            "+safe_line = True\n"
            " more context\n"
            "@@ -50,3 +51,4 @@\n"
            " context\n"
            '+KEY = "AKIAIOSFODNN7EXAMPLE"\n'
            " more context\n"
        )
        findings = scan_for_secrets(diff)
        aws_findings = [f for f in findings if "AWS" in f.title]
        assert len(aws_findings) >= 1
        assert aws_findings[0].line_start == 52


class TestScanMultipleFiles:
    """Scanner should handle diffs with multiple files."""

    def test_findings_across_files(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,4 @@\n"
            " context\n"
            '+KEY1 = "AKIAIOSFODNN7EXAMPLE"\n'
            " more\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1,3 +1,4 @@\n"
            " context\n"
            '+KEY2 = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"\n'
            " more\n"
        )
        findings = scan_for_secrets(diff)
        files = {f.file_path for f in findings}
        assert "a.py" in files
        assert "b.py" in files


class TestScanFindingMetadata:
    """All findings should have correct severity, category, confidence, and agents."""

    def test_finding_metadata(self):
        diff = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,3 +1,4 @@\n"
            " ctx\n"
            '+T = "AKIAIOSFODNN7EXAMPLE"\n'
            " ctx\n"
        )
        findings = scan_for_secrets(diff)
        assert len(findings) >= 1
        for f in findings:
            assert f.severity == Severity.CRITICAL
            assert f.category == Category.SECURITY
            assert f.confidence == 0.95
            assert f.consensus_score == 1.0
            assert f.agreeing_agents == ["secret-scanner"]


class TestScanEmptyAndClean:
    """Edge cases: empty diff, clean diff."""

    def test_empty_diff(self):
        assert scan_for_secrets("") == []

    def test_clean_diff_no_secrets(self):
        diff = (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,3 +1,4 @@\n"
            " context\n"
            "+x = 42\n"
            " more context\n"
        )
        assert scan_for_secrets(diff) == []

    def test_deduplication(self):
        diff = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1,3 +1,4 @@\n"
            " ctx\n"
            '+T = "AKIAIOSFODNN7EXAMPLE"\n'
            " ctx\n"
        )
        findings = scan_for_secrets(diff)
        aws_findings = [f for f in findings if "AWS Access Key ID" in f.title]
        assert len(aws_findings) == 1

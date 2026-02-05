"""Tests for review agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCursorClient:
    """Tests for the unified Cursor API client."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initializes with correct config."""
        from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig

        config = CursorConfig(
            api_key="test-key",
            base_url="https://api.cursor.com/v1",
            timeout=60,
        )
        client = CursorClient(config)

        assert client.config.api_key == "test-key"
        assert client.config.timeout == 60

    @pytest.mark.asyncio
    async def test_client_complete_request(self):
        """Test sending completion request."""
        from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig

        config = CursorConfig(api_key="test-key")
        client = CursorClient(config)

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Review response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.complete(
                model="claude-4.5-opus-high-thinking",
                system_prompt="You are a code reviewer",
                user_prompt="Review this code",
            )

            assert result == "Review response"
            mock_post.assert_called_once()


class TestSecurityAgent:
    """Tests for the security-focused review agent."""

    @pytest.mark.asyncio
    async def test_detects_sql_injection(self, sample_vulnerable_diff, mock_review_context):
        """Test that security agent detects SQL injection."""
        from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
        from ai_reviewer.agents.security import SecurityAgent
        from ai_reviewer.models.findings import Severity

        # Create agent with mocked client
        config = CursorConfig(api_key="test-key")
        client = CursorClient(config)

        # Mock response containing SQL injection finding
        mock_response = """
        {
            "findings": [
                {
                    "file_path": "auth/login.py",
                    "line_start": 15,
                    "line_end": 16,
                    "severity": "critical",
                    "category": "security",
                    "title": "SQL Injection Vulnerability",
                    "description": "User input directly interpolated into SQL query",
                    "suggested_fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE username = %s', (username,))",
                    "confidence": 0.95
                }
            ],
            "summary": "Found 1 critical security issue"
        }
        """

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            agent = SecurityAgent(client)
            review = await agent.review(
                diff=sample_vulnerable_diff,
                file_contents={},
                context=mock_review_context,
            )

            assert len(review.findings) >= 1
            # Should find the SQL injection
            sql_findings = [
                f for f in review.findings if "sql" in f.title.lower()]
            assert len(sql_findings) >= 1
            assert sql_findings[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_no_false_positives_on_safe_code(self, sample_secure_diff, mock_review_context):
        """Test that security agent doesn't flag safe code."""
        from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
        from ai_reviewer.agents.security import SecurityAgent

        config = CursorConfig(api_key="test-key")
        client = CursorClient(config)

        mock_response = """
        {
            "findings": [],
            "summary": "No security issues found"
        }
        """

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            agent = SecurityAgent(client)
            review = await agent.review(
                diff=sample_secure_diff,
                file_contents={},
                context=mock_review_context,
            )

            # Should not have critical findings for safe code
            critical_findings = [
                f for f in review.findings if f.severity.value == "critical"]
            assert len(critical_findings) == 0


class TestPerformanceAgent:
    """Tests for the performance-focused review agent."""

    @pytest.mark.asyncio
    async def test_detects_on2_complexity(self, sample_performance_diff, mock_review_context):
        """Test that performance agent detects O(n²) loops."""
        from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
        from ai_reviewer.agents.performance import PerformanceAgent
        from ai_reviewer.models.findings import Category

        config = CursorConfig(api_key="test-key")
        client = CursorClient(config)

        mock_response = """
        {
            "findings": [
                {
                    "file_path": "utils/processor.py",
                    "line_start": 10,
                    "line_end": 16,
                    "severity": "warning",
                    "category": "performance",
                    "title": "O(n²) Time Complexity",
                    "description": "Nested loops result in quadratic time complexity",
                    "suggested_fix": "Use a set for O(n) lookup: seen = set(); duplicates = [x for x in items if x in seen or seen.add(x)]",
                    "confidence": 0.9
                }
            ],
            "summary": "Found 1 performance issue"
        }
        """

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            agent = PerformanceAgent(client)
            review = await agent.review(
                diff=sample_performance_diff,
                file_contents={},
                context=mock_review_context,
            )

            perf_findings = [
                f for f in review.findings if f.category == Category.PERFORMANCE]
            assert len(perf_findings) >= 1

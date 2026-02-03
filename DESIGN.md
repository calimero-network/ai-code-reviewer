# AI Code Reviewer - Design Document

**Project:** Multi-Agent Code Review System  
**Version:** 0.1.0  
**Date:** February 2026  
**Status:** Draft

---

## 1. Executive Summary

AI Code Reviewer is a multi-agent system that orchestrates multiple LLMs to produce comprehensive, high-quality code reviews. Unlike single-agent approaches, this tool leverages the diversity of multiple AI models—each with different strengths—to generate consolidated reviews that catch more issues and provide better insights.

The system integrates with GitHub for PR reviews and can utilize Cursor's API for enhanced code understanding.

---

## 2. Goals & Non-Goals

### Goals
- **Multi-Agent Orchestration**: Spawn N parallel LLM agents to review code independently
- **Consensus-Based Reviews**: Aggregate findings with confidence scoring based on agent agreement
- **GitHub Integration**: Automatically review PRs and post consolidated feedback
- **Extensible Agent System**: Support multiple LLM providers (Claude, GPT-4, Cursor, local models)
- **Specialized Review Perspectives**: Different agents focus on security, performance, style, architecture
- **Actionable Output**: Generate clear, prioritized, and actionable review comments

### Non-Goals (v1)
- Automatic code fixes (that's what ai-bounty-hunter does)
- IDE plugins (may come in v2)
- Real-time streaming reviews
- Self-hosted LLM orchestration

---

## 3. Architecture Overview

The system uses **Cursor API as the unified gateway** to access multiple LLM models (Claude, GPT-4, etc.). This simplifies the architecture by:
- Single API key management
- Consistent request/response format across all models
- Leveraging Cursor's built-in rate limiting and infrastructure
- Access to codebase-aware features

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI Code Reviewer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │   Triggers   │    │      Cursor API (Unified Gateway)         │   │
│  │              │    │  ┌────────┐ ┌────────┐ ┌────────┐        │   │
│  │ • GitHub PR  │───▶│  │Claude  │ │ GPT-4  │ │ Other  │        │   │
│  │ • CLI        │    │  │ Agent  │ │ Agent  │ │ Models │        │   │
│  │ • API        │    │  └───┬────┘ └───┬────┘ └───┬────┘        │   │
│  └──────────────┘    │      │          │          │              │   │
│                      │      ▼          ▼          ▼              │   │
│                      │  ┌──────────────────────────────────┐    │   │
│                      │  │       Review Aggregator           │    │   │
│                      │  │  • Deduplication                  │    │   │
│                      │  │  • Consensus scoring              │    │   │
│                      │  │  • Priority ranking               │    │   │
│                      │  │  • Conflict resolution            │    │   │
│                      │  └──────────────┬───────────────────┘    │   │
│                      └─────────────────┼────────────────────────┘   │
│                                        │                             │
│  ┌──────────────────────────────────────▼──────────────────────┐    │
│  │                    Output Formatter                          │    │
│  │  • GitHub PR comments    • JSON report    • Markdown         │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Cursor API as the Single Gateway?

| Aspect | Direct Provider APIs | Cursor API (Chosen) |
|--------|---------------------|---------------------|
| API Keys | 3+ keys (Anthropic, OpenAI, etc.) | 1 key |
| Client Libraries | Multiple, different interfaces | Single unified client |
| Rate Limiting | Handle per provider | Cursor manages it |
| Response Format | Varies by provider | Consistent |
| Codebase Context | Manual implementation | Built-in support |
| Model Switching | Code changes required | Configuration only |

---

## 4. Core Components

### 4.1 Agent Abstraction Layer

Each agent implements a common interface, allowing easy addition of new LLM providers:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ReviewFinding:
    """A single finding from an agent's review."""
    file_path: str
    line_start: int
    line_end: Optional[int]
    severity: str  # "critical", "warning", "suggestion", "nitpick"
    category: str  # "security", "performance", "style", "logic", "architecture"
    title: str
    description: str
    suggested_fix: Optional[str]
    confidence: float  # 0.0 - 1.0

@dataclass
class AgentReview:
    """Complete review from a single agent."""
    agent_id: str
    agent_type: str  # "claude", "gpt4", "cursor", etc.
    focus_areas: List[str]
    findings: List[ReviewFinding]
    summary: str
    review_time_ms: int

class ReviewAgent(ABC):
    """Base class for all review agents."""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        pass
    
    @property
    @abstractmethod
    def focus_areas(self) -> List[str]:
        """Categories this agent specializes in."""
        pass
    
    @abstractmethod
    async def review(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext,
    ) -> AgentReview:
        """
        Perform code review and return findings.
        
        Args:
            diff: The git diff to review
            file_contents: Full contents of changed files
            context: Additional context (PR description, repo info, etc.)
        """
        pass
```

### 4.2 Agent Implementations (via Cursor API)

All agents use the **Cursor API** as the unified gateway, specifying different models and prompts:

#### 4.2.1 Cursor Client (Unified Interface)
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CursorConfig:
    """Configuration for Cursor API."""
    api_key: str
    base_url: str = "https://api.cursor.com/v1"
    timeout: int = 120

class CursorClient:
    """Unified client for accessing multiple models via Cursor API."""
    
    def __init__(self, config: CursorConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout,
        )
    
    async def complete(
        self,
        model: str,  # "claude-3-opus", "gpt-4-turbo", etc.
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send completion request to specified model via Cursor."""
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

#### 4.2.2 Claude Agent (Security & Architecture Focus)
```python
class ClaudeSecurityAgent(ReviewAgent):
    """Claude-based agent focused on security vulnerabilities."""
    
    MODEL = "claude-3-opus-20240229"  # Specified to Cursor API
    
    SYSTEM_PROMPT = """You are an expert security code reviewer. 
    Focus on:
    - SQL injection, XSS, CSRF vulnerabilities
    - Authentication/authorization flaws
    - Cryptographic misuse
    - Data exposure risks
    - Input validation issues
    
    Be thorough but avoid false positives. Only report issues with 
    concrete evidence in the code."""
    
    def __init__(self, cursor_client: CursorClient):
        self.client = cursor_client
    
    async def review(self, diff: str, files: dict, context: ReviewContext) -> AgentReview:
        response = await self.client.complete(
            model=self.MODEL,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._format_review_request(diff, context),
        )
        return self._parse_response(response)
```

#### 4.2.3 GPT-4 Agent (Performance & Logic Focus)
```python
class GPTPerformanceAgent(ReviewAgent):
    """GPT-4-based agent focused on performance and correctness."""
    
    MODEL = "gpt-4-turbo-preview"  # Specified to Cursor API
    
    SYSTEM_PROMPT = """You are an expert performance engineer reviewing code.
    Focus on:
    - Algorithm complexity issues (O(n²) where O(n) possible)
    - Memory leaks and resource management
    - Unnecessary computations
    - Race conditions and concurrency bugs
    - Edge cases and error handling"""
    
    def __init__(self, cursor_client: CursorClient):
        self.client = cursor_client
```

#### 4.2.4 Codebase-Aware Agent (Cursor's Unique Capability)
```python
class CodebaseContextAgent(ReviewAgent):
    """Uses Cursor's codebase indexing for context-aware reviews."""
    
    MODEL = "claude-3-opus-20240229"  # Can use any model
    
    SYSTEM_PROMPT = """You have full context of the codebase.
    Focus on:
    - Consistency with existing patterns
    - API contract violations
    - Breaking changes
    - Missing test coverage for critical paths
    - Integration concerns"""
    
    def __init__(self, cursor_client: CursorClient, repo_path: str):
        self.client = cursor_client
        self.repo_path = repo_path
    
    async def review(self, diff: str, files: dict, context: ReviewContext) -> AgentReview:
        # Cursor API can include codebase context automatically
        response = await self.client.complete(
            model=self.MODEL,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._format_review_request(diff, context),
            # Cursor-specific: include codebase context
            include_codebase=True,
            codebase_path=self.repo_path,
        )
        return self._parse_response(response)
```

### 4.3 Agent Orchestrator

The orchestrator manages parallel execution and handles agent failures gracefully:

```python
class AgentOrchestrator:
    """Coordinates multiple agents to review code in parallel."""
    
    def __init__(
        self,
        agents: List[ReviewAgent],
        timeout_seconds: int = 120,
        min_agents_required: int = 2,
    ):
        self.agents = agents
        self.timeout = timeout_seconds
        self.min_agents = min_agents_required
    
    async def review(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext,
    ) -> List[AgentReview]:
        """
        Execute all agents in parallel and collect results.
        
        Handles timeouts and failures gracefully—partial results
        are still valuable as long as min_agents succeed.
        """
        tasks = [
            asyncio.create_task(
                self._run_agent_with_timeout(agent, diff, file_contents, context)
            )
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_reviews = [r for r in results if isinstance(r, AgentReview)]
        
        if len(successful_reviews) < self.min_agents:
            raise InsufficientAgentsError(
                f"Only {len(successful_reviews)} agents succeeded, "
                f"minimum {self.min_agents} required"
            )
        
        return successful_reviews
```

### 4.4 Review Aggregator

The aggregator combines findings from multiple agents using consensus-based scoring:

```python
class ReviewAggregator:
    """Combines multiple agent reviews into a unified review."""
    
    def aggregate(self, reviews: List[AgentReview]) -> ConsolidatedReview:
        """
        Merge findings using semantic similarity and voting.
        
        Algorithm:
        1. Cluster similar findings using embedding similarity
        2. For each cluster, compute consensus score
        3. Merge descriptions from agreeing agents
        4. Rank by severity × consensus score
        5. Deduplicate and format final output
        """
        all_findings = self._extract_all_findings(reviews)
        clusters = self._cluster_similar_findings(all_findings)
        
        consolidated = []
        for cluster in clusters:
            merged = self._merge_cluster(cluster)
            merged.consensus_score = len(cluster.findings) / len(reviews)
            merged.agreeing_agents = [f.agent_id for f in cluster.findings]
            consolidated.append(merged)
        
        # Sort by priority (severity × consensus)
        consolidated.sort(
            key=lambda f: self._priority_score(f),
            reverse=True
        )
        
        return ConsolidatedReview(
            findings=consolidated,
            summary=self._generate_summary(consolidated),
            agent_count=len(reviews),
            review_quality_score=self._compute_quality_score(reviews),
        )
    
    def _priority_score(self, finding: ConsolidatedFinding) -> float:
        """Compute priority based on severity and consensus."""
        severity_weights = {
            "critical": 1.0,
            "warning": 0.6,
            "suggestion": 0.3,
            "nitpick": 0.1,
        }
        return severity_weights[finding.severity] * finding.consensus_score
```

---

## 5. GitHub Integration

### 5.1 PR Event Handling

```python
class GitHubPRHandler:
    """Handles GitHub PR events and posts reviews."""
    
    def __init__(self, github_token: str, orchestrator: AgentOrchestrator):
        self.gh = Github(github_token)
        self.orchestrator = orchestrator
        self.aggregator = ReviewAggregator()
    
    async def handle_pr_event(self, event: PREvent) -> None:
        """Process a PR event and post review."""
        pr = self.gh.get_repo(event.repo).get_pull(event.pr_number)
        
        # Extract diff and file contents
        diff = self._get_pr_diff(pr)
        files = self._get_changed_files(pr)
        context = self._build_context(pr)
        
        # Run multi-agent review
        agent_reviews = await self.orchestrator.review(diff, files, context)
        consolidated = self.aggregator.aggregate(agent_reviews)
        
        # Post to GitHub
        self._post_pr_review(pr, consolidated)
        self._post_inline_comments(pr, consolidated)
    
    def _post_pr_review(self, pr, review: ConsolidatedReview) -> None:
        """Post overall review comment."""
        body = self._format_review_summary(review)
        
        # Determine review action based on findings
        if any(f.severity == "critical" for f in review.findings):
            event = "REQUEST_CHANGES"
        elif len(review.findings) > 0:
            event = "COMMENT"
        else:
            event = "APPROVE"
        
        pr.create_review(body=body, event=event)
```

### 5.2 Webhook Server

```python
from fastapi import FastAPI, Request
from github import GithubIntegration

app = FastAPI()

@app.post("/webhook")
async def github_webhook(request: Request):
    """Handle GitHub webhook events."""
    payload = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    
    if event_type == "pull_request":
        action = payload["action"]
        if action in ("opened", "synchronize", "reopened"):
            pr_event = PREvent(
                repo=payload["repository"]["full_name"],
                pr_number=payload["pull_request"]["number"],
                action=action,
            )
            
            # Process async to respond quickly
            asyncio.create_task(pr_handler.handle_pr_event(pr_event))
    
    return {"status": "ok"}
```

---

## 6. Configuration System

### 6.1 YAML Configuration

```yaml
# config.yaml
version: 1

# Single API key for all LLM access via Cursor
cursor:
  api_key: ${CURSOR_API_KEY}
  base_url: https://api.cursor.com/v1  # or self-hosted
  timeout_seconds: 120

# GitHub Integration
github:
  token: ${GITHUB_TOKEN}  # Personal Access Token (simple)
  # Or use GitHub App for production:
  # app_id: ${GITHUB_APP_ID}
  # private_key_path: ./github-app.pem
  # webhook_secret: ${GITHUB_WEBHOOK_SECRET}

# Agent Configuration - All use Cursor API with different models
agents:
  - name: security-claude
    model: claude-3-opus-20240229  # Model selection via Cursor
    focus_areas: [security, architecture]
    max_tokens: 4096
    temperature: 0.3
    
  - name: performance-gpt4
    model: gpt-4-turbo-preview  # Different model, same Cursor API
    focus_areas: [performance, logic, edge_cases]
    max_tokens: 4096
    temperature: 0.3
    
  - name: patterns-claude
    model: claude-3-opus-20240229
    focus_areas: [consistency, patterns, integration]
    include_codebase_context: true  # Cursor's unique feature

orchestrator:
  timeout_seconds: 120
  min_agents_required: 2
  max_parallel_agents: 5

aggregator:
  similarity_threshold: 0.85  # For clustering similar findings
  min_consensus_for_critical: 0.5  # At least half must agree
  
output:
  include_agent_breakdown: true
  include_confidence_scores: true
  max_findings_per_file: 10
  
review_policy:
  auto_approve_if_no_findings: false
  block_on_critical: true
  require_human_review_for: [security]
```

### 6.2 Repository-Specific Overrides

Support `.ai-reviewer.yaml` in repositories for custom configuration:

```yaml
# .ai-reviewer.yaml (in repo root)
inherit_from: default

# Override agent focus for this repo
agents:
  - type: claude
    focus_areas: [security, crypto]  # Crypto-focused for this repo
    custom_prompt_append: |
      This is a Rust codebase using the `eyre` error handling crate.
      Pay special attention to unwrap() and expect() calls.

# Ignore certain paths
ignore:
  - "**/*.generated.rs"
  - "**/vendor/**"
  - "**/__tests__/**"

# Custom rules
rules:
  require_tests_for_new_functions: true
  max_function_length: 50
  required_review_categories: [security, performance]
```

---

## 7. Data Models

### 7.1 Core Types

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

class Severity(Enum):
    CRITICAL = "critical"    # Must fix before merge
    WARNING = "warning"      # Should fix, potential issues
    SUGGESTION = "suggestion"  # Nice to have improvements
    NITPICK = "nitpick"      # Style/formatting only

class Category(Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    STYLE = "style"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"

@dataclass
class ReviewContext:
    """Context provided to agents for informed reviews."""
    repo_name: str
    pr_number: int
    pr_title: str
    pr_description: str
    base_branch: str
    head_branch: str
    author: str
    changed_files_count: int
    additions: int
    deletions: int
    labels: List[str]
    repo_languages: List[str]
    custom_instructions: Optional[str] = None

@dataclass
class ConsolidatedFinding:
    """A finding that has been merged from multiple agents."""
    id: str
    file_path: str
    line_start: int
    line_end: Optional[int]
    severity: Severity
    category: Category
    title: str
    description: str
    suggested_fix: Optional[str]
    
    # Consensus metadata
    consensus_score: float  # 0.0 - 1.0 (% of agents that found this)
    agreeing_agents: List[str]
    confidence: float  # Average confidence across agents
    
    # Source tracking
    original_findings: List[ReviewFinding] = field(default_factory=list)

@dataclass
class ConsolidatedReview:
    """Final aggregated review output."""
    id: str
    created_at: datetime
    repo: str
    pr_number: int
    
    # Results
    findings: List[ConsolidatedFinding]
    summary: str
    
    # Metadata
    agent_count: int
    review_quality_score: float  # How confident we are in this review
    total_review_time_ms: int
    
    # Breakdown
    findings_by_severity: Dict[Severity, int]
    findings_by_category: Dict[Category, int]
    agent_reviews: List[AgentReview]  # Original reviews for transparency
```

---

## 8. CLI Interface

```bash
# Review a local diff
ai-reviewer review --diff ./changes.patch --output json

# Review a GitHub PR
ai-reviewer review-pr calimero-network/core 123

# Start webhook server
ai-reviewer serve --port 8080

# Validate configuration
ai-reviewer config validate

# List available agents
ai-reviewer agents list

# Test a single agent
ai-reviewer agents test claude --diff ./test.patch
```

### CLI Implementation

```python
import click
import asyncio

@click.group()
def cli():
    """AI Code Reviewer - Multi-agent code review system."""
    pass

@cli.command()
@click.option("--repo", required=True, help="Repository in owner/name format")
@click.option("--pr", required=True, type=int, help="PR number")
@click.option("--output", type=click.Choice(["github", "json", "markdown"]), default="github")
@click.option("--dry-run", is_flag=True, help="Don't post to GitHub")
async def review_pr(repo: str, pr: int, output: str, dry_run: bool):
    """Review a GitHub pull request."""
    config = load_config()
    orchestrator = create_orchestrator(config)
    
    click.echo(f"🔍 Reviewing PR #{pr} in {repo}...")
    click.echo(f"📡 Launching {len(orchestrator.agents)} agents...")
    
    # ... implementation
    
    click.echo("✅ Review complete!")

@cli.command()
@click.option("--port", default=8080, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
def serve(port: int, host: str):
    """Start the webhook server."""
    import uvicorn
    click.echo(f"🚀 Starting webhook server on {host}:{port}")
    uvicorn.run("ai_reviewer.webhook:app", host=host, port=port)

if __name__ == "__main__":
    cli()
```

---

## 9. Output Formats

### 9.1 GitHub PR Comment

```markdown
## 🤖 AI Code Review

**Reviewed by 3 agents** | Consensus score: 87% | Review time: 45s

### Summary
This PR introduces user authentication improvements. Found 2 critical issues 
that should be addressed, 3 warnings, and 5 suggestions.

---

### 🔴 Critical Issues (2)

#### 1. SQL Injection Vulnerability
**File:** `src/auth/login.py` (lines 45-48) | **Consensus:** 3/3 agents ✓

```python
# Current code
query = f"SELECT * FROM users WHERE username = '{username}'"
```

**Issue:** User input directly interpolated into SQL query.

**Suggested fix:**
```python
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, (username,))
```

> 🔒 *Found by: claude-security, gpt4-security, cursor-context*

---

#### 2. Missing Rate Limiting
**File:** `src/auth/login.py` (lines 12-15) | **Consensus:** 2/3 agents

Login endpoint lacks rate limiting, enabling brute force attacks.

> 🔒 *Found by: claude-security, gpt4-security*

---

### 🟡 Warnings (3)
<details>
<summary>Click to expand</summary>

1. **Inefficient loop** in `src/utils/parser.py:89` - O(n²) complexity
2. **Missing error handling** in `src/api/client.py:34`
3. **Deprecated API usage** in `src/auth/oauth.py:56`

</details>

### 💡 Suggestions (5)
<details>
<summary>Click to expand</summary>

1. Consider adding type hints to `process_user()`
2. Extract magic number `86400` to named constant
3. Add docstring to `AuthHandler` class
4. Consider using `pathlib` instead of `os.path`
5. Unused import `json` can be removed

</details>

---

<sub>🤖 Generated by [AI Code Reviewer](https://github.com/calimero-network/ai-code-reviewer) | 
[Configure](.ai-reviewer.yaml) | [Report Issue](https://github.com/calimero-network/ai-code-reviewer/issues)</sub>
```

### 9.2 JSON Output

```json
{
  "review_id": "rev_abc123",
  "created_at": "2026-02-03T10:30:00Z",
  "repo": "calimero-network/core",
  "pr_number": 123,
  "summary": "Found 2 critical issues, 3 warnings, 5 suggestions",
  "quality_score": 0.87,
  "agent_count": 3,
  "findings": [
    {
      "id": "find_001",
      "severity": "critical",
      "category": "security",
      "title": "SQL Injection Vulnerability",
      "file_path": "src/auth/login.py",
      "line_start": 45,
      "line_end": 48,
      "description": "User input directly interpolated into SQL query",
      "suggested_fix": "Use parameterized queries",
      "consensus_score": 1.0,
      "agreeing_agents": ["claude-security", "gpt4-security", "cursor-context"],
      "confidence": 0.95
    }
  ],
  "agent_reviews": [
    {
      "agent_id": "claude-security",
      "agent_type": "claude",
      "focus_areas": ["security", "architecture"],
      "findings_count": 8,
      "review_time_ms": 12500
    }
  ]
}
```

---

## 10. Project Structure

```
ai-code-reviewer/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── config.example.yaml
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── release.yaml
├── src/
│   └── ai_reviewer/
│       ├── __init__.py
│       ├── cli.py                 # CLI entry point
│       ├── config.py              # Configuration loading
│       ├── models/
│       │   ├── __init__.py
│       │   ├── findings.py        # ReviewFinding, ConsolidatedFinding
│       │   ├── review.py          # AgentReview, ConsolidatedReview
│       │   └── context.py         # ReviewContext
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py            # ReviewAgent ABC
│       │   ├── claude.py          # Claude implementation
│       │   ├── openai.py          # GPT-4 implementation
│       │   ├── cursor.py          # Cursor API implementation
│       │   └── prompts/
│       │       ├── security.txt
│       │       ├── performance.txt
│       │       └── style.txt
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── orchestrator.py    # AgentOrchestrator
│       │   └── aggregator.py      # ReviewAggregator
│       ├── github/
│       │   ├── __init__.py
│       │   ├── client.py          # GitHub API wrapper
│       │   ├── webhook.py         # FastAPI webhook server
│       │   └── formatter.py       # Output formatting
│       └── utils/
│           ├── __init__.py
│           ├── diff_parser.py     # Git diff parsing
│           └── embeddings.py      # For similarity clustering
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_orchestrator/
│   ├── test_aggregator/
│   └── fixtures/
│       └── sample_diffs/
└── docs/
    ├── configuration.md
    ├── custom_agents.md
    └── api_reference.md
```

---

## 11. Security Considerations

### 11.1 Secret Management
- API keys stored in environment variables, never in config files
- GitHub App private key stored securely (not in repo)
- Webhook secret validated for all incoming requests

### 11.2 Code Exposure
- Only diff and changed file contents sent to LLMs
- Option to exclude sensitive paths via config
- No credentials or secrets in review context

### 11.3 Rate Limiting
- Respect LLM provider rate limits
- Implement backoff and retry logic
- Queue large PRs to avoid burst costs

---

## 12. Metrics & Observability

```python
# Prometheus metrics
review_duration_seconds = Histogram(
    "ai_reviewer_review_duration_seconds",
    "Time to complete a review",
    ["repo", "agent_count"]
)

findings_total = Counter(
    "ai_reviewer_findings_total", 
    "Total findings by severity",
    ["severity", "category"]
)

agent_success_rate = Gauge(
    "ai_reviewer_agent_success_rate",
    "Success rate per agent type",
    ["agent_type"]
)
```

---

## 13. Future Enhancements (v2+)

1. **Learning from feedback**: Track which findings were accepted/dismissed
2. **Custom agent training**: Fine-tune models on repo-specific patterns
3. **IDE integration**: VS Code / Cursor extension for local reviews
4. **Self-hosted LLMs**: Support for Ollama, vLLM for air-gapped environments
5. **Review memory**: Remember past reviews for consistency
6. **Automatic fix suggestions**: Integration with ai-bounty-hunter for fixes

---

## 14. Implementation Roadmap

### Phase 1: Core (Week 1-2)
- [ ] Project setup (pyproject.toml, CI)
- [ ] Data models and types
- [ ] Agent base class and Claude implementation
- [ ] Basic orchestrator (parallel execution)
- [ ] Simple aggregator (deduplication)
- [ ] CLI for local diff review

### Phase 2: Multi-Agent (Week 3)
- [ ] GPT-4 agent implementation
- [ ] Cursor agent implementation
- [ ] Advanced aggregator (embeddings, consensus)
- [ ] Configuration system

### Phase 3: GitHub Integration (Week 4)
- [ ] GitHub App setup
- [ ] Webhook server
- [ ] PR comment formatting
- [ ] Inline comment support

### Phase 4: Polish (Week 5)
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Docker packaging
- [ ] Example workflows

---

## 15. Open Questions

1. **Embedding model for similarity**: Use OpenAI embeddings, sentence-transformers, or simpler text matching?
2. **Cost management**: How to balance quality vs. API costs for large PRs?
3. **Cursor API specifics**: What capabilities does the Cursor API expose for code understanding?
4. **Agent disagreement**: How to handle when agents have conflicting findings?

---

## Appendix A: Example Agent Prompts

### Security Review Prompt
```
You are a senior security engineer performing a code review. Your task is to 
identify security vulnerabilities in the provided code changes.

Focus areas:
- Injection attacks (SQL, command, XSS, etc.)
- Authentication and authorization flaws
- Cryptographic issues
- Data exposure and privacy
- Input validation
- Secure defaults

For each finding, provide:
1. File path and line numbers
2. Severity (critical/warning/suggestion)
3. Clear description of the vulnerability
4. Concrete suggestion for fixing it

Be thorough but precise. Only report issues you can clearly demonstrate in the code.
Do not speculate about issues that might exist elsewhere.

Code changes to review:
---
{diff}
---
```

---

*Document version: 0.1.0 | Last updated: 2026-02-03*

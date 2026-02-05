# AI Code Reviewer - Quick Context

> Read this file when you need a fast orientation to the codebase.

## What Is This?

**AI Code Reviewer** is a multi-agent system that reviews code using multiple LLMs in parallel, then combines their findings into a single consensus-based review.

```
PR Diff → [Claude, GPT-4, ...] → Aggregator → Consolidated Review
```

## Key Design Decisions

1. **Cursor API as single gateway** - All LLM access via Cursor, no direct provider SDKs
2. **Agents are independent** - Run in parallel, no shared state
3. **Consensus scoring** - Findings are weighted by how many agents agree
4. **Graceful degradation** - Works even if some agents fail

## Directory Map

```
src/ai_reviewer/
├── agents/           # LLM agents that perform reviews
│   ├── base.py       # ReviewAgent base class (read this first)
│   ├── cursor_client.py  # Unified LLM client
│   ├── security.py   # Security-focused agent
│   └── performance.py  # Performance-focused agent
├── orchestrator/     # Coordination layer
│   ├── orchestrator.py  # Runs agents in parallel
│   └── aggregator.py    # Combines results
├── github/           # GitHub integration
│   ├── client.py     # GitHub API wrapper
│   ├── webhook.py    # FastAPI webhook server
│   └── formatter.py  # Output formatting
├── models/           # Data structures (no logic)
│   ├── findings.py   # ReviewFinding, Severity, Category
│   ├── review.py     # AgentReview, ConsolidatedReview
│   └── context.py    # ReviewContext
├── config.py         # Configuration loading
└── cli.py            # CLI entry point
```

## Important Types

```python
# What agents produce
class AgentReview:
    agent_id: str
    findings: list[ReviewFinding]
    summary: str

# Individual finding
class ReviewFinding:
    file_path: str
    line_start: int
    severity: Severity  # critical, warning, suggestion, nitpick
    category: Category  # security, performance, logic, etc.
    title: str
    description: str

# Final output
class ConsolidatedReview:
    findings: list[ConsolidatedFinding]  # Deduplicated, scored
    agent_count: int
    quality_score: float
```

## Common Tasks

### Add a new agent type

1. Create `src/ai_reviewer/agents/new_agent.py`
2. Extend `ReviewAgent` base class
3. Define `MODEL`, `AGENT_TYPE`, `FOCUS_AREAS`, `SYSTEM_PROMPT`
4. Add to agent config in `config.yaml`

### Change how findings are aggregated

- Edit `src/ai_reviewer/orchestrator/aggregator.py`
- Key method: `ReviewAggregator.aggregate()`

### Modify GitHub output format

- Edit `src/ai_reviewer/github/formatter.py`
- Look for `to_github_markdown()` method

### Add new configuration option

1. Add field to appropriate dataclass in `config.py`
2. Parse it in `_parse_config()`
3. Document in `config.example.yaml`

## Testing

```bash
pytest                          # Run all tests
pytest tests/test_agents.py     # Test specific module
pytest -k "test_security"       # Run tests matching pattern
```

## Configuration

```yaml
# config.yaml
cursor:
  api_key: ${CURSOR_API_KEY}

agents:
  - name: security-reviewer
    model: claude-4.5-opus-high-thinking
    focus_areas: [security]
  - name: performance-reviewer
    model: gpt-5.2
    focus_areas: [performance]

orchestrator:
  min_agents_required: 2
  timeout_seconds: 120
```

## Key Invariants to Preserve

1. **All LLM calls go through `CursorClient`** - never direct API calls
2. **Agents are stateless** - each review is independent
3. **Async throughout** - no blocking I/O
4. **Graceful degradation** - some results better than none
5. **Type safety** - use enums for Severity/Category, not strings

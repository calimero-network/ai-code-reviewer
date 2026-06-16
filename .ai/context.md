# AI Code Reviewer - Quick Context

> Read this file when you need a fast orientation to the codebase.

## What Is This?

**AI Code Reviewer** is a multi-agent system that reviews code using multiple LLMs in parallel, then combines their findings into a single consensus-based review.

```
PR Diff → [Sonnet (Security), Sonnet (Logic), Sonnet (Patterns), Haiku (Style)] → Aggregator → Consolidated Review
```

## Key Design Decisions

1. **Anthropic Messages API** - All LLM access via `AnthropicClient` in `agents/anthropic_client.py`; models: `claude-sonnet-4-6` (main agents), `claude-haiku-4-5-20251001` (style/doc-gen)
2. **Agents are independent** - Run in parallel, no shared state
3. **Consensus scoring** - Findings are weighted by how many agents agree
4. **Graceful degradation** - Works even if some agents fail
5. **Protocol-based tool registry** - `ToolRegistryProtocol` defines the interface for tool access, enabling flexible tool implementations
6. **Strict import control** - Only `agents/anthropic_client.py` may import the Anthropic SDK; all other code routes LLM access through `AnthropicClient` (architecture invariant I1, enforced by ruff `flake8-tidy-imports` / `TID251`)
7. **Prompt caching** - Enabled by default when system prompts exceed minimum cacheable length (~1024 tokens); cache control breakpoints set on system blocks to optimize repeated requests

## Directory Map

```
src/ai_reviewer/
├── agents/           # LLM agents that perform reviews
│   ├── base.py           # ReviewAgent base class (read this first)
│   ├── anthropic_client.py  # Anthropic Messages API wrapper (tool-use loop, caching)
│   ├── security.py       # SecurityAgent + AuthenticationAgent (Sonnet)
│   ├── performance.py    # PerformanceAgent + LogicAgent (Sonnet)
│   └── patterns.py       # PatternsAgent (Sonnet) + StyleAgent (Haiku)
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
# Tool registry interface
class ToolRegistryProtocol(Protocol):
    """Structural interface the tool-use loop needs from a tool registry."""
    def tool_specs(self) -> list[dict[str, Any]]: ...
    async def execute(self, name: str, tool_input: dict[str, Any]) -> str: ...

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
pytest tests/integration -m integration  # Run live API integration tests
```

## Configuration

```yaml
# config.yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  default_model: claude-sonnet-4-6
  max_retries: 1
  max_combined_context_tokens: 80000
  enable_prompt_caching: true

agents:
  - name: security-reviewer
    model: claude-sonnet-4-6
    focus_areas: [security]
    max_tool_calls: 8
  - name: style-reviewer
    model: claude-haiku-4-5-20251001
    focus_areas: [style]
    allow_tool_use: false

orchestrator:
  min_agents_required: 2
  timeout_seconds: 120
```

## Key Invariants to Preserve

1. **Architecture Invariant I1: Single SDK importer** - Only `agents/anthropic_client.py` may import the Anthropic SDK (marked with `# noqa: TID251`). All other code must access LLMs through `AnthropicClient`. Enforced by ruff's `flake8-tidy-imports` (`TID251`) rule.
2. **Tool registries implement `ToolRegistryProtocol`** - enables structural typing and flexible tool implementations
3. **Agents are stateless** - each review is independent
4. **Async throughout** - no blocking I/O
5. **Graceful degradation** - some results better than none
6. **Type safety** - use enums for Severity/Category, not strings

## AnthropicClient Methods

**Main entry points:**
- `run_review(model, system, user_context, tools, ...)` → Full agent review with tool use, caching, and JSON schema
- `complete_simple(model, system, user, max_tokens, temperature)` → Lightweight completion with caching but no tools or schema; used for cross-review and other internal calls

Both methods log token usage and support prompt caching when `enable_prompt_caching` is true.
# Agents Module Rules

## Purpose

The `agents/` module contains all LLM agent implementations that perform code reviews. Each agent is specialized for particular review focuses. All LLM access goes through `AnthropicClient` (official Anthropic SDK).

## Key Types

```python
# Base class all agents inherit from
class ReviewAgent:
    MODEL: str              # Anthropic model ID (e.g. "claude-sonnet-4-6")
    AGENT_TYPE: str          # Agent classification (e.g. "security-reviewer")
    FOCUS_AREAS: list        # What this agent specializes in
    SYSTEM_PROMPT: str       # Instructions for the LLM
    THINKING_ENABLED: bool   # Enable adaptive thinking (default False)

# Return type for all reviews
@dataclass
class AgentReview:
    agent_id: str
    agent_type: str
    focus_areas: list[str]
    findings: list[ReviewFinding]
    summary: str
    review_time_ms: int
```

## File Structure

```
agents/
├── __init__.py              # Exports public API
├── anthropic_client.py      # AnthropicClient: Messages API wrapper with tool-use loop, caching
├── base.py                  # ReviewAgent base class
├── security.py              # SecurityAgent, AuthenticationAgent (Sonnet)
├── performance.py           # PerformanceAgent (Sonnet), LogicAgent (Sonnet)
└── patterns.py              # PatternsAgent (Sonnet), StyleAgent (Haiku)
```

## Invariants

### A1: All Agents Extend ReviewAgent

Never create standalone agent functions. Always inherit from `ReviewAgent`.

### A2: Agents Return JSON-Structured Findings

Output is enforced via `output_config.format = json_schema` on the Anthropic API. The schema is defined in `context/builder.py:FINDINGS_SCHEMA`.

### A3: Agents Are Stateless

No mutable state between `review()` calls. Each review is independent. Agents receive pre-built `system_blocks` and `user_blocks` at construction time.

### A4: Focus Areas Match System Prompt

If `FOCUS_AREAS = ["security"]`, the system prompt must emphasize security.

## Creating a New Agent

```python
# src/ai_reviewer/agents/new_focus.py

from ai_reviewer.agents.base import ReviewAgent

class NewFocusAgent(ReviewAgent):
    """Agent focused on [specific area]."""

    MODEL = "claude-sonnet-4-6"      # use claude-haiku-4-5-20251001 for style-only agents
    AGENT_TYPE = "new-focus-reviewer"
    FOCUS_AREAS = ["focus1", "focus2"]
    THINKING_ENABLED = False          # keep False — thinking adds quadratic cost in tool loops

    SYSTEM_PROMPT = """You are an expert in [area].
    Focus on:
    - Point 1
    - Point 2

    Be thorough but avoid false positives."""
```

Then register in `review.py`:
```python
_AGENT_CLASSES["new-focus-reviewer"] = NewFocusAgent
DEFAULT_AGENT_ORDER.append("new-focus-reviewer")
```

## AnthropicClient Usage

```python
# Agents don't call the client directly — base.py handles it.
# The review() method is inherited:

class MyAgent(ReviewAgent):
    MODEL = "claude-sonnet-4-6"
    THINKING_ENABLED = False

    # Override SYSTEM_PROMPT — that's usually all you need.
    SYSTEM_PROMPT = """..."""

# Construction happens in review.py:
agent = MyAgent(
    client=anthropic_client,
    agent_id="my-agent-0",
    system_blocks=system_blocks,     # Shared conventions + schema
    user_blocks=user_blocks,         # PR diff + files + neighbors
    tool_registry=registry,          # read_file/glob/grep tools
    thinking_enabled=True,           # Config override (optional)
)
review = await agent.review(diff="", file_contents={}, context=ctx)
```

## Severity Guidelines for Agents

| Severity     | When to Use                                  |
| ------------ | -------------------------------------------- |
| `critical`   | Security vulnerabilities, data loss, crashes |
| `warning`    | Bugs, performance issues, bad practices      |
| `suggestion` | Improvements, refactoring opportunities      |
| `nitpick`    | Style, formatting, minor preferences         |

## Anti-Patterns

1. **Don't parse raw LLM text** - Structured output via `output_config` handles this
2. **Don't catch all exceptions silently** - Let orchestrator handle failures
3. **Don't access GitHub/external APIs directly** - Use `ToolRegistry` for repo exploration
4. **Don't hardcode temperatures** - Use configuration
5. **Don't share state between reviews** - Create fresh state each call
6. **Don't import `anthropic` SDK directly** - Only `anthropic_client.py` should import it

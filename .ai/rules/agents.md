# Agents Module Rules

## Purpose

The `agents/` module contains all LLM agent implementations that perform code reviews. Each agent is specialized for particular review focuses.

## Key Types

```python
# Base class all agents inherit from
class ReviewAgent:
    MODEL: str           # LLM model identifier
    AGENT_TYPE: str      # Agent classification
    FOCUS_AREAS: list    # What this agent specializes in
    SYSTEM_PROMPT: str   # Instructions for the LLM

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
├── __init__.py          # Exports public API
├── base.py              # ReviewAgent base class
├── cursor_client.py     # Unified LLM access via Cursor API
├── security.py          # Security-focused agent
├── performance.py       # Performance-focused agent
└── patterns.py          # Pattern/consistency agent
```

## Invariants

### A1: All Agents Extend ReviewAgent

Never create standalone agent functions. Always inherit from `ReviewAgent`.

### A2: Agents Return JSON-Structured Findings

Use `client.complete_json()` to ensure structured output.

### A3: Agents Are Stateless

No mutable state between `review()` calls. Each review is independent.

### A4: Focus Areas Match System Prompt

If `FOCUS_AREAS = ["security"]`, the system prompt must emphasize security.

## Creating a New Agent

```python
# src/ai_reviewer/agents/new_focus.py

from ai_reviewer.agents.base import ReviewAgent

class NewFocusAgent(ReviewAgent):
    """Agent focused on [specific area]."""

    MODEL = "claude-4.5-opus-high-thinking"  # Model accessed via Cursor
    AGENT_TYPE = "new-focus"
    FOCUS_AREAS = ["focus1", "focus2"]

    SYSTEM_PROMPT = """You are an expert in [area].
    Focus on:
    - Point 1
    - Point 2

    Be thorough but avoid false positives."""
```

## CursorClient Usage

```python
# Always use the injected client
class MyAgent(ReviewAgent):
    def __init__(self, client: CursorClient, agent_id: str | None = None):
        super().__init__(client, agent_id)

    async def review(self, diff, files, context):
        # JSON responses for structured findings
        response = await self.client.complete_json(
            model=self.MODEL,
            system_prompt=self._get_system_prompt(),
            user_prompt=self._build_review_prompt(diff, files, context),
            temperature=0.3,  # Low temperature for consistency
        )
        return self._parse_response(response)
```

## Severity Guidelines for Agents

| Severity     | When to Use                                  |
| ------------ | -------------------------------------------- |
| `critical`   | Security vulnerabilities, data loss, crashes |
| `warning`    | Bugs, performance issues, bad practices      |
| `suggestion` | Improvements, refactoring opportunities      |
| `nitpick`    | Style, formatting, minor preferences         |

## Anti-Patterns

1. **Don't parse raw LLM text** - Use `complete_json()` for structured output
2. **Don't catch all exceptions silently** - Let orchestrator handle failures
3. **Don't access GitHub/external APIs** - Agents only process provided data
4. **Don't hardcode temperatures** - Use configuration
5. **Don't share state between reviews** - Create fresh state each call

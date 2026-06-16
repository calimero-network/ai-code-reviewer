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

# Structural interface for tool registries
class ToolRegistryProtocol(Protocol):
    def tool_specs(self) -> list[dict[str, Any]]: ...
    async def execute(self, name: str, tool_input: dict[str, Any]) -> str: ...
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

### I1: LLM SDK Access is Centralized

Only `agents/anthropic_client.py` may import the `anthropic` SDK directly. All other modules must access LLM functionality through `AnthropicClient`. This is enforced by `ruff` with the `flake8-tidy-imports` rule.

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
    tool_registry=registry,          # read_file/glob/grep tools (optional)
    thinking_enabled=True,           # Config override (optional)
)
review = await agent.review(diff="", file_contents={}, context=ctx)
```

### Simple Completions

For lightweight single-turn completions without tools or JSON schema (e.g., cross-review summarization), use `AnthropicClient.complete_simple()`:

```python
result = await client.complete_simple(
    model="claude-sonnet-4-6",
    system="You are a code reviewer.",
    user="Summarize these findings: ...",
    max_tokens=1024,
    temperature=0.2,
)
```

This method:
- Requires no tool registry
- Returns plain text (not structured JSON)
- Logs token usage on every call
- Respects prompt caching configuration (cache_control breakpoint on last system block)
- Is useful for operations that must go through `AnthropicClient` for invariant I1 compliance but don't need full agent infrastructure

## Tool Registry Interface

Tool registries implement `ToolRegistryProtocol`, providing:

- `tool_specs() -> list[dict[str, Any]]`: Returns JSON schema for available tools
- `async execute(name: str, tool_input: dict[str, Any]) -> str`: Executes a named tool with input

Pass `None` for `tool_registry` if the agent does not use tools.

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
6. **Don't import `anthropic` SDK directly** - Only `anthropic_client.py` should import it (enforced by invariant I1)
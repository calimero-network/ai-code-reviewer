# Architecture Rules

> **Canonical reference:** [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) is the comprehensive human-readable architecture documentation with flowcharts, scoring formulas, and the convergence state machine. This file is a quick-reference for AI agents — see the docs version for full detail.

## Purpose
AI Code Reviewer is a multi-agent system that orchestrates multiple LLMs to produce comprehensive, high-quality code reviews through consensus.

## Core Design Principles

### 1. Anthropic Messages API
- **All LLM access goes through `AnthropicClient`** in `agents/anthropic_client.py`
- Single API key (`ANTHROPIC_API_KEY`) manages all models
- Features: prompt caching, adaptive thinking, JSON-schema structured output, tool use
- Model switching via configuration (`claude-sonnet-4-6` / `claude-haiku-4-5-20251001`)

### 2. Agent Independence
- Each agent runs **independently** with no shared state during review
- Agents can fail without affecting others
- Results are aggregated post-execution
- Graceful degradation: review succeeds if `min_agents_required` complete

### 3. Consensus-Based Quality
- Multiple agents reviewing same code -> higher confidence findings
- Findings are **clustered by semantic similarity**
- Consensus score = (agents_agreeing / total_agents)
- Critical findings require higher consensus threshold

## System Flow

```
Input (PR/Diff)
       |
       v
+------------------+
|   Orchestrator   |---- Spawns N agents in parallel
+--------+---------+
         |
    +----+----+--------+--------+
    v         v        v        v
+--------+ +--------+ +--------+ +-------+
| Sonnet | | Sonnet | | Sonnet | | Haiku |  All via Anthropic API
| (Sec)  | | (Pat)  | | (Perf) | (Style)|
+---+---+ +---+---+ +---+---+ +---+---+
    |         |        |        |
    +----+----+--------+--------+
         v
+------------------+
|   Aggregator     |---- Clusters, deduplicates, scores
+--------+---------+
         |
         v
   ConsolidatedReview
```

## Invariants (Must Always Be True)

### I1: LLM Access Through AnthropicClient Only
```python
# Only agents/anthropic_client.py should import the SDK:
import anthropic  # OK in anthropic_client.py ONLY

# All other modules use the client:
from ai_reviewer.agents.anthropic_client import AnthropicClient
```

### I2: Agents Must Return Structured Data
All agents return `AgentReview` objects with typed `ReviewFinding` lists. Output schema enforced server-side via `output_config.format = json_schema`.

### I3: Async All The Way
All LLM calls and orchestration use `async/await`. No blocking calls in review path.

### I4: Graceful Degradation
System must produce valid output even if some agents fail, as long as `min_agents_required` succeed.

### I5: Configuration Over Code
Agent behavior (model, temperature, focus areas, thinking) controlled via config, not hardcoded. Per-agent `thinking_enabled` in YAML overrides class defaults.

## Module Responsibilities

| Module | Single Responsibility |
|--------|----------------------|
| `agents/` | LLM agent implementations + `AnthropicClient` |
| `context/` | Prompt assembly: system blocks, user blocks, neighbor heuristics, convention fetching |
| `tools/` | `ToolRegistry`: `read_file`/`glob`/`grep` for Claude tool use (GitHub Contents API) |
| `session.py` | Per-review transient state: GitHub quota, file cache, tree cache |
| `orchestrator/` | Parallel execution + result aggregation |
| `github/` | GitHub API integration only |
| `models/` | Data structures, no business logic |
| `config.py` | Configuration loading and validation |
| `cli.py` | CLI entry points only |
| `review.py` | Main pipeline: `review_pr()`, agent dispatch, cross-review, aggregation |

## Extension Points

To add new functionality:
- **New agent**: Add class in `agents/`, register in `review.py:_AGENT_CLASSES`
- **New review focus**: Create specialized agent with custom `SYSTEM_PROMPT`
- **New output format**: Add formatter in `github/formatter.py`
- **New trigger**: Add handler, keep core review logic unchanged
- **New tool**: Add to `tools/repo_tools.py:TOOL_SPECS` + implement in `ToolRegistry`

## Anti-Patterns to Avoid

1. **Coupling agents to GitHub** - Agents review diffs, not PR objects
2. **Blocking calls in async code** - Use `asyncio` throughout
3. **Hardcoding model names** - Use config
4. **Silent failures** - Log and propagate errors properly
5. **Shared mutable state between agents** - Agents are isolated
6. **Importing `anthropic` SDK outside `anthropic_client.py`** - Single gateway

# Architecture Rules

> **Canonical reference:** [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) is the comprehensive human-readable architecture documentation with flowcharts, scoring formulas, and the convergence state machine. This file is a quick-reference for AI agents — see the docs version for full detail.

## Purpose
AI Code Reviewer is a multi-agent system that orchestrates multiple LLMs to produce comprehensive, high-quality code reviews through consensus.

## Core Design Principles

### 1. Single Gateway Architecture
- **All LLM access goes through Cursor API** - never direct provider SDKs
- Single API key management simplifies configuration
- Consistent request/response format across models
- Model switching via configuration, not code changes

### 2. Agent Independence
- Each agent runs **independently** with no shared state during review
- Agents can fail without affecting others
- Results are aggregated post-execution
- Graceful degradation: review succeeds if `min_agents_required` complete

### 3. Consensus-Based Quality
- Multiple agents reviewing same code → higher confidence findings
- Findings are **clustered by semantic similarity**
- Consensus score = (agents_agreeing / total_agents)
- Critical findings require higher consensus threshold

## System Flow

```
Input (PR/Diff)
       │
       ▼
┌──────────────────┐
│   Orchestrator   │──── Spawns N agents in parallel
└────────┬─────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Claude │ │ GPT-4 │ │Claude │  Each via Cursor API
│(Sec)  │ │(Perf) │ │(Arch) │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    └────┬────┴────────┘
         ▼
┌──────────────────┐
│   Aggregator     │──── Clusters, deduplicates, scores
└────────┬─────────┘
         │
         ▼
   ConsolidatedReview
```

## Invariants (Must Always Be True)

### I1: No Direct LLM Provider Dependencies
```python
# ✅ Correct - use Cursor client
from ai_reviewer.agents.cursor_client import CursorClient

# ❌ Wrong - never direct imports
import anthropic
import openai
```

### I2: Agents Must Return Structured Data
All agents return `AgentReview` objects with typed `ReviewFinding` lists. Never raw strings or unstructured JSON.

### I3: Async All The Way
All LLM calls and orchestration use `async/await`. No blocking calls in review path.

### I4: Graceful Degradation
System must produce valid output even if some agents fail, as long as `min_agents_required` succeed.

### I5: Configuration Over Code
Agent behavior (model, temperature, focus areas) controlled via config, not hardcoded.

## Module Responsibilities

| Module | Single Responsibility |
|--------|----------------------|
| `agents/` | Individual LLM agent implementations |
| `orchestrator/` | Parallel execution + result aggregation |
| `github/` | GitHub API integration only |
| `models/` | Data structures, no business logic |
| `config.py` | Configuration loading and validation |
| `cli.py` | CLI entry points only |

## Extension Points

To add new functionality:
- **New LLM model**: Add agent class in `agents/`, configure via YAML
- **New review focus**: Create specialized agent with custom system prompt
- **New output format**: Add formatter in `github/formatter.py`
- **New trigger**: Add handler, keep core review logic unchanged

## Anti-Patterns to Avoid

1. **Coupling agents to GitHub** - Agents review diffs, not PR objects
2. **Blocking calls in async code** - Use `asyncio` throughout
3. **Hardcoding model names** - Use config
4. **Silent failures** - Log and propagate errors properly
5. **Shared mutable state between agents** - Agents are isolated

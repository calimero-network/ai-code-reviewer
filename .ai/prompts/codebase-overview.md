# Codebase Overview Prompt

You are an AI assistant helping with the AI Code Reviewer codebase.

## What This Project Does

AI Code Reviewer is a multi-agent system that:
1. Takes a code diff (from a PR or local changes)
2. Sends it to multiple LLM agents in parallel (Claude, GPT-4, etc.)
3. Each agent reviews from a different perspective (security, performance, etc.)
4. Aggregates findings using consensus scoring
5. Outputs a unified review with confidence scores

## Architecture Summary

```
User Input (PR/Diff)
        │
        ▼
┌───────────────────┐
│   CLI / Webhook   │  Entry points
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Orchestrator    │  Coordinates parallel agents
└─────────┬─────────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
┌──────┐ ┌──────┐ ┌──────┐
│Agent1│ │Agent2│ │Agent3│  All via Cursor API
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   └────────┼────────┘
            ▼
┌───────────────────┐
│    Aggregator     │  Dedupe, cluster, score
└─────────┬─────────┘
          │
          ▼
   ConsolidatedReview
          │
          ▼
┌───────────────────┐
│    Formatter      │  GitHub, JSON, CLI output
└───────────────────┘
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/ai_reviewer/agents/` | LLM agent implementations |
| `src/ai_reviewer/orchestrator/` | Parallel execution + aggregation |
| `src/ai_reviewer/github/` | GitHub API integration |
| `src/ai_reviewer/models/` | Data structures |
| `.cursor/rules/` | AI rules per module |
| `.ai/` | AI context and automation |

## Key Files to Understand

1. **`agents/base.py`** - ReviewAgent base class, core interface
2. **`orchestrator/orchestrator.py`** - Parallel execution logic
3. **`orchestrator/aggregator.py`** - Consensus and deduplication
4. **`config.py`** - Configuration system
5. **`cli.py`** - Entry points

## Important Patterns

1. **Single Gateway**: All LLM access via `CursorClient`, never direct APIs
2. **Async Everything**: All I/O is async/await
3. **Typed Data**: Dataclasses with type hints everywhere
4. **Agent Independence**: Agents have no shared state

## When Asked to Make Changes

1. Read the relevant `.cursor/rules/` file first
2. Follow existing patterns in the module
3. Maintain type safety with hints
4. Keep async patterns consistent
5. Update docs if public API changes

## Common Tasks

- **Add new agent**: See `.cursor/rules/agents.md`
- **Change aggregation**: See `.cursor/rules/orchestrator.md`
- **Modify GitHub output**: See `.cursor/rules/github.md`
- **Add config option**: See `config.py` patterns

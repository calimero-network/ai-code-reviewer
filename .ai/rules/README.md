# AI Rules for AI Code Reviewer

This directory contains structured rules and context that help AI agents (Cursor, review bots, documentation bots) understand and work with this codebase efficiently.

## Directory Structure

```
.ai/rules/
├── README.md           # This file - explains the rules system
├── architecture.md     # High-level architecture and design decisions
├── agents.md           # Rules for the agents module
├── orchestrator.md     # Rules for orchestration and aggregation
├── github.md           # GitHub integration patterns
├── models.md           # Data model conventions
└── conventions.md      # Coding style and conventions
```

## How to Use These Rules

### For AI Assistants (Cursor, Claude, etc.)
When working on this codebase, start by reading:
1. `architecture.md` - Understand the overall system
2. The specific module rule file for what you're working on
3. `conventions.md` - Follow consistent patterns

### For Review Agents
Review agents should validate against:
- Architecture invariants in `architecture.md`
- Module-specific rules in component files
- Coding conventions in `conventions.md`

### For Documentation Bot
The documentation bot should:
1. Read `.ai/doc-bot.md` for behavior instructions
2. Cross-reference code changes against these rules
3. Suggest updates when rules or docs become stale

## Keeping Rules Updated

Rules should be updated when:
- ✅ Adding new modules or major components
- ✅ Changing core architecture decisions
- ✅ Introducing new patterns or conventions
- ❌ NOT for minor implementation details

## Rule File Format

Each rule file follows this structure:
```markdown
# Module Name

## Purpose
What this module does and why it exists.

## Key Concepts
Important types, patterns, or abstractions.

## Invariants
Things that must always be true.

## Patterns
Common patterns to follow.

## Anti-Patterns
What to avoid.

## Dependencies
What this module depends on.
```

# AI Context & Automation

This directory contains files that help AI tools understand and work with this codebase.

## Directory Structure

```
.ai/
├── README.md        # This file
├── context.md       # Quick codebase overview (read first!)
├── doc-bot.md       # Documentation bot instructions
├── prompts/         # Reusable prompts for AI tasks
│   ├── codebase-overview.md
│   └── review-documentation.md
└── rules/           # Detailed rules per module
    ├── README.md
    ├── architecture.md
    ├── agents.md
    ├── orchestrator.md
    ├── github.md
    ├── models.md
    └── conventions.md
```

## Quick Context for AI Assistants

When you need a fast overview, read `context.md`. It provides:
- What the project does
- Key architectural decisions
- Important files to know about
- Common tasks and how to do them

For deeper module-specific work, check `rules/<module>.md`.

## Documentation Bot

The documentation bot (`doc-bot.md`) runs on PRs and:
1. Detects what files changed
2. Checks if documentation needs updates
3. Suggests changes to README, DESIGN.md, or AI rules
4. Posts a comment or creates a follow-up PR

## File Purposes

| File | Purpose | Read When |
|------|---------|-----------|
| `context.md` | Quick overview | Starting any task |
| `rules/*.md` | Module details | Deep work on specific module |
| `doc-bot.md` | Bot instructions | Configuring automation |
| `prompts/*.md` | AI prompts | Customizing AI behavior |

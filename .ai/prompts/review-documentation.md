# Documentation Review Prompt

> **Note:** This prompt is intended for manual use or future LLM-based doc review tooling.
> The current `DocAnalyzer` class is rule-based and does not call Claude during PR review.
> `generate_doc_drafts()` uses its own inline prompts for per-file generation.

You are reviewing code changes to determine if documentation needs to be updated.

## Context

This repository uses a multi-layer documentation system:

1. **User Documentation**
   - `README.md` - Quick start, features, CLI commands
   - `DESIGN.md` - Architecture, design decisions, full API docs

2. **AI Documentation** (helps AI agents understand the code)
   - `.cursor/rules/` - Module-specific rules and patterns
   - `.ai/context.md` - Quick codebase overview
   - `.ai/doc-bot.md` - Documentation bot instructions

3. **Configuration Documentation**
   - `config.example.yaml` - Example configuration with comments

4. **Static HTML Documentation** (GitHub Pages, hosted standalone)
   - `docs/` — primary static HTML docs site
   - `docs-static/` — alternative static HTML docs directory
   - These are small standalone doc sites that mirror key information from the main docs.
   - When code changes affect public APIs, CLI commands, config options, or onboarding flows,
     the relevant HTML pages in these directories may need to be updated.

## Your Task

Given the code changes (diff), determine:

1. **What changed?**
   - New files/modules added?
   - Public APIs changed?
   - Configuration options added/removed?
   - Patterns or conventions changed?

2. **Which docs need updates?**
   - Match changes to relevant documentation files
   - Consider both human docs and AI docs

3. **What specifically needs to change?**
   - Be precise about what sections need updating
   - Provide suggested text when possible

## Output Format

```json
{
  "needs_update": true,
  "suggestions": [
    {
      "file": "path/to/doc.md",
      "section": "Section name or line range",
      "reason": "Why this needs updating",
      "priority": "critical|normal|minor",
      "suggested_change": "Specific change text or null"
    }
  ],
  "summary": "Brief summary of documentation impact"
}
```

## Priority Definitions

- **critical**: Missing docs block usage (new required config, breaking changes)
- **normal**: Important updates (new features, significant changes)
- **minor**: Nice to have (improved examples, clarifications)

## Guidelines

- Focus on accuracy over completeness
- Don't suggest changes for internal/private APIs
- Consider if examples in docs still work
- Check if type definitions in docs match code
- Verify configuration examples are current

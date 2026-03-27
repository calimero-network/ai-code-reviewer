# AI Code Reviewer

**Multi-agent code review system that orchestrates multiple LLMs to produce comprehensive, consensus-based code reviews.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AI Code Reviewer takes a different approach to automated code review: instead of relying on a single model, it orchestrates **multiple specialized agents** that review code from different perspectives — security, performance, and code quality — then combines their findings into a unified, confidence-scored review.

### Key Features

- **Multi-Agent Architecture**: Run 2–5+ LLM agents in parallel, each with a specialized focus area
- **Consensus-Based Scoring**: Findings are weighted by how many agents agree, reducing false positives
- **Single API Key**: All models (Claude, GPT-4, etc.) accessed through the Cursor unified API
- **GitHub Integration**: Automatic PR reviews via webhooks, with inline comments and thread resolution
- **Incremental Reviews**: Delta tracking detects new, fixed, and open findings across pushes — with convergence logic that stops reviewing when findings stabilize

For the full technical deep-dive — pipeline flowcharts, scoring formulas, convergence state machine, and prompt engineering — see the **[Architecture Documentation](docs/ARCHITECTURE.md)**.

---

## Quick Start

```bash
# Install
pip install ai-code-reviewer

# Export credentials
export CURSOR_API_KEY=cur_...
export GITHUB_TOKEN=ghp_...

# Review a GitHub PR
ai-reviewer review-pr calimero-network/core 123

# Review a local diff
git diff main | ai-reviewer review --output markdown
```

---

## How It Works

All LLM agents access Claude, GPT-4, and other models through the Cursor unified API — a single key, consistent interface, and codebase context for pattern-aware reviews.

```mermaid
flowchart LR
    PR["PR Diff"] --> Cursor["Cursor API\n(Unified LLM Gateway)"]

    subgraph Agents["Parallel Agent Execution"]
        A1["Claude\n(Security)"]
        A2["GPT-4\n(Performance)"]
        A3["Claude\n(Patterns)"]
    end

    Cursor --> Agents

    Agents --> Agg["Review Aggregator\n• Cluster similar findings\n• Compute consensus scores\n• Rank by severity × agreement"]
    Agg --> Delta["Delta Tracking\n• New / fixed / open findings\n• Convergence detection"]
    Delta --> Out["Consolidated Review\n(GitHub / JSON / MD)"]
```

For a detailed breakdown of the pipeline, scoring formulas, and convergence logic, see the **[Architecture Documentation](docs/ARCHITECTURE.md)**.

---

## Configuration

Create `config.yaml`:

```yaml
cursor:
  api_key: ${CURSOR_API_KEY}

github:
  token: ${GITHUB_TOKEN}  # or Classic PAT for thread resolution (see below)

agents:
  - name: security-reviewer
    model: claude-4.5-opus-high-thinking
    focus_areas: [security, architecture]

  - name: performance-reviewer
    model: gpt-5.2
    focus_areas: [performance, logic]

  - name: patterns-reviewer
    model: claude-4.5-opus-high-thinking
    focus_areas: [consistency, patterns]
    include_codebase_context: true

orchestrator:
  timeout_seconds: 120
  min_agents_required: 2
```

---

## CLI Commands

```bash
# Review a GitHub PR
ai-reviewer review-pr <owner/repo> <pr-number>

# Review a local diff or specific commit
ai-reviewer review --diff <file>
ai-reviewer review --commit <sha>

# Start webhook server
ai-reviewer serve --port 8080

# Configuration
ai-reviewer config validate
ai-reviewer config show

# Agent utilities
ai-reviewer agents list
ai-reviewer agents test <type>
```

---

## Output Example

```
Reviewed by 3 agents  |  Quality score: 87%

CRITICAL (1)
  SQL Injection in auth/login.py:45  [3/3 agents]
  User input interpolated directly into SQL query without parameterization.

WARNING (2)
  Missing rate limiting on /api/login  [2/3 agents]
  Inefficient O(n²) loop in process_batch()  [2/3 agents]

SUGGESTION (3)
  Add type hints to process_user()
  Extract magic number 86400 to a named constant
  Add docstring to AuthHandler
```

---

## Repository Configuration

Add `.ai-reviewer.yaml` to the root of any reviewed repository to customize behavior:

```yaml
# Exclude generated or vendored files from review
ignore:
  - "**/*.generated.rs"
  - "**/vendor/**"

# Append custom instructions to a specific agent's prompt
agents:
  - name: security-reviewer
    custom_prompt_append: |
      This is a Rust codebase using eyre for errors.
      Flag all unwrap() calls.

# Review policy
policy:
  require_human_review_for: [security]
  block_on_critical: true
```

---

## GitHub Actions Setup

### Basic Setup (`GITHUB_TOKEN`)

The default `GITHUB_TOKEN` provided by GitHub Actions is sufficient for most features:

- Posting reviews and inline comments
- Adding reactions
- Posting "Resolved" replies

It cannot resolve review threads (collapsing them in the UI), which requires a Classic PAT.

### Full Features (Classic Personal Access Token)

> **Note:** Fine-grained PATs do not support the `resolveReviewThread` GraphQL mutation. Use a Classic PAT with `repo` scope.

1. Create a [Classic Personal Access Token](https://github.com/settings/tokens/new) with the `repo` scope.

2. Add it as a repository secret named `GH_PAT`:
   ```
   Settings → Secrets and variables → Actions → New repository secret
   Name: GH_PAT
   Value: ghp_xxxxxxxxxxxxxxxxxxxx
   ```

3. The workflow uses `GH_PAT` automatically when present, falling back to `GITHUB_TOKEN`.

For production deployments, prefer a dedicated service account and rotate tokens regularly. GitHub Apps with fine-grained permissions are the recommended long-term approach.

---

## Development

```bash
git clone https://github.com/calimero-network/ai-code-reviewer
cd ai-code-reviewer

pip install -e ".[dev]"

pytest
ruff check .
mypy src/
```

---

## AI Rules & Documentation

The repository ships structured AI context to help coding assistants work with the codebase effectively.

```
.ai/
├── context.md           # Codebase overview — read first
├── doc-bot.md           # Documentation bot instructions
├── prompts/             # Reusable AI prompts
└── rules/               # Per-module design rules
    ├── architecture.md  # High-level design & invariants
    ├── agents.md        # Agent module patterns
    ├── orchestrator.md  # Orchestration rules
    ├── github.md        # GitHub integration patterns
    ├── models.md        # Data model conventions
    └── conventions.md   # Coding style guide
```

PRs that modify source code automatically trigger a documentation bot that analyzes which docs may need updating and posts suggestions as PR comments. Configure via `.ai-reviewer.yaml`.

---

## Related Projects

- [ai-bounty-hunter](https://github.com/calimero-network/ai-bounty-hunter) - Automatic bounty fixing
- [pr-agent](https://github.com/Codium-ai/pr-agent) - Single-agent PR reviews
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) - GitHub issue automation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<sub>Built with ❤️ by [Calimero Network](https://github.com/calimero-network)</sub>

# 🤖 AI Code Reviewer

**Multi-agent code review system that orchestrates multiple LLMs to produce comprehensive, high-quality code reviews.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AI Code Reviewer takes a different approach to automated code review: instead of relying on a single AI model, it orchestrates **multiple specialized agents** that review code from different perspectives (security, performance, architecture, etc.) and combines their findings into a unified, confidence-scored review.

### Key Features

- 🔀 **Multi-Agent Architecture**: Run 2-5+ LLM agents in parallel, each with specialized focus
- 🎯 **Consensus-Based Scoring**: Findings are weighted by how many agents agree
- 🔑 **Single API Key**: All models (Claude, GPT-4, etc.) accessed via Cursor API
- 🐙 **GitHub Integration**: Automatic PR reviews via webhooks
- 📊 **Actionable Output**: Prioritized findings with suggested fixes

---

## Quick Start

```bash
# Install
pip install ai-code-reviewer

# Configure - Only 2 API keys needed!
# Cursor API provides unified access to Claude, GPT-4, and other models
export CURSOR_API_KEY=cur_...
export GITHUB_TOKEN=ghp_...

# Review a GitHub PR
ai-reviewer review-pr calimero-network/core 123

# Review a local diff
git diff main | ai-reviewer review --output markdown
```

---

## How It Works

All LLM agents access Claude, GPT-4, and other models through **Cursor's unified API**:
- ✅ **Single API key** for all models
- ✅ **Consistent interface** across different LLMs
- ✅ **Codebase context** for pattern-aware reviews

```
                            PR Diff
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              Cursor API (Unified LLM Gateway)                 │
│                                                               │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │   Claude    │   │    GPT-4    │   │   Claude    │       │
│   │ (Security)  │   │(Performance)│   │ (Patterns)  │       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│          │                 │                 │               │
│          └─────────────────┼─────────────────┘               │
│                            ▼                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                Review Aggregator                      │  │
│   │  • Cluster similar findings                           │  │
│   │  • Compute consensus scores                           │  │
│   │  • Rank by severity × agreement                       │  │
│   └──────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
              Consolidated Review (GitHub / JSON / MD)
```

---

## Configuration

Create `config.yaml`:

```yaml
# Single Cursor API key for all LLM models
cursor:
  api_key: ${CURSOR_API_KEY}

# GitHub integration
github:
  token: ${GITHUB_TOKEN}

# Agents - different models, same Cursor API
agents:
  - name: security-reviewer
    model: claude-3-opus-20240229
    focus_areas: [security, architecture]
    
  - name: performance-reviewer
    model: gpt-4-turbo-preview
    focus_areas: [performance, logic]
    
  - name: patterns-reviewer
    model: claude-3-opus-20240229
    focus_areas: [consistency, patterns]
    include_codebase_context: true

orchestrator:
  timeout_seconds: 120
  min_agents_required: 2
```

---

## CLI Commands

```bash
# Review Commands
ai-reviewer review-pr <owner/repo> <pr-number>  # Review GitHub PR
ai-reviewer review --diff <file>                 # Review local diff
ai-reviewer review --commit <sha>                # Review specific commit

# Server Commands
ai-reviewer serve --port 8080                    # Start webhook server

# Configuration
ai-reviewer config validate                      # Check configuration
ai-reviewer config show                          # Show active config

# Agents
ai-reviewer agents list                          # List available agents
ai-reviewer agents test <type>                   # Test single agent
```

---

## Output Example

```markdown
## 🤖 AI Code Review

**Reviewed by 3 agents** | Consensus score: 87%

### 🔴 Critical (1)
**SQL Injection** in `auth/login.py:45` | 3/3 agents ✓
> User input interpolated into SQL query

### 🟡 Warning (2)  
**Missing rate limiting** | 2/3 agents
**Inefficient O(n²) loop** | 2/3 agents

### 💡 Suggestions (3)
- Add type hints to `process_user()`
- Extract magic number to constant
- Add docstring to `AuthHandler`
```

---

## Repository Configuration

Add `.ai-reviewer.yaml` to your repo for custom settings:

```yaml
# Ignore generated files
ignore:
  - "**/*.generated.rs"
  - "**/vendor/**"

# Custom instructions for agents
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

## Development

```bash
# Clone
git clone https://github.com/calimero-network/ai-code-reviewer
cd ai-code-reviewer

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linters
ruff check .
mypy src/
```

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

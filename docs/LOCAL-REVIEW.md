# Local review, before the PR exists

Runs the same multi-agent pipeline the PR bot runs, against local changes, from
inside a Claude Code session.
No API key, no pull request, nothing posted to GitHub.

The reviewers are subagents in your session rather than separate processes, and
everything that is not an LLM call - clustering, consensus scoring, per-severity
confidence floors, cross-file dedup, the adaptive cap, fix validation - stays in
Python. So the local review applies the same rules as the PR review, not an
approximation of them.

## What it looks like

```
/ai-review
```

```
Reviewed working tree - 3 agent(s), 41200ms

CRITICAL (1)
  src/client.py:412  Missing timeout on the retry path
    conf 0.92 - 3/3 agents - fix ready (validated)

WARNING (2)
  src/review.py:1590  Ignore patterns applied after the secret scan
    conf 0.75 - 2/3 agents - prose fix only

4 lower-severity finding(s) collapsed - run with --all to expand
```

`fix ready (validated)` means the replacement was spliced into the file in memory
and the result still parses, so it can be applied mechanically.
`prose fix only` means a human or an agent has to write the fix.

## Setup

### One repository

Nothing to install if the package is already a dev dependency:

```bash
pip install -e .          # provides the `ai-reviewer` command
```

The skill and the reviewer agent are checked in under `.claude/`, so they are
available in this repository as soon as you have it cloned.

### Every repository, every session

Install the command once so it is on `PATH` everywhere:

```bash
uv tool install ai-code-reviewer      # or: pipx install ai-code-reviewer
ai-reviewer --version
```

Then copy the skill and the agent into your user-level Claude Code directory, so
every session in every project can use them:

```bash
CFG="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
mkdir -p "$CFG/skills" "$CFG/agents"
cp -r .claude/skills/ai-review "$CFG/skills/"
cp .claude/agents/code-reviewer-readonly.md "$CFG/agents/"
```

Start a new session and `/ai-review` is available anywhere.
Repositories with no `.ai-reviewer.yaml` get the built-in defaults.

## Scopes

| Command | Reviews |
| --- | --- |
| `/ai-review` | uncommitted changes, **including untracked files** |
| `/ai-review --staged` | the index only |
| `/ai-review --base main` | `main...HEAD` |
| `/ai-review --agents 2` | fewer reviewer profiles |
| `/ai-review --all` | expand suggestions and nitpicks |

Untracked files are included deliberately: `git diff` omits them, which would hide
brand-new files - the ones most likely to contain something worth catching.

Agent count scales down with diff size, the same way the PR path does, so a
two-line change does not spend three reviewers.

## Configuration

Per-repository settings live in `.ai-reviewer.yaml` and are shared with the PR
path, so both reviews behave the same:

```yaml
agents:                       # which profiles run, and on which model
  - name: security-reviewer
    model: claude-sonnet-5
  - name: logic-reviewer
    model: claude-sonnet-5
  - name: patterns-reviewer
    model: claude-sonnet-5

ignore:                       # never reviewed, locally or as a PR
  - "generated/**"
  - "**/vendor/**"

aggregator:                   # per-severity confidence floors
  min_confidence_critical: 0.5
  min_confidence_warning: 0.6
```

The local path runs no cross-review round, so it applies the conservative
confidence floors rather than the lower ones the PR path uses when three or more
agents cross-check each other.

## Using it without a Claude Code session

The two commands the skill drives are ordinary CLI commands, so any orchestrator
can use them:

```bash
D=$(mktemp -d) && mkdir -p "$D/out"

# 1. build one self-contained brief per reviewer profile (no LLM calls)
ai-reviewer prompts --out "$D" --agents 3

# 2. have your reviewers answer each brief, writing JSON to "$D/out/<agent-name>.json"
#    The filename must match the agent name - it carries the attribution that
#    drives consensus scoring.

# 3. consolidate (no LLM calls)
ai-reviewer consolidate "$D"/out/*.json --scope "working tree"
```

Each brief states the exact JSON shape required.
`ai-reviewer consolidate --output json` emits machine-readable findings including
`suggested_replacement` and `fix_validated`, which is what a fix loop needs.

## Safety

Reviewer agents are read-only by construction: the agent definition allows only
`Read`, `Grep` and `Glob`, and the harness enforces that allowlist rather than
trusting the prompt.

This matters because the diff under review is untrusted input sitting in an
agent's context while it runs on a real checkout.
The agent definition also tells reviewers that instructions embedded in a diff are
a finding to report, not an order to follow.

Repository reads are confined to the repository: `read_repo_file` resolves each
path and rejects anything that lands outside the root, which covers both `..`
traversal and absolute paths.
Nothing is committed or pushed - fixes land in the working tree for you to read as
one `git diff`.

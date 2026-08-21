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

Two pieces: the `ai-reviewer` command, and the skill plus reviewer agent that
drive it.

### Every repository, every session

```bash
uv tool install git+https://github.com/calimero-network/ai-code-reviewer
```

```
/plugin marketplace add calimero-network/ai-code-reviewer
/plugin install ai-review@calimero
```

Start a new session and `/ai-review` works in any project.
Repositories with no `.ai-reviewer.yaml` get the built-in defaults.

The package is not on PyPI, so install from the repository rather than by name -
`ai-code-reviewer` on PyPI is an unrelated project.

### This repository

`pip install -e .` provides the command, and the skill and agent are already
checked in under `.claude/`, so a clone needs no plugin install.

The plugin points at that same `.claude` directory rather than a copy, so the
project-local skill and the published plugin cannot drift apart.

### Updating an existing setup

The command and the plugin are separate installs, so both move:

```bash
uv tool install --force git+https://github.com/calimero-network/ai-code-reviewer
```

```
/plugin update ai-review@calimero
```

Start a new session afterwards. `plugin update` compares version strings, so a
change under `.claude/` only reaches an existing install once the version in
`pyproject.toml` moves - CI requires that in the same pull request.

### Without installing anything

`uvx` runs the command straight from the repository, which is enough for the
non-session workflow below:

```bash
uvx --from git+https://github.com/calimero-network/ai-code-reviewer ai-reviewer --version
```

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

## Reviewing a pull request

`/ai-review` reviews changes that have no pull request yet.
`/review-contribution` reviews one that does, and posts the result.

```
/review-contribution https://github.com/calimero-network/core/pull/3573
```

It resolves the pull request, checks it out as a detached worktree - your own
clone's HEAD, index and working tree are never touched - runs the same reviewer
subagents, and posts one GitHub review: a summary body plus inline comments, with
validated fixes as clickable suggestion blocks.

The worktree comes from a clone of that repository if you are standing in one, or
from a blobless cache clone under `~/.cache/ai-reviewer` if you are not.
`--repo-path <dir>` names the clone explicitly, and is remembered for next time.

Posting needs a GitHub token: `GITHUB_TOKEN`, `github.token` in the config, or
whatever `gh auth token` returns.
The review posts under **your** GitHub identity and never approves.
`--dry-run` prints exactly what would be posted and posts nothing.

Re-running on the same pull request is safe: findings unchanged since the last
review are not posted again, and comments for findings that are now fixed are
resolved.

Because the local path runs no cross-review round, published findings use the
conservative confidence floors - fewer comments, at a higher bar, than the API
path would post.

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
ai-reviewer consolidate "$D"/out/*.json
```

Pass the same scope flag (`--staged` / `--base <ref>`) to both commands.
The finding cap scales with the size of the reviewed diff, so consolidating a
staged review without `--staged` measures a clean working tree.

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

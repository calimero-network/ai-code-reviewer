---
name: review-contribution
description: Review a GitHub pull request with the project's multi-agent reviewer and post the review. Use when given a PR URL to review, or asked to review someone's contribution. Spawns read-only reviewer subagents, consolidates in Python, and posts inline comments to the PR.
---

# Review a pull request, and post it

Runs the repository's own review pipeline against a real pull request, using
subagents in this session rather than the Anthropic API, then posts the result as
a GitHub review with inline comments.

This is the pull request counterpart of `ai-review`. The difference that matters:
**this path never edits the contributor's code.** It reads a detached worktree and
posts comments. Validated fixes are published as GitHub suggestion blocks, which
the author applies themselves. Never offer to repair anything here.

## Arguments

| Invocation | Effect |
| --- | --- |
| `/review-contribution <pr-url>` | review and post |
| `/review-contribution <pr-url> --dry-run` | print what would be posted, post nothing |
| `/review-contribution <pr-url> --repo-path <dir>` | take the worktree from that clone |

`owner/repo#N` works anywhere a URL does.

## Steps

Create a todo per step and work through them in order.

**1. Prepare the pull request.**

```bash
D=$(mktemp -d) && mkdir -p "$D/out"
REPO_PATH=()                  # or (--repo-path <dir>) if the user gave one
ai-reviewer prompts --out "$D" --pr <pr-url> "${REPO_PATH[@]}"
```

Set `REPO_PATH` once, from any `--repo-path <dir>` the user gave.
If `ai-reviewer` is not on PATH, stop and give the user the one install line:
`uv tool install git+https://github.com/calimero-network/ai-code-reviewer`.
This resolves the PR, checks it out as a detached worktree, and writes the briefs.
Each stdout line is `<agent-name>\t<model>\t<brief path>`; the preparation summary
goes to stderr. Report what it printed rather than describing what you intend to do.
If it prints no agent lines, there is nothing to review - say so and stop.

**2. Spawn one reviewer subagent per brief, all in a single message so they run in parallel.**

For each line, launch `subagent_type: code-reviewer-readonly` with the `model`
from that line, and this prompt:

> Read the file `<brief path>` in full and follow its instructions exactly.
> Working directory for file reads: `<$D/wt>`.
> Do NOT modify any file. Your final message must be the single JSON object the
> brief specifies, with no prose and no code fence.

Never paste brief contents into the prompt.
Each brief is ~40k tokens; having the subagent read it keeps that out of this
session's context entirely.

**3. Save each result.**

Write each subagent's returned JSON to `$D/out/<agent-name>.json`.
The filename carries the attribution that drives consensus scoring, so it must
match the agent name from step 1.
If a subagent returned prose around the JSON, extract the JSON object.
If one returned nothing usable, skip it and note the gap - a missing agent must
never read as a clean review.

**4. Consolidate and post.**

```bash
ai-reviewer publish "$D"
```

Add `--dry-run` to print the review without posting; the worktree is kept so a
following real post needs no refetch. Add `--all` to expand suggestions and
nitpicks in the terminal report.

This clusters agreeing findings, applies confidence floors, dedups across files,
caps the total, validates structured replacements, then posts one GitHub review:
a summary body plus inline comments, with validated fixes as suggestion blocks.
Findings unchanged since a previous review are not posted again, and comments for
findings that are now fixed are resolved.

**5. Report what was posted**: the link `publish` printed, the review action and
inline comment count from its summary line, how many agents contributed, and
whether any failed. Quote the link it printed; never construct one.

## What this never does

- Edit, commit, or push anything in the contributor's branch.
- Approve the pull request. It posts under your GitHub identity, so the review
  action is always a comment.
- Touch your own clones: the review runs in a detached worktree, removed when
  `publish` finishes.

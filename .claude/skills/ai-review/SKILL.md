---
name: ai-review
description: Review local changes with the project's multi-agent reviewer, in this session. Use when asked to review uncommitted work, staged changes, a branch range, or to "run the reviewer" before opening a PR. Spawns read-only reviewer subagents, consolidates their findings in Python, and reports back.
---

# Local multi-agent review

Runs the repository's own review pipeline against local changes, using subagents
in this session rather than separate Claude Code processes.

Why it is built this way: a separate `claude` session per agent costs ~33k tokens
of environment preamble each, fires the user's SessionStart hooks, and writes
observations into their memory store. Subagents avoid all three. Everything that
is not an LLM call - clustering, consensus scoring, per-severity confidence
floors, cross-file dedup, the adaptive cap, and fix validation - stays in Python,
so it is real code rather than instructions.

## Arguments

| Invocation | Scope |
| --- | --- |
| `/ai-review` | uncommitted working-tree changes (includes untracked files) |
| `/ai-review --staged` | the index only |
| `/ai-review --base main` | `main...HEAD` |
| `/ai-review --agents 2` | fewer reviewer profiles (default 3) |
| `/ai-review --all` | include suggestions and nitpicks in the report |

## Steps

Create a todo per step and work through them in order.

**1. Build the briefs.**

```bash
D=$(mktemp -d) && mkdir -p "$D/out"
SCOPE=()                      # or (--staged), or (--base main)
ai-reviewer prompts --out "$D" --agents 3 --config .ai-reviewer.yaml "${SCOPE[@]}"
```

If `ai-reviewer` is not on PATH, stop and give the user the one install line:
`uv tool install git+https://github.com/calimero-network/ai-code-reviewer`.
Set `SCOPE` once to match the requested scope; step 4 reuses it.
Each output line is `<agent-name>\t<model>\t<brief path>`.
If it prints nothing, there are no changes to review - say so and stop.

**2. Spawn one reviewer subagent per brief, all in a single message so they run in parallel.**

For each line, launch `subagent_type: code-reviewer-readonly` with the `model`
from that line, and this prompt:

> Read the file `<brief path>` in full and follow its instructions exactly.
> Working directory for file reads: `<repo root>`.
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
If one returned nothing usable, skip it and note the gap in the report - a missing
agent must never read as a clean review.

**4. Consolidate in Python.**

```bash
ai-reviewer consolidate "$D"/out/*.json --config .ai-reviewer.yaml "${SCOPE[@]}"
```

Pass the same `SCOPE` as step 1. The finding cap and density penalty scale with the
size of the reviewed diff, so consolidating a staged review without `--staged`
measures a clean working tree and caps the report at five findings.
Add `--all` to expand suggestions and nitpicks.
This clusters agreeing findings, applies confidence floors, dedups across files,
caps the total, and validates any structured replacements.

**5. Report the output as-is**, then state how many agents contributed and whether
any failed.

## Offering fixes

After reporting, offer to repair - do not repair unasked.
When accepted, use `ai-reviewer consolidate ... --output json` to get the machine-
readable findings and apply this gate:

| Tier | Condition | Action |
| --- | --- | --- |
| 1 | `fix_validated: true` | splice `suggested_replacement` into `line_start..line_end` directly; no subagent needed, the replacement was already applied in memory and parse-checked |
| 2 | critical or warning, `confidence >= 0.7`, no validated replacement | one fix subagent **per file**, given every in-scope finding for that file |
| 3 | anything else | report only, never touched |

One subagent per file, never per finding: two agents editing one file race, and a
repair for one finding can invalidate another's premise.

Then run the project's test and lint gates once, and re-run the review to confirm
which findings actually closed. A subagent reporting success is not evidence.

Never commit or push. Changes land in the working tree for the user to read as
one `git diff`.

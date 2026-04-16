# Quality regression — Anthropic migration (2026-04)

Before declaring the Anthropic Messages API migration successful, run the
new reviewer against a fixed set of PRs that were previously reviewed under
the Cursor setup and confirm no quality regression.

## PR selection (5 PRs)

Pick 5 PRs covering the surface area we care about:

| # | Kind | Criteria |
|---|------|----------|
| 1 | Security fix | A PR that landed an auth/validation fix; confirm the new reviewer still flags the original vulnerability shape in pre-fix state |
| 2 | Performance optimization | Measurable perf change; confirm performance-reviewer notes the hot path and complexity |
| 3 | Refactor touching patterns / architecture | Multi-file refactor; confirm patterns-reviewer catches consistency issues (this is the main tool-use payoff) |
| 4 | Small typo / docs-only | Expect 0 findings, or only nitpicks; verify we don't fabricate issues |
| 5 | Large multi-file (>400 lines) | Stress-test context packing + truncation; verify diff is preserved and findings reference real lines |

Record each PR's `owner/repo` and PR number under `baseline/` below.

## Procedure

1. Set creds:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export GITHUB_TOKEN=ghp_...
   ```
2. For each pinned PR, run the reviewer in JSON mode so the output is
   comparable:
   ```bash
   ai-reviewer review-pr <owner/repo> <pr-number> --output json > \
     docs/migrations/runs/2026-04-<repo>-<pr>.json
   ```
3. Compare against the pre-migration baseline (stored under
   `docs/migrations/baseline/`). For each PR compute:
   - Count of findings by severity (critical / warning / suggestion / nitpick)
   - Count of findings that match a baseline finding (same file, ±3 lines,
     same category) — **true positives retained**
   - Count of new findings not in baseline — **either new true positives or
     new false positives**; manually classify
   - Count of baseline findings missing from the new run — **true positives
     dropped**

## Acceptance criteria

The migration is green when **all** of the following hold across the 5 PRs:

- True-positive retention ≥ 90% of baseline (no more than 10% drop)
- False-positive rate ≤ baseline FP rate + 2 absolute percentage points
- No critical finding from baseline disappears without a documented reason
- No run errors out or returns `all_agents_failed`
- Total review wall-clock time is within 2× the baseline (or better)

If any criterion fails, do not merge the migration PR. Iterate on:

1. Prompt content in `ReviewAgent` subclass `SYSTEM_PROMPT`s
2. Context budget / neighbor selection in `context/neighbors.py`
3. Agent mix / thinking budget / model choice

Then rerun.

## Recording results

Write results into `docs/migrations/2026-04-run-results.md` with a table
per PR. This file is the sign-off artifact for the migration PR.

## Baseline storage

Before rerunning, capture the existing reviewer output from the latest
Cursor-backed run of each pinned PR and drop them into
`docs/migrations/baseline/<repo>-<pr>.json`. If no baseline exists
(e.g., the PR predates the reviewer), manually curate an expected-findings
list and store it in the same location.

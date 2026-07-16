# The Best Code Reviewer: Phases A-E

Roadmap distilled from the 2026-07-13..15 reliability program (Sonnet 5 recall, sharding, queue, incident hardening) and the field review of the bot's output quality.
Tracking issues: #102 (fix validation), #103 (dismissal ledger), #104 (diff-aware doc bot), #105 (convergence verdict).

## North-star invariants

1. Never silent, never fake - every PR event ends in a posted verdict (findings, honest partial, or could-not-complete with a re-trigger path).
2. Every agent finishes - no PR size, burst, outage, or exploration depth may discard an agent's work.
3. Every finding is real, every fix is applyable - findings survive only with a reachable failure scenario; fixes ship only when they demonstrably work.
4. One substance, one thread, with a lifecycle - found once, tracked across pushes, auto-resolved when fixed, closed with an explicit verdict.
5. Proportional, predictable cost.
6. Measured, not vibes - recall/precision tracked against ground truth; the reviewer regression-tests itself.

## Phase A - context economics (this PR)

Problem: agents pre-load full contents of every changed file plus neighbors (~80k tokens on a 550-line PR) despite having read_file/grep tools.
Thinking agents then grow into the 160k circuit breaker on medium PRs and get discarded (observed: core #3256 logic-reviewer).

- A1 Pull-based context: changed files over `full_file_max_lines` (300) are excerpted to diff hunks with `hunk_context_lines` (60) of surrounding context; small files stay whole; neighbors capped to their first 40 lines; full files remain one read_file away.
  Instrumented: read_file on a trimmed path logs `context-trim readback` so tools-compensating behavior is measurable.
- A2 Stale-thinking strip: only the last assistant turn keeps its thinking block; the API ignores earlier turns' thinking, so re-sending it was pure dead weight (~1-2k tokens x rounds).
- A3 Soft-finalize at 75% of the breaker: past `0.75 x circuit_limit` the loop stops offering tools and forces the findings emission (same salvage as the tool-budget path), so near-breaches complete instead of being aborted at 100%.
- A4 Conventions block capped at `conventions_max_chars` (16k) - the last unbounded input.

Validation gate before trust: workflow_dispatch re-runs of core #3240 and #3256 plus two studio PRs; finding parity and logic-reviewer completing = ship; missing findings = widen the A1 window.

## Phase B - finding lifecycle and convergence (issues #103, #105)

- B1 Clean verdict: when the delta shows all prior findings fixed/resolved and the current pass is clean, post an explicit "all findings addressed" review.
- B2 `/ai-review` comment trigger on the Actions path (core), permission-gated, so a dismissal-only round can end in a verdict without a new commit.
- B3 Auto-resolve outdated threads on every review posting (generalize the LGTM-path resolution): a thread whose finding is in delta.fixed resolves itself.
- B4 Dismissal ledger: resolved-thread rationales are read back (GraphQL) and fed into cross-review keyed by substance fingerprint; a re-raise must rebut the recorded rationale or be marked invalid.

## Phase C - fix trustworthiness (issue #102)

- C1 (shipped) Cross-review sanity-checks suggested_fix; majority-flagged broken fixes are withdrawn.
- C2 Structured fixes: agents emit suggested_fix as an exact replacement for the cited line range; validation anchors it to the diff and parse-checks the result; an optional repo-configured fast lint command runs for the top findings on the CI path (which has a checkout).
- C3 Validated fixes post as GitHub ```suggestion blocks (one-click committable); unvalidated fixes stay prose so the reader always knows which is which.

## Phase D - evaluation flywheel (parallel track)

- D1 Golden set: 15-30 historical PRs with ground truth (shipped-then-hotfixed bugs = known misses; accepted findings = true positives; dismissed threads = false positives) as a versioned manifest.
- D2 Eval runner: substantive reviewer PRs replay the golden set dry-run; release gate is no recall/precision regression.
- D3 Telemetry: one structured log line per review (agents, tokens, drops, salvages, verdict) feeding a small dashboard.
- D4 A/B discipline: every knob change is justified by golden-set numbers.

## Phase E - productization

Config profiles (strict/balanced/economy), language rule packs beyond Rust, one-file repo onboarding, author commands (/ai-review focus=security).

## Deliberately not building

RAG/embedding repo index, fine-tuning, auto-pushed fixes, per-line style nit spam, review of vendored/generated code.

## KPIs

| KPI | Target |
| --- | --- |
| PR events ending in a posted verdict | >= 99.5%, zero silent |
| Agent work discarded | < 1% of agent runs |
| Precision (findings accepted/fixed) | >= 70% |
| Recall on golden set | no release regresses |
| Latency | p50 < 6 min, p95 < 20 min |
| Cost | median PR < $1.50, large < $8 |

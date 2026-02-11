# Code Review Standard – Enhancements from Google's Eng Practices

This document maps [Google's Standard of Code Review](https://google.github.io/eng-practices/review/) to concrete **prompt** and **feature** enhancements for the AI Code Reviewer, and extends our existing improvement list.

---

## 1. Core principle (already reflected in prompts)

> **In general, reviewers should favor approving a CL once it definitely improves the overall code health of the system, even if the CL isn't perfect.**

- **Prompts:** Instruct the model to prefer LGTM when the change improves code health; do not block on minor polish. Use `nitpick` only for optional improvements and prefix with "Nit:" so the author knows it's optional.
- **Features:** No change needed; the review action (APPROVE / COMMENT / REQUEST_CHANGES) already follows severity. We can add an explicit instruction that "approve when no critical/warning and code health improves."

---

## 2. What to look for (add to base prompt / agent focus)

| Google guideline | Prompt / feature addition |
|------------------|---------------------------|
| **Design** | Add to base prompt: "Consider overall design: do the interactions make sense? Does this belong here? Does it integrate well? Is now the right time for this change?" |
| **Functionality** | Already partly covered by security/performance/quality agents. Add: "Think about edge cases, concurrency, and whether the CL does what the author intended and is good for users." |
| **Complexity** | Add: "Is the CL more complex than needed? Avoid over-engineering; solve the current problem, not hypothetical future ones." |
| **Tests** | Already in quality-agent. Add: "Tests should be in the same CL; ensure tests are correct and would fail when code is broken." |
| **Naming** | Add: "Are names clear and long enough to communicate intent without being verbose?" |
| **Comments** | Add: "Comments should explain *why*, not *what*. Prefer simpler code over comments that explain complexity." |
| **Style** | Add: "Follow project style. For non-mandatory style preferences, use severity 'nitpick' and prefix with 'Nit:'." |
| **Consistency** | Add: "Prefer consistency with existing code when the style guide doesn't dictate otherwise." |
| **Documentation** | Add: "If the CL changes how users build/test/use the system, documentation should be updated." |
| **Every line** | Already implied by "only report issues on changed lines." Optional: "Review the full context of changed files when needed." |
| **Good things** | Add to summary instruction: "If something is done well (e.g. clear naming, good tests, clean design), mention it briefly in the summary." |

---

## 3. How to write comments (prompt additions)

| Guideline | Implementation |
|------------|-----------------|
| **Be kind** | Add: "Be courteous. Comment on the code, not the author." |
| **Explain why** | Add: "When asking for a change, briefly explain why it improves code health when helpful." |
| **Severity labels** | Map severities to intent: **critical/warning** = required; **suggestion** = consider/optional; **nitpick** = Nit, not mandatory. In output format, add: "For optional or style-only points use 'nitpick' and start the title with 'Nit: '." |
| **LGTM with comments** | Add: "You may report minor nits (e.g. typo, import sort) without blocking; use nitpick and the author can address later." |

---

## 4. Navigating the CL (prompt additions)

| Guideline | Implementation |
|------------|-----------------|
| **Broad view first** | Add: "First consider: does this change make sense? If the change shouldn't happen, say why and suggest an alternative." |
| **Main parts first** | Add: "Focus on the main design and logic first; then details. If design is wrong, other comments may be irrelevant." |
| **Speed** | N/A (we're automated). Optional: "Be concise so the author can act quickly." (already have "one short sentence per finding.") |

---

## 5. New items for the overall improvement list

These extend the P0/P1/P2 list we had:

| Priority | Enhancement | Source |
|----------|-------------|--------|
| P1 | **Base prompt: "Review standard" block** – Favor approval when code health improves; no perfectionism; use Nit for optional. | Standard of Code Review |
| P1 | **Base prompt: "What to look for"** – Short checklist: design, functionality, complexity, tests, naming, comments, style, consistency, docs. | What to look for |
| P1 | **Output rules: Severity semantics** – critical/warning = required; suggestion = consider; nitpick = Nit (optional), title prefix "Nit: ". | How to write comments |
| P1 | **Tone** – Comment on code not author; be kind; explain why when useful. | Courtesy, Explain why |
| P2 | **Summary: "Good things"** – If something is done well, add one brief positive line in the summary. | Good Things |
| P2 | **Design-first instruction** – "Consider overall design first; if the approach is wrong, say so before listing small issues." | Step Two |
| P2 | **Consistency rule** – "Prefer consistency with existing code; if style guide doesn't require it, don't block." | Consistency |
| P3 | **Optional: LGTM-with-comments** – In formatter/action logic, consider COMMENT (not REQUEST_CHANGES) when only nitpicks/suggestions remain, so author isn't blocked. | LGTM With Comments |

---

## 6. Severity semantics (canonical)

| Severity   | Meaning | Author action |
|-----------|---------|----------------|
| critical  | Must fix; security or correctness. | Required before merge. |
| warning   | Should fix; real risk or maintainability. | Expected to fix. |
| suggestion| Consider; improves code health. | Optional but recommended. |
| nitpick   | Nit; polish or style preference. | Optional; prefix with "Nit: ". |

---

## 7. Design principles (SOLID, DRY, KISS, etc.)

Incorporated into the base prompt and quality-agent so the reviewer can flag meaningful violations without nitpicking:

| Principle | What to flag (when relevant) |
|-----------|------------------------------|
| **SOLID** | SRP (one responsibility per class), OCP (extend don’t modify), LSP (subtypes substitutable), ISP (small interfaces), DIP (depend on abstractions). |
| **DRY** | Duplicate logic that should be extracted; repeated code that belongs in one place. |
| **KISS** | Over-engineering; unnecessary complexity or abstraction. |
| **YAGNI** | Code added for hypothetical future needs; speculative features. |
| **Composition over Inheritance** | Deep or rigid inheritance where composition would be clearer. |
| **Law of Demeter** | Long call chains (a.b.c.d); coupling to distant objects. |
| **Convention over Configuration** | Where conventions would simplify the codebase. |

Instruction in prompt: only flag violations that meaningfully hurt maintainability or clarity; use "Nit:" for minor style preferences.

---

## 8. Implemented in this repo

- **P0 (done):** Inline-first body, webhook body read once, concise prompts, docs/CI-aware.
- **Review standard & what to look for:** In base prompt; severity semantics (Nit:) and “good things” in summary in output format.
- **Design principles:** Base prompt includes SOLID, DRY, KISS, YAGNI, Composition over Inheritance, Law of Demeter, Convention over Configuration. Quality-agent explicitly focuses on these when running multi-agent.

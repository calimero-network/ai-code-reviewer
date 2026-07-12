# Sonnet 5 Reliability & Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the reviewer from posting fake "LGTM" reviews when an agent silently fails, and re-tune prompts/budgets so `claude-sonnet-5` actually surfaces findings instead of self-filtering them away.

**Architecture:** Three layers of fixes. (1) *Client layer* (`anthropic_client.py`): make thinking/temperature explicit per-model (Sonnet 5 silently turns thinking ON when the field is omitted, eating the token budget and producing empty responses), and tag every give-up path (tool-loop cap, parse error, truncation, circuit breaker) with a machine-readable marker. (2) *Pipeline layer* (`review.py`, `formatter.py`): treat marker-tagged agents as failed so the existing `failed_agents` machinery (which already blocks posting when all agents fail) kicks in, and make the PR comment honest when a review is partial. (3) *Prompt layer* (`context/builder.py`, agent classes): switch from "omit if unsure" (which Sonnet 5 obeys literally, tanking recall) to coverage-first reporting, letting the existing per-severity confidence thresholds and cross-review round do the filtering they were built for.

**Tech Stack:** Python 3.11/3.12, `anthropic` SDK (unpinned, `>=0.40.0`), pytest, ruff, mypy.

## Global Constraints

- Only `src/ai_reviewer/agents/anthropic_client.py` may import the `anthropic` SDK (architecture invariant I1, enforced via ruff TID251).
- Baseline: 559 tests pass; `ruff check`, `ruff format --check`, and `mypy` are clean. Every task must leave them clean.
- Run tests with the project venv: `.venv/bin/python -m pytest`.
- Conventional-commit messages (`fix:`, `feat:`, `test:`); each commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Work on branch `fix/sonnet5-reliability-and-recall` off up-to-date `origin/master`.
- Model facts this plan relies on (from Anthropic's Sonnet 5 migration guide): omitted `thinking` runs adaptive-on (4.6 ran off); explicit `{"type": "disabled"}` is valid on Sonnet 5/Opus 4.7/4.8/Haiku but **rejected (400) on Fable/Mythos**; `temperature`/`top_p`/`top_k` are rejected (400) on Sonnet 5/Opus 4.7+/Fable; the new tokenizer produces ~30% more tokens for the same text; conservative review prompts ("only report if confident") depress recall because Sonnet 5 follows them literally.

---

### Task 1: Explicit per-model thinking + temperature in one helper

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py` (the `_NO_SAMPLING_PARAMS_MODELS` block near the top; `run_completion` ~line 66; `complete_simple` ~line 103–110; `run_review` kwargs block ~line 173–187)
- Test: `tests/test_anthropic_client.py`

**Interfaces:**
- Produces: `_sampling_params(model: str, enable_thinking: bool, temperature: float | None) -> dict[str, Any]` — module-private helper; returns the `thinking` and (when applicable) `temperature` entries for a `messages.create` kwargs dict. All three client entry points route through it.

**Why:** `run_review` currently omits `thinking` when `enable_thinking=False`. On Sonnet 5 that silently enables adaptive thinking, which eats the 4096-token `max_tokens` budget before any JSON is emitted — the direct cause of the `Failed to parse JSON: ''` empty reviews observed on core runs 28747946371 and 28741074623.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anthropic_client.py` (reuse the file's existing `_make_client`/mock pattern — look at `test_run_review_with_thinking_enabled_sets_adaptive_config` around line 99 for the shape used to build a client with a mocked SDK and inspect `call_kwargs`):

```python
@pytest.mark.asyncio
async def test_run_review_thinking_off_sends_explicit_disabled():
    """Omitted `thinking` means adaptive-ON for Sonnet 5 — must send explicit disabled."""
    client, mock_create = _client_with_mocked_sdk()  # use the file's existing helper/pattern
    await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "temperature" not in kwargs  # sonnet-5 rejects it


@pytest.mark.asyncio
async def test_run_review_sonnet46_thinking_off_keeps_temperature():
    client, mock_create = _client_with_mocked_sdk()
    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
        temperature=0.3,
    )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_run_review_fable_omits_thinking_entirely():
    """Fable rejects explicit {"type": "disabled"} — the field must be absent."""
    client, mock_create = _client_with_mocked_sdk()
    await client.run_review(
        model="claude-fable-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    kwargs = mock_create.call_args.kwargs
    assert "thinking" not in kwargs
    assert "temperature" not in kwargs
```

If the file has no single shared `_client_with_mocked_sdk` helper, follow the inline mock construction used by the existing thinking tests instead — do not invent a new fixture style.

**Also flip the existing test** `test_run_review_without_thinking_omits_config` (~line 122): it asserts `"thinking" not in kwargs`, which encodes the bug. Rename it to `test_run_review_without_thinking_sends_disabled` and assert `kwargs["thinking"] == {"type": "disabled"}` (its model is `claude-sonnet-4-6`).

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/python -m pytest tests/test_anthropic_client.py -v -k "thinking or fable or sonnet46"`
Expected: new tests FAIL (`KeyError: 'thinking'` / assertion errors); the renamed test FAILS.

- [ ] **Step 3: Implement the helper and wire all three call sites**

In `src/ai_reviewer/agents/anthropic_client.py`, replace the `_accepts_temperature` block with:

```python
# Models that reject temperature/top_p/top_k outright (400 invalid_request_error).
# ponytail: hardcoded set, add the next rejecting model here when it ships.
_NO_SAMPLING_PARAMS_MODELS = {
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
}

# Models that reject an explicit thinking={"type": "disabled"} (thinking is always on).
_ALWAYS_THINKING_MODELS = {"claude-fable-5", "claude-mythos-5"}


def _accepts_temperature(model: str) -> bool:
    return model not in _NO_SAMPLING_PARAMS_MODELS


def _sampling_params(
    model: str, enable_thinking: bool, temperature: float | None
) -> dict[str, Any]:
    """Thinking + temperature request params that are safe for the target model.

    Sonnet 5 (and Opus 4.7+) treat an *omitted* `thinking` field as adaptive-ON,
    so thinking-off agents must send an explicit {"type": "disabled"} or thinking
    silently eats the max_tokens budget. Fable/Mythos reject explicit "disabled"
    (thinking is always on there) — omit the field for those instead. Temperature
    is only sent to models that still accept it.
    """
    params: dict[str, Any] = {}
    if enable_thinking:
        params["thinking"] = {"type": "adaptive"}
        if _accepts_temperature(model):
            params["temperature"] = 1.0  # API requires temp=1 alongside thinking
    else:
        if model not in _ALWAYS_THINKING_MODELS:
            params["thinking"] = {"type": "disabled"}
        if temperature is not None and _accepts_temperature(model):
            params["temperature"] = temperature
    return params
```

Wire it in:

1. `run_review` — delete the current temperature/thinking lines:
```python
            if _accepts_temperature(model):
                kwargs["temperature"] = 1.0 if enable_thinking else temperature
            if tools:
                kwargs["tools"] = tools
            if enable_thinking:
                kwargs["thinking"] = {"type": "adaptive"}
```
and replace with:
```python
            kwargs.update(_sampling_params(model, enable_thinking, temperature))
            if tools:
                kwargs["tools"] = tools
```

2. `complete_simple` — delete the trailing `if _accepts_temperature(model): kwargs["temperature"] = temperature` block and add after the kwargs dict:
```python
        kwargs.update(_sampling_params(model, enable_thinking=False, temperature=temperature))
```

3. `run_completion` — it currently passes kwargs inline with no temperature. Rewrite the call:
```python
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        kwargs.update(_sampling_params(model, enable_thinking=False, temperature=None))
        response = await self._sdk.messages.create(**kwargs)
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (previous count 559 + 3 new). If any *other* test asserts `"thinking" not in kwargs` or a bare temperature, update it to the new explicit-disabled contract — the new behavior is the spec.

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format src tests
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "fix: send explicit thinking config per model (Sonnet 5 defaults omitted field to adaptive-on)"
```

---

### Task 2: Machine-readable incomplete markers on every give-up path

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py` (circuit-breaker return ~line 166, terminal-response branch ~line 213, tool-loop-cap return ~line 277, `_parse_json` fallback ~line 348)
- Test: `tests/test_anthropic_client.py`

**Interfaces:**
- Produces: module constants `TOOL_LOOP_CAP_MARKER = "[tool loop cap]"`, `PARSE_ERROR_MARKER = "[parse error]"`, `CIRCUIT_BREAKER_MARKER = "[circuit breaker: context limit exceeded]"`, `TRUNCATED_MARKER = "[truncated at max_tokens]"`, and `INCOMPLETE_SUMMARY_MARKERS: tuple[str, ...]` containing all four. Task 3 imports `INCOMPLETE_SUMMARY_MARKERS`.

**Why:** The client already returns `findings: []` with magic summary strings on failure, but nothing downstream reads them — so a capped or empty-response agent is indistinguishable from a genuinely clean review. Centralizing the strings as constants makes them a contract instead of a coincidence. This also adds the one failure path currently missing entirely: `stop_reason == "max_tokens"` (truncated output).

- [ ] **Step 1: Write the failing tests**

```python
def test_incomplete_markers_cover_all_giveup_paths():
    from ai_reviewer.agents import anthropic_client as ac

    assert ac.TOOL_LOOP_CAP_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert ac.PARSE_ERROR_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert ac.TRUNCATED_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert any(m.startswith("[circuit breaker") for m in ac.INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_run_review_truncated_response_is_marked_incomplete():
    """stop_reason=max_tokens must not read as a clean zero-finding review."""
    # Build the mock response the way the file's existing tests do, but with
    # stop_reason="max_tokens" and an empty/partial text block.
    client, mock_create = _client_with_mocked_sdk(stop_reason="max_tokens", text="")
    result = await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    from ai_reviewer.agents.anthropic_client import TRUNCATED_MARKER

    assert TRUNCATED_MARKER in result.parsed["summary"]
```

(Adapt mock construction to the file's existing response-stub pattern; the key inputs are `stop_reason="max_tokens"` and empty text content.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_anthropic_client.py -v -k "incomplete or truncated"`
Expected: FAIL (`AttributeError: ... has no attribute 'TOOL_LOOP_CAP_MARKER'`).

- [ ] **Step 3: Implement**

Near the top of `anthropic_client.py` (after the model sets):

```python
# Summary markers for reviews that did NOT complete. aggregate_findings() treats
# any summary containing one of these as a failed agent — a give-up must never
# be indistinguishable from a genuinely clean review.
TOOL_LOOP_CAP_MARKER = "[tool loop cap]"
PARSE_ERROR_MARKER = "[parse error]"
CIRCUIT_BREAKER_MARKER = "[circuit breaker: context limit exceeded]"
TRUNCATED_MARKER = "[truncated at max_tokens]"
INCOMPLETE_SUMMARY_MARKERS: tuple[str, ...] = (
    TOOL_LOOP_CAP_MARKER,
    PARSE_ERROR_MARKER,
    "[circuit breaker",
    TRUNCATED_MARKER,
)
```

Then use the constants at the existing literal sites:
- Circuit-breaker return: `"summary": CIRCUIT_BREAKER_MARKER` (replacing the inline string).
- Tool-loop-cap return: `"summary": TOOL_LOOP_CAP_MARKER`.
- `_parse_json` fallback: `return {"findings": [], "summary": PARSE_ERROR_MARKER}`.

In the terminal branch of `run_review` (where `stop != "tool_use"` and the result is built), mark truncation before returning:

```python
                raw_text = _extract_text(response)
                parsed = _parse_json(raw_text)
                if stop == "max_tokens":
                    logger.warning(
                        "Response truncated at max_tokens=%d — marking review incomplete",
                        max_tokens,
                    )
                    parsed["summary"] = f"{TRUNCATED_MARKER} {parsed.get('summary', '')}".strip()
                return AnthropicReviewResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    usage=usage,
                    tool_calls=tool_calls,
                )
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. (`test_review.py` has no assertions on the literal `[tool loop cap]`/`[parse error]` strings — verified by grep — but if one surfaces, point it at the constant.)

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat: tag every review give-up path with a machine-readable incomplete marker"
```

---

### Task 3: Count incomplete agents as failed in aggregation

**Files:**
- Modify: `src/ai_reviewer/review.py` (`aggregate_findings`, the failed-agent detection ~line 681–684)
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `INCOMPLETE_SUMMARY_MARKERS` from Task 2.
- Produces: agents whose summary carries an incomplete marker land in `ConsolidatedReview.failed_agents`. Downstream this is free behavior: the CLI already refuses to post and exits non-zero when `all_agents_failed` (this is exactly what surfaced the original temperature bug), and cross-review already excludes `failed_agents`.

**Why:** Core runs a single agent. Today a capped/empty agent produces a "clean" review → "✅ No Issues Found... LGTM! 🎉" gets posted. With this change the same situation becomes `all_agents_failed` → nothing is posted, CI goes red, humans investigate — the honest outcome.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review.py`:

```python
def test_aggregate_findings_marks_incomplete_agent_as_failed():
    from ai_reviewer.agents.anthropic_client import TOOL_LOOP_CAP_MARKER, PARSE_ERROR_MARKER
    from ai_reviewer.review import aggregate_findings

    review = aggregate_findings(
        [
            ("security-reviewer", [], TOOL_LOOP_CAP_MARKER),
            ("logic-reviewer", [], PARSE_ERROR_MARKER),
            ("patterns-reviewer", [], "Reviewed thoroughly, code looks good."),
        ],
        "o/r",
        1,
    )
    assert "security-reviewer" in review.failed_agents
    assert "logic-reviewer" in review.failed_agents
    assert "patterns-reviewer" not in review.failed_agents
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_review.py::test_aggregate_findings_marks_incomplete_agent_as_failed -v`
Expected: FAIL (`assert 'security-reviewer' in []`).

- [ ] **Step 3: Implement**

In `src/ai_reviewer/review.py`, add to the imports from the client module (it already imports `AnthropicClient` from `ai_reviewer.agents.anthropic_client`):

```python
from ai_reviewer.agents.anthropic_client import AnthropicClient, INCOMPLETE_SUMMARY_MARKERS
```

Change the detection in `aggregate_findings`:

```python
        if (
            "Agent failed:" in summary
            or "401 Unauthorized" in summary
            or any(marker in summary for marker in INCOMPLETE_SUMMARY_MARKERS)
        ):
            failed_agents.append(agent_name)
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/review.py tests/test_review.py
git commit -m "fix: count incomplete agents (tool cap, parse error, truncation) as failed"
```

---

### Task 4: Honest PR comment when a review is partial

**Files:**
- Modify: `src/ai_reviewer/github/formatter.py` (both "No Issues Found" sites — `format_comment` ~line 54–62 and the second site ~line 300–305; locate both with `grep -n "No Issues Found" src/ai_reviewer/github/formatter.py`)
- Test: `tests/test_github.py`

**Interfaces:**
- Consumes: `ConsolidatedReview.failed_agents` (existing field, now populated by Task 3).
- Produces: no new API — changed rendering only.

**Why:** Even with Task 3, a *multi-agent* run where one agent fails and the others find nothing would still render "All agents reviewed the code and found no issues." That sentence must only appear when it is true.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_github.py` (match the file's existing formatter-test construction style — it already builds `ConsolidatedReview` objects for formatter tests):

```python
def _minimal_review(findings=None, failed_agents=None, agent_count=2):
    from datetime import datetime
    from ai_reviewer.models.review import ConsolidatedReview

    return ConsolidatedReview(
        id="review-test1234",
        created_at=datetime.now(),
        repo="o/r",
        pr_number=1,
        findings=findings or [],
        summary="s",
        agent_count=agent_count,
        review_quality_score=0.9,
        total_review_time_ms=1000,
        failed_agents=failed_agents or [],
    )


class TestPartialReviewHonesty:
    def test_no_findings_with_failed_agent_does_not_say_lgtm(self):
        formatter = ReviewFormatter(reviewer_name="MeroReviewer")
        review = _minimal_review(failed_agents=["security-reviewer-0"])
        comment = formatter.format_comment(review)
        assert "LGTM" not in comment
        assert "No Issues Found" not in comment
        assert "Review Incomplete" in comment
        assert "security-reviewer-0" in comment

    def test_no_findings_no_failures_still_says_lgtm(self):
        formatter = ReviewFormatter(reviewer_name="MeroReviewer")
        review = _minimal_review()
        comment = formatter.format_comment(review)
        assert "No Issues Found" in comment
```

(Use the actual formatter class name and constructor signature from the file — check the top of `tests/test_github.py` for how existing tests instantiate it, and reuse an existing review-builder helper if one exists rather than adding `_minimal_review`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_github.py -v -k "PartialReviewHonesty"`
Expected: first test FAILS (comment contains "LGTM").

- [ ] **Step 3: Implement**

In `format_comment` (and identically at the second "No Issues Found" site), replace:

```python
        if not review.findings:
            lines.extend(
                [
                    "### ✅ No Issues Found",
                    "",
                    "All agents reviewed the code and found no issues. LGTM! 🎉",
                    "",
                ]
            )
```

with:

```python
        if not review.findings:
            if review.failed_agents:
                lines.extend(
                    [
                        "### ⚠️ Review Incomplete",
                        "",
                        f"{len(review.failed_agents)} of {review.agent_count} agent(s) did not "
                        f"finish ({', '.join(review.failed_agents)}) and no findings were "
                        "produced. Treat this PR as **not yet reviewed**, not as approved.",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "### ✅ No Issues Found",
                        "",
                        "All agents reviewed the code and found no issues. LGTM! 🎉",
                        "",
                    ]
                )
```

And after the findings/severity-section branch (still inside the same method), add a partial-review note when there *are* findings but some agent failed:

```python
        if review.findings and review.failed_agents:
            lines.extend(
                [
                    f"> ⚠️ Partial review: {', '.join(review.failed_agents)} did not finish — "
                    "findings above may be incomplete.",
                    "",
                ]
            )
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including pre-existing formatter tests (none of them set `failed_agents`, so the LGTM path is unchanged for them).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/github/formatter.py tests/test_github.py
git commit -m "feat: render honest 'Review Incomplete' instead of LGTM when agents fail"
```

---

### Task 5: Budget raises for the Sonnet 5 tokenizer and agentic behavior

**Files:**
- Modify: `src/ai_reviewer/config.py` (`AgentConfig.max_tokens` default, line ~23; loader default `agent_raw.get("max_tokens", 4096)`, line ~265)
- Modify: `src/ai_reviewer/agents/base.py` (`__init__` `max_tokens: int = 4096`, line ~35)
- Modify: `src/ai_reviewer/agents/anthropic_client.py` (`run_review` `max_tool_rounds: int = 8`, line ~155)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: default `max_tokens` becomes **8192** everywhere an agent default exists; default `max_tool_rounds` becomes **20** (aligned with the existing `AgentConfig.max_tool_calls = 20`, which was unreachable behind an 8-round cap). Explicit YAML values still win.

**Why:** Two of the observed silent failures were budget failures. (a) Sonnet 5's tokenizer emits ~30% more tokens for the same content, so 4096 output tokens is tighter than it was on 4.6 — and repos like core with no `agents:` block get the dataclass default. (b) Sonnet 5 runs more tool rounds before concluding (observed: PR #3200 hit `max_tool_rounds=8` and returned zero findings).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_agent_defaults_sized_for_sonnet5():
    from ai_reviewer.config import AgentConfig

    cfg = AgentConfig(name="a", model="m", focus_areas=[])
    assert cfg.max_tokens == 8192
```

And to `tests/test_anthropic_client.py`:

```python
def test_run_review_default_tool_rounds_matches_tool_call_budget():
    import inspect
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    sig = inspect.signature(AnthropicClient.run_review)
    assert sig.parameters["max_tool_rounds"].default == 20
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_agent_defaults_sized_for_sonnet5 tests/test_anthropic_client.py::test_run_review_default_tool_rounds_matches_tool_call_budget -v`
Expected: both FAIL (4096 ≠ 8192; 8 ≠ 20).

- [ ] **Step 3: Implement**

- `config.py` `AgentConfig`: `max_tokens: int = 8192`
- `config.py` loader: `max_tokens=agent_raw.get("max_tokens", 8192),`
- `agents/base.py` `__init__`: `max_tokens: int = 8192,`
- `anthropic_client.py` `run_review`: `max_tool_rounds: int = 20,` with a comment:
```python
        # 20 matches AgentConfig.max_tool_calls — the registry's per-review call
        # budget is the binding cap; an 8-round loop cap below it made the last
        # 12 calls unreachable and silently truncated agentic reviews on Sonnet 5.
        max_tool_rounds: int = 20,
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. If any existing test asserts the old 4096 default (grep `4096` in `tests/test_config.py` / `tests/test_base_agent.py`), update it to 8192 — the raise is the spec.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/config.py src/ai_reviewer/agents/base.py src/ai_reviewer/agents/anthropic_client.py tests/test_config.py tests/test_anthropic_client.py
git commit -m "fix: raise default max_tokens to 8192 and tool rounds to 20 for Sonnet 5"
```

---

### Task 6: Coverage-first prompt retune

**Files:**
- Modify: `src/ai_reviewer/context/builder.py` (`REVIEW_STANDARD_BLOCK` lines ~60–82; `_pr_tuning_block` lines ~104–132)
- Modify: `src/ai_reviewer/agents/security.py` (closing lines of `SecurityAgent.SYSTEM_PROMPT`, ~52–54)
- Modify: `src/ai_reviewer/agents/performance.py` (no change to `PerformanceAgent`; `LogicAgent` prompt is already coverage-compatible — leave as is)
- Test: `tests/test_context_builder.py`

**Interfaces:**
- Consumes: nothing new. The downstream filters this prompt now leans on already exist: per-severity confidence thresholds (`aggregator.min_confidence_*`, applied in `review.py` ~line 1198) and the cross-review validation round.
- Produces: prompt text only; block order unchanged (order matters for prompt caching — do not reorder blocks).

**Why:** Anthropic's Sonnet 5 migration guide is explicit: review harnesses that say "if you are not confident a finding is real, omit it" see recall drop on Sonnet 5 because it obeys the filter literally — it finds the bugs, then declines to report them. The guide's recommended fix is coverage-first finding + downstream filtering, and this codebase *already has* the downstream filter stack. The current `REVIEW_STANDARD_BLOCK` ("Precision over volume… omit it") and the small-PR tuning ("report only findings you are confident about") are exactly the anti-pattern. This is also why post-switch core reviews cratered to zero findings even when the agent ran fine.

- [ ] **Step 1: Update the tests first**

`tests/test_context_builder.py` currently asserts on the old wording (it matched the grep for `Precision over volume`). Update the affected assertion(s) to the new contract and add:

```python
def test_review_standard_is_coverage_first():
    from ai_reviewer.context.builder import REVIEW_STANDARD_BLOCK

    text = REVIEW_STANDARD_BLOCK["text"]
    assert "omit it" not in text  # the old self-filter instruction is gone
    assert "confidence" in text.lower()
    assert "changed line" in text  # grounding rule preserved


def test_small_pr_tuning_no_longer_self_filters():
    from ai_reviewer.context.builder import _pr_tuning_block

    block = _pr_tuning_block(None, "small")
    assert block is not None
    assert "only findings you are confident about" not in block["text"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_context_builder.py -v`
Expected: the new tests FAIL against the current text.

- [ ] **Step 3: Rewrite the prompt blocks**

Replace `REVIEW_STANDARD_BLOCK` in `context/builder.py` with:

```python
# Shared review standard + severity rubric — every agent sees this so severity
# is calibrated consistently rather than decided per agent.
#
# Coverage-first by design: the pipeline has downstream filters (per-severity
# confidence thresholds + a cross-review validation round), so the finder's job
# is coverage, not self-censorship. Telling Sonnet-5-era models "omit unless
# confident" measurably suppresses recall — they obey it literally.
REVIEW_STANDARD_BLOCK: dict[str, Any] = {
    "type": "text",
    "text": (
        "## Review standard\n\n"
        "Favor approving when the change improves overall code health, even if "
        "imperfect — there is no perfect code, only better code. Do not block on "
        "minor polish. Technical facts and engineering principles outweigh personal "
        "preference: if the author's approach is a valid alternative, defer to it. "
        "Comment on the code, not the author, and explain *why* you ask for a change.\n\n"
        "Report every issue you find, including ones you are uncertain about — do "
        "not self-filter for importance or confidence. A separate validation step "
        "filters and ranks findings; your job at this stage is coverage. Signal "
        "certainty honestly through the `confidence` field (0.0-1.0) instead of "
        "omitting doubtful findings: a real-but-unproven concern belongs in the "
        "report at low confidence. Report each distinct issue exactly once, and do "
        "not flag mechanical formatting or import ordering that an "
        "autoformatter/linter already handles.\n\n"
        "Every finding must point to a specific changed line AND give a concrete fix "
        "or the precise reason the code is wrong. When a finding depends on code you "
        "cannot see in the diff (callers, definitions, configuration), use the "
        "provided repository tools (read_file / grep / glob) to check the actual "
        "code rather than guessing.\n\n"
        "**Severity:**\n"
        "- `critical` — must fix: security vulnerabilities or data-corruption/loss risks only.\n"
        "- `warning` — should fix: other correctness, concurrency, or serious maintainability issues.\n"
        "- `suggestion` — consider; an optional improvement.\n"
        '- `nitpick` — optional polish; prefix the title with "Nit: " (never blocking).\n\n'
        "**Grounding:** Only report issues on lines changed in this PR. Cite the "
        "file and line. Do not report issues in code outside the diff — but do use "
        "the tools to read surrounding code when it determines whether a changed "
        "line is correct."
    ),
}
```

In `_pr_tuning_block`, replace the two size branches:

```python
    if pr_size in ("trivial", "small"):
        parts.append(
            "Small change — the full diff fits comfortably in context, so verify it "
            "exhaustively. Do not pad the review with generic advice; every finding "
            "must still cite a specific changed line."
        )
    elif pr_size == "large":
        parts.append(
            "Large change — lead with high-severity issues (architecture, "
            "correctness, security), but still report lower-severity issues you "
            "notice with honest severity and confidence rather than omitting them."
        )
```

In `agents/security.py`, replace the last three lines of `SecurityAgent.SYSTEM_PROMPT`:

```
Be thorough but precise. Only report issues you can clearly demonstrate in the code.
Do not speculate about issues that might exist elsewhere.
Provide specific line numbers and concrete evidence for each finding.
```

with:

```
Report every security concern you find, including ones you are not fully certain
about — signal certainty through the confidence field instead of omitting the
finding. Ground every finding in the changed code with specific line numbers and
concrete evidence; when correctness depends on code outside the diff, use the
repository tools to confirm the actual behavior rather than speculating.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (fix any remaining string-assertions in `tests/test_prompts.py` or `tests/test_context_builder.py` that pinned the old wording — align them with the new text).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context/builder.py src/ai_reviewer/agents/security.py tests/test_context_builder.py tests/test_prompts.py
git commit -m "feat: coverage-first review prompts — report all findings, filter downstream"
```

---

### Task 7: Dogfood config — adaptive thinking A/B on this repo's own reviews

**Files:**
- Modify: `.ai-reviewer.yaml` (this repo's self-review config — the `agents:` list)

**Interfaces:**
- Consumes: the per-agent `thinking_enabled` and `max_tokens` YAML keys (already supported end-to-end; Task 1 made the thinking-on path safe on Sonnet 5 by dropping temperature).
- Produces: config only, no code.

**Why:** Anthropic's guidance for Sonnet 5 is that adaptive thinking improves review quality and tool-eagerness (thinking-off Sonnet 5 reaches for tools less — and these agents depend on `read_file`/`grep` for context). Cost goes up, so flip it on this repo first (dogfood), watch a few reviews, then propose the same to `core` in a separate PR. Class defaults stay thinking-off — this is an explicit, revertable config experiment, not a behavior change for every consumer.

- [ ] **Step 1: Edit the config**

For each Sonnet agent entry in `.ai-reviewer.yaml`'s `agents:` list (the four `model: claude-sonnet-5` entries — leave the haiku style agent alone), set:

```yaml
    thinking_enabled: true
    max_tokens: 16000
```

(16000 gives thinking + findings JSON room under the new tokenizer; still comfortably non-streaming within the 300s timeout.)

- [ ] **Step 2: Validate the config loads**

Run: `.venv/bin/python -c "from pathlib import Path; from ai_reviewer.config import load_config; c = load_config(Path('.ai-reviewer.yaml')); print([(a.name, a.thinking_enabled, a.max_tokens) for a in c.agents])"`
Expected: each sonnet agent prints `(name, True, 16000)`.

- [ ] **Step 3: Commit**

```bash
git add .ai-reviewer.yaml
git commit -m "chore: dogfood adaptive thinking + 16k budget for self-review agents"
```

---

### Task 8: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full local gate**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check src tests && .venv/bin/python -m ruff format --check src tests && .venv/bin/python -m mypy src`
Expected: all green.

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin fix/sonnet5-reliability-and-recall
gh pr create --repo calimero-network/ai-code-reviewer \
  --title "fix: Sonnet 5 reliability (silent LGTM failures) + coverage-first recall retune" \
  --body "<summarize Tasks 1-7; link the failing core runs 28749222479 / 28747946371 / 28741074623 as evidence>"
```

- [ ] **Step 3: Live self-review check (the PR reviews itself)**

Opening the PR triggers the MeroReviewer workflow on this repo. Verify:

```bash
gh run list --repo calimero-network/ai-code-reviewer --workflow "MeroReviewer" --limit 1
gh run view <run-id> --repo calimero-network/ai-code-reviewer --log | grep -iE "Review complete|Failed to parse|Tool-use loop|truncated"
```

Expected: no `Failed to parse JSON`, no `Tool-use loop exceeded`, no truncation marker; review posts either real findings or a genuine clean pass.

- [ ] **Step 4: Recall spot-check against a known-findings baseline**

```bash
gh workflow run "MeroReviewer" --repo calimero-network/ai-code-reviewer -f pr_number=89 -f agents=3
```

PR #89 previously produced findings under `claude-sonnet-4-6`-era prompts. Expected: the coverage-first Sonnet 5 run reports a non-zero, sane set of findings (evidence the recall retune works). If it returns 0 findings, investigate before merging — that is the failure mode this whole plan exists to eliminate.

- [ ] **Step 5: Follow-up (separate PRs, out of scope here)**

- `core`: after this merges and a few dogfood reviews look good, optionally add an `agents:` block or `thinking_enabled` guidance there; also note core's `anthropic.default_model` only drives cross-review + CLI display, not agent selection.
- If dogfood thinking-on reviews are materially better, consider flipping `THINKING_ENABLED = True` as the class default in a follow-up (cost discussion required — same rule as core's "no Opus without cost discussion" comment).

---

## Self-Review

- **Spec coverage:** empty-response parse failures → Task 1 (root cause) + Task 2 (detection); tool-loop cap → Task 5 (budget) + Task 2 (detection); fake LGTMs → Tasks 2–4 (marker → failed_agents → honest comment / no-post); Sonnet 5 recall collapse from conservative prompts → Task 6; tokenizer/budget sizing → Task 5; quality upside (thinking, tool-eagerness) → Task 7; proof it works → Task 8. ✔
- **Placeholder scan:** all steps carry concrete code/commands; the only "adapt to existing pattern" notes are test-fixture reuse instructions pointing at named existing tests. ✔
- **Type consistency:** `_sampling_params(model: str, enable_thinking: bool, temperature: float | None)` used identically in Tasks 1; `INCOMPLETE_SUMMARY_MARKERS: tuple[str, ...]` produced in Task 2 and imported by name in Task 3; `ConsolidatedReview.failed_agents` (existing `list[str]`) consumed in Task 4. ✔

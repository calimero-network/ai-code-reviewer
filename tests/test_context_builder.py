from ai_reviewer.context.builder import (
    FINDINGS_SCHEMA,
    _pr_tuning_block,
    build_system_blocks,
    build_user_blocks,
)


def test_build_system_blocks_includes_role_schema_and_conventions():
    convention_texts = {
        "AGENTS.md": "Always cite file:line.",
        "CONTRIBUTING.md": "Follow PEP8.",
    }
    repo_map = "Top-level: src/, tests/, docs/"
    blocks = build_system_blocks(
        agent_role="You review security.",
        convention_texts=convention_texts,
        repo_map=repo_map,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "You review security." in combined
    assert "findings" in combined
    assert "Always cite file:line." in combined
    assert "Follow PEP8." in combined
    assert "src/" in combined
    assert blocks[-1]["type"] == "text"


def test_build_system_blocks_includes_review_standard_and_few_shot():
    blocks = build_system_blocks(
        agent_role="You review security.",
        convention_texts={},
        repo_map="map",
    )
    combined = "\n".join(b["text"] for b in blocks)
    # Shared review standard + severity rubric
    assert "Favor approving" in combined
    assert "Nit: " in combined
    assert "critical" in combined
    # Coverage-first calibration (Sonnet-5 retune)
    assert "do not self-filter" in combined
    assert "defer to it" in combined
    # Few-shot quality anchors
    assert "SQL injection via string interpolation" in combined
    assert "DO NOT produce these" in combined


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


def test_build_system_blocks_includes_language_block_only_when_provided():
    with_rules = build_system_blocks(
        agent_role="r",
        convention_texts={},
        repo_map="m",
        language_rules="For Rust:\n- `.unwrap()` in non-test code.",
    )
    combined = "\n".join(b["text"] for b in with_rules)
    assert "Language-specific priorities" in combined
    assert ".unwrap()" in combined

    plain = "\n".join(
        b["text"] for b in build_system_blocks(agent_role="r", convention_texts={}, repo_map="m")
    )
    assert "Language-specific priorities" not in plain


def test_pr_tuning_block_docs_and_ci():
    docs = _pr_tuning_block("docs", "small")
    assert docs is not None and "factual" in docs["text"].lower()
    ci = _pr_tuning_block("ci", "trivial")
    assert ci is not None and "workflow correctness" in ci["text"].lower()


def test_pr_tuning_block_size_guidance():
    small = _pr_tuning_block("code", "small")
    assert small is not None and "exhaustively" in small["text"].lower()
    large = _pr_tuning_block("code", "large")
    assert large is not None and "high-severity" in large["text"].lower()


def test_pr_tuning_block_none_when_nothing_applies():
    assert _pr_tuning_block("code", "medium") is None
    assert _pr_tuning_block(None, None) is None


def test_build_system_blocks_includes_tuning_only_when_classified():
    tuned = build_system_blocks(
        agent_role="r", convention_texts={}, repo_map="m", pr_type="docs", pr_size="large"
    )
    tuned_combined = "\n".join(b["text"] for b in tuned).lower()
    assert "factual" in tuned_combined
    assert "high-severity" in tuned_combined

    plain = "\n".join(
        b["text"] for b in build_system_blocks(agent_role="r", convention_texts={}, repo_map="m")
    ).lower()
    assert "high-severity" not in plain
    assert "docs-only" not in plain


def test_findings_schema_is_complete():
    assert FINDINGS_SCHEMA["type"] == "object"
    assert "findings" in FINDINGS_SCHEMA["properties"]
    finding = FINDINGS_SCHEMA["$defs"]["Finding"]
    for required in ("file_path", "line_start", "severity", "category", "title"):
        assert required in finding["required"]


def test_build_user_blocks_contains_all_sections():
    blocks = build_user_blocks(
        pr_title="Fix auth bug",
        pr_body="Resolves #123",
        diff="@@ -1 +1 @@\n-old\n+new",
        changed_files={"src/a.py": "print('a')\n"},
        neighbor_files={"src/b.py": "print('b')\n"},
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "Fix auth bug" in combined
    assert "Resolves #123" in combined
    assert "```diff" in combined
    assert "src/a.py" in combined
    assert "src/b.py" in combined


def test_build_user_blocks_truncates_neighbors_first():
    big_neighbor = "x" * 50_000
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff="@@ -1 +1 @@",
        changed_files={"a.py": "keep-this"},
        neighbor_files={"n.py": big_neighbor},
        max_total_chars=5_000,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "keep-this" in combined
    assert "[... neighbors truncated ...]" in combined


# --- Hunk-context (pull-based) mode ---


def _numbered(n: int) -> str:
    """n lines uniquely labeled L0001.. so no label is a substring of another."""
    return "\n".join(f"L{i:04d}" for i in range(1, n + 1))


def test_hunk_mode_large_file_becomes_excerpt():
    content = _numbered(400)
    diff = "diff --git a/big.py b/big.py\n@@ -100,3 +100,4 @@\n+new\n"
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": content},
        neighbor_files={},
        full_file_max_lines=300,
        hunk_context_lines=60,
    )
    combined = "\n".join(b["text"] for b in blocks)
    # Window is [100-60, 100+4+60] = [40, 164]; 125 lines kept of 400.
    assert "[excerpt: 125 of 400 lines - hunks +/-60 context" in combined
    assert "L0100" in combined  # inside the hunk
    assert "L0040" in combined  # window low boundary
    assert "L0164" in combined  # window high boundary
    assert "L0039" not in combined  # just outside the window
    assert "L0400" not in combined  # distant content excluded


def test_hunk_mode_small_file_kept_full():
    content = _numbered(100)
    diff = "diff --git a/small.py b/small.py\n@@ -50,1 +50,2 @@\n+new\n"
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"small.py": content},
        neighbor_files={},
        full_file_max_lines=300,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "[excerpt:" not in combined
    assert "L0001" in combined
    assert "L0100" in combined


def test_hunk_mode_overlapping_hunks_merge_into_one_window():
    content = _numbered(400)
    diff = "diff --git a/big.py b/big.py\n@@ -100,2 +100,2 @@\n+a\n@@ -150,2 +150,2 @@\n+b\n"
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": content},
        neighbor_files={},
        full_file_max_lines=300,
        hunk_context_lines=60,
    )
    combined = "\n".join(b["text"] for b in blocks)
    # Windows [40,162] and [90,212] overlap -> single merged window, no separator.
    assert "\n...\n" not in combined
    assert "L0100" in combined
    assert "L0200" in combined


def test_hunk_mode_disjoint_hunks_keep_separator():
    content = _numbered(400)
    diff = "diff --git a/big.py b/big.py\n@@ -20,1 +20,1 @@\n+a\n@@ -380,1 +380,1 @@\n+b\n"
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": content},
        neighbor_files={},
        full_file_max_lines=300,
        hunk_context_lines=60,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "\n...\n" in combined  # two disjoint windows


def test_hunk_mode_trimmed_paths_out_records_only_excerpted():
    diff = (
        "diff --git a/big.py b/big.py\n@@ -100,1 +100,1 @@\n+x\n"
        "diff --git a/small.py b/small.py\n@@ -1,1 +1,1 @@\n+y\n"
    )
    trimmed: set[str] = set()
    build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": _numbered(400), "small.py": _numbered(100)},
        neighbor_files={},
        full_file_max_lines=300,
        trimmed_paths_out=trimmed,
    )
    assert trimmed == {"big.py"}


def test_hunk_mode_large_file_without_hunks_is_capped_not_full():
    content = _numbered(400)
    # Diff references a different path, so big.py has no discoverable hunks.
    diff = "diff --git a/other.py b/other.py\n@@ -1,1 +1,1 @@\n+x\n"
    trimmed: set[str] = set()
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": content},
        neighbor_files={},
        full_file_max_lines=300,
        trimmed_paths_out=trimmed,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "truncated to first 300 lines" in combined
    assert "L0300" in combined
    assert "L0301" not in combined  # full content must not leak through
    assert trimmed == {"big.py"}


def test_hunk_mode_caps_neighbors_to_first_lines():
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff="diff --git a/a.py b/a.py\n@@ -1,1 +1,1 @@\n+x\n",
        changed_files={"a.py": _numbered(10)},
        neighbor_files={"n.py": _numbered(100)},
        full_file_max_lines=300,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "truncated to first 40 lines" in combined
    assert "L0040" in combined
    assert "L0041" not in combined


def test_hunk_mode_none_preserves_full_content():
    content = _numbered(400)
    diff = "diff --git a/big.py b/big.py\n@@ -100,3 +100,4 @@\n+new\n"
    trimmed: set[str] = set()
    blocks = build_user_blocks(
        pr_title="t",
        pr_body="",
        diff=diff,
        changed_files={"big.py": content},
        neighbor_files={"n.py": _numbered(100)},
        full_file_max_lines=None,
        trimmed_paths_out=trimmed,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "[excerpt:" not in combined
    assert "truncated to first" not in combined
    assert content in combined  # full changed file verbatim
    assert _numbered(100) in combined  # full neighbor verbatim
    assert trimmed == set()


# --- Conventions aggregate cap ---


def test_conventions_block_truncated_over_aggregate_cap():
    convention_texts = {"A.md": "a\n" * 5000, "B.md": "b\n" * 5000}
    blocks = build_system_blocks(
        agent_role="r",
        convention_texts=convention_texts,
        repo_map="m",
        conventions_max_chars=2000,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "[conventions truncated at 2000 chars]" in combined


def test_conventions_block_untouched_under_cap():
    convention_texts = {"A.md": "always cite file:line", "B.md": "follow PEP8"}
    blocks = build_system_blocks(
        agent_role="r",
        convention_texts=convention_texts,
        repo_map="m",
        conventions_max_chars=16_000,
    )
    combined = "\n".join(b["text"] for b in blocks)
    assert "[conventions truncated at" not in combined
    assert "always cite file:line" in combined
    assert "follow PEP8" in combined

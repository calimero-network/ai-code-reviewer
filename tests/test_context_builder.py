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
    # Calibration additions from the research pass
    assert "Precision over volume" in combined
    assert "defer to it" in combined
    # Few-shot quality anchors
    assert "SQL injection via string interpolation" in combined
    assert "DO NOT produce these" in combined


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
    assert small is not None and "precision" in small["text"].lower()
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

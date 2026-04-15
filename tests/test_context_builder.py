from ai_reviewer.context.builder import (
    FINDINGS_SCHEMA,
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

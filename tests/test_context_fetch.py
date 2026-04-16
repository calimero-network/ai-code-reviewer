import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_reviewer.context.fetch import build_repo_map, fetch_conventions
from ai_reviewer.session import ReviewSession


def _encoded(content: str) -> MagicMock:
    m = MagicMock()
    m.content = base64.b64encode(content.encode()).decode()
    return m


def test_fetch_conventions_returns_texts():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=50)
    gh = MagicMock()

    def contents(repo_name, path, ref=None):  # noqa: ARG001
        mapping = {"AGENTS.md": "conv-a", "CONTRIBUTING.md": "conv-c"}
        if path in mapping:
            return _encoded(mapping[path])
        raise FileNotFoundError(path)

    gh.get_file_contents.side_effect = contents

    texts = fetch_conventions(
        session,
        gh,
        paths=["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"],
    )
    assert texts["AGENTS.md"] == "conv-a"
    assert texts["CONTRIBUTING.md"] == "conv-c"
    assert "CLAUDE.md" not in texts


def test_build_repo_map_lists_top_level():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=50)
    gh = MagicMock()
    tree = [
        SimpleNamespace(path="src", type="tree"),
        SimpleNamespace(path="tests", type="tree"),
        SimpleNamespace(path="README.md", type="blob"),
        SimpleNamespace(path="src/app/x.py", type="blob"),
    ]
    gh.get_tree.return_value.tree = tree

    out = build_repo_map(session, gh)
    assert "src/" in out
    assert "tests/" in out
    assert "README.md" in out

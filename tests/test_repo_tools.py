import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_reviewer.session import ReviewSession
from ai_reviewer.tools.repo_tools import ToolQuotaExceeded, ToolRegistry


@pytest.fixture
def session():
    return ReviewSession(repo="o/r", head_sha="abc", github_budget=5)


@pytest.fixture
def fake_gh():
    gh = MagicMock()
    contents = MagicMock()
    contents.content = base64.b64encode(b"print('hi')").decode()
    gh.get_file_contents.return_value = contents
    return gh


@pytest.mark.asyncio
async def test_read_file_returns_decoded_content(session, fake_gh):
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=10, per_file_max_bytes=512 * 1024)
    out = await reg.execute("read_file", {"path": "a.py"})
    assert out == "print('hi')"


@pytest.mark.asyncio
async def test_read_file_cache_hit_does_not_call_github(session, fake_gh):
    session.store_file("a.py", "cached-content")
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=10, per_file_max_bytes=512 * 1024)
    out = await reg.execute("read_file", {"path": "a.py"})
    assert out == "cached-content"
    fake_gh.get_repo.assert_not_called()


@pytest.mark.asyncio
async def test_per_agent_max_calls_enforced(session, fake_gh):
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=1, per_file_max_bytes=512 * 1024)
    await reg.execute("read_file", {"path": "a.py"})
    with pytest.raises(ToolQuotaExceeded):
        await reg.execute("read_file", {"path": "b.py"})


@pytest.fixture
def fake_gh_with_tree():
    gh = MagicMock()
    tree_items = [
        SimpleNamespace(path="src/a.py", type="blob"),
        SimpleNamespace(path="src/b.py", type="blob"),
        SimpleNamespace(path="README.md", type="blob"),
        SimpleNamespace(path="src/sub", type="tree"),
    ]
    gh.get_tree.return_value.tree = tree_items

    def _contents(repo_name, path, ref=None):  # noqa: ARG001
        payloads = {
            "src/a.py": b"import os\nprint('a')\n",
            "src/b.py": b"def f():\n    return 42\n",
            "README.md": b"# Readme\nprint-like example\n",
        }
        c = MagicMock()
        c.content = base64.b64encode(payloads.get(path, b"")).decode()
        return c

    gh.get_file_contents.side_effect = _contents
    return gh


@pytest.mark.asyncio
async def test_glob_filters_blobs_only(session, fake_gh_with_tree):
    reg = ToolRegistry(
        session, fake_gh_with_tree, agent_id="a1", max_calls=50, per_file_max_bytes=1024
    )
    out = await reg.execute("glob", {"pattern": "src/*.py"})
    assert "src/a.py" in out
    assert "src/b.py" in out
    assert "README.md" not in out
    assert "src/sub" not in out


@pytest.mark.asyncio
async def test_grep_returns_path_line_match(session, fake_gh_with_tree):
    reg = ToolRegistry(
        session, fake_gh_with_tree, agent_id="a1", max_calls=50, per_file_max_bytes=1024
    )
    out = await reg.execute("grep", {"pattern": r"print", "path_glob": "**/*.py"})
    assert "src/a.py:2: print('a')" in out


@pytest.mark.asyncio
async def test_grep_invalid_regex_returns_error(session, fake_gh_with_tree):
    reg = ToolRegistry(
        session, fake_gh_with_tree, agent_id="a1", max_calls=5, per_file_max_bytes=1024
    )
    out = await reg.execute("grep", {"pattern": "[unclosed", "path_glob": "*.py"})
    assert out.startswith("[error: invalid regex")

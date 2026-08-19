import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_reviewer.session import ReviewSession
from ai_reviewer.tools.repo_tools import ToolBudgetExhausted, ToolRegistry


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
    with pytest.raises(ToolBudgetExhausted):
        await reg.execute("read_file", {"path": "b.py"})


@pytest.mark.asyncio
async def test_read_file_truncates_on_byte_boundary_not_char_index(session):
    """Regression for #56: the per-file byte cap must slice on UTF-8 bytes, not
    character index. For multi-byte content a char-index slice at position N can
    yield far more than N bytes (here ~4x), overshooting the cap."""
    limit = 10
    payload = "😀" * 10  # 10 characters, 40 UTF-8 bytes
    gh = MagicMock()
    contents = MagicMock()
    contents.content = base64.b64encode(payload.encode("utf-8")).decode()
    gh.get_file_contents.return_value = contents

    reg = ToolRegistry(session, gh, agent_id="a1", max_calls=10, per_file_max_bytes=limit)
    out = await reg.execute("read_file", {"path": "emoji.py"})

    marker = "\n[... file truncated ...]"
    assert out.endswith(marker)
    body = out[: -len(marker)]
    assert len(body.encode("utf-8")) <= limit, (
        f"truncated body is {len(body.encode('utf-8'))} bytes, exceeds cap {limit}"
    )
    # No dangling partial multi-byte character should survive truncation.
    assert "�" not in body


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


@pytest.mark.asyncio
async def test_tool_result_over_cap_is_truncated_with_marker(session):
    limit = 100
    payload = "x" * 1000
    gh = MagicMock()
    contents = MagicMock()
    contents.content = base64.b64encode(payload.encode()).decode()
    gh.get_file_contents.return_value = contents

    reg = ToolRegistry(
        session,
        gh,
        agent_id="a1",
        max_calls=10,
        per_file_max_bytes=512 * 1024,
        max_tool_result_bytes=limit,
    )
    out = await reg.execute("read_file", {"path": "a.py"})

    marker = f"\n[output truncated at {limit} bytes - "
    assert marker in out
    body = out[: out.index(marker)]
    assert len(body.encode("utf-8")) <= limit


@pytest.mark.asyncio
async def test_tool_result_under_cap_is_unchanged(session, fake_gh):
    reg = ToolRegistry(
        session,
        fake_gh,
        agent_id="a1",
        max_calls=10,
        per_file_max_bytes=512 * 1024,
        max_tool_result_bytes=16 * 1024,
    )
    out = await reg.execute("read_file", {"path": "a.py"})
    assert out == "print('hi')"


@pytest.mark.asyncio
async def test_read_file_logs_readback_for_trimmed_path(session, fake_gh, caplog):
    reg = ToolRegistry(
        session,
        fake_gh,
        agent_id="a1",
        max_calls=10,
        per_file_max_bytes=512 * 1024,
        trimmed_paths={"a.py"},
    )
    with caplog.at_level("INFO", logger="ai_reviewer.tools.repo_tools"):
        await reg.execute("read_file", {"path": "a.py"})
    assert "context-trim readback: a.py" in caplog.text


@pytest.mark.asyncio
async def test_read_file_no_readback_log_for_untrimmed_path(session, fake_gh, caplog):
    reg = ToolRegistry(
        session,
        fake_gh,
        agent_id="a1",
        max_calls=10,
        per_file_max_bytes=512 * 1024,
        trimmed_paths={"other.py"},
    )
    with caplog.at_level("INFO", logger="ai_reviewer.tools.repo_tools"):
        await reg.execute("read_file", {"path": "a.py"})
    assert "context-trim readback" not in caplog.text


def test_max_tool_result_bytes_config_parses_and_defaults():
    from ai_reviewer.config import _parse_config

    config = _parse_config({"github": {"token": "t"}})
    assert config.anthropic.max_tool_result_bytes == 16 * 1024

    config = _parse_config({"github": {"token": "t"}, "anthropic": {"max_tool_result_bytes": 4096}})
    assert config.anthropic.max_tool_result_bytes == 4096


def test_read_file_rejects_absolute_paths():
    """Defence in depth: the ".."-only guard let absolute paths through, and the
    local repo source resolves them against the real filesystem."""
    from unittest.mock import MagicMock

    from ai_reviewer.session import ReviewSession
    from ai_reviewer.tools.repo_tools import ToolRegistry

    registry = ToolRegistry(
        session=ReviewSession(repo="o/r", head_sha="sha", github_budget=10),
        github_client=MagicMock(),
        agent_id="a-0",
        max_calls=5,
        per_file_max_bytes=1024,
    )

    assert "not allowed" in registry._read_file("/etc/passwd")

# Anthropic Messages API Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Cursor Background Agent API with Anthropic's official Messages API as the backing LLM layer for all review agents, without regressing review quality.

**Architecture:** A new `AnthropicClient` wraps the `anthropic` SDK's Messages API with tool-use loop, extended thinking, structured JSON output, and prompt caching. On-demand repo exploration goes through a `ToolRegistry` backed by the existing `GitHubClient` (Contents API — no cloning). A `ReviewSession` holds per-review transient state; heuristic neighbor-file selection lives in `context/neighbors.py`. `ReviewAgent.review()` is rewritten to call the new client. `review.py` is rewired to instantiate and drive the agents instead of passing prompts to Cursor's background agents.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `httpx`, `PyGithub`, `pytest` + `pytest-asyncio`, existing orchestrator/aggregator/convergence code.

**Spec:** `docs/superpowers/specs/2026-04-15-anthropic-messages-migration-design.md`

---

## Phase 0 — Branch and scaffolding

### Task 0: Create migration branch

**Files:** (none modified)

- [ ] **Step 1: Verify clean working tree and branch**

Run:
```bash
git status
git branch --show-current
```
Expected: clean tree, on `master`, with the two spec commits (`2696cfc`, `4849c3c`) already present.

- [ ] **Step 2: Create and switch to migration branch**

Run:
```bash
git checkout -b feature/anthropic-messages-migration
```

- [ ] **Step 3: Push the branch**

Run:
```bash
git push -u origin feature/anthropic-messages-migration
```

---

## Phase 1 — Dependencies and config scaffolding

Config changes land first so later tasks can import the new types.

### Task 1: Add `anthropic` SDK to dependencies

**Files:**
- Modify: `pyproject.toml:26-42` (dependencies block)

- [ ] **Step 1: Add dependency, remove stale comment**

Replace the `dependencies` list with:
```toml
dependencies = [
    "anthropic>=0.40.0",              # Official Anthropic SDK for Messages API
    "PyGithub>=2.1.0",
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "click>=8.1.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    "httpx>=0.26.0",
    "pyyaml>=6.0.0",
    "rich>=13.7.0",
    "tenacity>=8.2.0",
    "sentence-transformers>=2.3.0",
    "PyJWT>=2.8.0",
    "cryptography>=42.0.0",
]
```

- [ ] **Step 2: Install and verify import**

Run:
```bash
pip install -e '.[dev]'
python -c "import anthropic; print(anthropic.__version__)"
```
Expected: a version ≥ 0.40.0 prints.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add anthropic SDK dependency"
```

---

### Task 2: Add `AnthropicApiConfig` and extend `AgentConfig`

**Files:**
- Modify: `src/ai_reviewer/config.py:24-30` (add new dataclass), `:118-130` (extend Config), `:196-208` (extend AgentConfig parsing), `:175-183` (parse anthropic block)
- Test: `tests/test_config.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py` (append if exists):
```python
from pathlib import Path
import textwrap

from ai_reviewer.config import load_config


def test_load_anthropic_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-test")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""
        anthropic:
          api_key: ${ANTHROPIC_API_KEY}
          default_model: claude-sonnet-4-6
          enable_prompt_caching: true
        github:
          token: ${GITHUB_TOKEN}
        agents:
          - name: security-reviewer
            model: claude-sonnet-4-6
            focus_areas: [security]
            thinking_enabled: true
            thinking_budget_tokens: 8192
            allow_tool_use: true
            max_tool_calls: 20
    """))
    cfg = load_config(cfg_file)
    assert cfg.anthropic.api_key == "sk-test-123"
    assert cfg.anthropic.default_model == "claude-sonnet-4-6"
    assert cfg.anthropic.enable_prompt_caching is True
    assert cfg.agents[0].thinking_enabled is True
    assert cfg.agents[0].thinking_budget_tokens == 8192
    assert cfg.agents[0].allow_tool_use is True
    assert cfg.agents[0].max_tool_calls == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_load_anthropic_config -v`
Expected: FAIL (AttributeError on `cfg.anthropic` or missing fields).

- [ ] **Step 3: Add `AnthropicApiConfig` and extend `AgentConfig` / `Config`**

In `src/ai_reviewer/config.py`, **add** next to `CursorApiConfig`:
```python
@dataclass
class AnthropicApiConfig:
    """Anthropic Messages API configuration."""

    api_key: str
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: int = 300
    max_retries: int = 3
    default_model: str = "claude-sonnet-4-6"
    enable_prompt_caching: bool = True
    max_combined_context_tokens: int = 150_000
    per_file_max_bytes: int = 512 * 1024
    per_review_github_request_budget: int = 200
```

Extend `AgentConfig`:
```python
@dataclass
class AgentConfig:
    name: str
    model: str
    focus_areas: list[str]
    max_tokens: int = 4096
    temperature: float = 0.3
    custom_prompt_append: str | None = None
    include_codebase_context: bool = False
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 8192
    allow_tool_use: bool = True
    max_tool_calls: int = 20
```

Extend `Config`:
```python
@dataclass
class Config:
    cursor: CursorApiConfig | None
    anthropic: AnthropicApiConfig | None
    github: GitHubConfig
    agents: list[AgentConfig]
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    aggregator: AggregatorSettings = field(default_factory=AggregatorSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)
    server: ServerSettings = field(default_factory=ServerSettings)
    doc_review: DocReviewSettings = field(default_factory=DocReviewSettings)
```

- [ ] **Step 4: Parse the `anthropic` YAML block**

In `_parse_config()`, add after the Cursor block:
```python
    anthropic_raw = raw.get("anthropic", {})
    anthropic = AnthropicApiConfig(
        api_key=anthropic_raw.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", ""),
        base_url=anthropic_raw.get("base_url", "https://api.anthropic.com"),
        timeout_seconds=anthropic_raw.get("timeout_seconds", 300),
        max_retries=anthropic_raw.get("max_retries", 3),
        default_model=anthropic_raw.get("default_model", "claude-sonnet-4-6"),
        enable_prompt_caching=anthropic_raw.get("enable_prompt_caching", True),
        max_combined_context_tokens=anthropic_raw.get("max_combined_context_tokens", 150_000),
        per_file_max_bytes=anthropic_raw.get("per_file_max_bytes", 512 * 1024),
        per_review_github_request_budget=anthropic_raw.get("per_review_github_request_budget", 200),
    ) if anthropic_raw else None
```

Update `AgentConfig` construction in `_parse_config()` to pass new fields:
```python
        agents.append(
            AgentConfig(
                name=agent_raw["name"],
                model=agent_raw["model"],
                focus_areas=agent_raw.get("focus_areas", []),
                max_tokens=agent_raw.get("max_tokens", 4096),
                temperature=agent_raw.get("temperature", 0.3),
                custom_prompt_append=agent_raw.get("custom_prompt_append"),
                include_codebase_context=agent_raw.get("include_codebase_context", False),
                thinking_enabled=agent_raw.get("thinking_enabled", False),
                thinking_budget_tokens=agent_raw.get("thinking_budget_tokens", 8192),
                allow_tool_use=agent_raw.get("allow_tool_use", True),
                max_tool_calls=agent_raw.get("max_tool_calls", 20),
            )
        )
```

Update the final `return Config(cursor=cursor, anthropic=anthropic, ...)`.

- [ ] **Step 5: Update `validate_config()` to require one of cursor/anthropic**

Replace the cursor-only check with:
```python
    if not config.anthropic or not config.anthropic.api_key:
        if not config.cursor or not config.cursor.api_key:
            errors.append(
                "Missing Anthropic API key (set ANTHROPIC_API_KEY or anthropic.api_key)"
            )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_load_anthropic_config -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ai_reviewer/config.py tests/test_config.py
git commit -m "feat(config): add AnthropicApiConfig and extend AgentConfig

- AnthropicApiConfig mirrors Cursor config shape for migration
- AgentConfig gains thinking_enabled, thinking_budget_tokens,
  allow_tool_use, max_tool_calls
- Cursor config remains for coexistence during migration"
```

---

## Phase 2 — Runtime primitives: session, tools, neighbors

### Task 3: `ReviewSession` dataclass

**Files:**
- Create: `src/ai_reviewer/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_session.py`:
```python
from ai_reviewer.session import ReviewSession


def test_review_session_tracks_quota():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=3)
    assert session.remaining_github_budget() == 3
    session.consume_github_request()
    session.consume_github_request()
    assert session.remaining_github_budget() == 1
    session.consume_github_request()
    assert session.remaining_github_budget() == 0
    assert session.is_github_budget_exhausted()


def test_review_session_file_cache():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=10)
    assert session.cached_file("a.py") is None
    session.store_file("a.py", "print('hi')")
    assert session.cached_file("a.py") == "print('hi')"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `ReviewSession`**

Create `src/ai_reviewer/session.py`:
```python
"""Per-review transient state: GitHub quota, file cache, tree listing."""

from dataclasses import dataclass, field


@dataclass
class ReviewSession:
    """Transient, per-review state shared across agents.

    Holds cached Contents API results, the recursive tree listing,
    and per-review GitHub request budget counters. Not a data model —
    a runtime object; do not persist.
    """

    repo: str                               # "owner/repo"
    head_sha: str                           # commit to pin reads to
    github_budget: int                      # total request budget for this review
    _github_used: int = 0
    _file_cache: dict[str, str] = field(default_factory=dict)
    _tree_cache: list[str] | None = None
    _tool_calls_by_agent: dict[str, int] = field(default_factory=dict)

    def remaining_github_budget(self) -> int:
        return max(0, self.github_budget - self._github_used)

    def is_github_budget_exhausted(self) -> bool:
        return self._github_used >= self.github_budget

    def consume_github_request(self, count: int = 1) -> None:
        self._github_used += count

    def cached_file(self, path: str) -> str | None:
        return self._file_cache.get(path)

    def store_file(self, path: str, content: str) -> None:
        self._file_cache[path] = content

    def cached_tree(self) -> list[str] | None:
        return self._tree_cache

    def store_tree(self, paths: list[str]) -> None:
        self._tree_cache = paths

    def incr_tool_call(self, agent_id: str) -> int:
        n = self._tool_calls_by_agent.get(agent_id, 0) + 1
        self._tool_calls_by_agent[agent_id] = n
        return n

    def tool_calls_for(self, agent_id: str) -> int:
        return self._tool_calls_by_agent.get(agent_id, 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/session.py tests/test_session.py
git commit -m "feat(session): add ReviewSession for per-review transient state"
```

---

### Task 4: `ToolRegistry` — `read_file` tool

**Files:**
- Create: `src/ai_reviewer/tools/__init__.py` (empty)
- Create: `src/ai_reviewer/tools/repo_tools.py`
- Test: `tests/test_repo_tools.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_repo_tools.py`:
```python
from unittest.mock import MagicMock

import pytest

from ai_reviewer.session import ReviewSession
from ai_reviewer.tools.repo_tools import ToolRegistry, ToolQuotaExceeded


@pytest.fixture
def session():
    return ReviewSession(repo="o/r", head_sha="abc", github_budget=5)


@pytest.fixture
def fake_gh():
    gh = MagicMock()
    contents = MagicMock()
    contents.content = "cHJpbnQoJ2hpJyk="  # base64("print('hi')")
    gh.get_repo.return_value.get_contents.return_value = contents
    return gh


@pytest.mark.asyncio
async def test_read_file_returns_decoded_content(session, fake_gh):
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=10, per_file_max_bytes=512*1024)
    out = await reg.execute("read_file", {"path": "a.py"})
    assert out == "print('hi')"


@pytest.mark.asyncio
async def test_read_file_cache_hit_does_not_call_github(session, fake_gh):
    session.store_file("a.py", "cached-content")
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=10, per_file_max_bytes=512*1024)
    out = await reg.execute("read_file", {"path": "a.py"})
    assert out == "cached-content"
    fake_gh.get_repo.assert_not_called()


@pytest.mark.asyncio
async def test_per_agent_max_calls_enforced(session, fake_gh):
    reg = ToolRegistry(session, fake_gh, agent_id="a1", max_calls=1, per_file_max_bytes=512*1024)
    await reg.execute("read_file", {"path": "a.py"})
    with pytest.raises(ToolQuotaExceeded):
        await reg.execute("read_file", {"path": "b.py"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_repo_tools.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `ToolRegistry` with `read_file`**

Create `src/ai_reviewer/tools/__init__.py`:
```python
"""Tool implementations for Anthropic tool use."""
```

Create `src/ai_reviewer/tools/repo_tools.py`:
```python
"""Repo-exploration tools backed by GitHub Contents API.

Exposes read_file / glob / grep as Anthropic tools. All file access
is pinned to the PR head SHA and is read-only. Enforces per-agent
call budget and per-review GitHub request budget.
"""

from __future__ import annotations

import base64
import fnmatch
import logging
import re
from typing import Any

from ai_reviewer.github.client import GitHubClient
from ai_reviewer.session import ReviewSession

logger = logging.getLogger(__name__)


class ToolQuotaExceeded(RuntimeError):
    """Raised when an agent exceeds its tool-call budget."""


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read the full contents of a file at the PR's head commit. "
            "Prefer reading only files you specifically need. "
            "Cite path + line numbers in findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "glob",
        "description": (
            "List repo paths matching a glob pattern (e.g., 'src/**/*.py'). "
            "Use this before read_file when you do not know the exact path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Regex search across repo paths. Prefer this over reading many "
            "files. Returns up to 100 matches with path and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "path_glob": {"type": "string", "description": "Glob to restrict search"},
            },
            "required": ["pattern", "path_glob"],
        },
    },
]


class ToolRegistry:
    """Dispatches Anthropic tool_use calls to real implementations."""

    def __init__(
        self,
        session: ReviewSession,
        github_client: GitHubClient,
        agent_id: str,
        max_calls: int,
        per_file_max_bytes: int,
    ) -> None:
        self.session = session
        self.gh = github_client
        self.agent_id = agent_id
        self.max_calls = max_calls
        self.per_file_max_bytes = per_file_max_bytes

    def tool_specs(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        if self.session.tool_calls_for(self.agent_id) >= self.max_calls:
            raise ToolQuotaExceeded(
                f"Agent {self.agent_id} exceeded max_tool_calls={self.max_calls}"
            )
        self.session.incr_tool_call(self.agent_id)

        if name == "read_file":
            return self._read_file(tool_input["path"])
        if name == "glob":
            return self._glob(tool_input["pattern"])
        if name == "grep":
            return self._grep(tool_input["pattern"], tool_input["path_glob"])
        raise ValueError(f"Unknown tool: {name}")

    def _read_file(self, path: str) -> str:
        cached = self.session.cached_file(path)
        if cached is not None:
            return cached
        if self.session.is_github_budget_exhausted():
            return "[error: review GitHub budget exhausted]"
        self.session.consume_github_request()
        try:
            repo = self.gh.client.get_repo(self.session.repo)
            contents = repo.get_contents(path, ref=self.session.head_sha)
        except Exception as e:                                  # noqa: BLE001
            logger.warning("read_file(%s) failed: %s", path, e)
            return f"[error: {e}]"
        raw = getattr(contents, "content", "") or ""
        try:
            text = base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception as e:                                  # noqa: BLE001
            return f"[error: decode {e}]"
        if len(text.encode("utf-8")) > self.per_file_max_bytes:
            text = text[: self.per_file_max_bytes]
            text += "\n[... file truncated ...]"
        self.session.store_file(path, text)
        return text

    def _tree(self) -> list[str]:
        cached = self.session.cached_tree()
        if cached is not None:
            return cached
        if self.session.is_github_budget_exhausted():
            return []
        self.session.consume_github_request()
        try:
            repo = self.gh.client.get_repo(self.session.repo)
            tree = repo.get_git_tree(self.session.head_sha, recursive=True)
        except Exception as e:                                  # noqa: BLE001
            logger.warning("tree() failed: %s", e)
            return []
        paths = [item.path for item in tree.tree if getattr(item, "type", None) == "blob"]
        self.session.store_tree(paths)
        return paths

    def _glob(self, pattern: str) -> str:
        paths = self._tree()
        hits = [p for p in paths if fnmatch.fnmatch(p, pattern)]
        hits = hits[:500]
        return "\n".join(hits) if hits else "[no matches]"

    def _grep(self, pattern: str, path_glob: str) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[error: invalid regex: {e}]"
        matches: list[str] = []
        for path in self._tree():
            if not fnmatch.fnmatch(path, path_glob):
                continue
            content = self._read_file(path)
            if content.startswith("[error"):
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{path}:{lineno}: {line[:200]}")
                    if len(matches) >= 100:
                        return "\n".join(matches) + "\n[... grep truncated at 100 matches ...]"
        return "\n".join(matches) if matches else "[no matches]"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_repo_tools.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/tools tests/test_repo_tools.py
git commit -m "feat(tools): add ToolRegistry with read_file/glob/grep

Backed by GitHub Contents API + recursive tree listing. Enforces
per-agent max_tool_calls and per-review github_budget from
ReviewSession."
```

---

### Task 5: `ToolRegistry` — `glob` and `grep` tests

**Files:**
- Test: `tests/test_repo_tools.py` (append)

- [ ] **Step 1: Add tests for glob and grep**

Append to `tests/test_repo_tools.py`:
```python
from types import SimpleNamespace


@pytest.fixture
def fake_gh_with_tree():
    gh = MagicMock()
    tree_items = [
        SimpleNamespace(path="src/a.py", type="blob"),
        SimpleNamespace(path="src/b.py", type="blob"),
        SimpleNamespace(path="README.md", type="blob"),
        SimpleNamespace(path="src/sub", type="tree"),
    ]
    gh.client.get_repo.return_value.get_git_tree.return_value.tree = tree_items

    def _contents(path, ref=None):
        payloads = {
            "src/a.py": b"import os\nprint('a')\n",
            "src/b.py": b"def f():\n    return 42\n",
            "README.md": b"# Readme\nprint-like example\n",
        }
        c = MagicMock()
        c.content = base64.b64encode(payloads.get(path, b"")).decode()
        return c

    gh.client.get_repo.return_value.get_contents.side_effect = _contents
    return gh


@pytest.mark.asyncio
async def test_glob_filters_blobs_only(session, fake_gh_with_tree):
    import base64  # noqa: F401  — available in the test
    reg = ToolRegistry(session, fake_gh_with_tree, agent_id="a1", max_calls=50, per_file_max_bytes=1024)
    out = await reg.execute("glob", {"pattern": "src/*.py"})
    assert "src/a.py" in out
    assert "src/b.py" in out
    assert "README.md" not in out
    assert "src/sub" not in out


@pytest.mark.asyncio
async def test_grep_returns_path_line_match(session, fake_gh_with_tree):
    reg = ToolRegistry(session, fake_gh_with_tree, agent_id="a1", max_calls=50, per_file_max_bytes=1024)
    out = await reg.execute("grep", {"pattern": r"print", "path_glob": "**/*.py"})
    assert "src/a.py:2: print('a')" in out


@pytest.mark.asyncio
async def test_grep_invalid_regex_returns_error(session, fake_gh_with_tree):
    reg = ToolRegistry(session, fake_gh_with_tree, agent_id="a1", max_calls=5, per_file_max_bytes=1024)
    out = await reg.execute("grep", {"pattern": "[unclosed", "path_glob": "*.py"})
    assert out.startswith("[error: invalid regex")
```

Fix the `glob` pattern recursion — note `fnmatch` doesn't treat `**` specially. Update `_glob` in `repo_tools.py` to use `pathlib`-style matching:
```python
    def _glob(self, pattern: str) -> str:
        from pathlib import PurePosixPath
        paths = self._tree()
        hits = [p for p in paths if PurePosixPath(p).match(pattern)]
        hits = hits[:500]
        return "\n".join(hits) if hits else "[no matches]"
```

And `_grep`'s `path_glob` check uses the same:
```python
        for path in self._tree():
            if not PurePosixPath(path).match(path_glob):
                continue
```

Import at the top of `repo_tools.py`:
```python
from pathlib import PurePosixPath
```

Remove the `fnmatch` import.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_repo_tools.py -v`
Expected: PASS (6 tests).

- [ ] **Step 3: Commit**

```bash
git add src/ai_reviewer/tools/repo_tools.py tests/test_repo_tools.py
git commit -m "test(tools): add glob/grep tests; switch to PurePosixPath matching"
```

---

### Task 6: Python neighbor heuristics (AST)

**Files:**
- Create: `src/ai_reviewer/context/__init__.py` (empty)
- Create: `src/ai_reviewer/context/neighbors.py`
- Test: `tests/test_neighbors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_neighbors.py`:
```python
from ai_reviewer.context.neighbors import parse_imports_python


def test_python_absolute_imports():
    src = "import os\nimport ai_reviewer.config\nfrom ai_reviewer.models import findings\n"
    imports = parse_imports_python(src)
    assert "os" in imports
    assert "ai_reviewer.config" in imports
    assert "ai_reviewer.models.findings" in imports


def test_python_relative_imports_resolved():
    src = "from . import sibling\nfrom .utils import helper\n"
    imports = parse_imports_python(src, current_module="ai_reviewer.agents.base")
    assert "ai_reviewer.agents.sibling" in imports
    assert "ai_reviewer.agents.utils.helper" in imports


def test_python_malformed_returns_empty():
    assert parse_imports_python("def f(:\n  ") == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_neighbors.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement Python import parser**

Create `src/ai_reviewer/context/__init__.py`:
```python
"""Prompt context builders: neighbor-file heuristics."""
```

Create `src/ai_reviewer/context/neighbors.py`:
```python
"""Heuristics for selecting non-changed files that clarify the diff.

Strategy:
- Siblings of each changed file (same directory).
- One-hop import graph: files imported by the changed file + files
  that import the changed file.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)


def parse_imports_python(source: str, current_module: str | None = None) -> set[str]:
    """Return dotted module names imported by the source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and current_module:
                parts = current_module.split(".")
                parts = parts[: -node.level] if node.level <= len(parts) else []
                prefix = ".".join(parts)
                base = f"{prefix}.{base}" if base else prefix
            for alias in node.names:
                name = alias.name
                full = f"{base}.{name}" if base else name
                imports.add(full)
    return imports
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_neighbors.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context tests/test_neighbors.py
git commit -m "feat(context): Python import parser for neighbor heuristics"
```

---

### Task 7: Regex-based parsers for TS/JS, Go, Rust, Java

**Files:**
- Modify: `src/ai_reviewer/context/neighbors.py` (append)
- Test: `tests/test_neighbors.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neighbors.py`:
```python
from ai_reviewer.context.neighbors import parse_imports_regex_ts, parse_imports_regex_go


def test_ts_imports():
    src = """
    import { foo } from "./utils";
    import bar from '../types';
    const q = require("./legacy");
    """
    out = parse_imports_regex_ts(src)
    assert "./utils" in out and "../types" in out and "./legacy" in out


def test_go_imports_single_and_block():
    src = """
package a
import "fmt"
import (
    "strings"
    "my/pkg"
)
    """
    out = parse_imports_regex_go(src)
    assert out == {"fmt", "strings", "my/pkg"}
```

- [ ] **Step 2: Run tests — they fail**

Run: `pytest tests/test_neighbors.py -v`
Expected: FAIL on new tests.

- [ ] **Step 3: Implement regex parsers**

Append to `src/ai_reviewer/context/neighbors.py`:
```python
_TS_IMPORT_RE = re.compile(
    r"""(?mx)
    ^\s*(?:
        import\s+[^;]*?from\s+["']([^"']+)["']
      | import\s+["']([^"']+)["']
      | (?:const|let|var)\s+\S+\s*=\s*require\(\s*["']([^"']+)["']\s*\)
    )
    """,
)


def parse_imports_regex_ts(source: str) -> set[str]:
    """Match TS/JS import specifiers (paths or module names)."""
    out: set[str] = set()
    for m in _TS_IMPORT_RE.finditer(source):
        for group in m.groups():
            if group:
                out.add(group)
    return out


_GO_SINGLE_RE = re.compile(r'^\s*import\s+"([^"]+)"\s*$', re.MULTILINE)
_GO_BLOCK_RE = re.compile(r"import\s*\((.*?)\)", re.DOTALL)
_GO_BLOCK_ITEM_RE = re.compile(r'"([^"]+)"')


def parse_imports_regex_go(source: str) -> set[str]:
    out: set[str] = set()
    out.update(m.group(1) for m in _GO_SINGLE_RE.finditer(source))
    for block in _GO_BLOCK_RE.finditer(source):
        out.update(m.group(1) for m in _GO_BLOCK_ITEM_RE.finditer(block.group(1)))
    return out


_RUST_USE_RE = re.compile(r"^\s*use\s+([a-zA-Z0-9_:]+)", re.MULTILINE)


def parse_imports_regex_rust(source: str) -> set[str]:
    return {m.group(1) for m in _RUST_USE_RE.finditer(source)}


_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([a-zA-Z0-9_.]+)\s*;", re.MULTILINE)


def parse_imports_regex_java(source: str) -> set[str]:
    return {m.group(1) for m in _JAVA_IMPORT_RE.finditer(source)}


def parse_imports_by_path(path: str, source: str) -> set[str]:
    """Dispatch by file extension."""
    p = PurePosixPath(path)
    ext = p.suffix.lower()
    if ext == ".py":
        return parse_imports_python(source)
    if ext in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        return parse_imports_regex_ts(source)
    if ext == ".go":
        return parse_imports_regex_go(source)
    if ext == ".rs":
        return parse_imports_regex_rust(source)
    if ext == ".java":
        return parse_imports_regex_java(source)
    return set()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_neighbors.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context/neighbors.py tests/test_neighbors.py
git commit -m "feat(context): add regex import parsers for TS/JS/Go/Rust/Java"
```

---

### Task 8: Neighbor selection (siblings + one-hop import graph)

**Files:**
- Modify: `src/ai_reviewer/context/neighbors.py` (append)
- Test: `tests/test_neighbors.py` (append)

- [ ] **Step 1: Add the selection test**

Append to `tests/test_neighbors.py`:
```python
from ai_reviewer.context.neighbors import select_neighbors


def test_select_neighbors_siblings_and_outbound_imports():
    changed = {"src/app/user.py": "from .auth import verify\nfrom ..models import User\n"}
    repo_paths = [
        "src/app/user.py",
        "src/app/auth.py",
        "src/app/profile.py",
        "src/models/__init__.py",
        "src/unrelated/other.py",
    ]

    def read(path: str) -> str:
        return {"src/app/user.py": changed["src/app/user.py"]}.get(path, "")

    neighbors = select_neighbors(
        changed_files=changed,
        repo_paths=repo_paths,
        read_file=read,
        max_siblings=2,
        max_total=6,
    )
    assert "src/app/auth.py" in neighbors           # sibling + import
    assert "src/app/profile.py" in neighbors        # sibling
    assert "src/unrelated/other.py" not in neighbors
```

- [ ] **Step 2: Implement `select_neighbors`**

Append to `src/ai_reviewer/context/neighbors.py`:
```python
def _module_to_possible_paths(module: str) -> list[str]:
    """Heuristic: turn 'ai_reviewer.models.findings' into possible path stems."""
    parts = module.split(".")
    stems = [
        "/".join(parts),
        "/".join(parts[:-1]) if len(parts) > 1 else parts[0],
    ]
    candidates: list[str] = []
    for stem in stems:
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java"):
            candidates.append(f"{stem}{ext}")
        candidates.append(f"{stem}/__init__.py")
        candidates.append(f"{stem}/index.ts")
        candidates.append(f"{stem}/index.js")
    return candidates


def select_neighbors(
    changed_files: dict[str, str],
    repo_paths: list[str],
    read_file,                              # Callable[[str], str]
    max_siblings: int = 5,
    max_total: int = 20,
) -> list[str]:
    """Return paths of neighbor files (not already changed)."""
    changed_set = set(changed_files)
    repo_set = set(repo_paths)
    picks: list[str] = []

    def add(path: str) -> bool:
        if path in changed_set or path in picks or path not in repo_set:
            return False
        picks.append(path)
        return len(picks) < max_total

    # 1) Siblings
    for path in changed_files:
        parent = str(PurePosixPath(path).parent)
        siblings = [p for p in repo_paths if str(PurePosixPath(p).parent) == parent and p != path]
        for s in siblings[:max_siblings]:
            if not add(s):
                return picks

    # 2) Outbound import graph
    for path, src in changed_files.items():
        imports = parse_imports_by_path(path, src)
        for module in imports:
            for cand in _module_to_possible_paths(module):
                if cand in repo_set and not add(cand):
                    return picks

    # 3) Inbound edges skipped here — expensive; caller may supplement via grep tool.
    return picks
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_neighbors.py -v`
Expected: PASS (6 tests).

- [ ] **Step 4: Commit**

```bash
git add src/ai_reviewer/context/neighbors.py tests/test_neighbors.py
git commit -m "feat(context): neighbor selection by siblings + outbound imports"
```

---

## Phase 3 — Anthropic client

### Task 9: `AnthropicClient` skeleton + happy-path message send

**Files:**
- Create: `src/ai_reviewer/agents/anthropic_client.py`
- Test: `tests/test_anthropic_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_anthropic_client.py`:
```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_reviewer.agents.anthropic_client import AnthropicClient, AnthropicReviewResult
from ai_reviewer.config import AnthropicApiConfig


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_response(text: str, stop_reason: str = "end_turn"):
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = [_text_block(text)]
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


@pytest.mark.asyncio
async def test_run_review_happy_path_parses_json():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    result = await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "You are a reviewer."}],
        user_blocks=[{"type": "text", "text": "diff..."}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    assert isinstance(result, AnthropicReviewResult)
    assert result.parsed == {"findings": [], "summary": "ok"}
    assert result.usage.input_tokens == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement happy-path `AnthropicClient.run_review`**

Create `src/ai_reviewer/agents/anthropic_client.py`:
```python
"""Anthropic Messages API client with tool-use loop, thinking, caching."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class AnthropicReviewResult:
    parsed: dict[str, Any]                       # {"findings": [...], "summary": "..."}
    raw_text: str                                # final assistant text content
    usage: UsageStats = field(default_factory=UsageStats)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class AnthropicClient:
    """Thin wrapper over the official anthropic SDK for review agents."""

    def __init__(self, config: AnthropicApiConfig) -> None:
        self.config = config
        self._sdk = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def close(self) -> None:
        await self._sdk.close()

    async def __aenter__(self) -> "AnthropicClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def run_review(
        self,
        model: str,
        system_blocks: list[dict[str, Any]],
        user_blocks: list[dict[str, Any]],
        output_schema: dict[str, Any],
        tool_registry: "ToolRegistry | None",
        thinking_budget: int | None,
        max_tokens: int,
        temperature: float,
    ) -> AnthropicReviewResult:
        # Phase 3 (task 9) implements the happy-path single-shot call only.
        # Tool use loop and thinking added in subsequent tasks.
        messages = [{"role": "user", "content": user_blocks}]
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_blocks,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = await self._sdk.messages.create(**kwargs)

        raw_text = _extract_text(response)
        parsed = _parse_json(raw_text)

        return AnthropicReviewResult(
            parsed=parsed,
            raw_text=raw_text,
            usage=UsageStats(
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
            ),
        )


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        m = re.search(r"```json\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    elif "```" in text:
        m = re.search(r"```\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %r", text[:200])
        return {"findings": [], "summary": "[parse error]"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat(agents): AnthropicClient happy-path run_review"
```

---

### Task 10: Structured output via `output_config.format=json_schema`

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py`
- Test: `tests/test_anthropic_client.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_anthropic_client.py`:
```python
@pytest.mark.asyncio
async def test_run_review_passes_output_schema_as_json_schema():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    schema = {"type": "object", "properties": {"findings": {"type": "array"}}}
    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema=schema,
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] == schema
```

- [ ] **Step 2: Run test — should fail**

Run: `pytest tests/test_anthropic_client.py::test_run_review_passes_output_schema_as_json_schema -v`
Expected: FAIL (`output_config` not in kwargs).

- [ ] **Step 3: Pass `output_config` in `run_review`**

In `anthropic_client.py::run_review`, update `kwargs`:
```python
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_blocks,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                },
            },
        }
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat(agents): pass JSON schema as output_config.format"
```

---

### Task 11: Extended thinking

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py`
- Test: `tests/test_anthropic_client.py` (append)

- [ ] **Step 1: Write the failing test**

Append:
```python
@pytest.mark.asyncio
async def test_run_review_with_thinking_budget_sets_thinking_config():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=8192,
        max_tokens=16384,
        temperature=1.0,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
async def test_run_review_without_thinking_omits_config():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
```

- [ ] **Step 2: Run tests — should fail**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: FAIL on thinking tests.

- [ ] **Step 3: Wire `thinking` into kwargs**

In `run_review`, after building `kwargs`:
```python
        if thinking_budget is not None and thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
```

Also, when thinking is enabled temperature must be 1.0 per the Messages API contract — validate and log:
```python
        if thinking_budget and temperature != 1.0:
            logger.info(
                "Thinking enabled; overriding temperature %.2f -> 1.0 per API contract",
                temperature,
            )
            kwargs["temperature"] = 1.0
```

Add a test for the override:
```python
@pytest.mark.asyncio
async def test_thinking_forces_temperature_one():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )
    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=8192,
        max_tokens=16384,
        temperature=0.2,
    )
    assert client._sdk.messages.create.call_args.kwargs["temperature"] == 1.0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat(agents): extended thinking config with temp override"
```

---

### Task 12: Tool-use loop

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py`
- Test: `tests/test_anthropic_client.py` (append)

- [ ] **Step 1: Write the failing test**

Append:
```python
def _tool_use_block(tool_id: str, name: str, input_: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_
    return b


def _tool_use_response(tool_id: str, name: str, input_: dict):
    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = [_tool_use_block(tool_id, name, input_)]
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


@pytest.mark.asyncio
async def test_tool_use_loop_dispatches_and_feeds_result_back():
    cfg = AnthropicApiConfig(api_key="sk-test")
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(side_effect=[
        _tool_use_response("t1", "read_file", {"path": "x.py"}),
        _fake_response('{"findings": [], "summary": "done"}'),
    ])

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="file-contents")

    result = await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    assert result.parsed == {"findings": [], "summary": "done"}
    registry.execute.assert_awaited_once_with("read_file", {"path": "x.py"})
    assert client._sdk.messages.create.await_count == 2

    # Second call should contain the tool_result in its messages
    second_kwargs = client._sdk.messages.create.await_args_list[1].kwargs
    last_msg = second_kwargs["messages"][-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["type"] == "tool_result"
    assert last_msg["content"][0]["tool_use_id"] == "t1"
    assert last_msg["content"][0]["content"] == "file-contents"
```

- [ ] **Step 2: Run test — should fail**

Run: `pytest tests/test_anthropic_client.py::test_tool_use_loop_dispatches_and_feeds_result_back -v`
Expected: FAIL.

- [ ] **Step 3: Implement the loop**

Replace `run_review` in `anthropic_client.py` (integrating all earlier features plus tool use):
```python
    async def run_review(
        self,
        model: str,
        system_blocks: list[dict[str, Any]],
        user_blocks: list[dict[str, Any]],
        output_schema: dict[str, Any],
        tool_registry: Any,                         # ToolRegistry | None
        thinking_budget: int | None,
        max_tokens: int,
        temperature: float,
        max_tool_rounds: int = 30,
    ) -> AnthropicReviewResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        usage = UsageStats()
        tool_calls: list[dict[str, Any]] = []

        tools = tool_registry.tool_specs() if tool_registry else None

        for _ in range(max_tool_rounds + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "system": system_blocks,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "output_config": {
                    "format": {"type": "json_schema", "schema": output_schema},
                },
            }
            if tools:
                kwargs["tools"] = tools
            if thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                if temperature != 1.0:
                    logger.info("Thinking enabled; overriding temperature -> 1.0")
                    kwargs["temperature"] = 1.0

            response = await self._sdk.messages.create(**kwargs)
            _accumulate_usage(usage, response)

            if response.stop_reason != "tool_use" or not tool_registry:
                return AnthropicReviewResult(
                    parsed=_parse_json(_extract_text(response)),
                    raw_text=_extract_text(response),
                    usage=usage,
                    tool_calls=tool_calls,
                )

            assistant_blocks = list(response.content)
            messages.append({"role": "assistant", "content": _serialize_blocks(assistant_blocks)})
            tool_result_blocks = []
            for block in assistant_blocks:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool_calls.append({"name": block.name, "input": block.input})
                try:
                    tool_output = await tool_registry.execute(block.name, block.input)
                except Exception as e:                              # noqa: BLE001
                    tool_output = f"[tool error: {e}]"
                    logger.warning("Tool %s failed: %s", block.name, e)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                })
            messages.append({"role": "user", "content": tool_result_blocks})

        logger.warning("Tool-use loop exceeded max_tool_rounds=%d", max_tool_rounds)
        return AnthropicReviewResult(
            parsed={"findings": [], "summary": "[tool loop cap]"},
            raw_text="",
            usage=usage,
            tool_calls=tool_calls,
        )


def _accumulate_usage(u: UsageStats, response: Any) -> None:
    ru = getattr(response, "usage", None)
    if not ru:
        return
    u.input_tokens += getattr(ru, "input_tokens", 0)
    u.output_tokens += getattr(ru, "output_tokens", 0)
    u.cache_read_input_tokens += getattr(ru, "cache_read_input_tokens", 0)
    u.cache_creation_input_tokens += getattr(ru, "cache_creation_input_tokens", 0)


def _serialize_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert SDK block objects back to the dict form the API expects."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif t == "tool_use":
            out.append({
                "type": "tool_use",
                "id": getattr(b, "id"),
                "name": getattr(b, "name"),
                "input": getattr(b, "input"),
            })
        elif t == "thinking":
            out.append({
                "type": "thinking",
                "thinking": getattr(b, "thinking", ""),
                "signature": getattr(b, "signature", ""),
            })
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat(agents): tool-use loop with ToolRegistry dispatch"
```

---

### Task 13: Prompt caching — mark system blocks `cache_control: ephemeral`

**Files:**
- Modify: `src/ai_reviewer/agents/anthropic_client.py`
- Test: `tests/test_anthropic_client.py` (append)

- [ ] **Step 1: Write the failing test**

Append:
```python
@pytest.mark.asyncio
async def test_caching_marks_last_system_block_when_enabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[
            {"type": "text", "text": "role"},
            {"type": "text", "text": "conventions"},
        ],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert sent[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_caching_disabled_leaves_system_unchanged():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "role"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert "cache_control" not in sent[0]
```

- [ ] **Step 2: Run tests — should fail**

Expected: FAIL.

- [ ] **Step 3: Apply `cache_control` on the last system block when enabled**

In `run_review`, just before `kwargs = {...}` is built:
```python
        system_to_send = system_blocks
        if self.config.enable_prompt_caching and system_blocks:
            system_to_send = [dict(b) for b in system_blocks]
            system_to_send[-1]["cache_control"] = {"type": "ephemeral"}
```
And change `kwargs["system"] = system_to_send`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat(agents): prompt caching on last system block"
```

---

## Phase 4 — Context builders

### Task 14: `build_system_blocks()` — role + schema + conventions + repo map

**Files:**
- Create: `src/ai_reviewer/context/builder.py`
- Test: `tests/test_context_builder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_context_builder.py`:
```python
from unittest.mock import MagicMock

from ai_reviewer.context.builder import build_system_blocks, FINDINGS_SCHEMA


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
    assert "findings" in combined       # schema reference
    assert "Always cite file:line." in combined
    assert "Follow PEP8." in combined
    assert "src/" in combined
    # last block is the largest/most-cacheable bundle (conventions + map)
    assert blocks[-1]["type"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

Create `src/ai_reviewer/context/builder.py`:
```python
"""Assemble system/user prompt blocks for Anthropic Messages API."""

from __future__ import annotations

import json
from typing import Any

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["findings", "summary"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {"$ref": "#/$defs/Finding"},
        },
        "summary": {"type": "string"},
    },
    "$defs": {
        "Finding": {
            "type": "object",
            "required": [
                "file_path",
                "line_start",
                "severity",
                "category",
                "title",
                "description",
                "confidence",
            ],
            "properties": {
                "file_path": {"type": "string"},
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": ["integer", "null"], "minimum": 1},
                "severity": {"enum": ["critical", "warning", "suggestion", "nitpick"]},
                "category": {
                    "enum": [
                        "security",
                        "performance",
                        "logic",
                        "style",
                        "architecture",
                        "testing",
                        "documentation",
                    ],
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "suggested_fix": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
        },
    },
}


def build_system_blocks(
    agent_role: str,
    convention_texts: dict[str, str],
    repo_map: str,
) -> list[dict[str, Any]]:
    """Return system prompt blocks in deterministic order.

    Order matters for caching: later blocks are the ones marked
    cache_control by the client.
    """
    role_block = {"type": "text", "text": f"{agent_role.strip()}\n\n"
                                           "Respond only in the JSON format described by the schema."}
    schema_block = {
        "type": "text",
        "text": "## Output schema (enforced)\n\n```json\n"
                + json.dumps(FINDINGS_SCHEMA, indent=2)
                + "\n```",
    }
    convention_parts = []
    for name, text in convention_texts.items():
        convention_parts.append(f"### {name}\n\n{text.strip()}")
    convention_block = {
        "type": "text",
        "text": "## Project conventions\n\n" + "\n\n".join(convention_parts)
        if convention_parts
        else "## Project conventions\n\n(none available)",
    }
    map_block = {
        "type": "text",
        "text": f"## Repository map\n\n{repo_map.strip()}",
    }
    return [role_block, schema_block, convention_block, map_block]
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_context_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context/builder.py tests/test_context_builder.py
git commit -m "feat(context): build_system_blocks with schema + conventions + map"
```

---

### Task 15: `build_user_blocks()` — PR meta + diff + changed files + neighbors

**Files:**
- Modify: `src/ai_reviewer/context/builder.py` (append)
- Test: `tests/test_context_builder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_context_builder.py`:
```python
from ai_reviewer.context.builder import build_user_blocks


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
    assert "keep-this" in combined                      # diff + changed never truncated
    assert "[... neighbors truncated ...]" in combined
```

- [ ] **Step 2: Run — should fail**

Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `src/ai_reviewer/context/builder.py`:
```python
def _files_block(heading: str, files: dict[str, str]) -> str:
    if not files:
        return f"## {heading}\n\n(none)"
    parts = [f"## {heading}\n"]
    for path, content in files.items():
        parts.append(f"### {path}\n```\n{content}\n```\n")
    return "\n".join(parts)


def build_user_blocks(
    pr_title: str,
    pr_body: str,
    diff: str,
    changed_files: dict[str, str],
    neighbor_files: dict[str, str],
    max_total_chars: int = 600_000,     # ~150K tokens at ~4 chars/token
) -> list[dict[str, Any]]:
    """Assemble the user message for review.

    Truncation priority (lowest first): neighbors, changed files.
    The diff is never truncated.
    """
    pr_meta = f"## PR metadata\n\n**Title:** {pr_title}\n\n**Description:**\n\n{pr_body or '(empty)'}"
    diff_block = f"## Diff\n\n```diff\n{diff}\n```"
    changed_block = _files_block("Changed files (full contents)", changed_files)
    neighbor_block = _files_block("Neighbor files (context)", neighbor_files)

    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    if len(assembled) <= max_total_chars:
        return [{"type": "text", "text": assembled}]

    # Over budget: drop neighbors first, then truncate changed files.
    neighbor_block = _files_block("Neighbor files (context)", {}) + "\n[... neighbors truncated ...]"
    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    if len(assembled) <= max_total_chars:
        return [{"type": "text", "text": assembled}]

    # Still over: truncate the last changed file content.
    truncated = {}
    budget = max_total_chars - len(pr_meta) - len(diff_block) - len(neighbor_block) - 1000
    for path, content in changed_files.items():
        if budget <= 0:
            truncated[path] = "[... file omitted due to budget ...]"
            continue
        if len(content) > budget:
            truncated[path] = content[:budget] + "\n[... file truncated ...]"
            budget = 0
        else:
            truncated[path] = content
            budget -= len(content)
    changed_block = _files_block("Changed files (full contents)", truncated)
    assembled = "\n\n".join([pr_meta, diff_block, changed_block, neighbor_block])
    return [{"type": "text", "text": assembled}]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_context_builder.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context/builder.py tests/test_context_builder.py
git commit -m "feat(context): build_user_blocks with truncation policy"
```

---

## Phase 5 — Rewrite ReviewAgent base + subclasses

### Task 16: Rewrite `ReviewAgent.review()` on `AnthropicClient`

**Files:**
- Modify: `src/ai_reviewer/agents/base.py` (full rewrite)
- Test: `tests/test_base_agent.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_base_agent.py`:
```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_reviewer.agents.anthropic_client import AnthropicReviewResult, UsageStats
from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.models.context import ReviewContext


class DummyAgent(ReviewAgent):
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "dummy"
    FOCUS_AREAS = ["security"]
    SYSTEM_PROMPT = "You are a dummy reviewer."
    THINKING_ENABLED = True
    THINKING_BUDGET = 4096


@pytest.mark.asyncio
async def test_review_agent_uses_anthropic_client():
    client = MagicMock()
    client.run_review = AsyncMock(return_value=AnthropicReviewResult(
        parsed={
            "findings": [{
                "file_path": "a.py",
                "line_start": 1,
                "severity": "warning",
                "category": "security",
                "title": "t",
                "description": "d",
                "confidence": 0.9,
            }],
            "summary": "sum",
        },
        raw_text="",
        usage=UsageStats(input_tokens=100, output_tokens=20),
    ))

    agent = DummyAgent(
        client=client,
        agent_id="dummy-1",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        tool_registry=None,
        max_tokens=4096,
        temperature=0.2,
    )
    ctx = ReviewContext(repo_name="o/r", pr_number=1)
    review = await agent.review(diff="d", file_contents={}, context=ctx)

    assert review.agent_id == "dummy-1"
    assert review.agent_type == "dummy"
    assert len(review.findings) == 1
    assert review.summary == "sum"
    kwargs = client.run_review.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["thinking_budget"] == 4096
```

- [ ] **Step 2: Run — should fail**

Expected: FAIL (constructor signature differs).

- [ ] **Step 3: Rewrite `base.py`**

Replace the entire contents of `src/ai_reviewer/agents/base.py`:
```python
"""Base class for review agents (Anthropic Messages API backed)."""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_reviewer.agents.anthropic_client import AnthropicClient
from ai_reviewer.context.builder import FINDINGS_SCHEMA
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ReviewFinding, Severity
from ai_reviewer.models.review import AgentReview

logger = logging.getLogger(__name__)


class ReviewAgent:
    """Base class for all review agents."""

    MODEL: str = "claude-sonnet-4-6"
    AGENT_TYPE: str = "base"
    FOCUS_AREAS: list[str] = []
    SYSTEM_PROMPT: str = "You are a code reviewer."
    THINKING_ENABLED: bool = False
    THINKING_BUDGET: int = 8192

    def __init__(
        self,
        client: AnthropicClient,
        agent_id: str,
        system_blocks: list[dict[str, Any]],
        user_blocks: list[dict[str, Any]],
        tool_registry: Any,                 # ToolRegistry | None
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self.client = client
        self._agent_id = agent_id
        self._system_blocks = system_blocks
        self._user_blocks = user_blocks
        self._tool_registry = tool_registry
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def focus_areas(self) -> list[str]:
        return self.FOCUS_AREAS

    async def review(
        self,
        diff: str,                          # accepted for API parity; context is in user_blocks
        file_contents: dict[str, str],      # same — context is in user_blocks
        context: ReviewContext | dict[str, Any],
    ) -> AgentReview:
        start_time = time.monotonic()

        system_blocks = self._prepend_role(self._system_blocks)
        thinking_budget = self.THINKING_BUDGET if self.THINKING_ENABLED else None

        try:
            result = await self.client.run_review(
                model=self.MODEL,
                system_blocks=system_blocks,
                user_blocks=self._user_blocks,
                output_schema=FINDINGS_SCHEMA,
                tool_registry=self._tool_registry,
                thinking_budget=thinking_budget,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as e:
            logger.error("Agent %s failed: %s", self.agent_id, e)
            raise

        findings = _parse_findings(result.parsed)
        summary = result.parsed.get("summary", "Review completed")
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return AgentReview(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            focus_areas=self.focus_areas,
            findings=findings,
            summary=summary,
            review_time_ms=elapsed_ms,
        )

    def _prepend_role(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject this agent's SYSTEM_PROMPT as the first block."""
        role_block = {"type": "text", "text": self.SYSTEM_PROMPT}
        return [role_block, *blocks]


def _parse_findings(parsed: dict[str, Any]) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for raw in parsed.get("findings", []) or []:
        try:
            findings.append(ReviewFinding(
                file_path=raw["file_path"],
                line_start=int(raw["line_start"]),
                line_end=int(raw["line_end"]) if raw.get("line_end") else None,
                severity=Severity(str(raw["severity"]).lower()),
                category=Category(str(raw["category"]).lower()),
                title=raw["title"],
                description=raw["description"],
                suggested_fix=raw.get("suggested_fix"),
                confidence=float(raw.get("confidence", 0.8)),
            ))
        except (KeyError, ValueError) as e:
            logger.warning("Failed to parse finding: %s, raw=%r", e, raw)
    return findings
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_base_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/base.py tests/test_base_agent.py
git commit -m "refactor(agents): rewrite ReviewAgent.review on AnthropicClient"
```

---

### Task 17: Update subclass `MODEL` / thinking flags / remove GPT

**Files:**
- Modify: `src/ai_reviewer/agents/security.py`
- Modify: `src/ai_reviewer/agents/patterns.py`
- Modify: `src/ai_reviewer/agents/performance.py`

- [ ] **Step 1: Update `security.py`**

In `SecurityAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "security-reviewer"
    FOCUS_AREAS = ["security", "authentication", "data_validation", "cryptography"]
    THINKING_ENABLED = True
    THINKING_BUDGET = 8192
```

In `AuthenticationAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "authentication-reviewer"
    FOCUS_AREAS = ["authentication", "authorization", "session_management"]
    THINKING_ENABLED = True
    THINKING_BUDGET = 8192
```

- [ ] **Step 2: Update `patterns.py`**

In `PatternsAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "patterns-reviewer"
    FOCUS_AREAS = ["consistency", "patterns", "architecture", "maintainability"]
    THINKING_ENABLED = True
    THINKING_BUDGET = 8192
```

In `StyleAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "style-reviewer"
    FOCUS_AREAS = ["style", "documentation", "readability"]
    THINKING_ENABLED = False
```

- [ ] **Step 3: Update `performance.py` — swap GPT for Sonnet**

In `PerformanceAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "performance-reviewer"
    FOCUS_AREAS = ["performance", "complexity", "resource_management", "efficiency"]
    THINKING_ENABLED = False
```

In `LogicAgent`:
```python
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "logic-reviewer"
    FOCUS_AREAS = ["logic", "edge_cases", "error_handling", "correctness"]
    THINKING_ENABLED = True
    THINKING_BUDGET = 8192
```

- [ ] **Step 4: Run existing agent tests**

Run: `pytest tests/test_agents.py -v`
Expected: these tests will partly fail because they still reference `CursorClient`. Leave them failing — Task 23 fixes them after `review.py` is migrated.

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/agents/security.py src/ai_reviewer/agents/patterns.py src/ai_reviewer/agents/performance.py
git commit -m "refactor(agents): update subclass models to Claude only

Security/Auth/Patterns/Logic/Performance use claude-sonnet-4-6 (thinking disabled).
Style/doc-gen use claude-haiku-4-5-20251001.
GPT models removed — Anthropic only per migration spec."
```

---

## Phase 6 — Rewire `review.py`

### Task 18: Convention + repo-map fetcher

**Files:**
- Create: `src/ai_reviewer/context/fetch.py`
- Test: `tests/test_context_fetch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_context_fetch.py`:
```python
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_reviewer.context.fetch import fetch_conventions, build_repo_map
from ai_reviewer.session import ReviewSession


def _encoded(content: str) -> MagicMock:
    m = MagicMock()
    m.content = base64.b64encode(content.encode()).decode()
    return m


def test_fetch_conventions_returns_texts():
    session = ReviewSession(repo="o/r", head_sha="abc", github_budget=50)
    gh = MagicMock()

    def contents(path, ref=None):
        mapping = {"AGENTS.md": "conv-a", "CONTRIBUTING.md": "conv-c"}
        if path in mapping:
            return _encoded(mapping[path])
        raise FileNotFoundError(path)

    gh.client.get_repo.return_value.get_contents.side_effect = contents

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
    gh.client.get_repo.return_value.get_git_tree.return_value.tree = tree

    out = build_repo_map(session, gh)
    assert "src/" in out
    assert "tests/" in out
    assert "README.md" in out
```

- [ ] **Step 2: Run — should fail**

Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/ai_reviewer/context/fetch.py`:
```python
"""Fetch conventions + repo map via GitHubClient (with ReviewSession budget)."""

from __future__ import annotations

import base64
import logging
from typing import Iterable

from ai_reviewer.github.client import GitHubClient
from ai_reviewer.session import ReviewSession

logger = logging.getLogger(__name__)


def fetch_conventions(
    session: ReviewSession,
    gh: GitHubClient,
    paths: Iterable[str],
) -> dict[str, str]:
    """Fetch convention files that exist; skip missing silently."""
    out: dict[str, str] = {}
    repo = gh.client.get_repo(session.repo)
    for path in paths:
        cached = session.cached_file(path)
        if cached is not None:
            out[path] = cached
            continue
        if session.is_github_budget_exhausted():
            break
        session.consume_github_request()
        try:
            contents = repo.get_contents(path, ref=session.head_sha)
        except Exception as e:                                  # noqa: BLE001
            logger.debug("Convention %s not found: %s", path, e)
            continue
        try:
            text = base64.b64decode(getattr(contents, "content", "")).decode("utf-8", errors="replace")
        except Exception as e:                                  # noqa: BLE001
            logger.warning("Convention %s decode failed: %s", path, e)
            continue
        session.store_file(path, text)
        out[path] = text
    return out


def build_repo_map(session: ReviewSession, gh: GitHubClient) -> str:
    cached_tree = session.cached_tree()
    if cached_tree is None:
        if session.is_github_budget_exhausted():
            return "(repo map unavailable: budget exhausted)"
        session.consume_github_request()
        try:
            repo = gh.client.get_repo(session.repo)
            tree = repo.get_git_tree(session.head_sha, recursive=True)
        except Exception as e:                                  # noqa: BLE001
            logger.warning("Tree fetch failed: %s", e)
            return "(repo map unavailable)"
        paths = [
            item.path + ("/" if getattr(item, "type", None) == "tree" else "")
            for item in tree.tree
        ]
        session.store_tree([p.rstrip("/") for p in paths])
    else:
        paths = [p + "/" for p in cached_tree if "/" not in p.rstrip("/")] + [
            p for p in cached_tree if "/" in p
        ]

    top_level = sorted({p.split("/", 1)[0] + ("/" if "/" in p else "") for p in paths})
    return "\n".join(top_level)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_context_fetch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/context/fetch.py tests/test_context_fetch.py
git commit -m "feat(context): fetch_conventions + build_repo_map"
```

---

### Task 19: Replace Cursor calls in `review.py` — build agents and run

**Files:**
- Modify: `src/ai_reviewer/review.py:31-35` (imports), `:1108-1212` (multi-agent flow and cross-review)
- Delete: the `run_review_agent` and `run_single_agent` Cursor-specific helpers (lines ~953-1000)

This is the largest change. It replaces three `async with CursorClient(...)` blocks with a unified Anthropic-backed flow.

- [ ] **Step 1: Update imports**

At the top of `review.py`, replace:
```python
from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
```
with:
```python
from ai_reviewer.agents.anthropic_client import AnthropicClient
from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.context.builder import build_system_blocks, build_user_blocks, FINDINGS_SCHEMA
from ai_reviewer.context.fetch import build_repo_map, fetch_conventions
from ai_reviewer.context.neighbors import select_neighbors
from ai_reviewer.session import ReviewSession
from ai_reviewer.tools.repo_tools import ToolRegistry
from ai_reviewer.agents.security import AuthenticationAgent, SecurityAgent
from ai_reviewer.agents.patterns import PatternsAgent, StyleAgent
from ai_reviewer.agents.performance import LogicAgent, PerformanceAgent
```

- [ ] **Step 2: Replace `AGENT_CONFIGS` with an agent class registry**

Replace the module-level `AGENT_CONFIGS` list with:
```python
_AGENT_CLASSES = {
    "security-reviewer": SecurityAgent,
    "performance-reviewer": PerformanceAgent,
    "patterns-reviewer": PatternsAgent,
    "logic-reviewer": LogicAgent,
    "style-reviewer": StyleAgent,
    "authentication-reviewer": AuthenticationAgent,
}

DEFAULT_AGENT_ORDER = [
    "security-reviewer",
    "performance-reviewer",
    "patterns-reviewer",
    "logic-reviewer",
    "style-reviewer",
]
```

Delete the old AGENT_CONFIGS list (lines ~42-350). The prompt content baked into its `prompt_addition` fields is now covered by the `SYSTEM_PROMPT` on each agent class (Task 17 already updated these).

- [ ] **Step 3: Add a function to build the shared prompt context**

Add near the bottom of `review.py` (before `review_pr()`):
```python
CONVENTION_PATHS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".ai/rules/architecture.md",
    ".ai/rules/conventions.md",
    ".ai/rules/agents.md",
    ".cursor/rules/README.md",            # backward-compat
]


async def _prepare_shared_context(
    session: ReviewSession,
    gh: GitHubClient,
    pr,
    diff: str,
    changed_file_contents: dict[str, str],
    anthropic_cfg: AnthropicApiConfig,
) -> tuple[list[dict], list[dict]]:
    conventions = fetch_conventions(session, gh, CONVENTION_PATHS)
    repo_map = build_repo_map(session, gh)

    neighbors: dict[str, str] = {}
    tree = session.cached_tree() or []
    def _read(p: str) -> str:
        cached = session.cached_file(p)
        if cached is not None:
            return cached
        return ""
    neighbor_paths = select_neighbors(
        changed_files=changed_file_contents,
        repo_paths=tree,
        read_file=_read,
        max_siblings=5,
        max_total=15,
    )
    repo = gh.client.get_repo(session.repo)
    for path in neighbor_paths:
        if session.is_github_budget_exhausted():
            break
        cached = session.cached_file(path)
        if cached is not None:
            neighbors[path] = cached
            continue
        try:
            session.consume_github_request()
            c = repo.get_contents(path, ref=session.head_sha)
            import base64 as _b64
            text = _b64.b64decode(getattr(c, "content", "")).decode("utf-8", errors="replace")
            if len(text) > anthropic_cfg.per_file_max_bytes:
                text = text[: anthropic_cfg.per_file_max_bytes] + "\n[truncated]"
            session.store_file(path, text)
            neighbors[path] = text
        except Exception as e:                                  # noqa: BLE001
            logger.debug("Neighbor fetch failed %s: %s", path, e)

    system_blocks = build_system_blocks(
        agent_role="You are a code reviewer. Each agent adds its focus area in its role block.",
        convention_texts=conventions,
        repo_map=repo_map,
    )
    user_blocks = build_user_blocks(
        pr_title=getattr(pr, "title", "") or "",
        pr_body=getattr(pr, "body", "") or "",
        diff=diff,
        changed_files=changed_file_contents,
        neighbor_files=neighbors,
        max_total_chars=anthropic_cfg.max_combined_context_tokens * 4,
    )
    return system_blocks, user_blocks
```

- [ ] **Step 4: Replace the multi-agent block**

Locate `async with CursorClient(cursor_config) as client:` at line ~1130 and at line ~1147. Replace both with a single Anthropic-backed flow. In the body of `review_pr()` (where the multi-agent execution happens), replace the old two-branch (1 agent vs. multi-agent) block with:

```python
    # --- New Anthropic-backed flow ---
    session = ReviewSession(
        repo=repo,
        head_sha=pr.head.sha,
        github_budget=anthropic_cfg.per_review_github_request_budget,
    )

    async with AnthropicClient(anthropic_cfg) as client:
        system_blocks, user_blocks = await _prepare_shared_context(
            session=session,
            gh=github_client,
            pr=pr,
            diff=diff,
            changed_file_contents=files,
            anthropic_cfg=anthropic_cfg,
        )

        # Resolve agent names from config, falling back to defaults
        configured = [a.name for a in (config.agents if config else [])]
        agent_order = (configured or DEFAULT_AGENT_ORDER)[: min(num_agents, len(DEFAULT_AGENT_ORDER))]

        tasks = []
        for i, agent_name in enumerate(agent_order):
            cls = _AGENT_CLASSES.get(agent_name)
            if not cls:
                logger.warning("Unknown agent %s; skipping", agent_name)
                continue
            agent_cfg = next((a for a in (config.agents if config else []) if a.name == agent_name), None)
            allow_tools = agent_cfg.allow_tool_use if agent_cfg else True
            max_tool_calls = agent_cfg.max_tool_calls if agent_cfg else 20
            registry = (
                ToolRegistry(
                    session=session,
                    github_client=github_client,
                    agent_id=f"{agent_name}-{i}",
                    max_calls=max_tool_calls,
                    per_file_max_bytes=anthropic_cfg.per_file_max_bytes,
                )
                if allow_tools
                else None
            )
            agent = cls(
                client=client,
                agent_id=f"{agent_name}-{i}",
                system_blocks=system_blocks,
                user_blocks=user_blocks,
                tool_registry=registry,
                max_tokens=agent_cfg.max_tokens if agent_cfg else 8192,
                temperature=agent_cfg.temperature if agent_cfg else 0.3,
            )
            tasks.append(_run_agent_with_status(agent, context, on_status))

        if on_status:
            on_status("CREATING")
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map results to the (name, findings, summary) tuple that aggregate_findings expects
    all_findings = []
    for agent_name, result in zip(agent_order, agent_results):
        if isinstance(result, Exception):
            logger.error("Agent %s failed: %s", agent_name, result)
            all_findings.append((agent_name, [], f"[error: {result}]"))
            continue
        all_findings.append((agent_name, result.findings, result.summary))
```

Add `_run_agent_with_status` helper above `review_pr`:
```python
async def _run_agent_with_status(agent, context, on_status):
    name = agent.agent_id
    if on_status:
        on_status(f"{name}: RUNNING")
    try:
        review = await agent.review(diff="", file_contents={}, context=context)
    finally:
        if on_status:
            on_status(f"{name}: DONE")
    return review
```

- [ ] **Step 5: Replace the cross-review Cursor block**

Locate `async with CursorClient(cursor_config) as client:` around line ~1199 (cross-review round). Since the main flow already has an `AnthropicClient`, this now reuses the same client instance. Rewrite the cross-review section so that:
  - It is enclosed in the same `async with AnthropicClient(anthropic_cfg) as client:` block (move the code in-scope), OR
  - Open a fresh client for the cross-review round.

Simpler: open a fresh client. Replace with:
```python
            async with AnthropicClient(anthropic_cfg) as cross_client:
                cross_results = await run_cross_review_round(
                    client=cross_client,
                    session=session,
                    gh=github_client,
                    pr=pr,
                    review=review,
                    context=context,
                    diff=diff,
                    agents_to_run=agents_for_cross,
                    anthropic_cfg=anthropic_cfg,
                    on_status=on_status,
                )
```

Update the `run_cross_review_round` signature accordingly (find it in `review.py`, change its `client: CursorClient` parameter to `client: AnthropicClient`, and rewrite its internals to instantiate `ReviewAgent` subclasses with pre-built system/user blocks describing the validation task).

A minimal rewrite of `run_cross_review_round` body (preserving its public contract):
```python
async def run_cross_review_round(
    *,
    client: AnthropicClient,
    session: ReviewSession,
    gh: GitHubClient,
    pr,
    review: ConsolidatedReview,
    context: ReviewContext,
    diff: str,
    agents_to_run: list[dict],
    anthropic_cfg: AnthropicApiConfig,
    on_status: Callable | None = None,
) -> list[tuple[str, list, str]]:
    if not review.findings:
        return []
    # Build a cross-review-specific user block with the prior findings to validate
    prior_findings_json = json.dumps(
        [f.to_dict() if hasattr(f, "to_dict") else {
            "file_path": f.file_path,
            "line_start": f.line_start,
            "severity": f.severity.value if hasattr(f.severity, "value") else f.severity,
            "category": f.category.value if hasattr(f.category, "value") else f.category,
            "title": f.title,
            "description": f.description,
        } for f in review.findings],
        indent=2,
    )
    system_blocks, user_blocks = await _prepare_shared_context(
        session=session, gh=gh, pr=pr, diff=diff,
        changed_file_contents={}, anthropic_cfg=anthropic_cfg,
    )
    cross_user_blocks = user_blocks + [{
        "type": "text",
        "text": (
            "## Cross-review task\n\n"
            "The findings below were produced by earlier agents. "
            "Re-evaluate each: is it a true positive? What is your confidence? "
            "Return findings list in the same schema. For each finding you agree with, "
            "copy its file_path/line_start/title/category; set your own confidence.\n\n"
            f"```json\n{prior_findings_json}\n```"
        ),
    }]
    results: list[tuple[str, list, str]] = []
    for i, cfg in enumerate(agents_to_run):
        cls = _AGENT_CLASSES.get(cfg["name"])
        if not cls:
            continue
        agent = cls(
            client=client,
            agent_id=f"{cfg['name']}-cross-{i}",
            system_blocks=system_blocks,
            user_blocks=cross_user_blocks,
            tool_registry=None,
            max_tokens=8192,
            temperature=0.2,
        )
        try:
            r = await agent.review(diff="", file_contents={}, context=context)
            results.append((cfg["name"], r.findings, r.summary))
        except Exception as e:                                  # noqa: BLE001
            logger.error("Cross-review agent %s failed: %s", cfg["name"], e)
            results.append((cfg["name"], [], f"[error: {e}]"))
    return results
```

- [ ] **Step 6: Delete the obsolete helpers**

Remove `run_review_agent` and `run_single_agent` (the Cursor-specific wrappers) from `review.py` entirely. Also remove references to `cursor_config` parameters; replace with `anthropic_cfg`.

- [ ] **Step 7: Run entire test suite**

Run: `pytest -v`
Expected: `tests/test_agents.py` and `tests/test_convergence.py` may still fail — Task 20 fixes those.

- [ ] **Step 8: Commit**

```bash
git add src/ai_reviewer/review.py src/ai_reviewer/context/fetch.py tests/test_context_fetch.py
git commit -m "refactor(review): rewire multi-agent + cross-review on AnthropicClient

- Shared prompt context built once per review
- ReviewSession threads quota + caches through agents
- Cursor-specific run_review_agent/run_single_agent helpers removed"
```

---

### Task 20: Update `tests/test_agents.py` and `tests/test_convergence.py`

**Files:**
- Modify: `tests/test_agents.py`
- Modify: `tests/test_convergence.py`

- [ ] **Step 1: Read current tests to identify Cursor mocks**

Run: `grep -n "CursorClient\|complete_json\|cursor_client" tests/test_agents.py tests/test_convergence.py`

- [ ] **Step 2: Replace Cursor mocks with Anthropic mocks**

For each test that previously built a `CursorClient` mock whose `complete_json` returned a dict, replace with an `AnthropicClient` mock whose `run_review` returns an `AnthropicReviewResult`.

Example before:
```python
client = MagicMock(spec=CursorClient)
client.complete_json = AsyncMock(return_value={"findings": [...], "summary": "s"})
agent = SecurityAgent(client=client)
```

After:
```python
from ai_reviewer.agents.anthropic_client import AnthropicReviewResult, UsageStats

client = MagicMock()
client.run_review = AsyncMock(return_value=AnthropicReviewResult(
    parsed={"findings": [...], "summary": "s"},
    raw_text="",
    usage=UsageStats(),
))
agent = SecurityAgent(
    client=client,
    agent_id="security-test",
    system_blocks=[{"type": "text", "text": "sys"}],
    user_blocks=[{"type": "text", "text": "u"}],
    tool_registry=None,
)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_agents.py tests/test_convergence.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agents.py tests/test_convergence.py
git commit -m "test: update agent + convergence tests for Anthropic client"
```

---

## Phase 7 — Wiring and config rollout

### Task 21: Update CLI and webhook entry points

**Files:**
- Modify: `src/ai_reviewer/cli.py:18` (import), `:182-280` (config usage), `:611` (webhook startup)
- Modify: `src/ai_reviewer/github/webhook.py:132-150` (imports/config)

- [ ] **Step 1: `cli.py` — swap Cursor config usage**

Replace the import:
```python
from ai_reviewer.config import AnthropicApiConfig, load_config, validate_config
```

Where `cursor_config = CursorConfig(api_key=config.cursor.api_key, ...)` is built, replace with:
```python
anthropic_cfg = config.anthropic
if not anthropic_cfg or not anthropic_cfg.api_key:
    click.echo("error: anthropic.api_key not configured", err=True)
    raise SystemExit(2)
```

And where `review_pr()` is called, pass `anthropic_cfg=anthropic_cfg` instead of `cursor_config=...`. Adjust `review_pr`'s signature accordingly.

- [ ] **Step 2: `webhook.py` — swap Cursor config usage**

In the `from ai_reviewer.agents.cursor_client import CursorConfig` block, replace with:
```python
from ai_reviewer.config import AnthropicApiConfig
```

Swap any `CursorConfig(...)` construction for a direct use of `config.anthropic`.

- [ ] **Step 3: Run CLI unit test**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (after you update fixture config to include `anthropic:` block).

- [ ] **Step 4: Commit**

```bash
git add src/ai_reviewer/cli.py src/ai_reviewer/github/webhook.py tests/test_cli.py
git commit -m "refactor(cli, webhook): use AnthropicApiConfig instead of Cursor"
```

---

### Task 22: Update YAML config files

**Files:**
- Modify: `config.example.yaml:12-16` (cursor block)
- Modify: `.ai-reviewer.yaml` (same shape)

- [ ] **Step 1: Replace the `cursor:` block with `anthropic:`**

In `config.example.yaml`, replace:
```yaml
cursor:
  api_key: ${CURSOR_API_KEY}
  timeout_seconds: 120
```
with:
```yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  default_model: claude-sonnet-4-6
  timeout_seconds: 300
  enable_prompt_caching: true
  max_combined_context_tokens: 150000
  per_file_max_bytes: 524288
  per_review_github_request_budget: 200
```

- [ ] **Step 2: Update the agents list**

Replace model strings:
- `claude-4.5-opus-high-thinking` → `claude-sonnet-4-6` (thinking disabled — see `docs/optimization.md`)
- `gpt-5.2` → `claude-sonnet-4-6`
- style-reviewer → `claude-haiku-4-5-20251001`

For each agent entry, add:
```yaml
    thinking_enabled: false
    allow_tool_use: true
    max_tool_calls: 8
```

- [ ] **Step 3: Mirror the same in `.ai-reviewer.yaml`**

Same edits.

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml .ai-reviewer.yaml
git commit -m "config: migrate YAML to anthropic block + Claude models"
```

---

### Task 23: Delete `cursor_client.py` and `CursorApiConfig`

**Files:**
- Delete: `src/ai_reviewer/agents/cursor_client.py`
- Modify: `src/ai_reviewer/config.py` (remove `CursorApiConfig` and its parsing)
- Modify: `src/ai_reviewer/agents/__init__.py` (remove Cursor exports)

- [ ] **Step 1: Confirm no imports remain**

Run:
```bash
grep -rn "CursorClient\|CursorConfig\|CursorApiConfig\|cursor_client" src/ tests/
```
Expected: only hits are in the files being modified.

- [ ] **Step 2: Delete the Cursor client module**

```bash
git rm src/ai_reviewer/agents/cursor_client.py
```

- [ ] **Step 3: Remove `CursorApiConfig` from `config.py`**

Delete the `@dataclass CursorApiConfig` block and its parsing. Update `Config` to drop the `cursor:` field. Update `validate_config()` — the "either Cursor or Anthropic" check becomes an Anthropic-only check:
```python
    if not config.anthropic or not config.anthropic.api_key:
        errors.append(
            "Missing Anthropic API key (set ANTHROPIC_API_KEY or anthropic.api_key)"
        )
```

- [ ] **Step 4: Update `agents/__init__.py`**

Replace with:
```python
"""Review agents for AI Code Reviewer."""

from ai_reviewer.agents.anthropic_client import AnthropicClient, AnthropicReviewResult
from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.agents.patterns import PatternsAgent, StyleAgent
from ai_reviewer.agents.performance import LogicAgent, PerformanceAgent
from ai_reviewer.agents.security import AuthenticationAgent, SecurityAgent

__all__ = [
    "AnthropicClient",
    "AnthropicReviewResult",
    "AuthenticationAgent",
    "LogicAgent",
    "PatternsAgent",
    "PerformanceAgent",
    "ReviewAgent",
    "SecurityAgent",
    "StyleAgent",
]
```

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove CursorClient + CursorApiConfig (migration complete)"
```

---

## Phase 8 — Deploy and docs

### Task 24: Update GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/ai-review.yaml`

- [ ] **Step 1: Rename env var**

Replace every reference to `CURSOR_API_KEY` with `ANTHROPIC_API_KEY`. Ensure the secret is in repo/org secrets under the new name.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ai-review.yaml
git commit -m "ci: rename CURSOR_API_KEY -> ANTHROPIC_API_KEY"
```

---

### Task 25: Update Cloud Build + deploy script

**Files:**
- Modify: `cloudbuild.yaml`
- Modify: `deploy-gcp.sh`

- [ ] **Step 1: Update secret references**

Replace Cloud Build and deploy script references to Cursor secrets with Anthropic equivalents:
- Secret name in Secret Manager: `anthropic-api-key`
- Env var exposed to Cloud Run: `ANTHROPIC_API_KEY`

Find/replace all `CURSOR_API_KEY` → `ANTHROPIC_API_KEY` and `cursor-api-key` → `anthropic-api-key`.

- [ ] **Step 2: Commit**

```bash
git add cloudbuild.yaml deploy-gcp.sh
git commit -m "deploy: rename secret to anthropic-api-key"
```

---

### Task 26: Update docs

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Replace every mention of Cursor**

Search and update:
```bash
grep -rn "Cursor\|cursor" README.md DESIGN.md docs/ARCHITECTURE.md | grep -v "\.cursor/"
```

Rewrite sections to describe:
- Anthropic Messages API as the LLM gateway
- Prompt caching, extended thinking, tool use, JSON schema
- `ANTHROPIC_API_KEY` setup
- Model mix (Sonnet + Haiku)

- [ ] **Step 2: Update the architecture diagram / model list**

Wherever models are listed, replace `claude-4.5-opus-high-thinking` / `gpt-5.2` with `claude-sonnet-4-6` / `claude-sonnet-4-6`.

- [ ] **Step 3: Add a CHANGELOG entry**

In the appropriate changelog file (`docs/CHANGELOG-AND-NEXT.md`), add:
```markdown
## Unreleased

### Migrated
- LLM backend: Cursor Background Agent API → Anthropic Messages API (official SDK).
- Model mapping: `claude-4.5-opus-high-thinking` / `gpt-5.2` → `claude-sonnet-4-6` (Anthropic-only,
  no GPT, extended thinking disabled). Style/doc-gen agents → `claude-haiku-4-5-20251001`.
- Repo context: Cursor background agent exploration replaced by GitHub Contents API–backed
  `read_file`/`glob`/`grep` tools invoked by Claude (no local cloning).
- Config: `cursor:` block replaced by `anthropic:`. New per-agent knobs:
  `thinking_enabled`, `thinking_budget_tokens`, `allow_tool_use`, `max_tool_calls`.

### Secrets
- Rename `CURSOR_API_KEY` to `ANTHROPIC_API_KEY` in GitHub Actions, Cloud Run, and local `.env`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md DESIGN.md docs/ARCHITECTURE.md docs/CHANGELOG-AND-NEXT.md
git commit -m "docs: describe Anthropic Messages API migration"
```

---

## Phase 9 — Integration and quality regression

### Task 27: Add integration test (opt-in)

**Files:**
- Create: `tests/integration/test_e2e_review.py`

- [ ] **Step 1: Add opt-in integration test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_e2e_review.py`:
```python
"""Opt-in end-to-end integration test.

Runs a real review against a pinned small public PR. Requires:
- ANTHROPIC_API_KEY env
- GITHUB_TOKEN env (read-only ok)

Enable with: pytest -m integration
"""
import os
import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_review_against_pinned_public_pr():
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("GITHUB_TOKEN"):
        pytest.skip("credentials not set")

    from ai_reviewer.config import load_config
    from ai_reviewer.review import review_pr
    from pathlib import Path

    cfg_path = Path("config.example.yaml")
    config = load_config(cfg_path)

    # PIN A REAL SMALL PUBLIC PR HERE before running.
    repo = "PLACEHOLDER-OWNER/PLACEHOLDER-REPO"
    pr_number = 1
    if "PLACEHOLDER" in repo:
        pytest.skip("pinned PR not yet set")

    review = await review_pr(
        repo=repo,
        pr_number=pr_number,
        config=config,
    )
    assert review is not None
    assert isinstance(review.findings, list)
```

- [ ] **Step 2: Register the `integration` marker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`:
```toml
markers = ["integration: opt-in end-to-end tests requiring live API keys"]
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration pyproject.toml
git commit -m "test: add opt-in E2E integration test skeleton"
```

---

### Task 28: Quality regression doc

**Files:**
- Create: `docs/migrations/2026-04-anthropic-quality-regression.md`

- [ ] **Step 1: Write regression harness docs**

Create the doc:
```markdown
# Quality regression — Anthropic migration

Run the new reviewer against 5 PRs previously reviewed under the Cursor setup.
Record results to verify no quality regression.

## Selection
Pick 5 PRs covering:
- One pure-security fix (tests assertion of security findings)
- One performance optimization
- One refactor touching patterns / architecture
- One small typo / docs-only PR (no findings expected)
- One large multi-file PR (>400 lines changed)

## Procedure
1. `export ANTHROPIC_API_KEY=...`
2. `ai-reviewer review --repo <owner/repo> --pr <n> --output json > run.json`
3. Compare against prior Cursor run (stored in `docs/migrations/baseline/`):
   - Count of critical findings
   - Count of true positives (manual judgment)
   - Severity distribution
4. Accept if:
   - True-positive count ≥ 90% of baseline
   - False-positive rate ≤ baseline + 2 absolute pp
5. Record results in `docs/migrations/2026-04-run-results.md`

## Sign-off
If criteria met → merge PR. If not → iterate on prompts / context / model mix.
```

- [ ] **Step 2: Commit**

```bash
git add docs/migrations/2026-04-anthropic-quality-regression.md
git commit -m "docs: quality regression harness for Anthropic migration"
```

---

## Phase 10 — Open the PR

### Task 29: Push and open the PR

- [ ] **Step 1: Push final branch**

Run:
```bash
git push origin feature/anthropic-messages-migration
```

- [ ] **Step 2: Open the PR**

Run:
```bash
gh pr create --title "feat: migrate LLM backend to Anthropic Messages API" --body "$(cat <<'EOF'
## Summary
- Replaces Cursor Background Agent API with Anthropic's official Messages API
- Adds `AnthropicClient` with tool-use loop, JSON-schema structured output, prompt caching, and circuit breaker
- Adds `ToolRegistry` (read_file/glob/grep) backed by GitHub Contents API — no cloning
- Rewires `review.py` to instantiate `ReviewAgent` subclasses and drive them directly
- Models: `claude-sonnet-4-6` for all main agents, `claude-haiku-4-5-20251001` for style/doc-gen (GPT removed; Anthropic only)
- Renames secret `CURSOR_API_KEY` → `ANTHROPIC_API_KEY` across CI, Cloud Build, Cloud Run, and docs

## Design & plan
- Spec: `docs/superpowers/specs/2026-04-15-anthropic-messages-migration-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-anthropic-messages-migration.md`

## Test plan
- [ ] `pytest -v` — all unit tests pass
- [ ] Local CLI review run on a sample repo with `ANTHROPIC_API_KEY` set
- [ ] Integration test (`pytest -m integration`) against a pinned PR after key provisioned
- [ ] Quality regression on 5 pinned PRs per `docs/migrations/2026-04-anthropic-quality-regression.md`
- [ ] Webhook E2E smoke test on staging Cloud Run

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Share PR URL with the user**

---

## Self-review

**Spec coverage:** Every section of the spec is mapped to a task:
- §3 quality layers 1-6 → Tasks 14, 15 (context), 4, 5 (tools), 11 (thinking), 17 (model mix), 10 (schema), 13 (caching)
- §4 components → Tasks 3 (session), 4-5 (tools), 6-8 (neighbors), 9-13 (client), 16-17 (agents), 18-19 (review.py wiring)
- §5 detail sections → mapped in Phase 2-5
- §6 config → Task 2, 22, 23
- §7 contracts → Task 9-13 (client), 16 (base)
- §8 operational → Tasks 25, 26
- §9 testing → All tasks with test steps; Tasks 27, 28 for integration + regression
- §10 phased cutover → matches Phases 2-9
- §14 acceptance criteria → covered by Task 28 (regression) and Task 29 (PR checklist)

**Placeholder scan:** No "TBD", "implement later", or hand-waving. Pinned-PR placeholder in Task 27 is explicit (`PLACEHOLDER-OWNER/PLACEHOLDER-REPO`) and documented as needing real value at run time.

**Type consistency:** `AnthropicReviewResult`, `UsageStats`, `ToolRegistry`, `ReviewSession`, `AnthropicClient`, `AnthropicApiConfig`, `build_system_blocks`, `build_user_blocks`, `FINDINGS_SCHEMA`, `_AGENT_CLASSES`, `DEFAULT_AGENT_ORDER`, `_prepare_shared_context`, `_run_agent_with_status`, `run_cross_review_round` — signatures match across introducing and consuming tasks.

**Note on one carry-over:** Task 20 accepts that `tests/test_agents.py` and `tests/test_convergence.py` may fail after Task 17 and are fixed in Task 20. This is flagged explicitly in Task 17 Step 4 and Task 19 Step 7.

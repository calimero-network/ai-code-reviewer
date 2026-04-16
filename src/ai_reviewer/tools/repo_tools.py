"""Repo-exploration tools backed by GitHub Contents API.

Exposes read_file / glob / grep as Anthropic tools. All file access
is pinned to the PR head SHA and is read-only. Enforces per-agent
call budget and per-review GitHub request budget.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import PurePosixPath
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
            contents = self.gh.get_file_contents(self.session.repo, path, ref=self.session.head_sha)
        except Exception as e:  # noqa: BLE001
            logger.warning("read_file(%s) failed: %s", path, e)
            return f"[error: {e}]"
        raw = getattr(contents, "content", "") or ""
        try:
            text = base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
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
            tree = self.gh.get_tree(self.session.repo, self.session.head_sha, recursive=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("tree() failed: %s", e)
            return []
        paths = [item.path for item in tree.tree if getattr(item, "type", None) == "blob"]
        self.session.store_tree(paths)
        return paths

    def _glob(self, pattern: str) -> str:
        paths = self._tree()
        hits = [p for p in paths if PurePosixPath(p).match(pattern)]
        hits = hits[:500]
        return "\n".join(hits) if hits else "[no matches]"

    def _grep(self, pattern: str, path_glob: str) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[error: invalid regex: {e}]"
        matches: list[str] = []
        for path in self._tree():
            if not PurePosixPath(path).match(path_glob):
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

"""Per-review transient state: GitHub quota, file cache, tree listing."""

from dataclasses import dataclass, field


@dataclass
class ReviewSession:
    """Transient, per-review state shared across agents.

    Holds cached Contents API results, the recursive tree listing,
    and per-review GitHub request budget counters. Not a data model —
    a runtime object; do not persist.
    """

    repo: str
    head_sha: str
    github_budget: int
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

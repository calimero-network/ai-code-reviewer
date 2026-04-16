"""Fetch conventions + repo map via GitHubClient (with ReviewSession budget)."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterable

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
    repo = gh._gh.get_repo(session.repo)
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
        except Exception as e:  # noqa: BLE001
            logger.debug("Convention %s not found: %s", path, e)
            continue
        try:
            text = base64.b64decode(getattr(contents, "content", "")).decode(
                "utf-8", errors="replace"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Convention %s decode failed: %s", path, e)
            continue
        session.store_file(path, text)
        out[path] = text
    return out


def build_repo_map(session: ReviewSession, gh: GitHubClient) -> str:
    """Build a compact top-level directory listing for the system prompt."""
    cached_tree = session.cached_tree()
    if cached_tree is None:
        if session.is_github_budget_exhausted():
            return "(repo map unavailable: budget exhausted)"
        session.consume_github_request()
        try:
            repo = gh._gh.get_repo(session.repo)
            tree = repo.get_git_tree(session.head_sha, recursive=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Tree fetch failed: %s", e)
            return "(repo map unavailable)"
        paths_with_kind: list[tuple[str, str]] = []
        for item in tree.tree:
            kind = getattr(item, "type", None)
            if kind in {"blob", "tree"}:
                paths_with_kind.append((item.path, kind))
        session.store_tree([p for p, k in paths_with_kind if k == "blob"])
    else:
        paths_with_kind = [(p, "blob") for p in cached_tree]

    top_level: set[str] = set()
    for path, kind in paths_with_kind:
        head = path.split("/", 1)[0]
        if "/" in path or kind == "tree":
            top_level.add(head + "/")
        else:
            top_level.add(head)
    return "\n".join(sorted(top_level))

"""Preparing a pull request for review in a local checkout.

Kept out of ``local_source`` so the working-tree path stays free of any GitHub or
network concern: that module must keep working with no remote at all.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Where clones are fetched from. A module constant so tests can point it at a
# local path instead of the network.
_GITHUB_URL = "https://github.com"
# Fallback clones, and the record of clones found elsewhere on this machine.
CLONE_CACHE = Path.home() / ".cache" / "ai-reviewer"
CLONE_INDEX = CLONE_CACHE / "clones.json"

_PR_URL = re.compile(r"^https?://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)/?$")
_PR_SHORT = re.compile(r"^([\w.-]+/[\w.-]+)#(\d+)$")
_REMOTE_SLUG = re.compile(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?/?$")


def parse_pr_target(target: str) -> tuple[str, int]:
    """``("owner/repo", number)`` from a PR URL or ``owner/repo#N``."""
    for pattern in (_PR_URL, _PR_SHORT):
        match = pattern.match(target.strip())
        if match:
            return match.group(1), int(match.group(2))
    raise ValueError(
        f"not a pull request: {target!r} "
        "(expected https://github.com/owner/repo/pull/N or owner/repo#N)"
    )


def resolve_clone(slug: str, repo_path: str | None = None) -> Path:
    """A local clone of *slug*, never mutated - worktrees are taken from it.

    Ordered by how likely it is to be what the developer meant: the checkout they
    ran from, one they named before, then a cache clone.
    """
    if repo_path:
        named = Path(repo_path).expanduser().resolve()
        if _remote_slug(named) != slug:
            raise ValueError(f"{named} is not a clone of {slug}")
        _remember(slug, named)
        return named

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists() and _remote_slug(candidate) == slug:
            return candidate

    remembered = _recall(slug)
    if remembered is not None and _remote_slug(remembered) == slug:
        return remembered

    return _cache_clone(slug)


def _remote_slug(path: Path) -> str | None:
    """``owner/repo`` for a checkout's origin, or None when it has no usable one."""
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    match = _REMOTE_SLUG.search(proc.stdout.strip())
    return match.group(1) if match else None


def _cache_clone(slug: str) -> Path:
    """Clone *slug* under the cache, blobless and without a checkout.

    It is only ever an object store for worktrees, so the blobs and the working
    copy are both wasted work on a repository of any size.
    """
    target = CLONE_CACHE / slug
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s into %s", slug, target)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            f"{_GITHUB_URL}/{slug}.git",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


def _load_index() -> dict[str, str]:
    try:
        loaded = json.loads(CLONE_INDEX.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _remember(slug: str, path: Path) -> None:
    CLONE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index = _load_index()
    index[slug] = str(path)
    CLONE_INDEX.write_text(json.dumps(index, indent=2))


def _recall(slug: str) -> Path | None:
    recorded = _load_index().get(slug)
    return Path(recorded) if recorded else None

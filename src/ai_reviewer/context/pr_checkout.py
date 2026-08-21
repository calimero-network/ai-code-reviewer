"""Preparing a pull request for review in a local checkout.

Kept out of ``local_source`` so the working-tree path stays free of any GitHub or
network concern: that module must keep working with no remote at all.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Where clones are fetched from. A module constant so tests can point it at a
# local path instead of the network.
_GITHUB_URL = "https://github.com"
# Fallback clones, when the developer is not standing in one.
CLONE_CACHE = Path.home() / ".cache" / "ai-reviewer"

_PR_URL = re.compile(r"^https?://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)/?$")
_PR_SHORT = re.compile(r"^([\w.-]+/[\w.-]+)#(\d+)$")
_REMOTE_SLUG = re.compile(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?/?$")


def parse_pr_target(target: str) -> tuple[str, int]:
    """``("owner/repo", number)`` from a PR URL or ``owner/repo#N``."""
    for pattern in (_PR_URL, _PR_SHORT):
        match = pattern.match(target.strip())
        # The slug is used as a path under the clone cache, so a relative segment
        # would aim the clone and the index entry outside it.
        if match and not any(part in (".", "..") for part in match.group(1).split("/")):
            return match.group(1), int(match.group(2))
    raise ValueError(
        f"not a pull request: {target!r} "
        "(expected https://github.com/owner/repo/pull/N or owner/repo#N)"
    )


def resolve_clone(slug: str, repo_path: str | None = None) -> Path:
    """A local clone of *slug*, never mutated - worktrees are taken from it.

    Ordered by how likely it is to be what the developer meant: one they named,
    the checkout they ran from, then a cache clone.
    """
    if repo_path:
        named = Path(repo_path).expanduser().resolve()
        if _remote_slug(named) != slug:
            raise ValueError(f"{named} is not a clone of {slug}")
        return named

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists() and _remote_slug(candidate) == slug:
            return candidate

    return _cache_clone(slug)


def _remote_slug(path: Path) -> str | None:
    """``owner/repo`` for a checkout's origin, or None when it has no usable one.

    A missing directory counts as unusable: repo_path typos and stale index
    entries must fall through cleanly instead of crashing on a bad cwd.
    """
    if not path.is_dir():
        return None
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
    _git(
        target.parent,
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        f"{_GITHUB_URL}/{slug}.git",
        str(target),
    )
    return target


@dataclass
class PreparedPR:
    """What is under review, recorded so the second phase cannot disagree with the first.

    Written by ``prompts --pr`` and read by ``publish``: the scope is recorded
    rather than re-typed, so the two phases cannot measure different diffs.
    """

    repo: str
    number: int
    clone: str
    root: str
    base_sha: str
    head_sha: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read(cls, path: Path) -> PreparedPR:
        return cls(**json.loads(path.read_text()))


def create_pr_worktree(
    clone: Path,
    slug: str,
    number: int,
    base_ref: str,
    root: Path,
) -> PreparedPR:
    """Check the pull request out at *root* as a detached worktree of *clone*.

    The base branch is fetched alongside the PR head because ``--base <sha>``
    diffs ``<sha>...HEAD``, which is the pull request's range only when that
    commit is present locally.

    Fetching by remote name rather than a constructed URL uses whatever protocol
    and credentials the clone is already configured with; ``resolve_clone`` has
    verified that remote points at *slug*.
    """
    head_ref = f"refs/ai-reviewer/{number}/head"
    base_local = f"refs/ai-reviewer/{number}/base"
    # A session that never reached publish leaves an administrative entry behind.
    _git(clone, "worktree", "prune")
    _git(
        clone,
        "fetch",
        "--no-tags",
        "origin",
        f"+refs/pull/{number}/head:{head_ref}",
        f"+refs/heads/{base_ref}:{base_local}",
    )
    head_sha = _git(clone, "rev-parse", head_ref).strip()
    base_sha = _git(clone, "rev-parse", base_local).strip()
    _git(clone, "worktree", "add", "--detach", str(root), head_sha)
    return PreparedPR(
        repo=slug,
        number=number,
        clone=str(clone),
        root=str(root),
        base_sha=base_sha,
        head_sha=head_sha,
    )


def remove_pr_worktree(prepared: PreparedPR) -> None:
    """Remove the worktree and the refs it needed. Safe to call more than once."""
    clone = Path(prepared.clone)
    _git_quiet(clone, "worktree", "remove", "--force", prepared.root)
    shutil.rmtree(prepared.root, ignore_errors=True)
    for ref in (
        f"refs/ai-reviewer/{prepared.number}/head",
        f"refs/ai-reviewer/{prepared.number}/base",
    ):
        _git_quiet(clone, "update-ref", "-d", ref)
    _git_quiet(clone, "worktree", "prune")


def _git(repo: Path, *args: str) -> str:
    """Output is captured, so an exit status on its own would be all the caller sees."""
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit status {proc.returncode}"
        raise RuntimeError(f"git {' '.join(args)}: {detail}")
    return proc.stdout


def _git_quiet(repo: Path, *args: str) -> None:
    """For cleanup, where the thing being removed may already be gone."""
    subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)

"""Working-tree review inputs: a GitHubClient stand-in backed by the checkout.

``_prepare_shared_context`` reaches GitHub through exactly two methods -
``get_file_contents`` and ``get_tree`` - so matching those is enough to reuse the
whole context builder against a local repository. Content is returned
base64-encoded because every caller decodes it that way.
"""

from __future__ import annotations

import base64
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from ai_reviewer.models.context import ReviewContext

logger = logging.getLogger(__name__)

# Language detection for the context's repo_languages, by extension.
_LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".rs": "Rust",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".java": "Java",
}


@dataclass
class _Head:
    sha: str


@dataclass
class LocalPR:
    """Stands in for a PullRequest so shared review code can stay unchanged."""

    title: str
    body: str
    head: _Head


@dataclass(frozen=True)
class PRMeta:
    """The pull request a local review stands in for, when there is one."""

    repo: str
    number: int
    title: str
    body: str


@dataclass
class _Contents:
    content: str


@dataclass
class _TreeItem:
    path: str
    type: str


@dataclass
class _Tree:
    tree: list[_TreeItem]


class LocalGitSource:
    """Serves repository reads from the working tree instead of the GitHub API."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    def get_file_contents(self, repo: str, path: str, ref: str | None = None) -> _Contents:
        """Read *path* from the working tree, confined to the repository.

        *ref* is accepted for the RepoSource protocol and ignored: this source
        always reflects the current checkout. Callers needing a specific revision
        use ``changed_files``, which reads the diffed snapshot via ``git show``.

        Reads must be resolved and checked against the root: ``Path(root) / path``
        silently discards the root when *path* is absolute, and this source backs
        the LLM-facing read_file tool with real filesystem access. An escape is
        reported as a missing file so callers that already skip absent convention
        files need no new handling.
        """
        text = read_repo_file(str(self._root), path)
        if text is None:
            raise FileNotFoundError(path)
        return _Contents(content=base64.b64encode(text.encode()).decode())

    def get_tree(self, repo: str, sha: str | None = None, recursive: bool = True) -> _Tree:
        """List tracked files at the current checkout; *sha* is accepted for the
        RepoSource protocol and deliberately ignored, as with ``ref`` above.

        Untracked files are excluded on purpose - the repo map should describe the
        project, not build output or scratch files.
        """
        out = _run_git(self._root, "ls-files")
        return _Tree(tree=[_TreeItem(path=p, type="blob") for p in out.splitlines() if p])


def read_repo_file(root: str, path: str) -> str | None:
    """Read *path* confined to *root*; None when outside the repo or absent.

    ``Path(root) / path`` silently discards the root when *path* is absolute, and
    ``..`` segments walk out of it on disk. Both are reachable from untrusted
    input - the LLM-facing read_file tool, and ``file_path`` in agent-produced
    findings JSON - so confinement lives here, once, rather than at each caller.
    """
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        return None
    return target.read_text(errors="replace")


def local_diff(root: str, staged: bool = False, base: str | None = None) -> str:
    """Unified diff for the working tree, the index, or base...HEAD.

    Working-tree mode also covers untracked files. ``git diff`` omits them, which
    would make new files - where bugs are most likely - invisible to the review.
    """
    if base:
        return _run_git(Path(root), "diff", f"{base}...HEAD")
    if staged:
        return _run_git(Path(root), "diff", "--cached")

    sections = [_run_git(Path(root), "diff")]
    sections.extend(_new_file_diff(Path(root), p) for p in _untracked(Path(root)))
    return "".join(s for s in sections if s)


def changed_files(root: str, staged: bool = False, base: str | None = None) -> dict[str, str]:
    """Contents of every path the same diff touches, from the snapshot diffed.

    Paths come from git rather than from parsing the diff text, so renames and
    binary files follow git's own rules. Content must come from the revision that
    was diffed, not always from disk: ``git diff --cached`` reviews the index, so a
    file with staged edits plus further unstaged edits would otherwise be reviewed
    against a version the diff never showed - and a suggested replacement would be
    validated against the wrong snapshot.
    """
    args = ["diff", "--name-only"]
    rev: str | None = None
    if base:
        args.append(f"{base}...HEAD")
        rev = "HEAD"
    elif staged:
        args.append("--cached")
        rev = ""  # "git show :path" reads the index

    paths = _run_git(Path(root), *args).splitlines()
    if rev is None:
        paths += _untracked(Path(root))

    files: dict[str, str] = {}
    for path in paths:
        if not path:
            continue
        content = _read_at_revision(Path(root), path, rev)
        if content is not None:
            files[path] = content
    return files


def _read_at_revision(root: Path, path: str, rev: str | None) -> str | None:
    """Read *path* from *rev* (None = working tree). None when absent or deleted."""
    if rev is None:
        target = root / path
        return target.read_text(errors="replace") if target.is_file() else None
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,  # a deleted path has no blob at that revision
    )
    return proc.stdout if proc.returncode == 0 else None


def _untracked(root: Path) -> list[str]:
    """Untracked files, honouring .gitignore via --exclude-standard."""
    out = _run_git(root, "ls-files", "--others", "--exclude-standard")
    return [p for p in out.splitlines() if p]


def _new_file_diff(root: Path, path: str) -> str:
    """A new-file diff for an untracked path, without touching the index."""
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--", "/dev/null", path],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,  # --no-index exits 1 whenever the files differ
    )
    return proc.stdout


def build_local_context(
    root: str, diff: str, files: dict[str, str], *, pr: PRMeta | None = None
) -> ReviewContext:
    """A ReviewContext for a diff, with or without a pull request behind it.

    ``repo_name`` comes from *pr* when there is one: a pull request is reviewed in
    a worktree named for the session directory, not for the repository.
    """
    additions = sum(
        1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    languages = sorted(
        {lang for p in files if (lang := _LANGUAGE_BY_SUFFIX.get(Path(p).suffix.lower()))}
    )
    meta = pr or PRMeta(
        repo=Path(root).name,
        number=0,
        title="Local changes",
        body="Uncommitted work reviewed before a pull request exists.",
    )
    return ReviewContext(
        repo_name=meta.repo.split("/")[-1],
        pr_number=meta.number,
        pr_title=meta.title,
        pr_description=meta.body,
        base_branch=_current_branch(Path(root)),
        head_branch=_current_branch(Path(root)),
        author="local",
        changed_files_count=len(files),
        additions=additions,
        deletions=deletions,
        repo_languages=languages,
    )


def scope_label(staged: bool = False, base: str | None = None) -> str:
    """What a given scope is called, wherever it is shown to a reader."""
    return f"{base}...HEAD" if base else "the index" if staged else "the working tree"


def load_local_repo_config(root: str) -> dict | None:
    """Best-effort ``.ai-reviewer.yaml`` from the checkout, mirroring the API load."""
    config_file = Path(root) / ".ai-reviewer.yaml"
    if not config_file.is_file():
        return None
    try:
        parsed = yaml.safe_load(config_file.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read .ai-reviewer.yaml: %s", e)
        return None
    return parsed if isinstance(parsed, dict) else None


def build_local_pr(
    root: str, staged: bool = False, base: str | None = None, *, pr: PRMeta | None = None
) -> LocalPR:
    """A PullRequest stand-in describing what is being reviewed."""
    return LocalPR(
        title=pr.title if pr else f"Local review of {scope_label(staged, base)}",
        body=pr.body if pr else "",
        head=_Head(sha=_head_sha(Path(root))),
    )


def _head_sha(root: Path) -> str:
    try:
        return _run_git(root, "rev-parse", "HEAD").strip() or "HEAD"
    except subprocess.CalledProcessError:
        return "HEAD"


def _current_branch(root: Path) -> str:
    try:
        return _run_git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() or "HEAD"
    except subprocess.CalledProcessError:
        return "HEAD"


def _run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout

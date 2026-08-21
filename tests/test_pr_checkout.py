"""Preparing a pull request for local review.

Real git repositories rather than mocks: this module exists to agree with git,
so a mocked git would test nothing.  The clone source is a local path, so no
test touches the network.
"""

from __future__ import annotations

import subprocess

import pytest

from ai_reviewer.context import pr_checkout
from ai_reviewer.context.pr_checkout import (
    PreparedPR,
    create_pr_worktree,
    parse_pr_target,
    remove_pr_worktree,
    resolve_clone,
)


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def origin(tmp_path):
    """A repository standing in for github.com/acme/widget."""
    path = tmp_path / "origin" / "acme" / "widget.git"
    path.parent.mkdir(parents=True)
    work = tmp_path / "origin-work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@e.st")
    _git(work, "config", "user.name", "t")
    (work / "a.py").write_text("x = 1\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "init")
    _git(work, "clone", "-q", "--bare", str(work), str(path))
    return path


@pytest.fixture
def clone(tmp_path, origin):
    """A developer's clone of that repository."""
    path = tmp_path / "dev" / "widget"
    path.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(path)], check=True, capture_output=True)
    # origin stays the local bare repo: laid out as <root>/<owner>/<repo>.git it
    # reads as acme/widget, so no test needs the network.
    return path


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never read or write the real ~/.cache/ai-reviewer during tests."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(pr_checkout, "CLONE_CACHE", cache)
    return cache


@pytest.mark.parametrize(
    "target",
    [
        "https://github.com/acme/widget/pull/42",
        "https://github.com/acme/widget/pull/42/",
        "http://github.com/acme/widget/pull/42",
        "acme/widget#42",
    ],
)
def test_parse_pr_target_accepts_url_and_short_forms(target):
    assert parse_pr_target(target) == ("acme/widget", 42)


@pytest.mark.parametrize(
    "target",
    [
        "https://github.com/acme/widget",
        "https://github.com/acme/widget/issues/42",
        "acme/widget",
        "42",
        "",
    ],
)
def test_parse_pr_target_rejects_anything_else(target):
    with pytest.raises(ValueError, match="not a pull request"):
        parse_pr_target(target)


@pytest.mark.parametrize(
    "target",
    [
        "../b#1",
        "a/..#1",
        "./b#1",
        "https://github.com/../evil/pull/1",
        "https://github.com/acme/../../evil/pull/1",
    ],
)
def test_parse_pr_target_rejects_relative_segments(target):
    """The slug is used as a path under the clone cache, so a traversing one would
    aim the clone and the index entry outside it."""
    with pytest.raises(ValueError, match="not a pull request"):
        parse_pr_target(target)


def test_resolve_clone_finds_the_clone_you_are_standing_in(clone, monkeypatch):
    monkeypatch.chdir(clone)

    assert resolve_clone("acme/widget") == clone.resolve()


def test_resolve_clone_walks_up_from_a_subdirectory(clone, monkeypatch):
    nested = clone / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert resolve_clone("acme/widget") == clone.resolve()


def test_resolve_clone_ignores_a_clone_of_a_different_repo(
    clone,  # noqa: ARG001 - present so its origin bare repo exists on disk
    tmp_path,
    monkeypatch,
):
    """Standing in some other project must not review its files."""
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "remote", "add", "origin", "https://github.com/acme/other.git")
    monkeypatch.chdir(other)
    monkeypatch.setattr(pr_checkout, "_GITHUB_URL", str(tmp_path / "origin"))

    resolved = resolve_clone("acme/widget")

    assert resolved != other.resolve()
    assert resolved.is_relative_to(tmp_path / "cache")


def test_repo_path_pointing_at_the_wrong_repo_is_rejected(tmp_path, monkeypatch):
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    _git(wrong, "init", "-q")
    _git(wrong, "remote", "add", "origin", "https://github.com/acme/other.git")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not a clone of acme/widget"):
        resolve_clone("acme/widget", repo_path=str(wrong))


def test_repo_path_that_does_not_exist_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not a clone of acme/widget"):
        resolve_clone("acme/widget", repo_path=str(tmp_path / "no" / "such" / "path"))


def test_falls_back_to_a_cache_clone(
    tmp_path,
    origin,  # noqa: ARG001 - present so the clone source exists on disk
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pr_checkout, "_GITHUB_URL", str(tmp_path / "origin"))

    resolved = resolve_clone("acme/widget")

    assert resolved == tmp_path / "cache" / "acme" / "widget"
    assert (resolved / ".git").exists()


def test_a_second_call_reuses_the_cache_clone(
    tmp_path,
    origin,  # noqa: ARG001 - present so the clone source exists on disk
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pr_checkout, "_GITHUB_URL", str(tmp_path / "origin"))
    first = resolve_clone("acme/widget")
    marker = first / ".git" / "ai-reviewer-marker"
    marker.write_text("kept")

    assert resolve_clone("acme/widget") == first
    assert marker.read_text() == "kept"


def test_remote_slug_returns_none_without_an_origin_remote(tmp_path):
    repo = tmp_path / "no-origin"
    repo.mkdir()
    _git(repo, "init", "-q")

    assert pr_checkout._remote_slug(repo) is None


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/acme/widget.git", "acme/widget"),
        ("https://github.com/acme/widget", "acme/widget"),
        ("git@github.com:acme/widget.git", "acme/widget"),
        ("ssh://git@github.com/acme/widget.git", "acme/widget"),
    ],
)
def test_remote_slug_reads_every_url_form(tmp_path, url, expected):
    repo = tmp_path / f"r{abs(hash(url))}"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "remote", "add", "origin", url)

    assert pr_checkout._remote_slug(repo) == expected


def _open_pr(origin_bare, tmp_path, number: int = 42) -> None:
    """Push a branch and publish it as refs/pull/<n>/head, as GitHub does."""
    work = tmp_path / f"contrib{number}"
    subprocess.run(
        ["git", "clone", "-q", str(origin_bare), str(work)], check=True, capture_output=True
    )
    _git(work, "config", "user.email", "c@e.st")
    _git(work, "config", "user.name", "c")
    (work / "a.py").write_text("x = 1\ny = 2\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "add y")
    _git(work, "push", "-q", "origin", f"HEAD:refs/pull/{number}/head")


def test_worktree_is_detached_at_the_pr_head(clone, origin, tmp_path):
    _open_pr(origin, tmp_path)
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()

    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", root)

    assert (root / "a.py").read_text() == "x = 1\ny = 2\n"
    assert prepared.head_sha != prepared.base_sha
    remove_pr_worktree(prepared)


def test_the_developer_head_is_not_moved(clone, origin, tmp_path):
    _open_pr(origin, tmp_path)
    before = _git(clone, "rev-parse", "HEAD").strip()
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()

    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", root)

    assert _git(clone, "rev-parse", "HEAD").strip() == before
    assert _git(clone, "status", "--porcelain") == ""
    remove_pr_worktree(prepared)


def test_the_base_commit_is_fetched_so_the_diff_range_resolves(clone, origin, tmp_path):
    """``--base <sha>`` diffs ``<sha>...HEAD``; without the base commit locally
    that range does not exist."""
    from ai_reviewer.context.local_source import local_diff

    _open_pr(origin, tmp_path)
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()

    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", root)

    diff = local_diff(str(root), base=prepared.base_sha)
    assert "+y = 2" in diff
    remove_pr_worktree(prepared)


def test_remove_cleans_the_worktree_and_the_temporary_refs(clone, origin, tmp_path):
    _open_pr(origin, tmp_path)
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()
    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", root)

    remove_pr_worktree(prepared)

    assert not root.exists()
    # The worktree path is a pytest tmpdir, so it must be matched by its own path;
    # the refs carry the module's namespace and are matched by that.
    assert str(root) not in _git(clone, "worktree", "list")
    assert "refs/ai-reviewer/" not in _git(clone, "for-each-ref", "--format=%(refname)")


def test_remove_is_safe_to_call_twice(clone, origin, tmp_path):
    """publish removes in a finally, and a failed run may already have removed it."""
    _open_pr(origin, tmp_path)
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()
    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", root)
    remove_pr_worktree(prepared)

    remove_pr_worktree(prepared)


def test_a_leaked_worktree_does_not_block_the_next_run(clone, origin, tmp_path):
    """A session that never reached publish leaves a stale administrative entry;
    the next run must clear it rather than pile up a fresh one alongside it."""
    import shutil

    _open_pr(origin, tmp_path)
    first = tmp_path / "s1" / "wt"
    first.parent.mkdir()
    create_pr_worktree(clone, "acme/widget", 42, "main", first)
    shutil.rmtree(first.parent)

    second = tmp_path / "s2" / "wt"
    second.parent.mkdir()
    prepared = create_pr_worktree(clone, "acme/widget", 42, "main", second)

    assert (second / "a.py").exists()
    # git's admin dir for linked worktrees: one leaked entry plus one fresh one
    # would leave two, if the leading `worktree prune` were ever dropped.
    assert len(list((clone / ".git" / "worktrees").iterdir())) == 1
    remove_pr_worktree(prepared)


def test_prepared_pr_round_trips_through_json(tmp_path):
    prepared = PreparedPR(
        repo="acme/widget",
        number=42,
        clone="/clones/widget",
        root="/tmp/s/wt",
        base_sha="b" * 40,
        head_sha="h" * 40,
    )
    path = tmp_path / "target.json"

    prepared.write(path)

    assert PreparedPR.read(path) == prepared


def test_an_unfetchable_pr_ref_says_what_git_said(clone, tmp_path):
    """An absent ref, being offline, or a clone whose origin needs credentials all
    land here, and "exit status 128" on its own names none of them."""
    root = tmp_path / "session" / "wt"
    root.parent.mkdir()

    # "fatal:" comes from git; the ref alone would also match the arguments we
    # passed it, which is not evidence that anything was surfaced.
    with pytest.raises(RuntimeError, match=r"fatal: .*refs/pull/999/head"):
        create_pr_worktree(clone, "acme/widget", 999, "main", root)


def test_a_clone_that_cannot_be_made_says_what_git_said(tmp_path, monkeypatch):
    """A private repository with no credentials, or being offline, lands here."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pr_checkout, "_GITHUB_URL", str(tmp_path / "nothing-here"))

    with pytest.raises(RuntimeError, match="fatal:"):
        resolve_clone("acme/widget")

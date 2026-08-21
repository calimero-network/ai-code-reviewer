"""Working-tree review inputs: a GitHubClient stand-in backed by the checkout.

Real git repositories are used rather than mocks - the whole point of this module
is that it agrees with git, so a mocked git would test nothing.
"""

from __future__ import annotations

import base64
import subprocess

import pytest

from ai_reviewer.context.local_source import (
    LocalGitSource,
    build_local_context,
    changed_files,
    local_diff,
)


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.st")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_get_file_contents_returns_base64_like_the_github_api(repo):
    source = LocalGitSource(str(repo))

    contents = source.get_file_contents("o/r", "a.py", ref="HEAD")

    assert base64.b64decode(contents.content).decode() == "x = 1\n"


def test_get_file_contents_raises_for_a_missing_path(repo):
    """fetch_conventions probes for files that usually do not exist and relies on
    an exception to skip them."""
    source = LocalGitSource(str(repo))

    with pytest.raises(FileNotFoundError):
        source.get_file_contents("o/r", "CLAUDE.md", ref="HEAD")


def test_get_tree_lists_tracked_files_as_blobs(repo):
    source = LocalGitSource(str(repo))

    paths = {item.path for item in source.get_tree("o/r", "HEAD", recursive=True).tree}

    assert paths == {"a.py", "pkg/b.py"}


def test_local_diff_reports_unstaged_edits(repo):
    (repo / "a.py").write_text("x = 2\n")

    diff = local_diff(str(repo))

    assert "a.py" in diff
    assert "+x = 2" in diff


def test_local_diff_staged_reports_only_staged_edits(repo):
    (repo / "a.py").write_text("x = 2\n")
    _git(repo, "add", "a.py")
    (repo / "pkg" / "b.py").write_text("y = 99\n")

    diff = local_diff(str(repo), staged=True)

    assert "+x = 2" in diff
    assert "y = 99" not in diff


def test_changed_files_reads_current_content_for_each_changed_path(repo):
    (repo / "a.py").write_text("x = 2\n")

    files = changed_files(str(repo))

    assert files == {"a.py": "x = 2\n"}


def test_local_context_has_no_pr_number_and_counts_real_line_changes(repo):
    (repo / "a.py").write_text("x = 2\n")
    diff = local_diff(str(repo))

    context = build_local_context(str(repo), diff, changed_files(str(repo)))

    assert context.pr_number == 0
    assert context.changed_files_count == 1
    assert context.additions == 1
    assert context.deletions == 1


def test_untracked_files_are_reviewed_too(repo):
    """A pre-PR review whose blind spot is new files would miss the riskiest code."""
    (repo / "brand_new.py").write_text("def added():\n    return 1\n")

    diff = local_diff(str(repo))
    files = changed_files(str(repo))

    assert "brand_new.py" in diff
    assert "+def added():" in diff
    assert files["brand_new.py"] == "def added():\n    return 1\n"


def test_gitignored_files_are_not_reviewed(repo):
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "secret.txt").write_text("nope\n")

    # .gitignore is itself untracked, so its *content* legitimately appears in the
    # diff; assert on the file being reviewed, not on the word appearing anywhere.
    assert "+++ b/secret.txt" not in local_diff(str(repo))
    assert "secret.txt" not in changed_files(str(repo))


def test_untracked_files_are_excluded_from_staged_and_range_modes(repo):
    """--staged means the index, and --base means committed history; an untracked
    file belongs to neither."""
    (repo / "brand_new.py").write_text("x = 1\n")

    assert "brand_new.py" not in local_diff(str(repo), staged=True)
    assert "brand_new.py" not in changed_files(str(repo), staged=True)


def test_absolute_paths_cannot_escape_the_repository(repo):
    """Path(root) / "/etc/passwd" discards the root entirely, so an absolute path
    would read anything on the machine. This source backs the LLM-facing read_file
    tool with real filesystem access, so containment has to be here."""
    outside = repo.parent / "outside_secret.txt"
    outside.write_text("ssh-key-material\n")
    source = LocalGitSource(str(repo))

    with pytest.raises(FileNotFoundError):
        source.get_file_contents("o/r", str(outside))


def test_parent_traversal_cannot_escape_the_repository(repo):
    (repo.parent / "outside_secret.txt").write_text("ssh-key-material\n")
    source = LocalGitSource(str(repo))

    with pytest.raises(FileNotFoundError):
        source.get_file_contents("o/r", "../outside_secret.txt")


def test_paths_inside_the_repository_still_resolve(repo):
    source = LocalGitSource(str(repo))

    contents = source.get_file_contents("o/r", "pkg/b.py")

    assert base64.b64decode(contents.content).decode() == "y = 2\n"


def test_staged_mode_reads_the_index_not_the_working_tree(repo):
    """git diff --cached reviews the staged snapshot, so the file content handed to
    agents (and used for fix validation) must be the indexed blob. Reading disk
    instead lets a partially-staged file be reviewed against the wrong version."""
    (repo / "a.py").write_text("staged = True\n")
    _git(repo, "add", "a.py")
    (repo / "a.py").write_text("unstaged = True\n")

    files = changed_files(str(repo), staged=True)

    assert files["a.py"] == "staged = True\n"


def test_base_mode_reads_head_not_the_working_tree(repo):
    """base...HEAD reviews committed history, so content must come from HEAD."""
    (repo / "a.py").write_text("committed = True\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "second")
    (repo / "a.py").write_text("dirty = True\n")

    files = changed_files(str(repo), base="HEAD~1")

    assert files["a.py"] == "committed = True\n"


def test_read_repo_file_is_the_single_guarded_reader(repo):
    """One confined reader, used by both LocalGitSource and the consolidate
    command's fix-validation callback - so the guard cannot be fixed in one place
    and missed in the other."""
    from ai_reviewer.context.local_source import read_repo_file

    outside = repo.parent / "outside_secret.txt"
    outside.write_text("ssh-key-material\n")

    assert read_repo_file(str(repo), "pkg/b.py") == "y = 2\n"
    assert read_repo_file(str(repo), str(outside)) is None
    assert read_repo_file(str(repo), "../outside_secret.txt") is None
    assert read_repo_file(str(repo), "does_not_exist.py") is None


def test_build_local_context_defaults_to_the_working_tree_wording(repo):
    from ai_reviewer.context.local_source import build_local_context

    context = build_local_context(str(repo), "", {})

    assert context.pr_title == "Local changes"
    assert context.pr_number == 0
    assert context.repo_name == repo.name


def test_build_local_context_uses_the_pull_request_when_given_one(repo):
    """A worktree is named for the session directory, not the repository, so the
    repo name has to come from the pull request too."""
    from ai_reviewer.context.local_source import PRMeta, build_local_context

    meta = PRMeta(repo="acme/widget", number=42, title="fix: the thing", body="Because.")

    context = build_local_context(str(repo), "", {}, pr=meta)

    assert context.repo_name == "widget"
    assert context.pr_number == 42
    assert context.pr_title == "fix: the thing"
    assert context.pr_description == "Because."


def test_build_local_pr_carries_the_title_and_body(repo):
    from ai_reviewer.context.local_source import PRMeta, build_local_pr

    meta = PRMeta(repo="acme/widget", number=42, title="fix: the thing", body="Because.")

    pr = build_local_pr(str(repo), pr=meta)

    assert pr.title == "fix: the thing"
    assert pr.body == "Because."


def test_build_local_pr_without_a_pull_request_is_unchanged(repo):
    from ai_reviewer.context.local_source import build_local_pr

    pr = build_local_pr(str(repo))

    assert pr.title == "Local review of the working tree"
    assert pr.body == ""

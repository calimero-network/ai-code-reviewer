"""The consolidated local report: severity-grouped, with fix tiers.

Subagents in the coding session emit raw findings as JSON; consolidation and
formatting stay in Python so clustering, consensus and confidence floors are real
code rather than instructions.
"""

from __future__ import annotations

import json

from ai_reviewer.github.formatter import format_local_report
from ai_reviewer.review import consolidate_agent_findings


def _raw(**overrides) -> dict:
    base = {
        "file_path": "src/client.py",
        "line_start": 412,
        "line_end": 412,
        "severity": "critical",
        "category": "logic",
        "title": "Missing timeout on the retry path",
        "description": "A retry without a timeout can hang forever.",
        "suggested_fix": "Pass timeout=30.",
        "confidence": 0.92,
    }
    base.update(overrides)
    return base


def _write(tmp_path, name: str, findings: list[dict], summary: str = "ok"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"findings": findings, "summary": summary}))
    return path


def test_agreement_across_agents_becomes_a_consensus_score(tmp_path):
    """Two agents reporting the same issue must collapse into one finding."""
    a = _write(tmp_path, "security-reviewer", [_raw()])
    b = _write(tmp_path, "logic-reviewer", [_raw()])

    review = consolidate_agent_findings([a, b], repo="demo")

    assert len(review.findings) == 1
    assert review.findings[0].consensus_score == 1.0
    assert sorted(review.findings[0].agreeing_agents) == ["logic-reviewer", "security-reviewer"]


def test_agent_name_comes_from_the_filename(tmp_path):
    a = _write(tmp_path, "patterns-reviewer", [_raw()])

    review = consolidate_agent_findings([a], repo="demo")

    assert review.findings[0].agreeing_agents == ["patterns-reviewer"]


def test_report_groups_by_severity_with_counts(tmp_path):
    a = _write(
        tmp_path,
        "security-reviewer",
        [
            _raw(),
            _raw(file_path="src/review.py", line_start=1590, severity="warning", title="Ordering"),
        ],
    )
    review = consolidate_agent_findings([a], repo="demo")

    report = format_local_report(review, scope="working tree")

    assert "CRITICAL (1)" in report
    assert "WARNING (1)" in report
    assert "src/client.py:412" in report


def test_report_marks_a_validated_fix_as_ready(tmp_path):
    a = _write(tmp_path, "security-reviewer", [_raw()])
    review = consolidate_agent_findings([a], repo="demo")
    review.findings[0].fix_validated = True

    assert "fix ready (validated)" in format_local_report(review, scope="working tree")


def test_report_marks_an_unvalidated_fix_as_prose(tmp_path):
    a = _write(tmp_path, "security-reviewer", [_raw()])
    review = consolidate_agent_findings([a], repo="demo")

    assert "prose fix only" in format_local_report(review, scope="working tree")


def test_low_severity_is_collapsed_until_asked_for(tmp_path):
    a = _write(
        tmp_path,
        "security-reviewer",
        [_raw(severity="suggestion", title="Consider renaming", confidence=0.8)],
    )
    review = consolidate_agent_findings([a], repo="demo")

    collapsed = format_local_report(review, scope="working tree")
    expanded = format_local_report(review, scope="working tree", show_all=True)

    assert "--all" in collapsed
    assert "Consider renaming" not in collapsed
    assert "Consider renaming" in expanded


def test_a_clean_review_says_so(tmp_path):
    a = _write(tmp_path, "security-reviewer", [], summary="nothing found")
    review = consolidate_agent_findings([a], repo="demo")

    assert "No findings" in format_local_report(review, scope="working tree")


class TestPromptsCommand:
    """`ai-reviewer prompts` emits one self-contained prompt per reviewer profile,
    so the coding session can spawn subagents without reimplementing prompt logic."""

    def _repo(self, tmp_path):
        import subprocess

        def git(*a):
            subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, check=True)

        git("init", "-q")
        git("config", "user.email", "t@e.st")
        git("config", "user.name", "t")
        (tmp_path / "a.py").write_text("x = 1\n")
        git("add", "-A")
        git("commit", "-qm", "init")
        # Large enough that the size-based scaling keeps all requested agents.
        (tmp_path / "a.py").write_text("\n".join(f"x{i} = {i}" for i in range(600)) + "\n")
        return tmp_path

    def _invoke(self, tmp_path, *args):
        import os

        from click.testing import CliRunner

        import ai_reviewer.cli as cli

        out = tmp_path / "prompts"
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = CliRunner().invoke(cli.cli, ["prompts", "--out", str(out), *args])
        finally:
            os.chdir(cwd)
        return result, out

    def test_a_trivial_diff_scales_down_to_one_reviewer(self, tmp_path):
        """Matches the API path: three reviewers on a two-line change is waste."""
        import subprocess

        def git(*a):
            subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, check=True)

        git("init", "-q")
        git("config", "user.email", "t@e.st")
        git("config", "user.name", "t")
        (tmp_path / "a.py").write_text("x = 1\n")
        git("add", "-A")
        git("commit", "-qm", "init")
        (tmp_path / "a.py").write_text("x = 2\n")

        _result, out = self._invoke(tmp_path, "--agents", "3")

        assert len(list(out.glob("*.md"))) == 1

    def test_writes_one_prompt_file_per_agent(self, tmp_path):
        repo = self._repo(tmp_path)
        result, out = self._invoke(repo, "--agents", "3")

        assert result.exit_code == 0, result.output
        written = sorted(p.name for p in out.glob("*.md"))
        assert len(written) == 3
        assert "security-reviewer.md" in written

    def test_prompt_contains_the_shared_standard_and_the_diff(self, tmp_path):
        repo = self._repo(tmp_path)
        _result, out = self._invoke(repo, "--agents", "1")

        text = (out / "security-reviewer.md").read_text()
        assert "Review standard" in text
        assert "+x599 = 599" in text

    def test_prints_agent_names_and_models_for_the_orchestrator(self, tmp_path):
        repo = self._repo(tmp_path)
        result, _out = self._invoke(repo, "--agents", "2")

        assert "security-reviewer" in result.output
        assert "claude-" in result.output


class TestConsolidateCommand:
    def _files(self, tmp_path):
        return [_write(tmp_path, "security-reviewer", [_raw()])]

    def test_prints_the_severity_grouped_report(self, tmp_path):
        from click.testing import CliRunner

        import ai_reviewer.cli as cli

        paths = [str(p) for p in self._files(tmp_path)]
        result = CliRunner().invoke(cli.cli, ["consolidate", *paths])

        assert result.exit_code == 0, result.output
        assert "CRITICAL (1)" in result.output

    def test_json_output_is_machine_readable(self, tmp_path):
        from click.testing import CliRunner

        import ai_reviewer.cli as cli

        paths = [str(p) for p in self._files(tmp_path)]
        result = CliRunner().invoke(cli.cli, ["consolidate", *paths, "--output", "json"])

        assert json.loads(result.output)["findings"][0]["line_start"] == 412


def test_the_finding_cap_uses_the_real_diff_size_not_the_finding_count(tmp_path):
    """total_lines must describe the diff. Deriving it from how many findings were
    reported is self-referential: the cap would scale with the reviewers' output
    rather than with the size of the change."""
    # Distinct titles, or cross-file dedup collapses them before the cap runs.
    many = [
        _raw(file_path=f"src/f{i}.py", line_start=i + 1, severity="warning", title=f"Issue {i}")
        for i in range(30)
    ]
    a = _write(tmp_path, "security-reviewer", many)

    small_diff = consolidate_agent_findings([a], repo="demo", total_lines=50)
    large_diff = consolidate_agent_findings([a], repo="demo", total_lines=3000)

    assert len(small_diff.findings) == 5
    assert len(large_diff.findings) == 20


def test_a_malformed_findings_file_names_itself(tmp_path):
    """The skill extracts JSON from subagent replies, so bad input is expected in
    practice; a raw traceback would not say which agent's file was bad."""
    import pytest

    bad = tmp_path / "logic-reviewer.json"
    bad.write_text("not json at all")

    with pytest.raises(ValueError, match="logic-reviewer.json"):
        consolidate_agent_findings([bad], repo="demo")


def test_ignored_paths_are_not_sent_to_reviewers(tmp_path):
    """Parity with review_pr: files a repo excludes via .ai-reviewer.yaml must not
    be reviewed locally either."""
    import asyncio
    import subprocess

    from ai_reviewer.review import build_agent_prompts

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, check=True)

    git("init", "-q")
    git("config", "user.email", "t@e.st")
    git("config", "user.name", "t")
    (tmp_path / ".ai-reviewer.yaml").write_text('version: 1\nignore:\n  - "generated/**"\n')
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "big.py").write_text("AUTOGENERATED_MARKER = 1\n")
    (tmp_path / "real.py").write_text("REAL_CODE_MARKER = 1\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    (tmp_path / "generated" / "big.py").write_text("AUTOGENERATED_MARKER = 2\n")
    (tmp_path / "real.py").write_text("REAL_CODE_MARKER = 2\n")

    built = asyncio.run(build_agent_prompts(root=str(tmp_path), num_agents=1))
    prompt = next(iter(built.values()))["prompt"]

    assert "REAL_CODE_MARKER" in prompt
    assert "AUTOGENERATED_MARKER" not in prompt


def test_consolidate_measures_the_scope_it_was_given(tmp_path):
    """The cap scales with the reviewed diff. Consolidating a staged review without
    --staged measures a clean working tree, silently capping the report at five."""
    import os
    import subprocess

    from click.testing import CliRunner

    import ai_reviewer.cli as cli

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, check=True)

    git("init", "-q")
    git("config", "user.email", "t@e.st")
    git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 0\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    (tmp_path / "a.py").write_text("".join(f"x{i} = {i}\n" for i in range(2000)))
    git("add", "-A")

    many = [
        _raw(file_path=f"src/f{i}.py", line_start=i + 1, severity="warning", title=f"Issue {i}")
        for i in range(30)
    ]
    findings = str(_write(tmp_path, "security-reviewer", many))

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        staged = CliRunner().invoke(cli.cli, ["consolidate", findings, "--staged"])
        unscoped = CliRunner().invoke(cli.cli, ["consolidate", findings])
    finally:
        os.chdir(cwd)

    assert staged.exit_code == 0, staged.output
    assert "Reviewed the index" in staged.output
    assert "WARNING (20)" in staged.output
    assert "WARNING (5)" in unscoped.output


def test_the_skill_passes_one_scope_to_both_commands():
    """Consolidating with a different scope than the prompts were built for measures
    the wrong diff, so the two invocations must share one variable."""
    from pathlib import Path

    skill = Path(__file__).resolve().parents[1] / ".claude/skills/ai-review/SKILL.md"
    invocations = [
        line
        for line in skill.read_text().splitlines()
        if line.startswith(("ai-reviewer prompts", "ai-reviewer consolidate"))
    ]

    assert len(invocations) == 2
    assert all('"${SCOPE[@]}"' in line for line in invocations), invocations


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def test_the_plugin_ships_the_skill_and_agent_it_promises():
    """The marketplace source is a path, so a wrong one publishes a plugin that
    installs cleanly and provides nothing."""
    import json

    root = _repo_root()
    entry = json.loads((root / ".claude-plugin/marketplace.json").read_text())["plugins"][0]
    plugin_root = (root / entry["source"]).resolve()

    assert (plugin_root / ".claude-plugin/plugin.json").is_file()
    assert (plugin_root / "skills/ai-review/SKILL.md").is_file()
    assert (plugin_root / "agents/code-reviewer-readonly.md").is_file()


def test_the_documented_install_name_is_the_one_that_resolves():
    """`/plugin install <plugin>@<marketplace>` is what the docs tell people to run,
    so both halves have to match the manifests."""
    import json

    root = _repo_root()
    marketplace = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    entry = marketplace["plugins"][0]
    plugin = json.loads((root / entry["source"] / ".claude-plugin/plugin.json").read_text())

    assert entry["name"] == plugin["name"]
    documented = f"/plugin install {plugin['name']}@{marketplace['name']}"
    for doc in ("README.md", "docs/LOCAL-REVIEW.md"):
        assert documented in (root / doc).read_text(), doc


def test_the_plugin_version_tracks_the_package_version():
    """Two hand-maintained versions drift; the plugin's must follow pyproject."""
    import json
    import re

    root = _repo_root()
    declared = re.search(
        r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M
    ).group(1)
    plugin = json.loads((root / ".claude/.claude-plugin/plugin.json").read_text())

    assert plugin["version"] == declared


def test_the_reported_version_is_the_packaged_one():
    """--version answers "am I current?", so a hardcoded copy that drifts is worse
    than no answer. It had already fallen a release behind once."""
    import re

    from ai_reviewer import __version__

    declared = re.search(
        r'^version = "([^"]+)"', (_repo_root() / "pyproject.toml").read_text(), re.M
    ).group(1)

    assert __version__ == declared

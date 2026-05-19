#!/usr/bin/env python3
"""Local smoke-test for the update-docs AI generation pipeline.

Bypasses GitHub entirely — reads real local doc files, uses a fake diff,
calls Claude directly via generate_doc_drafts, and prints the result.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python scripts/test_docs_local.py

Optional env vars:
    TARGET_FILE   — which doc file to test (default: README.md)
    SOURCE_GLOB   — which source glob triggered it (default: src/ai_reviewer/cli.py)
    DIFF_FILE     — path to a real .diff file to use instead of the fake one
    REPO_NAME     — repo slug passed to generate_doc_drafts (default: calimero-network/ai-code-reviewer)

Does NOT need GITHUB_TOKEN.
"""

import asyncio
import difflib
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

# Run from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_reviewer.config import AnthropicApiConfig  # noqa: E402
from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts  # noqa: E402

FAKE_DIFF = textwrap.dedent("""\
    diff --git a/src/ai_reviewer/cli.py b/src/ai_reviewer/cli.py
    index abc1234..def5678 100644
    --- a/src/ai_reviewer/cli.py
    +++ b/src/ai_reviewer/cli.py
    @@ -55,6 +55,15 @@ def cli(verbose: bool) -> None:
     @cli.command("review-pr")
    +@cli.command("update-docs")
    +@click.argument("repo")
    +@click.argument("pr_number", type=int)
    +@click.option("--dry-run", is_flag=True)
    +def update_docs_cmd(repo, pr_number, dry_run):
    +    \"\"\"Generate and commit AI-drafted doc updates for a merged PR.\"\"\"
    +    asyncio.run(_update_docs_async(repo=repo, pr_number=pr_number, dry_run=dry_run))
    +
""")


def _make_gh_mock(target_file: str, current_content: str) -> MagicMock:
    """Return a GitHub client mock that serves current_content for target_file."""
    gh = MagicMock()

    file_mock = MagicMock()
    file_mock.decoded_content = current_content.encode()

    def get_file_contents(_repo_name, path, _ref):
        if path == target_file:
            return file_mock
        raise Exception(f"404: {path} not found")

    gh.get_file_contents.side_effect = get_file_contents
    return gh


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    target_file = os.environ.get("TARGET_FILE", "README.md")
    source_glob = os.environ.get("SOURCE_GLOB", "src/ai_reviewer/cli.py")
    repo_name = os.environ.get("REPO_NAME", "calimero-network/ai-code-reviewer")

    # Resolve and validate TARGET_FILE stays within the repo root
    repo_root_resolved = REPO_ROOT.resolve()
    local_path = (REPO_ROOT / target_file).resolve()
    if not local_path.is_relative_to(repo_root_resolved):
        print(f"ERROR: TARGET_FILE {target_file!r} resolves outside repo root", file=sys.stderr)
        sys.exit(1)
    if not local_path.exists():
        print(f"ERROR: {local_path} does not exist", file=sys.stderr)
        sys.exit(1)
    current_content = local_path.read_text()

    # Use provided diff or fallback to fake one
    diff_file = os.environ.get("DIFF_FILE")
    if diff_file:
        diff_path = Path(diff_file).resolve()
        if not diff_path.is_relative_to(repo_root_resolved):
            print(f"ERROR: DIFF_FILE {diff_file!r} resolves outside repo root", file=sys.stderr)
            sys.exit(1)
        diff = diff_path.read_text()
    else:
        diff = FAKE_DIFF

    print(f"Target file : {target_file} ({len(current_content)} chars)")
    print(f"Source glob : {source_glob}")
    print(f"Diff size   : {len(diff)} chars")
    print()

    suggestion = DocSuggestion(
        file=target_file,
        reason=(
            f"Files matching `{source_glob}` were changed but `{target_file}` "
            "was not updated (per source_to_docs_mapping)."
        ),
    )

    anthropic_cfg = AnthropicApiConfig(api_key=api_key)
    gh = _make_gh_mock(target_file, current_content)

    print("Calling Claude to generate doc draft...")
    drafts = await generate_doc_drafts(
        suggestions=[suggestion],
        diff=diff,
        repo_name=repo_name,
        ref="HEAD",
        anthropic_cfg=anthropic_cfg,
        gh=gh,
    )

    if not drafts:
        print("\nResult: Claude says NO_UPDATE_NEEDED — file is already accurate.")
        return

    draft = drafts[0]
    if draft.error:
        print(f"\nResult: ERROR — {draft.error}")
        sys.exit(1)

    lines = draft.updated_content.splitlines()
    original_lines = current_content.splitlines()
    print(
        f"\nResult: Claude generated an update ({len(lines)} lines vs {len(original_lines)} original)"
    )
    print()

    diff_output = list(
        difflib.unified_diff(
            original_lines,
            lines,
            fromfile=f"original/{target_file}",
            tofile=f"updated/{target_file}",
            lineterm="",
            n=3,
        )
    )
    if diff_output:
        print("\n".join(diff_output[:80]))
        if len(diff_output) > 80:
            print(f"... ({len(diff_output) - 80} more diff lines)")
    else:
        print("(No textual differences — content identical)")


if __name__ == "__main__":
    asyncio.run(main())

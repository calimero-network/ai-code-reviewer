"""Opt-in end-to-end integration test.

Runs a real review against a pinned small public PR. Requires:
- ANTHROPIC_API_KEY env var
- GITHUB_TOKEN env var (read-only ok)

Enable with: pytest -m integration

Before running, set PINNED_REPO and PINNED_PR below to a real PR.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


# Update these before running the integration test.
PINNED_REPO = "PLACEHOLDER-OWNER/PLACEHOLDER-REPO"
PINNED_PR = 1


@pytest.mark.asyncio
async def test_review_against_pinned_public_pr() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("GITHUB_TOKEN"):
        pytest.skip("credentials not set")
    if "PLACEHOLDER" in PINNED_REPO:
        pytest.skip("pinned PR not configured")

    from ai_reviewer.config import load_config
    from ai_reviewer.review import review_pr

    config = load_config()
    assert config.anthropic is not None and config.anthropic.api_key

    review = await review_pr(
        repo=PINNED_REPO,
        pr_number=PINNED_PR,
        anthropic_cfg=config.anthropic,
        github_token=os.environ["GITHUB_TOKEN"],
        num_agents=2,
        enable_cross_review=False,
        config=config,
    )

    assert review is not None
    assert isinstance(review.findings, list)
    # Quality floor: should not error out on a normal small PR
    assert not review.all_agents_failed

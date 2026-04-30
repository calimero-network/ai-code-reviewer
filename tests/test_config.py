"""Config loading tests."""

import textwrap
from pathlib import Path

from ai_reviewer.config import load_config


def test_load_anthropic_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-test")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        textwrap.dedent("""
        anthropic:
          api_key: ${ANTHROPIC_API_KEY}
          default_model: claude-sonnet-4-6
          enable_prompt_caching: true
        github:
          token: ${GITHUB_TOKEN}
        agents:
          - name: security-reviewer
            model: claude-sonnet-4-6
            focus_areas: [security]
            thinking_enabled: true
            thinking_budget_tokens: 8192
            allow_tool_use: true
            max_tool_calls: 20
    """)
    )
    cfg = load_config(cfg_file)
    assert cfg.anthropic is not None
    assert cfg.anthropic.api_key == "sk-test-123"
    assert cfg.anthropic.default_model == "claude-sonnet-4-6"
    assert cfg.anthropic.enable_prompt_caching is True
    assert cfg.agents[0].thinking_enabled is True
    assert cfg.agents[0].thinking_budget_tokens == 8192
    assert cfg.agents[0].allow_tool_use is True
    assert cfg.agents[0].max_tool_calls == 20

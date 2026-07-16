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
    assert cfg.agents[0].allow_tool_use is True
    assert cfg.agents[0].max_tool_calls == 20


def test_agent_defaults_sized_for_sonnet5():
    from ai_reviewer.config import AgentConfig

    cfg = AgentConfig(name="a", model="m", focus_areas=[])
    assert cfg.max_tokens == 8192


def test_anthropic_retries_default_is_resilient():
    # Long review calls drop connections transiently; 1 retry was too few.
    from ai_reviewer.config import AnthropicApiConfig

    assert AnthropicApiConfig(api_key="x").max_retries == 3


def test_pull_context_fields_default():
    from ai_reviewer.config import _parse_config

    cfg = _parse_config({"github": {"token": "t"}})
    assert cfg.anthropic.full_file_max_lines == 300
    assert cfg.anthropic.hunk_context_lines == 60
    assert cfg.anthropic.conventions_max_chars == 16_000


def test_pull_context_fields_parse_from_yaml():
    from ai_reviewer.config import _parse_config

    cfg = _parse_config(
        {
            "github": {"token": "t"},
            "anthropic": {
                "full_file_max_lines": 120,
                "hunk_context_lines": 25,
                "conventions_max_chars": 4096,
            },
        }
    )
    assert cfg.anthropic.full_file_max_lines == 120
    assert cfg.anthropic.hunk_context_lines == 25
    assert cfg.anthropic.conventions_max_chars == 4096

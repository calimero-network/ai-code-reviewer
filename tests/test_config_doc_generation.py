from __future__ import annotations

from ai_reviewer.config import DocGenerationSettings


def test_new_defaults():
    s = DocGenerationSettings()
    assert s.understanding_model == "claude-sonnet-4-6"
    assert s.apply_model == "claude-haiku-4-5-20251001"
    assert s.verify_model == "claude-haiku-4-5-20251001"
    assert s.max_understanding_diff_chars == 250_000
    assert s.allow_new_pages is True
    assert s.allow_new_sections is True
    assert s.verify_confidence_threshold == "medium"


def test_apply_model_falls_back_to_legacy_model_field():
    # `model` stays as the legacy/back-compat apply model default.
    s = DocGenerationSettings()
    assert s.apply_model == s.model

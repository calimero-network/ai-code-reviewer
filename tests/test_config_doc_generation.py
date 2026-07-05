from __future__ import annotations

from ai_reviewer.config import DocGenerationSettings


def test_new_defaults():
    s = DocGenerationSettings()
    assert s.understanding_model == "claude-sonnet-5"
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


def test_loader_apply_verify_fall_back_to_legacy_model():
    """When only `model` is set in YAML, apply_model/verify_model inherit it (loader)."""
    from ai_reviewer.config import _parse_config

    cfg = _parse_config({"doc_generation": {"model": "claude-legacy"}})
    assert cfg.doc_generation.apply_model == "claude-legacy"
    assert cfg.doc_generation.verify_model == "claude-legacy"
    # An explicit apply_model still wins over the legacy fallback.
    cfg2 = _parse_config(
        {"doc_generation": {"model": "claude-legacy", "apply_model": "claude-apply"}}
    )
    assert cfg2.doc_generation.apply_model == "claude-apply"
    assert cfg2.doc_generation.verify_model == "claude-legacy"

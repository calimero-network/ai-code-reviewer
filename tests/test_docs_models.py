from __future__ import annotations

import pytest

from ai_reviewer.docs.models import (
    Change,
    ChangeSummary,
    DocAction,
    DocDraft,
    FileWrite,
    Verdict,
    extract_json,
    meets_threshold,
)


def test_change_is_frozen():
    c = Change("fix", "t", "what", "why", ["sym"], ["f.rs"], "impact")
    with pytest.raises(AttributeError):
        c.title = "x"  # type: ignore[misc]


def test_change_summary_holds_changes():
    c = Change("fix", "t", "w", "y", [], [], "i")
    cs = ChangeSummary(pr_intent="intent", changes=[c])
    assert cs.changes[0] is c


def test_docdraft_defaults():
    d = DocDraft(action="add_section", target_path="architecture/x.html", updated_content="<html>")
    assert d.aux_edits == []
    assert d.error is None and d.flagged_reason is None


def test_filewrite_frozen():
    fw = FileWrite(path="nav.js", content="x")
    with pytest.raises(AttributeError):
        fw.path = "y"  # type: ignore[misc]


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose_around_object():
    assert extract_json('Here is the result:\n{"a": [1,2]}\nDone.') == {"a": [1, 2]}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_meets_threshold():
    assert meets_threshold("high", "medium")
    assert meets_threshold("medium", "medium")
    assert not meets_threshold("low", "medium")


def test_docaction_fields():
    c = Change("new_feature", "t", "w", "y", [], [], "i")
    a = DocAction(
        change=c,
        action="create_page",
        target_path="architecture/x.html",
        anchor=None,
        best_fit_reason="no existing home",
    )
    assert a.action == "create_page"


def test_verdict_fields():
    v = Verdict(reflects_change=False, confidence="low", notes="missed the invariant")
    assert v.reflects_change is False


def test_extract_json_rejects_bare_array():
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")


def test_extract_json_rejects_fenced_invalid():
    with pytest.raises(ValueError):
        extract_json("```json\n{not valid}\n```")


def test_meets_threshold_boundaries():
    assert meets_threshold("low", "low")
    assert meets_threshold("high", "high")


def test_docaction_optional_fields_assigned():
    c = Change("new_feature", "t", "w", "y", [], [], "i")
    a = DocAction(
        change=c,
        action="create_page",
        target_path="x.html",
        anchor=None,
        best_fit_reason="no existing home",
    )
    assert a.anchor is None
    assert a.best_fit_reason == "no existing home"

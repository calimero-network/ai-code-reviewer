"""Tests for structured-replacement validation and GitHub suggestion rendering."""

import logging

from ai_reviewer.agents.base import _parse_findings
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.review import aggregate_findings
from ai_reviewer.validation.fix_check import validate_finding_fixes, validate_replacement

PY_FILE = "def f():\n    x = 1\n    return x\n"


def _finding(**overrides) -> ConsolidatedFinding:
    base = {
        "id": "f1",
        "file_path": "src/foo.py",
        "line_start": 2,
        "line_end": 2,
        "severity": Severity.WARNING,
        "category": Category.LOGIC,
        "title": "Issue",
        "description": "desc",
        "suggested_fix": "prose",
        "consensus_score": 1.0,
        "agreeing_agents": ["a"],
        "confidence": 0.9,
    }
    base.update(overrides)
    return ConsolidatedFinding(**base)


# --- validate_replacement -------------------------------------------------


def test_validate_replacement_py_pass():
    assert validate_replacement(PY_FILE, 2, 2, "    x = 2", "src/foo.py") is True


def test_validate_replacement_py_fail_syntax():
    # Unbalanced paren makes ast.parse raise.
    assert validate_replacement(PY_FILE, 2, 2, "    x = (1", "src/foo.py") is False


def test_validate_replacement_json_pass_and_fail():
    original = '{\n  "a": 1\n}\n'
    assert validate_replacement(original, 2, 2, '  "a": 2', "cfg.json") is True
    assert validate_replacement(original, 2, 2, '  "a": ', "cfg.json") is False


def test_validate_replacement_yaml_pass_and_fail():
    original = "a: 1\nb: 2\n"
    assert validate_replacement(original, 2, 2, "b: 3", "cfg.yaml") is True
    # A tab in indentation is invalid YAML.
    assert validate_replacement(original, 2, 2, "b:\n\t- x", "cfg.yml") is False


def test_validate_replacement_toml_pass_and_fail():
    original = "a = 1\nb = 2\n"
    assert validate_replacement(original, 2, 2, "b = 3", "pyproject.toml") is True
    assert validate_replacement(original, 2, 2, "b = ", "pyproject.toml") is False


def test_validate_replacement_unknown_ext_balance_pass_and_fail():
    original = "fn main() {\n    call();\n}\n"
    # Balanced replacement keeps the same bracket deltas.
    assert validate_replacement(original, 2, 2, "    call(arg);", "main.rs") is True
    # Dropping a closing paren changes the balance.
    assert validate_replacement(original, 2, 2, "    call(arg;", "main.rs") is False


def test_validate_replacement_out_of_range_anchor():
    assert validate_replacement(PY_FILE, 99, 99, "    x = 2", "src/foo.py") is False
    assert validate_replacement(PY_FILE, 0, None, "    x = 2", "src/foo.py") is False


def test_validate_replacement_multi_line():
    original = "def f():\n    a = 1\n    b = 2\n    return a\n"
    ok = validate_replacement(original, 2, 3, "    a = 10\n    b = 20\n    c = 30", "src/foo.py")
    assert ok is True


# --- validate_finding_fixes driver ---------------------------------------


def test_driver_success_sets_fix_validated():
    findings = [_finding(suggested_replacement="    x = 2")]
    validate_finding_fixes(findings, lambda _p: PY_FILE)
    assert findings[0].fix_validated is True
    assert findings[0].suggested_replacement == "    x = 2"


def test_driver_failure_demotes_to_prose_and_logs(caplog):
    findings = [_finding(suggested_replacement="    x = (1")]
    with caplog.at_level(logging.INFO):
        validate_finding_fixes(findings, lambda _p: PY_FILE)
    assert findings[0].fix_validated is False
    assert findings[0].suggested_replacement is None
    assert any("demoting to prose" in r.message for r in caplog.records)


def test_driver_none_content_demotes():
    findings = [_finding(suggested_replacement="    x = 2")]
    validate_finding_fixes(findings, lambda _p: None)
    assert findings[0].fix_validated is False
    assert findings[0].suggested_replacement is None


def test_driver_respects_cap_in_severity_order():
    # Two criticals (valid), then a warning whose fix is broken. cap=2 => the
    # warning is never checked, so its (broken) replacement is left untouched.
    findings = [
        _finding(id="w", severity=Severity.WARNING, suggested_replacement="    x = (1"),
        _finding(id="c1", severity=Severity.CRITICAL, suggested_replacement="    x = 2"),
        _finding(id="c2", severity=Severity.CRITICAL, suggested_replacement="    x = 3"),
    ]
    validate_finding_fixes(findings, lambda _p: PY_FILE, max_checks=2)
    by_id = {f.id: f for f in findings}
    assert by_id["c1"].fix_validated is True
    assert by_id["c2"].fix_validated is True
    # Warning was beyond the cap: untouched, not demoted.
    assert by_id["w"].fix_validated is False
    assert by_id["w"].suggested_replacement == "    x = (1"


# --- schema round-trip ----------------------------------------------------


def test_parse_findings_accepts_and_tolerates_replacement():
    parsed = _parse_findings(
        {
            "findings": [
                {
                    "file_path": "a.py",
                    "line_start": 1,
                    "severity": "warning",
                    "category": "logic",
                    "title": "t",
                    "description": "d",
                    "suggested_fix": "prose",
                    "suggested_replacement": "x = 2",
                    "confidence": 0.7,
                },
                {
                    "file_path": "b.py",
                    "line_start": 1,
                    "severity": "warning",
                    "category": "logic",
                    "title": "t2",
                    "description": "d2",
                    "confidence": 0.7,
                },
            ]
        }
    )
    assert parsed[0].suggested_replacement == "x = 2"
    assert parsed[1].suggested_replacement is None


def test_replacement_survives_consolidation():
    raw = {
        "file_path": "a.py",
        "line_start": 5,
        "severity": "warning",
        "category": "logic",
        "title": "t",
        "description": "d",
        "suggested_fix": "prose",
        "suggested_replacement": "x = 2",
        "confidence": 0.9,
    }
    review = aggregate_findings([("agent-1", [raw], "s")], "o/r", 1)
    assert review.findings[0].suggested_replacement == "x = 2"
    assert review.findings[0].fix_validated is False


# --- formatter / suggestion block -----------------------------------------


def _body(finding: ConsolidatedFinding) -> str:
    return GitHubClient._build_review_comments([finding])[0]["body"]


def test_validated_finding_renders_suggestion_block():
    body = _body(_finding(suggested_replacement="    x = 2", fix_validated=True))
    assert "```suggestion\n    x = 2\n```" in body


def test_unvalidated_replacement_renders_prose_only():
    body = _body(_finding(suggested_replacement="    x = 2", fix_validated=False))
    assert "suggestion" not in body
    assert "**Suggested fix:**" in body


def test_triple_backtick_replacement_falls_back_to_prose():
    body = _body(_finding(suggested_replacement="x = '```'", fix_validated=True))
    assert "```suggestion" not in body


def test_multi_line_range_finding_falls_back_to_prose():
    body = _body(
        _finding(
            line_start=2,
            line_end=4,
            suggested_replacement="    x = 2\n    y = 3",
            fix_validated=True,
        )
    )
    assert "```suggestion" not in body


def test_json_export_carries_the_fields_a_fix_loop_needs():
    """The local fix loop reads findings from --output json. Without these two
    fields every fix must be re-derived by a model instead of applied."""
    from ai_reviewer.github.formatter import format_review_as_json

    validated = _finding(suggested_replacement="    x = 2", fix_validated=True)
    prose_only = _finding(id="f2", suggested_replacement=None, fix_validated=False)
    review = aggregate_findings([("a", [], "ok")], "o/r", 1)
    review.findings = [validated, prose_only]

    exported = format_review_as_json(review)["findings"]

    assert exported[0]["suggested_replacement"] == "    x = 2"
    assert exported[0]["fix_validated"] is True
    assert exported[1]["suggested_replacement"] is None
    assert exported[1]["fix_validated"] is False

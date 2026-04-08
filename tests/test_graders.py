"""
tests/test_graders.py

Failing tests for server/graders.py — written BEFORE implementation (TDD).

State objects are built with SimpleNamespace so tests have no dependency on
models.py (currently empty).  Each grader must accept any object that exposes
the documented attributes.

Grader signatures:
    grade_style_review(state)   -> float in [0.0, 1.0]
    grade_logic_bugs(state)     -> float in [0.0, 1.0]
    grade_security_audit(state) -> float in [0.0, 1.0]
"""

import sys
import os
import pytest
from types import SimpleNamespace

# Allow importing from repo root (graders.py lives in server/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.graders import grade_style_review, grade_logic_bugs, grade_security_audit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_state(**kwargs):
    """Return a SimpleNamespace with sensible defaults, overridden by kwargs."""
    defaults = dict(
        task_id="task_1",
        comments=[],
        ground_truth_issues=[],
        files_read=[],
        step_count=0,
        info={},
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# Ground-truth issues mirroring the plan's pr_dataset for task_1
TASK1_ISSUES = [
    {"file": "auth/utils.py", "line": 12, "type": "missing_docstring",  "severity": "low"},
    {"file": "auth/utils.py", "line": 24, "type": "camel_case_naming",  "severity": "low"},
    {"file": "auth/utils.py", "line": 37, "type": "line_too_long",      "severity": "low"},
    {"file": "auth/login.py", "line":  5, "type": "unused_import",      "severity": "low"},
    {"file": "auth/login.py", "line": 18, "type": "missing_type_hints", "severity": "low"},
]

# Ground-truth issues for task_2 (logic bugs in JS LRU cache)
TASK2_ISSUES = [
    {"file": "lru_cache.js", "line": 42, "type": "off_by_one",      "severity": "medium"},
    {"file": "lru_cache.js", "line": 67, "type": "race_condition",  "severity": "high"},
    {"file": "lru_cache.js", "line": 88, "type": "edge_case",       "severity": "medium"},
]

# Ground-truth issues for task_3 (security vulns in Python API)
TASK3_ISSUES = [
    {"file": "api/db.py",     "line": 31, "type": "sql_injection",            "severity": "critical"},
    {"file": "api/upload.py", "line": 14, "type": "insecure_deserialization",  "severity": "critical"},
    {"file": "api/upload.py", "line": 57, "type": "path_traversal",            "severity": "high"},
    {"file": "api/config.py", "line":  8, "type": "hardcoded_secret",          "severity": "medium"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared contract: all graders must return float in [0.0, 1.0]
# ─────────────────────────────────────────────────────────────────────────────

class TestGraderReturnTypeContract:
    """Every grader must return a float clamped to [0.0, 1.0]."""

    def test_grade_style_review_returns_float(self):
        state = make_state(task_id="task_1", ground_truth_issues=TASK1_ISSUES)
        result = grade_style_review(state)
        assert isinstance(result, float)

    def test_grade_style_review_within_bounds(self):
        state = make_state(task_id="task_1", ground_truth_issues=TASK1_ISSUES)
        result = grade_style_review(state)
        assert 0.0 <= result <= 1.0

    def test_grade_logic_bugs_returns_float(self):
        state = make_state(task_id="task_2", ground_truth_issues=TASK2_ISSUES)
        result = grade_logic_bugs(state)
        assert isinstance(result, float)

    def test_grade_logic_bugs_within_bounds(self):
        state = make_state(task_id="task_2", ground_truth_issues=TASK2_ISSUES)
        result = grade_logic_bugs(state)
        assert 0.0 <= result <= 1.0

    def test_grade_security_audit_returns_float(self):
        state = make_state(task_id="task_3", ground_truth_issues=TASK3_ISSUES)
        result = grade_security_audit(state)
        assert isinstance(result, float)

    def test_grade_security_audit_within_bounds(self):
        state = make_state(task_id="task_3", ground_truth_issues=TASK3_ISSUES)
        result = grade_security_audit(state)
        assert 0.0 <= result <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# grade_style_review
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeStyleReview:
    """
    Scoring formula:
        score = (correct_line_comments / total_known_issues) * 0.7
              + coverage_bonus * 0.3   # 0.3 if ALL issue-files were read
              - 0.05 * false_positive_count
        score = clamp(score, 0.0, 1.0)
    """

    # ── Minimum score (0 issues found, no files read) ────────────────────────

    def test_min_score_no_comments_no_files_read(self):
        """Agent posted no comments and read no files → score should be 0.0."""
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=[],
            files_read=[],
        )
        assert grade_style_review(state) == 0.0

    def test_only_false_positives_score_not_above_zero(self):
        """False positives with no correct comments → clamped to 0.0."""
        # Ten false-positive comments: 10 * 0.05 = 0.50 penalty, no positives
        fps = [{"file": "other.py", "line": i, "text": "wrong"} for i in range(10)]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=fps,
            files_read=[],
        )
        assert grade_style_review(state) == 0.0

    # ── Partial score ────────────────────────────────────────────────────────

    def test_partial_score_two_correct_no_coverage_bonus(self):
        """
        2 of 5 correct, no coverage bonus, no false positives.
        Expected = (2/5) * 0.7 = 0.28.
        """
        correct_comments = [
            {"file": "auth/utils.py", "line": 12, "text": "missing docstring"},
            {"file": "auth/utils.py", "line": 24, "text": "use snake_case"},
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=correct_comments,
            files_read=["auth/utils.py"],  # only one of two issue-files read
        )
        result = grade_style_review(state)
        assert abs(result - 0.28) < 1e-9

    def test_partial_score_with_coverage_bonus(self):
        """
        2 of 5 correct + coverage bonus (all files read), no false positives.
        Expected = (2/5) * 0.7 + 0.3 = 0.28 + 0.30 = 0.58.
        """
        correct_comments = [
            {"file": "auth/utils.py", "line": 12, "text": "missing docstring"},
            {"file": "auth/utils.py", "line": 24, "text": "use snake_case"},
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=correct_comments,
            files_read=["auth/utils.py", "auth/login.py"],
        )
        result = grade_style_review(state)
        assert abs(result - 0.58) < 1e-9

    def test_false_positives_reduce_score(self):
        """
        2 of 5 correct, coverage bonus, 2 false positives.
        Expected = 0.28 + 0.30 - 2*0.05 = 0.48.
        """
        comments = [
            {"file": "auth/utils.py", "line": 12, "text": "missing docstring"},
            {"file": "auth/utils.py", "line": 24, "text": "use snake_case"},
            {"file": "other.py",      "line": 99, "text": "hallucinated issue"},
            {"file": "other.py",      "line": 55, "text": "another hallucination"},
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=comments,
            files_read=["auth/utils.py", "auth/login.py"],
        )
        result = grade_style_review(state)
        assert abs(result - 0.48) < 1e-9

    # ── Maximum score (all issues found + coverage bonus) ────────────────────

    def test_max_score_all_issues_all_files_read(self):
        """
        5 of 5 correct + coverage bonus, no false positives.
        Expected = 1.0 * 0.7 + 0.3 = 1.0.
        """
        all_correct = [
            {"file": issue["file"], "line": issue["line"], "text": "found it"}
            for issue in TASK1_ISSUES
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=all_correct,
            files_read=["auth/utils.py", "auth/login.py"],
        )
        result = grade_style_review(state)
        assert abs(result - 1.0) < 1e-9

    # ── Coverage bonus edge cases ─────────────────────────────────────────────

    def test_coverage_bonus_requires_all_issue_files(self):
        """Reading only one of the two issue files does NOT grant the coverage bonus."""
        all_correct = [
            {"file": issue["file"], "line": issue["line"], "text": "found it"}
            for issue in TASK1_ISSUES
        ]
        state_partial_read = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=all_correct,
            files_read=["auth/utils.py"],          # auth/login.py missing
        )
        state_full_read = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=all_correct,
            files_read=["auth/utils.py", "auth/login.py"],
        )
        score_partial = grade_style_review(state_partial_read)
        score_full    = grade_style_review(state_full_read)
        assert score_full > score_partial

    def test_coverage_bonus_not_granted_when_extra_files_read(self):
        """Reading extra non-issue files beyond the required set still grants bonus."""
        all_correct = [
            {"file": issue["file"], "line": issue["line"], "text": "found it"}
            for issue in TASK1_ISSUES
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=all_correct,
            files_read=["auth/utils.py", "auth/login.py", "README.md", "tests/conftest.py"],
        )
        result = grade_style_review(state)
        # Full bonus should still apply — extra reads do no harm
        assert abs(result - 1.0) < 1e-9

    # ── Determinism ──────────────────────────────────────────────────────────

    def test_grade_style_review_is_deterministic(self):
        """Same state must produce the same score on repeated calls."""
        correct_comments = [
            {"file": "auth/utils.py", "line": 12, "text": "missing docstring"},
        ]
        state = make_state(
            task_id="task_1",
            ground_truth_issues=TASK1_ISSUES,
            comments=correct_comments,
            files_read=["auth/utils.py"],
        )
        assert grade_style_review(state) == grade_style_review(state)


# ─────────────────────────────────────────────────────────────────────────────
# grade_logic_bugs
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeLogicBugs:
    """
    Scoring formula:
        score = sum(bug_weight for each correctly identified bug)
              + 0.10 if any comment contains semantic fix keywords
              + 0.10 if state.info["assigned_score"] <= 4
        score = clamp(score, 0.0, 1.0)

    Bug weights: off_by_one=0.25, race_condition=0.40, edge_case=0.35
    Semantic keywords: ">=" (off-by-one), "await" (race condition), "null" (edge case)
    """

    # ── Minimum score ────────────────────────────────────────────────────────

    def test_min_score_no_bugs_found(self):
        """No comments, no rejection decision → score must be 0.0."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[],
            info={"assigned_score": 10},   # wrong rejection (approve)
        )
        assert grade_logic_bugs(state) == 0.0

    # ── Partial scores per bug type ───────────────────────────────────────────

    def test_off_by_one_only_scores_025(self):
        """Finding only the off-by-one bug (no fix keywords, no rejection bonus)."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 42, "text": "wrong comparison"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        assert abs(result - 0.25) < 1e-9

    def test_race_condition_only_scores_040(self):
        """Finding only the race condition bug (no fix keywords, no rejection bonus)."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 67, "text": "concurrent access issue"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        assert abs(result - 0.40) < 1e-9

    def test_edge_case_only_scores_035(self):
        """Finding only the edge case bug (no fix keywords, no rejection bonus)."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 88, "text": "returns undefined"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        assert abs(result - 0.35) < 1e-9

    def test_semantic_fix_bonus_for_off_by_one(self):
        """Comment containing '>=' for off-by-one grants +0.10 semantic fix bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 42, "text": "use >= instead of >"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        # 0.25 (off-by-one) + 0.10 (semantic fix)
        assert abs(result - 0.35) < 1e-9

    def test_semantic_fix_bonus_for_race_condition(self):
        """Comment containing 'await' for race condition grants +0.10 semantic fix bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 67, "text": "add await before lock.acquire()"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        # 0.40 (race condition) + 0.10 (semantic fix)
        assert abs(result - 0.50) < 1e-9

    def test_semantic_fix_bonus_for_edge_case(self):
        """Comment containing 'null' for edge case grants +0.10 semantic fix bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 88, "text": "return null not undefined"}],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        # 0.35 (edge_case) + 0.10 (semantic fix)
        assert abs(result - 0.45) < 1e-9

    def test_rejection_bonus_for_score_at_most_4(self):
        """assign_score <= 4 grants +0.10 rejection bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 42, "text": "use >="}],
            info={"assigned_score": 3},
        )
        result = grade_logic_bugs(state)
        # 0.25 + 0.10 (semantic) + 0.10 (rejection)
        assert abs(result - 0.45) < 1e-9

    def test_rejection_bonus_boundary_score_4(self):
        """assign_score exactly 4 still grants rejection bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[],
            info={"assigned_score": 4},
        )
        result = grade_logic_bugs(state)
        assert abs(result - 0.10) < 1e-9

    def test_rejection_bonus_not_granted_for_score_5(self):
        """assign_score = 5 does NOT grant rejection bonus."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[],
            info={"assigned_score": 5},
        )
        result = grade_logic_bugs(state)
        assert result == 0.0

    def test_semantic_fix_bonus_granted_only_once_regardless_of_bugs_found(self):
        """Semantic fix bonus is +0.10 total even when multiple valid fix keywords present."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[
                {"file": "lru_cache.js", "line": 42, "text": "use >="},
                {"file": "lru_cache.js", "line": 67, "text": "add await"},
                {"file": "lru_cache.js", "line": 88, "text": "return null"},
            ],
            info={"assigned_score": 10},
        )
        result = grade_logic_bugs(state)
        # 0.25 + 0.40 + 0.35 + 0.10 (single bonus) = 1.10 → clamped to 1.0
        assert result == 1.0

    # ── Maximum score ─────────────────────────────────────────────────────────

    def test_max_score_all_bugs_fix_and_rejection(self):
        """All three bugs + semantic fix keyword + rejection decision → 1.0 (clamped)."""
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[
                {"file": "lru_cache.js", "line": 42, "text": "change > to >="},
                {"file": "lru_cache.js", "line": 67, "text": "missing await"},
                {"file": "lru_cache.js", "line": 88, "text": "return null for empty cache"},
            ],
            info={"assigned_score": 2},
        )
        assert grade_logic_bugs(state) == 1.0

    # ── Determinism ──────────────────────────────────────────────────────────

    def test_grade_logic_bugs_is_deterministic(self):
        state = make_state(
            task_id="task_2",
            ground_truth_issues=TASK2_ISSUES,
            comments=[{"file": "lru_cache.js", "line": 42, "text": "use >="}],
            info={"assigned_score": 3},
        )
        assert grade_logic_bugs(state) == grade_logic_bugs(state)


# ─────────────────────────────────────────────────────────────────────────────
# grade_security_audit
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeSecurityAudit:
    """
    Scoring formula:
        score = sum(vuln_weight for each found vulnerability)
              + 0.05 * correctly_classified_severities
              + 0.05 * semantically_valid_fix_suggestions
              + 0.15 if SQL injection found AND assigned_score == 0
        score = clamp(score, 0.0, 1.0)

    Vulnerability weights:
        sql_injection=0.25, insecure_deserialization=0.25,
        path_traversal=0.20, hardcoded_secret=0.15

    Severity classification (comment text must contain):
        sql_injection/insecure_deserialization → "CRITICAL" or "critical"
        path_traversal                         → "HIGH" or "high"
        hardcoded_secret                       → "MEDIUM" or "medium"
    """

    # ── Minimum score ────────────────────────────────────────────────────────

    def test_min_score_no_vulns_found(self):
        """No comments posted → score must be 0.0."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[],
            info={"assigned_score": 5},
        )
        assert grade_security_audit(state) == 0.0

    # ── Per-vulnerability detection scores ───────────────────────────────────

    def test_sql_injection_detection_scores_025(self):
        """Finding SQL injection (no severity/fix/block bonus) → 0.25."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "SQL issue here"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        assert abs(result - 0.25) < 1e-9

    def test_insecure_deserialization_detection_scores_025(self):
        """Finding insecure deserialization → 0.25."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/upload.py", "line": 14, "text": "unsafe pickle"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        assert abs(result - 0.25) < 1e-9

    def test_path_traversal_detection_scores_020(self):
        """Finding path traversal → 0.20."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/upload.py", "line": 57, "text": "path not validated"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        assert abs(result - 0.20) < 1e-9

    def test_hardcoded_secret_detection_scores_015(self):
        """Finding hardcoded secret → 0.15."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/config.py", "line": 8, "text": "API key in code"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        assert abs(result - 0.15) < 1e-9

    # ── Severity classification bonus ─────────────────────────────────────────

    def test_correct_severity_sql_injection_grants_005(self):
        """SQL injection comment containing 'CRITICAL' grants severity bonus (+0.05)."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "CRITICAL: SQL injection via f-string"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        # 0.25 (detection) + 0.05 (severity) = 0.30
        assert abs(result - 0.30) < 1e-9

    def test_correct_severity_path_traversal_grants_005(self):
        """Path traversal comment containing 'HIGH' grants severity bonus (+0.05)."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/upload.py", "line": 57, "text": "HIGH: path traversal missing validation"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        # 0.20 + 0.05 = 0.25
        assert abs(result - 0.25) < 1e-9

    def test_correct_severity_hardcoded_secret_grants_005(self):
        """Hardcoded secret comment containing 'MEDIUM' grants severity bonus (+0.05)."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/config.py", "line": 8, "text": "MEDIUM severity: hardcoded API key"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        # 0.15 + 0.05 = 0.20
        assert abs(result - 0.20) < 1e-9

    def test_wrong_severity_label_does_not_grant_bonus(self):
        """SQL injection comment with 'HIGH' instead of 'CRITICAL' → no severity bonus."""
        state_wrong = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "HIGH: SQL injection"}],
            info={"assigned_score": 5},
        )
        state_correct = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "CRITICAL: SQL injection"}],
            info={"assigned_score": 5},
        )
        assert grade_security_audit(state_wrong) < grade_security_audit(state_correct)

    def test_severity_bonus_counts_each_correct_classification(self):
        """Each correctly classified vuln adds +0.05 (tested for two vulns together)."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[
                {"file": "api/db.py",     "line": 31, "text": "CRITICAL: SQL injection"},
                {"file": "api/upload.py", "line": 14, "text": "CRITICAL: insecure deserialization"},
            ],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        # 0.25 + 0.25 (detection) + 0.05 + 0.05 (severity) = 0.60
        assert abs(result - 0.60) < 1e-9

    # ── SQL injection block bonus ─────────────────────────────────────────────

    def test_sql_injection_block_bonus_requires_score_zero(self):
        """
        SQL injection found AND assigned_score == 0 → +0.15 block bonus.
        assigned_score != 0 must NOT grant the bonus.
        """
        state_blocked = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "SQL injection found"}],
            info={"assigned_score": 0},
        )
        state_not_blocked = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "SQL injection found"}],
            info={"assigned_score": 1},
        )
        blocked_score     = grade_security_audit(state_blocked)
        not_blocked_score = grade_security_audit(state_not_blocked)
        # 0.25 + 0.15 vs 0.25 + 0.00
        assert abs(blocked_score - 0.40) < 1e-9
        assert abs(not_blocked_score - 0.25) < 1e-9

    def test_block_bonus_not_granted_without_sql_injection_found(self):
        """assigned_score == 0 without finding SQL injection → no block bonus."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/config.py", "line": 8, "text": "hardcoded secret"}],
            info={"assigned_score": 0},
        )
        result = grade_security_audit(state)
        # Only 0.15 (hardcoded secret), no block bonus
        assert abs(result - 0.15) < 1e-9

    # ── Maximum score ─────────────────────────────────────────────────────────

    def test_max_score_all_vulns_correct_severity_and_block(self):
        """
        All 4 vulns detected, all severities correctly classified, all fixes
        semantically valid, SQL injection found and assigned_score == 0.

        Base: 0.25 + 0.25 + 0.20 + 0.15 = 0.85
        Severity bonuses: 4 × 0.05 = 0.20
        Block bonus: +0.15
        Total before clamp: 1.20 → clamped to 1.0
        """
        comments = [
            {"file": "api/db.py",     "line": 31, "text": "CRITICAL: SQL injection, use parameterized queries"},
            {"file": "api/upload.py", "line": 14, "text": "CRITICAL: insecure deserialization, avoid pickle"},
            {"file": "api/upload.py", "line": 57, "text": "HIGH: path traversal, validate file path"},
            {"file": "api/config.py", "line":  8, "text": "MEDIUM: hardcoded secret, use environment variable"},
        ]
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=comments,
            info={"assigned_score": 0},
        )
        assert grade_security_audit(state) == 1.0

    # ── Determinism ──────────────────────────────────────────────────────────

    def test_grade_security_audit_is_deterministic(self):
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "CRITICAL: SQL injection"}],
            info={"assigned_score": 0},
        )
        assert grade_security_audit(state) == grade_security_audit(state)

    # ── Lowercase severity keyword accepted ──────────────────────────────────

    def test_lowercase_critical_accepted_for_severity_bonus(self):
        """'critical' (lowercase) must be accepted, not just 'CRITICAL'."""
        state = make_state(
            task_id="task_3",
            ground_truth_issues=TASK3_ISSUES,
            comments=[{"file": "api/db.py", "line": 31, "text": "critical severity: SQL injection"}],
            info={"assigned_score": 5},
        )
        result = grade_security_audit(state)
        # 0.25 (detection) + 0.05 (severity lowercase accepted)
        assert abs(result - 0.30) < 1e-9

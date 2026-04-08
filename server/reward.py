"""
Reward computation — server-side only, never exposed to client.
"""
from models import Action, ActionType, CodeReviewState


def compute_step_reward(action: Action, state: CodeReviewState, result: str) -> float:
    """Compute per-step reward based on action type and current state."""
    reward = 0.0

    if action.action_type == ActionType.READ_FILE:
        if action.file_path and action.file_path not in state.files_read:
            reward = 0.02  # new file read
        else:
            reward = 0.0   # repeat read

    elif action.action_type == ActionType.GET_DIFF:
        reward = 0.01

    elif action.action_type == ActionType.POST_COMMENT:
        reward = _score_comment(action, state)

    elif action.action_type == ActionType.CHECK_LINT:
        if result and result.strip():
            reward = 0.03

    elif action.action_type == ActionType.SEARCH_PATTERN:
        reward = 0.0  # neutral — no reward/penalty

    elif action.action_type == ActionType.ASSIGN_SCORE:
        reward = _score_assign(action, state)

    # Penalty: repeated identical action (same type + same file + same line)
    for prev in state.comments:
        if (action.action_type == ActionType.POST_COMMENT
                and prev.get("file") == action.file_path
                and prev.get("line") == action.line_number
                and prev.get("text") == action.comment):
            reward = -0.10
            break

    return reward


def _score_comment(action: Action, state: CodeReviewState) -> float:
    """Return reward for a POST_COMMENT action."""
    ground_truth = {(i["file"], i["line"]) for i in state.ground_truth_issues}
    key = (action.file_path, action.line_number)

    if key in ground_truth:
        # Scale by severity
        for issue in state.ground_truth_issues:
            if issue["file"] == action.file_path and issue["line"] == action.line_number:
                sev = issue.get("severity", "low").lower()
                if sev == "critical":
                    return 0.25
                elif sev == "high":
                    return 0.20
                elif sev == "medium":
                    return 0.15
                else:
                    return 0.10
    return -0.05  # false positive


def _score_assign(action: Action, state: CodeReviewState) -> float:
    """Bonus if the assigned score reflects correct severity judgment."""
    if not state.comments:
        return -0.20  # no comments posted before assigning

    # Bonus if score range is appropriate for the task difficulty
    task_id = state.task_id
    score = action.score or 5.0

    if task_id == "task_1":
        # Style issues — should score 5–8 (approve with suggestions)
        if 5.0 <= score <= 8.0:
            return 0.10
    elif task_id == "task_2":
        # Logic bugs — should reject (score ≤ 4)
        if score <= 4.0:
            return 0.10
    elif task_id == "task_3":
        # Security vulns — must block (score = 0)
        if score == 0.0:
            return 0.10

    return 0.0


def compute_terminal_reward(grader_score: float) -> float:
    """Terminal reward on episode end: grader_score × 0.5"""
    return grader_score * 0.5

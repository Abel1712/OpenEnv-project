"""
CodeReviewEnvironment — implements the OpenEnv Gymnasium-style API.

INVARIANT: reset()/step()/state are for infrastructure only.
The agent interacts exclusively via MCP tools (mapped to Action types).
"""
import re
import uuid
from typing import Optional

from openenv.core.env_server import Environment

from models import Action, ActionType, CodeReviewState, Observation
from server.pr_dataset import PR_DATASET
from server.reward import compute_step_reward, compute_terminal_reward
from server.graders import grade_style_review, grade_logic_bugs, grade_security_audit

MAX_STEPS = 50


class CodeReviewEnvironment(Environment[Action, Observation, CodeReviewState]):

    def __init__(self):
        self._state: Optional[CodeReviewState] = None

    # ── Public Gym API ────────────────────────────────────────────────────────

    def reset(self, seed=None, episode_id: Optional[str] = None) -> Observation:
        task_id = self._resolve_task_id(episode_id)
        pr_data = PR_DATASET[task_id]

        self._state = CodeReviewState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            pr_data=pr_data,
            files_read=[],
            comments=[],
            step_count=0,
            ground_truth_issues=pr_data["ground_truth_issues"],
            total_reward=0.0,
            assigned_score=None,
        )

        return Observation(
            action_result="PR loaded. Use get_diff to see changes or read_file to inspect files.",
            pr_metadata={
                "title":         pr_data["title"],
                "difficulty":    pr_data["difficulty"],
                "changed_files": pr_data["changed_files"],
                "task_id":       task_id,
            },
            review_progress=self._progress(),
            reward=0.0,
            done=False,
            info={"task_id": task_id, "episode_id": self._state.episode_id},
        )

    def step(self, action: Action) -> Observation:
        assert self._state is not None, "Call reset() before step()"
        self._state.step_count += 1

        result, reward, done, info = self._dispatch(action)

        # Timeout penalty
        if self._state.step_count > MAX_STEPS and not done:
            reward -= 0.30
            done = True
            info["timeout"] = True

        self._state.total_reward += reward

        return Observation(
            action_result=result,
            pr_metadata={
                "title":         self._state.pr_data["title"],
                "difficulty":    self._state.pr_data["difficulty"],
                "changed_files": self._state.pr_data["changed_files"],
                "task_id":       self._state.task_id,
            },
            review_progress=self._progress(),
            reward=reward,
            done=done,
            info=info,
        )

    @property
    def state(self) -> CodeReviewState:
        return self._state

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, action: Action):
        """Route action to handler. Returns (result_str, reward, done, info)."""
        info = {"task_id": self._state.task_id, "episode_id": self._state.episode_id}

        if action.action_type == ActionType.READ_FILE:
            result, reward = self._handle_read_file(action)
        elif action.action_type == ActionType.GET_DIFF:
            result, reward = self._handle_get_diff(action)
        elif action.action_type == ActionType.POST_COMMENT:
            result, reward = self._handle_post_comment(action)
        elif action.action_type == ActionType.CHECK_LINT:
            result, reward = self._handle_check_lint(action)
        elif action.action_type == ActionType.SEARCH_PATTERN:
            result, reward = self._handle_search_pattern(action)
        elif action.action_type == ActionType.ASSIGN_SCORE:
            result, reward, done, grader_score = self._handle_assign_score(action)
            info["grader_score"] = grader_score
            info["assigned_score"] = action.score
            return result, reward, done, info
        else:
            result, reward = f"Unknown action type: {action.action_type}", -0.05

        return result, reward, False, info

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_read_file(self, action: Action):
        files = self._state.pr_data.get("files", {})
        path = action.file_path or ""

        if path not in files:
            return f"File not found: {path}. Available: {list(files.keys())}", -0.01

        reward = compute_step_reward(action, self._state, "")
        if path not in self._state.files_read:
            self._state.files_read.append(path)

        # Include line numbers so the agent can post comments on exact lines
        raw = files[path]
        numbered = "\n".join(
            f"{i:4}: {line}" for i, line in enumerate(raw.splitlines(), start=1)
        )
        return numbered, reward

    def _handle_get_diff(self, action: Action):
        diff = self._state.pr_data.get("diff", "")
        if action.file_path:
            # Filter diff to lines relevant to the requested file
            lines = diff.splitlines()
            in_file = False
            filtered = []
            for line in lines:
                if line.startswith("diff --git") and action.file_path in line:
                    in_file = True
                elif line.startswith("diff --git"):
                    in_file = False
                if in_file:
                    filtered.append(line)
            diff = "\n".join(filtered) if filtered else f"No diff found for {action.file_path}"

        reward = compute_step_reward(action, self._state, diff)
        return diff, reward

    def _handle_post_comment(self, action: Action):
        if not action.file_path or action.line_number is None or not action.comment:
            return "post_comment requires file_path, line_number, and comment.", -0.05

        # Detect duplicate comment
        for prev in self._state.comments:
            if (prev["file"] == action.file_path
                    and prev["line"] == action.line_number
                    and prev["text"] == action.comment):
                return "Duplicate comment — already posted.", -0.10

        self._state.comments.append({
            "file": action.file_path,
            "line": action.line_number,
            "text": action.comment,
        })

        reward = compute_step_reward(action, self._state, "")
        return f"Comment posted on {action.file_path}:{action.line_number}", reward

    def _handle_check_lint(self, action: Action):
        files = self._state.pr_data.get("files", {})
        path = action.file_path or ""

        if path not in files:
            return f"File not found: {path}", -0.01

        content = files[path]
        issues = []

        for i, line in enumerate(content.splitlines(), start=1):
            if len(line) > 120:
                issues.append(f"Line {i}: line too long ({len(line)} chars)")
            if re.search(r'\bcamelCase\b|[a-z][A-Z]', line):
                pass  # heuristic too noisy; leave to agent
            if re.match(r'^import\s+', line) or re.match(r'^from\s+\S+\s+import\s+', line):
                pass  # unused import detection is complex; surface raw

        result = "\n".join(issues) if issues else "No obvious style issues detected."
        reward = compute_step_reward(action, self._state, result)
        return result, reward

    def _handle_search_pattern(self, action: Action):
        pattern = action.pattern or ""
        files = self._state.pr_data.get("files", {})

        if action.file_path:
            search_files = {action.file_path: files.get(action.file_path, "")}
        else:
            search_files = files

        matches = []
        for fname, content in search_files.items():
            for i, line in enumerate(content.splitlines(), start=1):
                if re.search(pattern, line):
                    matches.append(f"{fname}:{i}: {line.strip()}")

        result = "\n".join(matches) if matches else f"No matches for pattern: {pattern}"
        return result, 0.0

    def _handle_assign_score(self, action: Action):
        score = action.score or 0.0
        self._state.assigned_score = score

        # Inject assigned_score into state so graders can read it
        grader_score = self._run_grader()
        terminal = compute_terminal_reward(grader_score)

        step_reward = compute_step_reward(action, self._state, "")
        total_reward = step_reward + terminal

        result = (
            f"Review complete. Score: {score:.1f}/10. "
            f"Grader score: {grader_score:.3f}. Terminal reward: {terminal:.2f}."
        )
        return result, total_reward, True, grader_score

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_grader(self) -> float:
        task_id = self._state.task_id
        if task_id == "task_1":
            return grade_style_review(self._state)
        elif task_id == "task_2":
            return grade_logic_bugs(self._state)
        elif task_id == "task_3":
            return grade_security_audit(self._state)
        return 0.0

    def _progress(self) -> dict:
        return {
            "files_read":      len(self._state.files_read) if self._state else 0,
            "comments_posted": len(self._state.comments) if self._state else 0,
            "step_count":      self._state.step_count if self._state else 0,
        }

    @staticmethod
    def _resolve_task_id(episode_id: Optional[str]) -> str:
        if episode_id and episode_id.startswith("task_"):
            task_id = episode_id.split("_")[0] + "_" + episode_id.split("_")[1]
            if task_id in PR_DATASET:
                return task_id
        import random
        return random.choice(list(PR_DATASET.keys()))

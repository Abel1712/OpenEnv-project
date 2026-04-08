# Code Review OpenEnv — Complete Project Plan

> **Hackathon Goal:** Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.
> **Judge Hint:** *"Long running tasks with multiple trajectories will be rewarded."*

---

## 1. What We're Building

A **Code Review / PR Review Environment** where an AI agent reviews pull requests, identifies bugs, suggests fixes, and assigns quality scores.

**Why this domain wins every judging criterion:**

| Judging Criterion | Weight | Why Code Review Wins |
|---|---|---|
| Real-world utility | 30% | Every software team does PR reviews daily — immediate, obvious value |
| Task & grader quality | 25% | 3 tasks: easy (style), medium (logic bugs), hard (security vulns) |
| Environment design | 20% | Multi-step review = dense reward signal, clean episode boundaries |
| Code quality & spec | 15% | Full OpenEnv spec + Dockerfile + HF Space deployment |
| Creativity & novelty | 10% | First code-review env in OpenEnv — genuinely novel domain |

**Why long trajectories:** Each PR review episode has 15–40 steps. The agent reads files, examines diffs, detects issues, writes comments, suggests fixes, and finally assigns a score. This creates:
- Multiple decision points per trajectory (dense reward signal throughout)
- Branching strategies: thorough-first vs. quick-scan vs. security-focused
- Learning signal at every step, not just episode end
- Realistic simulation of how senior engineers actually review code

---

## 2. Architecture Overview

### 2.1 OpenEnv Two-Interface Model

Following the mandatory dual-API boundary from `INVARIANTS.md`:

| Boundary | API | Purpose |
|---|---|---|
| **Agent (AI Reviewer)** | MCP Tools | `read_file`, `get_diff`, `post_comment`, `check_lint`, `assign_score` |
| **Infrastructure (Orchestrator)** | WebSocket Gym API | `reset()`, `step()`, `state()` — agent cannot access this |

> **CRITICAL INVARIANT:** The agent must NOT be able to call `reset()`. Only the infrastructure orchestrator calls it.

### 2.2 File Structure

```
code_review_env/
├── __init__.py                      # Export Action, Observation, Client
├── models.py                        # Pydantic: Action, Observation, State
├── client.py                        # EnvClient[Action, Observation, State]
├── openenv.yaml                     # Environment manifest
├── pyproject.toml                   # Dependencies
├── inference.py                     # ⚠️ REQUIRED — root level, exact stdout format
├── README.md                        # Full documentation
└── server/
    ├── code_review_environment.py   # Environment implementation
    ├── pr_dataset.py                # Hardcoded PR dataset (3 PRs with ground truth)
    ├── graders.py                   # Task graders (easy/medium/hard)
    ├── reward.py                    # Step + terminal reward computation
    ├── app.py                       # FastAPI + WebSocket server
    ├── requirements.txt             # Docker deps (pinned versions)
    └── Dockerfile                   # Container definition
```

### 2.3 Episode Data Flow

```
reset() → loads a PR (diff + files + metadata) → returns initial Observation
    ↓
Agent reads files with read_file action → +0.02 reward per new file
    ↓
Agent posts comments with post_comment → +0.10–0.25 per correct issue found
    ↓
Agent checks for specific bug types → reward for detection rate
    ↓
Agent calls assign_score → episode ends, grader runs, terminal reward added
    ↓
state() → returns current review progress, files read, comments posted
```

---

## 3. Data Models (`models.py`)

### 3.1 Action Model

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    READ_FILE      = "read_file"
    GET_DIFF       = "get_diff"
    POST_COMMENT   = "post_comment"
    CHECK_LINT     = "check_lint"
    SEARCH_PATTERN = "search_pattern"
    ASSIGN_SCORE   = "assign_score"

class Action(BaseModel):
    action_type:  ActionType
    file_path:    Optional[str]   = None
    line_number:  Optional[int]   = None
    comment:      Optional[str]   = None
    score:        Optional[float] = Field(None, ge=0, le=10)
    summary:      Optional[str]   = None
    pattern:      Optional[str]   = None
```

| Action Type | Key Fields | Description |
|---|---|---|
| `read_file` | `file_path` | Read a file from the PR diff |
| `get_diff` | `file_path` (optional) | Get unified diff for a file or whole PR |
| `post_comment` | `file_path`, `line_number`, `comment` | Post an inline review comment |
| `check_lint` | `file_path` | Run static analysis on a file |
| `search_pattern` | `pattern`, `file_path` | Search for a pattern in code |
| `assign_score` | `score` (0–10), `summary` | Finalize review and end episode |

### 3.2 Observation Model

```python
class Observation(BaseModel):
    action_result:   str    # Output of the action (file content, diff, lint output, etc.)
    pr_metadata:     dict   # PR title, author, branch, description, changed_files list
    review_progress: dict   # files_read, comments_posted, issues_found counts
    reward:          float  # Step reward (partial progress signal)
    done:            bool   # True when assign_score is called or step_count > 50
    info:            dict   # episode_id, task_id, difficulty, grader_score (on done)
```

### 3.3 State Model

```python
class CodeReviewState(BaseModel):
    episode_id:          str
    task_id:             str        # "task_1" | "task_2" | "task_3"
    pr_data:             dict       # Ground truth PR with known bugs/issues
    files_read:          list[str]
    comments:            list[dict] # All posted {file, line, text}
    step_count:          int
    ground_truth_issues: list[dict] # Used by grader — NEVER exposed to agent via MCP
    total_reward:        float
```

---

## 4. Three Tasks with Graders

Each task uses a different **hardcoded PR** — deterministic, no external API deps, known ground truth.
All graders score **0.0–1.0** with clear, reproducible logic.

### Task 1: Style & Formatting Review (Easy)

**PR:** A Python PR with 5 intentional style violations:
- Missing docstrings on 2 functions
- `camelCase` naming instead of `snake_case` (2 instances)
- Line > 120 chars (1 instance)
- Unused import (1 instance)
- Missing type hints on public function (1 instance)

**Agent Objective:** Identify at least 4 of the 5 style issues, post comments on correct lines.

**Grader Logic:**
```
score = (correct_line_comments / total_known_issues) × 0.7
      + coverage_bonus × 0.3          # +0.3 if agent read ALL changed files
      - 0.05 × false_positive_count   # penalty per hallucinated issue
score = clamp(score, 0.0, 1.0)
```

**Expected baseline score:** 0.65–0.80

---

### Task 2: Logic Bug Detection (Medium)

**PR:** A JavaScript LRU cache implementation with 3 subtle logic bugs:

| Bug | Description | Grader Weight |
|---|---|---|
| Off-by-one | `>` should be `>=` in eviction condition | 0.25 |
| Race condition | Missing `await` in async lock handling | 0.40 |
| Edge case | Returns `undefined` instead of `null` for empty cache | 0.35 |

**Agent Objective:** Find all 3 bugs, post comments explaining each with a suggested fix, assign PR score ≤ 4 (reject).

**Grader Logic:**
```
score = sum(bug_weight for each correctly identified bug)
      + 0.10 if suggested fix is semantically correct (any bug)
      + 0.10 if assign_score ≤ 4 (correct rejection decision)
score = clamp(score, 0.0, 1.0)
```

**Expected baseline score:** 0.40–0.60

---

### Task 3: Security Vulnerability Audit (Hard)

**PR:** A Python web API with 4 security vulnerabilities:

| Vulnerability | Description | Severity |
|---|---|---|
| SQL injection | f-string formatting in raw query | CRITICAL |
| Insecure deserialization | User-supplied `pickle` data loaded directly | CRITICAL |
| Path traversal | Missing validation on file upload endpoint | HIGH |
| Hardcoded secret | API key committed in config file | MEDIUM |

**Agent Objective:** Identify all 4 vulnerabilities, classify severity correctly, suggest secure fixes, assign PR score = 0 (block merge).

**Grader Logic:**
```
score = sum(proportional score per found vulnerability)
      + 0.05 × correctly_classified_severities
      + 0.05 × semantically_valid_fix_suggestions
      + 0.15 if SQL injection found AND assign_score == 0
score = clamp(score, 0.0, 1.0)
```

**Expected baseline score:** 0.20–0.40

---

## 5. Reward Function (`server/reward.py`)

> **INVARIANT:** Reward computation must stay inside the environment boundary — server-side only, never exposed to client.

### 5.1 Step-Level Rewards

| Action | Reward | Rationale |
|---|---|---|
| `read_file` (new file) | +0.02 | Encourage full coverage of changed files |
| `read_file` (repeat) | 0.0 | No reward for redundant reads |
| `get_diff` | +0.01 | Small reward for examining changes |
| `post_comment` (correct line + issue) | +0.10 to +0.25 | Scaled by issue severity |
| `post_comment` (wrong / hallucinated) | -0.05 | Penalize false positives |
| `check_lint` (useful result returned) | +0.03 | Reward using available tools |
| `assign_score` (correct severity range) | +0.10 | Bonus for correct final judgment |
| Repeated identical action | -0.10 | Penalize loops |

### 5.2 Episode-End Terminal Reward

On `assign_score` (`done=True`), the grader runs:

```
terminal_reward      = grader_score × 0.5
total_episode_reward = accumulated_step_rewards + terminal_reward
```

Partial progress is always rewarded — even poor final grading doesn't erase step-level gains.

### 5.3 Undesirable Behavior Penalties

| Condition | Penalty |
|---|---|
| Same action repeated (same file, same line) | -0.10 |
| More than 50 steps without `assign_score` | -0.30 |
| `assign_score` with zero comments posted | -0.20 |

---

## 6. `server/code_review_environment.py`

```python
class CodeReviewEnvironment(Environment[Action, Observation, CodeReviewState]):

    def reset(self, seed=None, episode_id=None) -> Observation:
        # Pick task (from episode_id prefix or random)
        # Load PR data from pr_dataset.PR_DATASET[task_id]
        # Initialize blank state (files_read=[], comments=[], step_count=0)
        # Return initial Observation with pr_metadata, empty review_progress

    def step(self, action: Action) -> Observation:
        # Increment step_count
        # Route to handler based on action.action_type
        # Compute step reward via reward.py
        # Check done: assign_score called OR step_count > 50
        # Return Observation

    @property
    def state(self) -> CodeReviewState:
        return self._state

    # ── Action handlers (each returns (result_str, reward_float)) ──────────
    def _handle_read_file(self, action):      ...  # returns file content, +0.02 if first read
    def _handle_get_diff(self, action):       ...  # returns unified diff
    def _handle_post_comment(self, action):   ...  # checks line vs ground_truth_issues
    def _handle_check_lint(self, action):     ...  # runs simple pattern-based lint
    def _handle_search_pattern(self, action): ...  # searches file content
    def _handle_assign_score(self, action):   ...  # runs grader, sets done=True, stores grader_score in info
```

---

## 7. `server/pr_dataset.py`

Hardcode all 3 PRs as Python dicts. No external API calls ever.

```python
PR_DATASET = {
    "task_1": {
        "task_id":    "task_1",
        "title":      "Refactor auth utils",
        "difficulty": "easy",
        "files": {
            "auth/utils.py": "<full file content with style violations>",
            "auth/login.py": "<supporting file>",
        },
        "diff": "<unified diff string>",
        "ground_truth_issues": [
            {"file": "auth/utils.py", "line": 12, "type": "missing_docstring",  "severity": "low"},
            {"file": "auth/utils.py", "line": 24, "type": "camel_case_naming",  "severity": "low"},
            {"file": "auth/utils.py", "line": 37, "type": "line_too_long",      "severity": "low"},
            {"file": "auth/login.py", "line":  5, "type": "unused_import",      "severity": "low"},
            {"file": "auth/login.py", "line": 18, "type": "missing_type_hints", "severity": "low"},
        ]
    },
    "task_2": { ... },   # JS LRU cache — 3 logic bugs
    "task_3": { ... },   # Python API — 4 security vulns
}
```

---

## 8. `inference.py` — Baseline Script (EXACT HACKATHON FORMAT)

> **Non-negotiable rules from the hackathon sample script:**
> - Named `inference.py`, placed in **root directory**
> - Uses `OpenAI` client for ALL LLM calls — no other client
> - Reads env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `LOCAL_IMAGE_NAME`
> - Defaults: `API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"`
> - Defaults: `MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"`
> - Emits exactly `[START]`, `[STEP]`, `[END]` to stdout — no other formats accepted
> - `reward` and `rewards` → **2 decimal places**
> - `done` and `success` → lowercase `true` / `false`
> - `error` → raw error string or literal `null`
> - All fields on **a single line**, no newlines within a line
> - Score must be clamped to `[0, 1]`
> - `[END]` must always be emitted — even on exception (use `finally`)
> - `score` uses `.3f` in the sample — match exactly
> - Runtime < 20 min on 2 vCPU / 8 GB

### Exact stdout format (from hackathon sample):

```
[START] task=task_1 env=code-review-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=read_file reward=0.02 done=false error=null
[STEP] step=2 action=post_comment reward=0.15 done=false error=null
[STEP] step=3 action=assign_score reward=0.10 done=true error=null
[END] success=true steps=3 score=0.720 rewards=0.02,0.15,0.10
```

### Full `inference.py` (modelled exactly on hackathon sample):

```python
"""
inference.py — Code Review OpenEnv Baseline
Place in ROOT of repo. Run: python inference.py
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from code_review_env import CodeReviewEnvClient, Action, ActionType

# ── Env vars (match hackathon sample pattern exactly) ────────────────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")           # docker image name
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "code-review-env"
MAX_STEPS    = 30
TEMPERATURE  = 0.7
MAX_TOKENS   = 600
SUCCESS_SCORE_THRESHOLD = 0.3
TASKS        = ["task_1", "task_2", "task_3"]

# ── Logging helpers (exact hackathon format) ─────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Note: score uses .3f to match hackathon sample exactly
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert code reviewer. Review pull requests step by step.

    Available actions — respond with a single JSON object only, no markdown fences:
      {"action_type": "read_file",     "file_path": "path/to/file.py"}
      {"action_type": "get_diff",      "file_path": "path/to/file.py"}
      {"action_type": "post_comment",  "file_path": "path.py", "line_number": 12, "comment": "Issue: ..."}
      {"action_type": "check_lint",    "file_path": "path/to/file.py"}
      {"action_type": "search_pattern","pattern": "regex_pattern", "file_path": "path.py"}
      {"action_type": "assign_score",  "score": 7.5, "summary": "Overall review summary"}

    Review strategy:
    1. Call get_diff first to see what changed.
    2. read_file for each changed file.
    3. post_comment for every issue found (include file, exact line number, explanation).
    4. assign_score when done reviewing (0=block merge, 10=approve without changes).

    Respond ONLY with a JSON object — no explanation, no markdown.
""").strip()


def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Task: {obs.get('info', {}).get('task_id', 'unknown')}
        PR Title: {obs.get('pr_metadata', {}).get('title', 'Unknown')}
        Changed files: {obs.get('pr_metadata', {}).get('changed_files', [])}
        Last action result:
        {obs.get('action_result', '')[:800]}
        Review progress: {obs.get('review_progress', {})}
        Previous steps:
        {history_block}
        What is your next action?
    """).strip()


def get_next_action(client: OpenAI, step: int, obs: dict, history: List[str]) -> Action:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model adds them
        raw = raw.strip("```json").strip("```").strip()
        data = json.loads(raw)
        return Action(**data)
    except Exception as exc:
        print(f"[DEBUG] Action parse failed: {exc}", flush=True)
        # Safe fallback — always valid
        return Action(action_type=ActionType.GET_DIFF)


# ── Per-task runner (mirrors hackathon sample structure exactly) ──────────────

async def run_task(client: OpenAI, task_id: str) -> None:
    env = await CodeReviewEnvClient.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env.reset(episode_id=task_id)   # OpenEnv .reset()
        obs        = result.observation.model_dump()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_next_action(client, step, obs, history)
            result = await env.step(action)                  # OpenEnv .step()
            obs    = result.observation.model_dump()

            reward = result.reward or 0.0
            done   = result.done
            error  = obs.get("info", {}).get("error")        # None → "null" in log

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.action_type!r} -> reward {reward:+.2f}")

            if done:
                # grader_score injected into info by _handle_assign_score()
                score = obs.get("info", {}).get("grader_score", 0.0)
                break

        score   = float(min(max(score, 0.0), 1.0))   # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. `server/Dockerfile`

The validator script (`validate-submission.sh`) looks for Dockerfile in **repo root OR `server/`**. Put it in `server/`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Layer cache — deps first
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# HF Spaces requires exactly port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### `server/requirements.txt` (pin everything)

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
websockets==12.0
openenv-core
```

**Test locally before pushing:**
```bash
docker build -t code-review-env -f server/Dockerfile .
docker run -p 7860:7860 code-review-env
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  http://localhost:7860/reset
# Must print: 200
```

---

## 10. `openenv.yaml`

```yaml
name: code-review-env
version: 0.1.0
description: "AI agent reviews pull requests: detects bugs, suggests fixes, assigns scores"
author: your-hf-username
tags:
  - openenv
  - code-review
  - software-engineering
  - security
entry_point: server/app.py
port: 7860
tasks:
  - id: task_1
    name: "Style & Formatting Review"
    difficulty: easy
    max_steps: 30
  - id: task_2
    name: "Logic Bug Detection"
    difficulty: medium
    max_steps: 40
  - id: task_3
    name: "Security Vulnerability Audit"
    difficulty: hard
    max_steps: 50
```

---

## 11. Pre-Submission Validation

The hackathon provides `validate-submission.sh`. Run it **before submitting**:

```bash
# Download from hackathon repo, then:
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space ./code_review_env

# Or pipe directly:
curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh \
  | bash -s -- https://your-space.hf.space .
```

**What the script checks (all 3 must pass — any failure stops and disqualifies):**

| Step | Check | Pass Condition | Timeout |
|---|---|---|---|
| 1/3 | HF Space live | `POST /reset` returns HTTP 200 | 30s |
| 2/3 | Docker build | `docker build` exits 0 | 600s |
| 3/3 | openenv validate | `openenv validate` exits 0 | — |

> The script looks for Dockerfile in repo root first, then `server/` — both work.

**Install prerequisites:**
```bash
pip install openenv-core
# Docker: https://docs.docker.com/get-docker/
```

### Full Pre-Submission Checklist

| Check | Requirement | How We Meet It |
|---|---|---|
| HF Space deploys | POST `/reset` returns 200 | FastAPI on port 7860, `/reset` endpoint |
| OpenEnv spec | `openenv.yaml` + typed models + endpoints | Full spec compliance |
| Dockerfile builds | `docker build` succeeds | Tested locally, versions pinned |
| Baseline reproduces | `inference.py` runs, produces scores | All 3 tasks complete without error |
| 3+ tasks with graders | Scores in `[0.0, 1.0]` | style / logic / security tasks |
| `API_BASE_URL` | Env var defined | HF Space secret |
| `MODEL_NAME` | Env var defined | HF Space secret |
| `HF_TOKEN` | Env var defined | HF Space secret |
| `LOCAL_IMAGE_NAME` | Env var defined | HF Space secret |
| `inference.py` in root | Root directory only | ✅ |
| OpenAI client | All LLM calls via `openai.OpenAI` | ✅ |
| Exact log format | `[START]`/`[STEP]`/`[END]` single lines | ✅ matched to sample |
| `reward` → 2 decimal | `{reward:.2f}` | ✅ |
| `score` → 3 decimal | `{score:.3f}` | ✅ matched to sample |
| `done`/`success` lowercase | `str(done).lower()` | ✅ |
| `[END]` always emitted | `finally:` block | ✅ |
| Runtime < 20 min | ~6 min/task × 3 = ~18 min | `MAX_STEPS=30` |
| Scores in `[0, 1]` | `min(max(score, 0.0), 1.0)` | ✅ |

---

## 12. Alignment with OpenEnv Invariants

| Invariant | How We Comply |
|---|---|
| Gymnasium API signatures | `reset(seed?, episode_id?) → Observation`, `step(action) → Observation`, `state → State` |
| Generic type safety | `EnvClient[Action, Observation, State]` and `Environment[Action, Observation, State]` |
| Pydantic serialization | All models extend `BaseModel`, JSON-compatible |
| Agent isolation | Agent cannot call `reset()` — only gets MCP tools |
| Rewards inside environment | `reward.py` is server-side only |
| Client-server separation | `client.py` never imports from `server/` |
| One env = one trajectory | No multiplexing — one PR per episode |

---

## 13. Execution Order for Claude Code

Run in **strict order** — each step depends on the previous:

```
1.  models.py
        ↳ Define all Pydantic types first — everything else imports from here

2.  server/pr_dataset.py
        ↳ Hardcode 3 PRs as Python dicts with ground_truth_issues
        ↳ Make the bugs realistic — real code, real line numbers

3.  server/graders.py  ← TDD: write tests BEFORE implementation
        ↳ grade_style_review(state) → float [0,1]
        ↳ grade_logic_bugs(state)   → float [0,1]
        ↳ grade_security_audit(state) → float [0,1]
        ↳ All must be deterministic and reproducible

4.  server/reward.py
        ↳ compute_step_reward(action, state) → float
        ↳ compute_terminal_reward(grader_score) → float = grader_score × 0.5
        ↳ Write unit tests for each reward case

5.  server/code_review_environment.py
        ↳ Implement reset() / step() / state property
        ↳ Wire all action handlers → reward.py
        ↳ _handle_assign_score() must call grader and store grader_score in info

6.  server/app.py
        ↳ create_app(env, Action, Observation) via openenv-core
        ↳ Verify /reset, /step, /state all respond

7.  client.py
        ↳ CodeReviewEnvClient(EnvClient[Action, Observation, CodeReviewState])
        ↳ _step_payload() and _parse_result() methods

8.  openenv.yaml
        ↳ name, version, tasks (3), entry_point, port: 7860

9.  server/requirements.txt
        ↳ Pin all dep versions

10. server/Dockerfile
        ↳ FROM python:3.11-slim, EXPOSE 7860
        ↳ Test: docker build -t code-review-env -f server/Dockerfile .
        ↳ Test: docker run -p 7860:7860 code-review-env
        ↳ Test: curl -X POST -d '{}' http://localhost:7860/reset → 200

11. inference.py  (root level)
        ↳ Exact [START]/[STEP]/[END] format — match hackathon sample byte-for-byte
        ↳ score uses .3f, reward uses .2f
        ↳ [END] in finally block — always emitted
        ↳ Test locally against running container

12. openenv validate
        ↳ Fix any issues before continuing

13. ./validate-submission.sh https://your-space.hf.space .
        ↳ All 3 steps must pass — if any fail, fix before submitting

14. Deploy to HF Space
        ↳ Tag with "openenv"
        ↳ Set secrets: API_BASE_URL, MODEL_NAME, HF_TOKEN, LOCAL_IMAGE_NAME

15. Run inference.py against live HF Space URL
        ↳ Record actual baseline scores for all 3 tasks

16. README.md
        ↳ Fill in real baseline scores from step 15
        ↳ Include: env description, action/obs space, task descriptions, setup instructions
```

---

## 14. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Grader edge cases (correct fix, wrong format) | Medium | Normalize before grading, semantic matching not exact string |
| `validate-submission.sh` Step 1 fails | Low | Test `/reset` with curl locally before pushing to HF Space |
| Inference script > 20 min | Medium | `MAX_STEPS=30`, `MAX_TOKENS=600`, sequential tasks ~6 min each |
| LLM outputs non-JSON / markdown-wrapped JSON | Medium | Strip fences, `try/except` with `GET_DIFF` fallback action |
| HF Space OOM on 8GB | Low | Pure Python + FastAPI, hardcoded PRs in memory, no heavy ML libs |
| `openenv validate` fails | Low | Run locally during dev, not just at submission time |
| `[END]` not emitted on exception | Low | Always use `finally:` block — matches hackathon sample pattern |

---

## 15. Projected Evaluation Score

| Criterion | Weight | Our Score | Rationale |
|---|---|---|---|
| Real-world utility | 30% | 26–28 / 30 | Code review is universal, immediate production value |
| Task & grader quality | 25% | 21–23 / 25 | 3 well-defined tasks, deterministic graders, clear difficulty range |
| Environment design | 20% | 16–18 / 20 | Dense reward, clean episodes, sensible action/obs space |
| Code quality & spec | 15% | 13–14 / 15 | Full OpenEnv spec, tested, Dockerfile works |
| Creativity & novelty | 10% | 8–9 / 10 | First code review env in OpenEnv, clever multi-step reward design |

**Projected Total: 84–92 / 100**


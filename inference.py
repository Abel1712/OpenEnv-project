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
from client import CodeReviewEnvClient
from models import Action, ActionType

# ── Env vars ─────────────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "code-review-env"
MAX_STEPS    = 30
TEMPERATURE  = 0.7
MAX_TOKENS   = 600
SUCCESS_SCORE_THRESHOLD = 0.3
TASKS        = ["task_1", "task_2", "task_3"]

# ── Logging (exact hackathon format) ─────────────────────────────────────────

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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
        raw = raw.strip("```json").strip("```").strip()
        data = json.loads(raw)
        return Action(**data)
    except Exception as exc:
        print(f"[DEBUG] Action parse failed: {exc}", flush=True)
        return Action(action_type=ActionType.GET_DIFF)


# ── Per-task runner ───────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str) -> None:
    env = await CodeReviewEnvClient.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(episode_id=task_id)
        obs    = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_next_action(client, step, obs, history)
            result = await env.step(action)
            obs    = result.observation.model_dump()

            reward = result.reward or 0.0
            done   = result.done
            error  = obs.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.action_type!r} -> reward {reward:+.2f}")

            if done:
                score = obs.get("info", {}).get("grader_score", 0.0)
                break

        score   = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())

"""
inference.py — Code Review OpenEnv Baseline
Place in ROOT of repo. Run: python inference.py
"""

import asyncio
import os
import textwrap
import json
from typing import List, Optional, Set, Dict

from openai import OpenAI
from client import CodeReviewEnvClient
from models import Action, ActionType

# ── Env vars ─────────────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.getenv("ENV_URL", "https://abeljames-code-review-env.hf.space")
BENCHMARK    = "code-review-env"
MAX_STEPS    = 20          # 3 tasks × 20 steps × ~10s/call ≈ 10 min — safely under 20 min limit
TEMPERATURE  = 0.7
MAX_TOKENS   = 900           # FIX 1: was 600 — too tight for comment+summary JSON
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
    1. get_diff — see what changed, note exact line numbers shown in the diff.
    2. read_file — read each changed file ONCE to confirm exact line numbers (line 1 = first line of file).
    3. post_comment — ONLY comment on lines you have SEEN in file output. Use the EXACT
       line number from the file content. Never guess. Each comment must reference a real
       issue on that specific line.
    4. assign_score — call this after posting all comments. Score guide:
       0-3: block merge (security/critical), 4-6: needs changes, 7-10: approve.

    CRITICAL RULES:
    - Never read a file you have already read — check "Files already read" before read_file.
    - Never post duplicate comments on the same file+line.
    - Never post a comment unless you have READ the file and confirmed the issue on that line.
    - Call assign_score within 15 steps — do not loop forever.
    - Respond ONLY with a JSON object — no explanation, no markdown fences.
""").strip()


def build_user_prompt(
    step: int,
    obs: dict,
    history: List[str],
    files_read: Set[str],
    comments_posted: Set[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    steps_left = MAX_STEPS - step
    urgency = " — CALL assign_score NOW" if steps_left <= 2 else (
              " — finish comments and call assign_score soon" if steps_left <= 6 else "")

    # FIX 2: Raise truncation limit so model sees enough content to identify real line numbers
    action_result = obs.get('action_result', '')
    action_result_preview = action_result[:2500] if len(action_result) > 2500 else action_result

    # FIX 3: Expose files_read and comments_posted so model avoids re-reads and duplicates
    files_read_str    = ", ".join(sorted(files_read))    if files_read    else "none"
    comments_str      = ", ".join(sorted(comments_posted)) if comments_posted else "none"

    return textwrap.dedent(f"""
        Step: {step}/{MAX_STEPS} (steps remaining: {steps_left}{urgency})
        Task: {obs.get('info', {}).get('task_id', 'unknown')}
        PR Title: {obs.get('pr_metadata', {}).get('title', 'Unknown')}
        Changed files: {obs.get('pr_metadata', {}).get('changed_files', [])}
        Files already read (do NOT re-read these): {files_read_str}
        Comments already posted (file:line — do NOT duplicate): {comments_str}
        Last action result:
        {action_result_preview}
        Review progress: {obs.get('review_progress', {})}
        Previous steps:
        {history_block}
        What is your next action?
    """).strip()


def get_next_action(
    client: OpenAI,
    step: int,
    obs: dict,
    history: List[str],
    files_read: Set[str],
    comments_posted: Set[str],
) -> Action:
    user_prompt = build_user_prompt(step, obs, history, files_read, comments_posted)
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
        # FIX 4: More robust JSON fence stripping (handles ```json\n...\n``` cleanly)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        return Action(**data)
    except Exception as exc:
        print(f"[DEBUG] Action parse failed: {exc}", flush=True)
        return Action(action_type=ActionType.GET_DIFF)


# ── Per-task runner ───────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str) -> None:
    history: List[str]          = []
    rewards: List[float]        = []
    files_read: Set[str]        = set()
    comments_posted: Set[str]   = set()
    steps_taken = 0
    score   = 0.0
    success = False
    env     = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Use local Docker image when validator provides LOCAL_IMAGE_NAME;
        # fall back to the hosted HF Space URL otherwise.
        if IMAGE_NAME:
            env = await CodeReviewEnvClient.from_docker_image(IMAGE_NAME)
        else:
            env = CodeReviewEnvClient(base_url=ENV_URL)

        result = await env.reset(episode_id=task_id)
        obs    = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_next_action(client, step, obs, history, files_read, comments_posted)
            result = await env.step(action)
            obs    = result.observation.model_dump()

            reward = result.reward or 0.0
            done   = result.done
            error  = obs.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.action_type.value, reward=reward, done=done, error=error)

            if action.action_type == ActionType.READ_FILE and action.file_path:
                files_read.add(action.file_path)

            if action.action_type == ActionType.POST_COMMENT and action.file_path and action.line_number:
                comments_posted.add(f"{action.file_path}:{action.line_number}")

            detail = ""
            if action.file_path:
                detail += f" file={action.file_path}"
            if action.line_number:
                detail += f" line={action.line_number}"
            if action.comment:
                detail += f" comment='{action.comment[:60]}'"
            history.append(
                f"Step {step}: {action.action_type.value}{detail} -> reward {reward:+.2f}"
            )

            if done:
                score = obs.get("info", {}).get("grader_score", 0.0)
                break

        # Fallback: if done was never True, try to read grader_score from final obs
        if score == 0.0:
            score = float(obs.get("info", {}).get("grader_score", 0.0))

        score   = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task({task_id}) error: {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        try:
            await run_task(client, task_id)
        except Exception as exc:
            print(f"[DEBUG] Unhandled error for {task_id}: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
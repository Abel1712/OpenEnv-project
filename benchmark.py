"""
benchmark.py — Local benchmarking harness for inference.py

Usage:
    python benchmark.py                          # build image + run all tasks
    python benchmark.py --image code-review-env  # skip build, use existing image
    python benchmark.py --skip-docker            # run against ENV_URL (HF Space / local server)

Requires:
    - Docker installed and running  (unless --skip-docker)
    - HF_TOKEN or API_KEY env var set for the LLM
    - API_BASE_URL / MODEL_NAME optionally set (defaults match inference.py)
"""

import argparse
import os
import re
import subprocess
import sys
import time

# ── Constants ────────────────────────────────────────────────────────────────

IMAGE_TAG          = "code-review-env-bench"
DOCKERFILE_PATH    = "Dockerfile"           # root Dockerfile (HF Space style)
TASKS              = ["task_1", "task_2", "task_3"]
SUCCESS_THRESHOLD  = 0.3

# Regex patterns for each log line type
RE_START = re.compile(
    r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>\S+)$"
)
RE_STEP = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>\S+) "
    r"reward=(?P<reward>-?\d+\.\d{2}) done=(?P<done>true|false) error=(?P<error>.+)$"
)
RE_END = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) "
    r"score=(?P<score>\d+\.\d{3}) rewards=(?P<rewards>.*)$"
)


# ── Docker helpers ───────────────────────────────────────────────────────────

def build_image(tag: str) -> bool:
    print(f"[benchmark] Building Docker image '{tag}' ...")
    result = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[benchmark] ERROR: docker build failed.", file=sys.stderr)
        return False
    print(f"[benchmark] Image '{tag}' built successfully.\n")
    return True


# ── Output parser ────────────────────────────────────────────────────────────

class TaskResult:
    def __init__(self, task_id: str):
        self.task_id    = task_id
        self.score      = 0.0
        self.steps      = 0
        self.success    = False
        self.rewards: list[float] = []
        self.format_errors: list[str] = []
        self.start_seen = False
        self.end_seen   = False

    @property
    def valid_format(self) -> bool:
        return len(self.format_errors) == 0 and self.start_seen and self.end_seen


def parse_output(stdout: str) -> list[TaskResult]:
    results: list[TaskResult] = []
    current: TaskResult | None = None

    for lineno, raw_line in enumerate(stdout.splitlines(), 1):
        line = raw_line.strip()

        # Ignore debug lines
        if line.startswith("[DEBUG]"):
            continue

        if line.startswith("[START]"):
            m = RE_START.match(line)
            if not m:
                if current:
                    current.format_errors.append(f"line {lineno}: malformed [START]: {line!r}")
                continue
            current = TaskResult(m.group("task"))
            current.start_seen = True
            results.append(current)

        elif line.startswith("[STEP]"):
            if current is None:
                continue
            m = RE_STEP.match(line)
            if not m:
                current.format_errors.append(f"line {lineno}: malformed [STEP]: {line!r}")
                continue
            # Validate action is a plain snake_case string (not an enum repr)
            action = m.group("action")
            if action.startswith("ActionType.") or "<" in action:
                current.format_errors.append(
                    f"line {lineno}: action is enum object, not string: {action!r}"
                )
            # Validate reward has exactly 2 decimal places
            reward_str = m.group("reward")
            if not re.match(r"^-?\d+\.\d{2}$", reward_str):
                current.format_errors.append(
                    f"line {lineno}: reward not 2 d.p.: {reward_str!r}"
                )
            # Validate done is lowercase
            done_str = m.group("done")
            if done_str not in ("true", "false"):
                current.format_errors.append(
                    f"line {lineno}: done must be 'true' or 'false', got {done_str!r}"
                )

        elif line.startswith("[END]"):
            if current is None:
                continue
            m = RE_END.match(line)
            if not m:
                current.format_errors.append(f"line {lineno}: malformed [END]: {line!r}")
                continue
            current.end_seen   = True
            current.success    = m.group("success") == "true"
            current.steps      = int(m.group("steps"))
            current.score      = float(m.group("score"))
            # Validate score has exactly 3 decimal places
            score_str = m.group("score")
            if not re.match(r"^\d+\.\d{3}$", score_str):
                current.format_errors.append(
                    f"line {lineno}: score not 3 d.p.: {score_str!r}"
                )
            # Parse individual rewards
            rewards_raw = m.group("rewards")
            if rewards_raw:
                for r in rewards_raw.split(","):
                    try:
                        current.rewards.append(float(r))
                    except ValueError:
                        current.format_errors.append(
                            f"line {lineno}: bad reward value in rewards list: {r!r}"
                        )

    return results


# ── Reporter ─────────────────────────────────────────────────────────────────

def print_report(results: list[TaskResult], elapsed: float) -> int:
    """Prints the benchmark report. Returns exit code (0=all pass, 1=failures)."""
    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)

    any_failure = False

    for r in results:
        status = "PASS" if r.success else "FAIL"
        fmt    = "OK" if r.valid_format else "FORMAT ERR"
        print(
            f"  {r.task_id:<8}  score={r.score:.3f}  steps={r.steps:<3}  "
            f"success={str(r.success).lower():<5}  [{status}]  format=[{fmt}]"
        )
        for err in r.format_errors:
            print(f"             !! {err}")
        if not r.end_seen:
            print(f"             !! [END] line never emitted — inference.py likely crashed")
        if not r.success or not r.valid_format:
            any_failure = True

    if results:
        avg = sum(r.score for r in results) / len(results)
        passing = sum(1 for r in results if r.success)
        print()
        print(f"  Tasks passed : {passing}/{len(results)}")
        print(f"  Average score: {avg:.3f}")
        print(f"  Elapsed time : {elapsed:.1f}s")
    else:
        print("  No task results found — inference.py produced no [START]/[END] lines.")
        any_failure = True

    print("=" * 60)
    return 1 if any_failure else 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference.py locally")
    parser.add_argument("--image",        default=IMAGE_TAG,
                        help="Docker image tag to use (default: builds a fresh one)")
    parser.add_argument("--skip-build",   action="store_true",
                        help="Skip docker build, use existing image")
    parser.add_argument("--skip-docker",  action="store_true",
                        help="Don't use Docker at all — run against ENV_URL")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print full inference.py stdout")
    args = parser.parse_args()

    # ── 1. Prepare Docker image ───────────────────────────────────────────────
    image_name = None
    if not args.skip_docker:
        image_name = args.image
        if not args.skip_build:
            if not build_image(image_name):
                return 1
        else:
            print(f"[benchmark] Using existing image '{image_name}'\n")

    # ── 2. Build env for inference.py ────────────────────────────────────────
    env = os.environ.copy()
    if image_name:
        env["LOCAL_IMAGE_NAME"] = image_name
    # Ensure LLM credentials are forwarded
    for key in ("HF_TOKEN", "API_KEY", "API_BASE_URL", "MODEL_NAME"):
        if key in os.environ:
            env[key] = os.environ[key]

    # ── 3. Run inference.py ──────────────────────────────────────────────────
    print("[benchmark] Running inference.py ...")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.time() - start

    stdout = proc.stdout
    stderr = proc.stderr

    if args.verbose or proc.returncode != 0:
        print("\n── inference.py stdout ──────────────────────────────────")
        print(stdout or "(empty)")
        if stderr:
            print("── inference.py stderr ──────────────────────────────────")
            print(stderr)
        print("─────────────────────────────────────────────────────────\n")

    if proc.returncode != 0:
        print(f"[benchmark] ERROR: inference.py exited with code {proc.returncode}")

    # ── 4. Parse + report ────────────────────────────────────────────────────
    results = parse_output(stdout)
    return print_report(results, elapsed)


if __name__ == "__main__":
    sys.exit(main())

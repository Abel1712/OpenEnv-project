# Claude Code — Prompt to Paste at Session Start
# ================================================
# Copy everything between the === lines and paste into Claude Code

==========================================================================

You are building a complete OpenEnv environment called "Code Review Env" for a hackathon.
All project documentation is in CodeReview_OpenEnv_ProjectPlan.md in this repo. Read it first before doing anything.

## Your identity for this project
- HF username: AbelJames
- HF Space URL: https://abeljames-code-review-env.hf.space
- GitHub repo: https://github.com/Abel1712/OpenEnv-project
- Python: 3.11
- OS: Windows (use forward slashes in paths, avoid Linux-only commands)

## Environment variables (already set in HF Space secrets)
- HF_TOKEN: set in HF Space secrets (never hardcode)
- API_BASE_URL: https://router.huggingface.co/v1
- MODEL_NAME: Qwen/Qwen2.5-72B-Instruct
- LOCAL_IMAGE_NAME: code-review-env
- HF_SPACE_URL: https://abeljames-code-review-env.hf.space

## Your first action
1. Read PLAN.md completely
2. Read INVARIANTS.md completely  
3. Read PATTERNS.md completely
4. Then start Step 1 of the execution order

## Execution order — follow strictly, do not skip steps
1.  models.py — all Pydantic types (Action, Observation, CodeReviewState)
2.  server/pr_dataset.py — hardcode 3 PRs with realistic code and ground_truth_issues
3.  server/graders.py — TDD: write failing tests FIRST, then implement graders
4.  server/reward.py — step rewards + terminal reward (grader_score × 0.5)
5.  server/code_review_environment.py — reset() / step() / state property
6.  server/app.py — wire to FastAPI via create_app()
7.  client.py — CodeReviewEnvClient extending EnvClient
8.  openenv.yaml — manifest with all 3 tasks
9.  server/requirements.txt — pin ALL dependency versions
10. server/Dockerfile — FROM python:3.11-slim, EXPOSE 7860
11. inference.py — exact [START]/[STEP]/[END] stdout format (see PLAN.md Section 8)
12. Run: openenv validate — fix any issues
13. Test Docker locally:
    docker build -t code-review-env -f server/Dockerfile .
    docker run -p 7860:7860 code-review-env
    curl -X POST -H "Content-Type: application/json" -d "{}" http://localhost:7860/reset
14. README.md — fill in after baseline scores are recorded

## Critical rules — never violate these
- NEVER expose reset()/step()/state() to the agent via MCP tools
- NEVER import server/ code from client.py
- ALWAYS compute rewards inside the environment (server-side only)
- ALWAYS use Environment[Action, Observation, CodeReviewState] generics
- ALWAYS use EnvClient[Action, Observation, CodeReviewState] generics
- ALL wire types must be Pydantic BaseModel
- inference.py MUST use openai.OpenAI client — no other LLM client
- inference.py MUST be in the ROOT directory
- [END] log MUST be in a finally: block — always emitted even on exception
- score uses {score:.3f}, reward uses {reward:.2f} — exact hackathon format
- Docker MUST expose port 7860
- All grader scores MUST be clamped: min(max(score, 0.0), 1.0)

## TDD enforcement for graders.py (Step 3)
Write tests FIRST in tests/test_graders.py before writing any grader code.
Each grader must:
- Return float in [0.0, 1.0]
- Be fully deterministic (same input → same output always)
- Have at least 3 test cases: min score, partial score, max score

## ALIGNMENT FLAGS
If any change would violate INVARIANTS.md, stop immediately and output:
**ALIGNMENT FLAG**: [description]
- **Invariant at risk**: [which one]
- **The concern**: [what would be violated]
Do not proceed until confirmed.

## PR Dataset requirements (server/pr_dataset.py)
Write REAL, realistic code for all 3 PRs — not placeholder strings.

Task 1 PR must contain actual Python code with these exact violations:
- Missing docstring on a function (pick a real line number)
- camelCase variable name instead of snake_case
- Line over 120 characters
- Unused import at top of file
- Missing type hint on a public function

Task 2 PR must contain actual JavaScript LRU cache code with:
- Off-by-one bug: > should be >= in size check
- Missing await on an async call
- Returns undefined instead of null for empty cache lookup

Task 3 PR must contain actual Python Flask/FastAPI code with:
- SQL query built with f-string (SQL injection)
- pickle.loads() called on user input (insecure deserialization)
- File path from user input with no validation (path traversal)
- Hardcoded API key string in the file

## When you finish each step
After completing each numbered step, output:
✅ Step N complete: [one line summary of what was built]
Then move immediately to the next step without waiting.

## When everything is built
Run the validate-submission checklist from PLAN.md Section 11.
Then output a summary:
- All files created
- openenv validate result
- Docker build result  
- curl /reset result
- What still needs manual action (HF Space deployment)

==========================================================================

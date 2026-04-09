"""
CodeReviewEnvClient — connects to the running server container.

INVARIANT: Never import from server/ directory.
All shared types come from models.py only.
"""
from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from models import Action, ActionType, CodeReviewState, Observation


class CodeReviewEnvClient(EnvClient[Action, Observation, CodeReviewState]):

    def _step_payload(self, action: Action) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})
        obs_data.setdefault("reward", 0.0)
        obs_data.setdefault("done", False)
        obs = Observation(**obs_data)
        # Prefer top-level done/reward if present (some openenv versions put them there)
        done   = payload.get("done",   obs.done)
        reward = payload.get("reward", obs.reward)
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> CodeReviewState:
        return CodeReviewState(**payload)
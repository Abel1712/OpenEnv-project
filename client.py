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
        obs = Observation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict) -> CodeReviewState:
        return CodeReviewState(**payload)

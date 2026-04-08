"""
FastAPI server — wires CodeReviewEnvironment to the OpenEnv HTTP API.
"""
import uvicorn
from openenv.core.env_server import create_app

from models import Action, Observation
from server.code_review_environment import CodeReviewEnvironment

app = create_app(CodeReviewEnvironment, Action, Observation)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

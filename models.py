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


class Observation(BaseModel):
    action_result:   str
    pr_metadata:     dict
    review_progress: dict
    reward:          float = 0.0
    done:            bool  = False
    info:            dict


class CodeReviewState(BaseModel):
    episode_id:          str
    task_id:             str
    pr_data:             dict
    files_read:          list[str]
    comments:            list[dict]
    step_count:          int
    ground_truth_issues: list[dict]
    total_reward:        float
    assigned_score:      Optional[float] = None
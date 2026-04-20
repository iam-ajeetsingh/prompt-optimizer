from typing import List
from pydantic import BaseModel, field_validator


class RunRequest(BaseModel):
    task_description: str
    model_target: str = "google/gemini-2.5-flash-lite"
    constraints: List[str] = []
    iteration_max: int = 3


class ReviewRequest(BaseModel):
    thread_id: str
    decision: str
    feedback: str = ""

    @field_validator("decision")
    @classmethod
    def decision_must_be_valid(cls, v: str) -> str:
        if v not in ("approve", "retry"):
            raise ValueError("decision must be 'approve' or 'retry'")
        return v


class RunResponse(BaseModel):
    thread_id: str
    variants: List[str]
    eval_scores: List[dict]
    critique_results: List[dict]
    status: str = "awaiting_review"

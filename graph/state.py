from typing import TypedDict, Optional, List


class PromptWorkspaceState(TypedDict):
    task_description: str
    model_target: str
    constraints: List[str]
    prompt_variants: List[str]
    critique_results: List[dict]
    eval_scores: List[dict]
    human_feedback: Optional[str]
    human_decision: str
    winning_prompt: Optional[str]
    iteration_count: int
    iteration_max: int

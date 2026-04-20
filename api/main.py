"""FastAPI app exposing /run and /review endpoints for the prompt optimizer graph."""
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI

from api.schemas import RunRequest, ReviewRequest, RunResponse
from graph.builder import graph

load_dotenv()

app = FastAPI(title="Prompt Optimizer")


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    """Start a new optimization run and pause at the first HITL checkpoint."""
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "task_description": request.task_description,
        "model_target": request.model_target,
        "constraints": request.constraints,
        "iteration_max": request.iteration_max,
        "human_feedback": None,
        "human_decision": "",
        "winning_prompt": None,
        "iteration_count": 0,
    }

    graph.invoke(initial_state, config=config)
    snapshot = graph.get_state(config)

    return RunResponse(
        thread_id=thread_id,
        variants=snapshot.values["prompt_variants"],
        eval_scores=snapshot.values["eval_scores"],
        critique_results=snapshot.values["critique_results"],
    )


@app.post("/review")
def review(request: ReviewRequest):
    """Submit a human review decision and resume the graph from the HITL checkpoint."""
    config = {"configurable": {"thread_id": request.thread_id}}

    snapshot = graph.get_state(config)

    graph.update_state(config, {
        "human_decision": request.decision,
        "human_feedback": request.feedback,
        "iteration_count": snapshot.values["iteration_count"] + 1,
    })

    graph.invoke(None, config=config)
    snapshot = graph.get_state(config)

    if snapshot.values.get("winning_prompt"):
        return {"status": "complete", "winning_prompt": snapshot.values["winning_prompt"]}

    return RunResponse(
        thread_id=request.thread_id,
        variants=snapshot.values["prompt_variants"],
        eval_scores=snapshot.values["eval_scores"],
        critique_results=snapshot.values["critique_results"],
        status="awaiting_review",
    )

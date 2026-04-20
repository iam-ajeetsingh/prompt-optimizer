from pathlib import Path
from typing import List

#from langchain_anthropic import ChatAnthropic
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel

from graph.state import PromptWorkspaceState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluator.txt"

RUBRIC: dict[str, float] = {
    "clarity": 0.25,
    "specificity": 0.20,
    "robustness": 0.25,
    "efficiency": 0.15,
    "alignment": 0.15,
}


class VariantScores(BaseModel):
    variant_index: int
    scores: dict[str, int]


class EvaluatorOutput(BaseModel):
    evaluations: List[VariantScores]


#_LLM = ChatAnthropic(model="claude-sonnet-4-5").with_structured_output(EvaluatorOutput)
#_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(EvaluatorOutput)

#_LLM = ChatOpenAI(
#    model="google/gemini-2.5-flash-lite",
#    openai_api_key=os.environ["OPENROUTER_API_KEY"],
#    openai_api_base="https://openrouter.ai/api/v1",
#).with_structured_output(EvaluatorOutput)

_LLM = None

def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model="google/gemini-2.5-flash-lite",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        ).with_structured_output(EvaluatorOutput)
    return _LLM

def _compute_composite(scores: dict[str, int]) -> float:
    """Compute weighted composite score from dimension scores and RUBRIC weights."""
    return round(sum(scores[dim] * weight for dim, weight in RUBRIC.items()), 2)


def evaluator_node(state: PromptWorkspaceState) -> dict:
    """Score each prompt variant on 5 dimensions, then compute a weighted composite.

    Args:
        state: current graph state; reads prompt_variants and critique_results.

    Returns:
        dict with key 'eval_scores' containing a list of score dicts, each with
        variant_index, scores (5 dimensions), and composite (float).
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    critique_map: dict[int, list] = {
        c["variant_index"]: c["weaknesses"]
        for c in state["critique_results"]
    }

    sections = []
    for i, variant in enumerate(state["prompt_variants"]):
        weaknesses = critique_map.get(i, [])
        weaknesses_block = "\n".join(f"  - {w}" for w in weaknesses) or "  None identified"
        sections.append(
            f"Variant {i}:\n{variant}\n\nCritique weaknesses:\n{weaknesses_block}"
        )

    user_message = "\n\n---\n\n".join(sections)

    result: EvaluatorOutput = _get_llm().invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return {
        "eval_scores": [
            e.model_dump() | {"composite": _compute_composite(e.scores)}
            for e in result.evaluations
        ]
    }

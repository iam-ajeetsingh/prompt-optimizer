from pathlib import Path
from typing import List

#from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel

from graph.state import PromptWorkspaceState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "optimizer.txt"


class OptimizerOutput(BaseModel):
    final_prompt: str


#_LLM = ChatAnthropic(model="claude-sonnet-4-5").with_structured_output(OptimizerOutput)
#_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(OptimizerOutput)

#_LLM = ChatOpenAI(
#    model="google/gemini-2.5-flash-lite",
#    openai_api_key=os.environ["OPENROUTER_API_KEY"],
#    openai_api_base="https://openrouter.ai/api/v1",
#).with_structured_output(OptimizerOutput)


_LLM = None

def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model="google/gemini-2.5-flash-lite",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        ).with_structured_output(OptimizerOutput)
    return _LLM



def _find_best_variant(prompt_variants: List[str], eval_scores: List[dict]) -> str:
    """Return the prompt variant with the highest composite eval score.

    Args:
        prompt_variants: list of prompt strings indexed by position.
        eval_scores: list of score dicts each containing 'variant_index' and 'composite'.

    Returns:
        The prompt string corresponding to the highest composite score.

    Raises:
        ValueError: if eval_scores is empty.
    """
    if not eval_scores:
        raise ValueError("eval_scores is empty — cannot determine best variant")

    best = max(eval_scores, key=lambda e: e["composite"])
    return prompt_variants[best["variant_index"]]


def optimizer_node(state: PromptWorkspaceState) -> dict:
    """Produce a final polished prompt from the best variant, all critiques, and human feedback.

    Args:
        state: current graph state; reads prompt_variants, eval_scores,
               critique_results, and human_feedback.

    Returns:
        dict with key 'winning_prompt' containing the final prompt string.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    best_variant = _find_best_variant(state["prompt_variants"], state["eval_scores"])

    all_weaknesses = [
        weakness
        for critique in state["critique_results"]
        for weakness in critique["weaknesses"]
    ]
    weaknesses_block = "\n".join(f"- {w}" for w in all_weaknesses) or "None identified"

    human_feedback = state.get("human_feedback") or "None"

    user_message = (
        f"Best variant:\n{best_variant}\n\n"
        f"All critique weaknesses:\n{weaknesses_block}\n\n"
        f"Human reviewer notes:\n{human_feedback}"
    )

    result: OptimizerOutput = _get_llm().invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return {"winning_prompt": result.final_prompt}

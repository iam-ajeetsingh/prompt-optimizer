from pathlib import Path
from typing import List

#from langchain_anthropic import ChatAnthropic
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel

from graph.state import PromptWorkspaceState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "generator.txt"

class GeneratorOutput(BaseModel):
    variants: List[str]

#_LLM = ChatAnthropic(model="claude-sonnet-4-5").with_structured_output(GeneratorOutput)
#_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(GeneratorOutput)


#_LLM = ChatOpenAI(
#    model="google/gemini-2.5-flash-lite",
#    openai_api_key=os.environ["OPENROUTER_API_KEY"],
#    openai_api_base="https://openrouter.ai/api/v1",
#).with_structured_output(GeneratorOutput)


_LLM = None

def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model="google/gemini-2.5-flash-lite",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        ).with_structured_output(GeneratorOutput)
    return _LLM


def generator_node(state: PromptWorkspaceState) -> dict:
    """Generate 4 prompt variants from task description, constraints, and optional human feedback.

    Args:
        state: current graph state containing task_description, model_target,
               constraints, and human_feedback.

    Returns:
        dict with key 'prompt_variants' containing a list of 4 prompt strings.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    constraints_block = "\n".join(f"- {c}" for c in state["constraints"]) or "None"
    human_feedback = state.get("human_feedback") or "None"

    user_message = (
        f"task_description: {state['task_description']}\n"
        f"model_target: {state['model_target']}\n"
        f"constraints:\n{constraints_block}\n"
        f"human_feedback: {human_feedback}"
    )
    result: GeneratorOutput = _get_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return {"prompt_variants": result.variants}

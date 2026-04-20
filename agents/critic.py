from pathlib import Path
from typing import List

#from langchain_anthropic import ChatAnthropic
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

from pydantic import BaseModel

from graph.state import PromptWorkspaceState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "critic.txt"


class CritiqueItem(BaseModel):
    variant_index: int
    weaknesses: List[str]
    severity: str


class CriticOutput(BaseModel):
    critiques: List[CritiqueItem]


#_LLM = ChatAnthropic(model="claude-sonnet-4-5").with_structured_output(CriticOutput)
#_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash").with_structured_output(CriticOutput)

_LLM = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
).with_structured_output(CriticOutput)



def critic_node(state: PromptWorkspaceState) -> dict:
    """Audit each prompt variant for weaknesses across 5 dimensions.

    Args:
        state: current graph state; reads prompt_variants.

    Returns:
        dict with key 'critique_results' containing a list of critique dicts,
        one per variant, each with variant_index, weaknesses, and severity.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    variants_block = "\n\n".join(
        f"{i}. {variant}" for i, variant in enumerate(state["prompt_variants"])
    )
    user_message = f"Here are the prompt variants to audit:\n\n{variants_block}"

    result: CriticOutput = _LLM.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    return {"critique_results": [c.model_dump() for c in result.critiques]}

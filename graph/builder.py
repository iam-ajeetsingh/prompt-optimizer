import sqlite3
import config  # loads .env before any agent module is imported

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.generator import generator_node
from agents.critic import critic_node
from agents.evaluator import evaluator_node
from agents.optimizer import optimizer_node
from graph.state import PromptWorkspaceState


def should_retry(state: PromptWorkspaceState) -> str:
    """Route from hitl: approve or max iterations reached sends to optimizer, otherwise retry."""
    if state["human_decision"] == "approve":
        return "optimizer"
    if state["iteration_count"] >= state["iteration_max"]:
        return "optimizer"
    return "generator"


def hitl_node(state: PromptWorkspaceState) -> dict:
    """Interrupt checkpoint for human-in-the-loop review.

    This node contains no logic. LangGraph pauses execution here (via
    interrupt_before) so an external caller can inject human_feedback and
    human_decision into the state before resuming the graph.
    """
    return {}


def build_graph():
    """Compile and return the prompt-optimization StateGraph with SQLite checkpointing."""
    workflow = StateGraph(PromptWorkspaceState)

    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("optimizer", optimizer_node)

    workflow.set_entry_point("generator")

    workflow.add_edge("generator", "critic")
    workflow.add_edge("critic", "evaluator")
    workflow.add_edge("evaluator", "hitl")

    workflow.add_conditional_edges(
        "hitl",
        should_retry,
        {"generator": "generator", "optimizer": "optimizer"},
    )

    workflow.add_edge("optimizer", END)

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory, interrupt_before=["hitl"])


graph = build_graph()

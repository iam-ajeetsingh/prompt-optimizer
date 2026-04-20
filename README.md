# Prompt Engineering Workspace

A multi-agent system that iteratively designs, critiques, scores, and optimises LLM prompts — with a human-in-the-loop review step between iterations.

---

## Overview

Writing a good prompt is rarely a one-shot task. This workspace automates the iterative process:

1. **Generator** produces 4 prompt variants using different strategies (direct instruction, chain-of-thought, role-play, few-shot).
2. **Critic** audits each variant across 5 risk dimensions and assigns a severity rating.
3. **Evaluator** scores each variant on 5 quality dimensions and computes a weighted composite score.
4. **Human review** — you inspect the scored variants, optionally add feedback, and decide to approve or retry.
5. **Optimizer** takes the best-scoring variant, all critique findings, and your feedback, and produces a final polished prompt.

If you choose **Retry**, the loop restarts with your notes fed back into the Generator. The loop also terminates automatically when the iteration ceiling is reached.

---

## Architecture

```
POST /run
   └─► generator_node   →  critic_node  →  evaluator_node  →  [HITL pause]
                                                                      │
                                                               human reviews
                                                                      │
                                             ┌─── approve ────────────┘
                                             │
                                       optimizer_node  →  winning_prompt
                                             │
                                        POST /review
                                        returns result
```

**State machine** — built with [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`. The full `PromptWorkspaceState` TypedDict lives in `graph/state.py` and is the single source of truth passed between every node.

**Checkpointing** — LangGraph's `SqliteSaver` persists graph state to `checkpoints.db` after every node, enabling the HTTP pause-resume pattern used by the HITL node.

**API** — FastAPI exposes two endpoints. `/run` starts a new session and blocks until the graph pauses at the HITL node. `/review` resumes from the checkpoint, injects the human decision, and runs to the next pause or completion.

**UI** — Streamlit single-page app with three screens: input form → variant review → winning prompt display.

---

## Project Structure

```
prompt-optimizer/
├── agents/
│   ├── generator.py       # Produces 4 prompt variants via structured output
│   ├── critic.py          # Audits each variant for weaknesses
│   ├── evaluator.py       # Scores variants; computes weighted composite
│   └── optimizer.py       # Final polish pass on the best variant
├── graph/
│   ├── state.py           # PromptWorkspaceState TypedDict (single source of truth)
│   └── builder.py         # StateGraph wiring, HITL node, should_retry routing
├── api/
│   ├── main.py            # FastAPI app — /run and /review endpoints
│   └── schemas.py         # Pydantic request/response models
├── prompts/
│   ├── generator.txt      # System prompt for the Generator agent
│   ├── critic.txt         # System prompt for the Critic agent
│   ├── evaluator.txt      # System prompt for the Evaluator agent
│   └── optimizer.txt      # System prompt for the Optimizer agent
├── tests/
│   ├── conftest.py        # pytest env setup (dummy API keys before import)
│   └── test_agents.py     # Unit tests for pure functions (no LLM calls)
├── app.py                 # Streamlit UI
├── requirements.txt
└── checkpoints.db         # Auto-created at runtime by SqliteSaver
```

---

## Evaluator Rubric

| Dimension   | Weight | What it measures |
|-------------|--------|-----------------|
| Clarity     | 25 %   | Unambiguous, easy to follow |
| Robustness  | 25 %   | Handles edge cases and adversarial inputs |
| Specificity | 20 %   | Constrains model behaviour precisely |
| Efficiency  | 15 %   | No wasted words |
| Alignment   | 15 %   | Matches task description and constraints |

Composite score = weighted sum (1–10 scale). Computed in Python — never delegated to the LLM.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 0.2+ |
| LLM calls | LangChain (Anthropic / OpenAI-compatible / Google Gemini) |
| Checkpointing | LangGraph SqliteSaver → `checkpoints.db` |
| API | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| UI | Streamlit |
| Testing | pytest |

---

## Setup

### Prerequisites

- Python 3.11
- Conda (recommended) or any virtual environment manager

### Installation

```bash
git clone https://github.com/your-username/prompt-optimizer.git
cd prompt-optimizer

conda create -n prompt-optimizer python=3.11
conda activate prompt-optimizer

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
# Required — at least one provider key depending on which agents are active
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...   # for OpenRouter-routed models (Gemini etc.)

# Optional — override the API base URL for the Streamlit app
API_BASE_URL=http://localhost:8000
```

---

## Running the App

Open two terminals, both with the conda env activated.

**Terminal 1 — FastAPI backend**

```bash
conda activate prompt-optimizer
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend**

```bash
conda activate prompt-optimizer
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

```bash
conda activate prompt-optimizer
pytest tests/ -v
```

All 11 tests are pure unit tests — no LLM calls, no API calls, no graph invocation. `tests/conftest.py` sets dummy API keys before any agent module is imported, so the test suite runs without credentials.

---

## API Reference

### `POST /run`

Start a new optimization run. Blocks until the graph pauses at the HITL checkpoint.

**Request body**

```json
{
  "task_description": "Write a prompt that summarises a research paper in 3 bullets",
  "model_target": "google/gemini-2.5-flash-lite",
  "constraints": ["Output in English", "Max 50 words per bullet"],
  "iteration_max": 3
}
```

**Response** — `RunResponse`

```json
{
  "thread_id": "d538acbd-...",
  "variants": ["...", "...", "...", "..."],
  "eval_scores": [{"variant_index": 0, "scores": {...}, "composite": 7.25}, ...],
  "critique_results": [{"variant_index": 0, "weaknesses": [...], "severity": "medium"}, ...],
  "status": "awaiting_review"
}
```

---

### `POST /review`

Submit a human decision and resume the graph.

**Request body**

```json
{
  "thread_id": "d538acbd-...",
  "decision": "approve",
  "feedback": "Ask the model to explain WHY each weakness matters"
}
```

`decision` must be `"approve"` or `"retry"`.

**Response — approved / complete**

```json
{
  "status": "complete",
  "winning_prompt": "You are a senior analyst..."
}
```

**Response — retry / awaiting next review**

```json
{
  "thread_id": "d538acbd-...",
  "variants": [...],
  "eval_scores": [...],
  "critique_results": [...],
  "status": "awaiting_review"
}
```

---

## Coding Conventions

- Every agent is a standalone function in its own file under `agents/`
- System prompts live as `.txt` files in `prompts/` — never hardcoded in Python
- `PromptWorkspaceState` lives **only** in `graph/state.py`
- All LLM outputs are parsed with Pydantic models — no raw string parsing
- Model names are never hardcoded — always read from `state["model_target"]`
- Agent logic and API logic are kept strictly separate

---

## License

MIT
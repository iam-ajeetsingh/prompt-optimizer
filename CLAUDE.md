# Prompt Engineering Workspace

## What this project does
A multi-agent system built with LangGraph and FastAPI that helps engineers
iteratively design, critique, evaluate, and optimize LLM prompts.
Agents: Generator → Critic + Evaluator (parallel) → Human-in-the-loop → Optimizer.

## Tech stack
- Python 3.11
- LangGraph (agent state machine + checkpointing)
- LangChain Google GenAI (LLM calls via Gemini)
- FastAPI + uvicorn (human-in-the-loop API)
- SQLite via SqliteSaver (LangGraph checkpoint persistence)

## Environment
- OS: Windows, terminal: cmd
- Conda env name: prompt-optimizer
- Activate: conda activate prompt-optimizer
- Run API: uvicorn api.main:app --reload --port 8000
- Run tests: pytest tests/ -v

## Project structure
prompt-optimizer/
├── agents/
│   ├── __init__.py
│   ├── generator.py
│   ├── critic.py
│   ├── evaluator.py
│   └── optimizer.py
├── graph/
│   ├── __init__.py
│   ├── state.py
│   └── builder.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── prompts/
│   ├── generator.txt
│   ├── critic.txt
│   ├── evaluator.txt
│   └── optimizer.txt
├── tests/
│   ├── __init__.py
│   └── test_agents.py
├── CLAUDE.md
├── requirements.txt
└── checkpoints.db  ← auto-created at runtime

## Coding conventions (follow these strictly)
- Every agent is a standalone function in its own file under agents/
- System prompts live as .txt files in prompts/ — never hardcode them in Python
- State TypedDict lives ONLY in graph/state.py — never inline it elsewhere
- All LLM outputs must be parsed with Pydantic models — no raw string parsing
- Never hardcode model names — always read from state["model_target"]
- Every function must have a docstring explaining inputs and outputs
- Do not mix agent logic with API logic — keep agents/ and api/ completely separate

## Agents summary
- Generator : reads task + constraints → outputs 3-4 prompt variants as JSON list
- Critic    : reads each variant → outputs weaknesses dict (severity: low/medium/high)
- Evaluator : scores each variant on 5 dimensions → outputs score card with composite
- Optimizer : reads best variant + all critiques + human notes → outputs final prompt string

## Key files to never modify directly
- graph/state.py (ask before changing the state schema)
- checkpoints.db (runtime file, do not touch)
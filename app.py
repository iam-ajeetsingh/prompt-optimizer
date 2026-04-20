import requests
import streamlit as st
import os

#API_BASE = "http://localhost:8000"
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Prompt Engineering Workspace", layout="wide")

st.markdown(
    "<style>textarea[disabled]{color:#f1f5f9 !important;-webkit-text-fill-color:#f1f5f9 !important;opacity:1 !important;}</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init():
    defaults = {
        "screen": "input",
        "run_response": None,
        "thread_id": None,
        "winning_prompt": None,
        "task_description": "",
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

_init()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _score_color(composite: float) -> str:
    if composite >= 7:
        return "#22c55e"
    if composite >= 5:
        return "#f59e0b"
    return "#ef4444"


def _submit_review(decision: str, feedback: str):
    with st.spinner("Running agents..."):
        try:
            resp = requests.post(f"{API_BASE}/review", json={
                "thread_id": st.session_state.thread_id,
                "decision": decision,
                "feedback": feedback,
            }, timeout=300)
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach API: {exc}")
            return

    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    if data["status"] == "complete":
        st.session_state.winning_prompt = data["winning_prompt"]
        st.session_state.screen = "complete"
    else:
        st.session_state.run_response = data
        st.session_state.thread_id = data["thread_id"]
        st.session_state.screen = "review"
    st.rerun()


# ---------------------------------------------------------------------------
# Screen 1 — Input
# ---------------------------------------------------------------------------

def render_input():
    st.title("Prompt Engineering Workspace")
    st.caption("Multi-agent prompt optimizer")

    task = st.text_area(
        "Task description",
        height=160,
        value=(
            "Write a prompt that instructs an LLM to analyze a failed product launch "
            "and produce a concise post-mortem report for a non-technical executive audience."
        ),
    )

    constraints_raw = st.text_area(
        "Constraints (one per line)",
        height=120,
        value=(
            "Report must be under 300 words\n"
            "Use plain English — no technical jargon\n"
            "Structure: Root Cause, Business Impact, Lessons Learned\n"
            "Tone must be neutral and factual, not defensive"
        ),
    )
    constraints = [c.strip() for c in constraints_raw.splitlines() if c.strip()]

    st.markdown("")
    iteration_max = st.slider("Max iterations", min_value=1, max_value=5, value=3)

    if st.button("Optimize Prompt", type="primary"):
        if not task.strip():
            st.error("Please enter a task description.")
            return
        with st.spinner("Running agents..."):
            try:
                resp = requests.post(f"{API_BASE}/run", json={
                    "task_description": task.strip(),
                    "model_target": "google/gemini-2.5-flash-lite",
                    "constraints": constraints,
                    "iteration_max": iteration_max,
                }, timeout=300)
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach API: {exc}")
                return

        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return

        data = resp.json()
        st.session_state.run_response = data
        st.session_state.thread_id = data["thread_id"]
        st.session_state.task_description = task.strip()
        st.session_state.run_constraints = constraints
        st.session_state.screen = "review"
        st.rerun()


# ---------------------------------------------------------------------------
# Screen 2 — Review
# ---------------------------------------------------------------------------

def render_review():
    with st.sidebar:
        st.caption("Current run")
        st.code(st.session_state.thread_id[:8] + "...", language=None)
        run_constraints = st.session_state.get("run_constraints", [])
        if run_constraints:
            st.caption("Constraints")
            for c in run_constraints:
                st.markdown(f"- {c}")

    st.header(st.session_state.task_description)

    data = st.session_state.run_response
    variants: list = data["variants"]
    eval_scores: list = data["eval_scores"]
    critique_results: list = data["critique_results"]

    score_map = {e["variant_index"]: e for e in eval_scores}
    critique_map = {c["variant_index"]: c for c in critique_results}
    best_idx = max(score_map, key=lambda i: score_map[i].get("composite", 0))

    for row in range(2):
        cols = st.columns(2)
        for col_pos in range(2):
            vi = row * 2 + col_pos
            if vi >= len(variants):
                break
            scores_entry = score_map.get(vi, {})
            composite = scores_entry.get("composite", 0.0)
            critique_entry = critique_map.get(vi, {})
            weaknesses = critique_entry.get("weaknesses", [])
            severity = critique_entry.get("severity", "low")
            is_best = vi == best_idx

            border = "#22c55e" if is_best else "#cbd5e1"
            badge_color = _score_color(composite)
            best_badge = (
                ' <span style="background:#16a34a;color:white;'
                'padding:2px 8px;border-radius:8px;font-size:0.78em">Best</span>'
                if is_best else ""
            )
            with cols[col_pos]:
                st.markdown(
                    f'<div style="border:2px solid {border};border-radius:10px;'
                    f'padding:18px 20px;margin-bottom:4px;'
                    f'background:rgba(255,255,255,0.04)">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:12px">'
                    f'<span style="font-weight:700;font-size:1.05em">'
                    f'Variant {vi + 1}{best_badge}</span>'
                    f'<span style="background:{badge_color};color:white;padding:3px 14px;'
                    f'border-radius:12px;font-weight:700;font-size:1em">{composite:.2f}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.text_area(
                    label=f"variant_{vi}_text",
                    value=variants[vi],
                    height=220,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"variant_text_{vi}",
                )
                if weaknesses:
                    with st.expander(f"⚠ {len(weaknesses)} weakness(es) · {severity}"):
                        for w in weaknesses:
                            st.warning(w)

    st.divider()
    feedback = st.text_area("Add notes for the next iteration (optional)", height=100)

    col_approve, col_retry = st.columns(2)
    with col_approve:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            _submit_review("approve", feedback)
    with col_retry:
        if st.button("🔄 Retry", use_container_width=True):
            _submit_review("retry", feedback)


# ---------------------------------------------------------------------------
# Screen 3 — Complete
# ---------------------------------------------------------------------------

def render_complete():
    st.success("Winning Prompt Ready")
    st.header("Winning Prompt")

    st.code(st.session_state.winning_prompt, language=None)

    data = st.session_state.run_response
    if data and data.get("eval_scores"):
        st.subheader("Composite Scores")
        chart_data = {
            f"Variant {e['variant_index'] + 1}": e.get("composite", 0.0)
            for e in data["eval_scores"]
        }
        st.bar_chart(chart_data)

    if st.button("Start Over"):
        _reset()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

screen = st.session_state.screen
if screen == "input":
    render_input()
elif screen == "review":
    render_review()
elif screen == "complete":
    render_complete()

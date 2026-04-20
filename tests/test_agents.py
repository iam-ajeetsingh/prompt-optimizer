import pytest
import config   # loads .env before any agent module is imported

from agents.evaluator import RUBRIC, _compute_composite
from agents.optimizer import _find_best_variant
from graph.builder import should_retry


# ---------------------------------------------------------------------------
# _find_best_variant
# ---------------------------------------------------------------------------

def test_find_best_variant_returns_correct_variant():
    variants = ["prompt-zero", "prompt-one", "prompt-two"]
    eval_scores = [
        {"variant_index": 0, "composite": 6.50},
        {"variant_index": 1, "composite": 8.75},
        {"variant_index": 2, "composite": 7.10},
    ]
    assert _find_best_variant(variants, eval_scores) == "prompt-one"


def test_find_best_variant_single_entry():
    variants = ["only-variant"]
    eval_scores = [{"variant_index": 0, "composite": 5.00}]
    assert _find_best_variant(variants, eval_scores) == "only-variant"


def test_find_best_variant_raises_on_empty_scores():
    with pytest.raises(ValueError, match="eval_scores is empty"):
        _find_best_variant(["prompt-a"], [])


# ---------------------------------------------------------------------------
# _compute_composite
# ---------------------------------------------------------------------------

def test_compute_composite_perfect_scores():
    scores = {"clarity": 10, "specificity": 10, "robustness": 10, "efficiency": 10, "alignment": 10}
    expected = round(sum(10 * w for w in RUBRIC.values()), 2)
    assert _compute_composite(scores) == expected


def test_compute_composite_known_values():
    # clarity=8, specificity=6, robustness=7, efficiency=9, alignment=5
    # 8*0.25 + 6*0.20 + 7*0.25 + 9*0.15 + 5*0.15
    # = 2.00 + 1.20 + 1.75 + 1.35 + 0.75 = 7.05
    scores = {"clarity": 8, "specificity": 6, "robustness": 7, "efficiency": 9, "alignment": 5}
    assert _compute_composite(scores) == 7.05


def test_compute_composite_zero_scores():
    scores = {"clarity": 0, "specificity": 0, "robustness": 0, "efficiency": 0, "alignment": 0}
    assert _compute_composite(scores) == 0.0


def test_compute_composite_weights_sum_to_one():
    assert round(sum(RUBRIC.values()), 10) == 1.0


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> dict:
    base = {
        "task_description": "test task",
        "model_target": "google/gemini-2.5-flash-lite",
        "constraints": [],
        "prompt_variants": [],
        "critique_results": [],
        "eval_scores": [],
        "human_feedback": None,
        "human_decision": "",
        "winning_prompt": None,
        "iteration_count": 1,
        "iteration_max": 3,
    }
    base.update(overrides)
    return base


def test_should_retry_approve_routes_to_optimizer():
    state = _make_state(human_decision="approve", iteration_count=1, iteration_max=3)
    assert should_retry(state) == "optimizer"


def test_should_retry_retry_below_max_routes_to_generator():
    state = _make_state(human_decision="retry", iteration_count=1, iteration_max=3)
    assert should_retry(state) == "generator"


def test_should_retry_at_iteration_max_routes_to_optimizer():
    state = _make_state(human_decision="retry", iteration_count=3, iteration_max=3)
    assert should_retry(state) == "optimizer"


def test_should_retry_exceeds_iteration_max_routes_to_optimizer():
    state = _make_state(human_decision="retry", iteration_count=5, iteration_max=3)
    assert should_retry(state) == "optimizer"

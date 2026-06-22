"""Scorer registry.

Deterministic scorers + LLM-judge scorers share one signature:
    score(artifact, item, pack, provider) -> CriterionScore
so the harness can treat them uniformly.
"""

from __future__ import annotations

from collections.abc import Callable

from lesson_loom.core.schemas import JUDGE_SCORERS
from lesson_loom.evals.scorers import (
    format_validity,
    grounding,
    hallucination,
    objective_coverage,
    readability,
)
from lesson_loom.evals.scorers.llm_judge import make_judge

REGISTRY: dict[str, Callable] = {
    "readability": readability.score,
    "factual_grounding": grounding.score,
    "objective_coverage": objective_coverage.score,
    "format_validity": format_validity.score,
    "no_hallucination": hallucination.score,
    **{name: make_judge(name) for name in JUDGE_SCORERS},
}


def get_scorer(name: str) -> Callable:
    return REGISTRY[name]

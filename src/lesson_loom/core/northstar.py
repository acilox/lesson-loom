"""The frozen north-star metric.

This is the immutable definition of "good educational content" used by the
promotion gate and the portfolio KPI. The optimizer can edit its own
`RewardWeights` all it likes, but it can NEVER touch these weights — so it
cannot win by moving the goalposts. The north-star is defined only over the
fixed BASE_SCORERS; tools the optimizer synthesizes never count toward it.
"""

from __future__ import annotations

from lesson_loom.core.schemas import BASE_SCORERS, ItemScorecard

# Fixed weights, sum to 1.0. Emphasis on grounding, objective coverage, and
# grade-appropriate readability — the things that make a lesson actually usable.
NORTHSTAR_WEIGHTS: dict[str, float] = {
    "factual_grounding": 0.22,
    "objective_coverage": 0.22,
    "readability": 0.18,
    "no_hallucination": 0.15,
    "format_validity": 0.08,
    "pedagogical_soundness": 0.07,
    "clarity": 0.05,
    "engagement": 0.03,
}

assert set(NORTHSTAR_WEIGHTS) == set(BASE_SCORERS), "north-star must cover the base scorers"
assert abs(sum(NORTHSTAR_WEIGHTS.values()) - 1.0) < 1e-9, "north-star weights must sum to 1.0"


def northstar_score(item: ItemScorecard) -> float:
    """Frozen weighted sum over the base scorers only (ignores synthesized tools)."""
    return sum(
        NORTHSTAR_WEIGHTS[name] * item.scores[name].normalized
        for name in NORTHSTAR_WEIGHTS
        if name in item.scores
    )

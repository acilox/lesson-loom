"""Failure analysis — turn a scorecard into a ranked list of gaps.

Gaps are criteria whose mean falls below a per-criterion threshold. They are
ranked by severity (how far below threshold), with the agent's *mutable* reward
weight as a tiebreaker — so reshaping the reward reprioritizes the search."""

from __future__ import annotations

from dataclasses import dataclass

from lesson_loom.core.schemas import BASE_SCORERS, Scorecard, SystemConfig

# A criterion counts as "failing" below these means.
THRESHOLDS: dict[str, float] = {
    "readability": 0.9,
    "factual_grounding": 0.9,
    "objective_coverage": 0.9,
    "format_validity": 0.9,
    "no_hallucination": 0.9,
    "pedagogical_soundness": 0.8,
    "clarity": 0.8,
    "engagement": 0.8,
}


@dataclass
class Gap:
    criterion: str
    mean: float
    severity: float  # threshold - mean
    weighted: float  # severity * reward weight (priority)


def analyze(scorecard: Scorecard, config: SystemConfig) -> list[Gap]:
    gaps: list[Gap] = []
    for name in BASE_SCORERS:
        mean = scorecard.criterion_mean(name)
        thr = THRESHOLDS[name]
        if mean < thr:
            sev = thr - mean
            gaps.append(Gap(name, mean, sev, sev * max(config.reward_weights.get(name), 1e-6)))
    # primary: severity; tiebreak: reward-weighted priority
    gaps.sort(key=lambda g: (round(g.severity, 6), g.weighted), reverse=True)
    return gaps

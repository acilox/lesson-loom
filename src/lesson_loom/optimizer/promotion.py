"""The promotion gate — the discipline that turns "iteration" into "improvement".

All judgments use the FROZEN north-star (never the agent's mutable reward), so the
agent cannot win by reshaping its own objective.

Prompt edits (which change generation) must:
  - raise train north-star by >= MIN_IMPROVEMENT,
  - not drop held-out north-star by more than REGRESSION_FLOOR,
  - not let any single held-out base criterion collapse (catastrophe guard).

Tool synthesis (which does not change generation, so cannot move the north-star)
is accepted only if the tool is informative (catches real failures) and the
held-out north-star is unchanged — instrumenting a weakness without harm.
"""

from __future__ import annotations

from lesson_loom.core.northstar import NORTHSTAR_WEIGHTS
from lesson_loom.core.schemas import PromotionDecision, Scorecard
from lesson_loom.optimizer import tools

MIN_IMPROVEMENT = 0.02
REGRESSION_FLOOR = -0.01
CATASTROPHE_FLOOR = -0.05


def evaluate(
    axis: str,
    target: str,
    base_train: Scorecard,
    base_heldout: Scorecard,
    cand_train: Scorecard,
    cand_heldout: Scorecard,
) -> PromotionDecision:
    train_delta = round(cand_train.northstar - base_train.northstar, 4)
    heldout_delta = round(cand_heldout.northstar - base_heldout.northstar, 4)
    goodhart_gap = round(train_delta - heldout_delta, 4)

    if axis == "tool":
        tool_name = _last_tool(cand_train)
        informative = bool(tool_name) and tools.is_informative(tool_name, cand_train)
        promoted = informative and heldout_delta >= REGRESSION_FLOOR
        reason = (
            "tool is informative and held-out did not regress"
            if promoted
            else "tool not informative or held-out regressed"
        )
        return PromotionDecision(
            promoted=promoted, reason=reason,
            northstar_train_delta=train_delta, heldout_delta=heldout_delta, goodhart_gap=goodhart_gap,
        )

    # prompt edit (generative)
    catastrophe = _worst_base_criterion_drop(base_heldout, cand_heldout)
    if train_delta < MIN_IMPROVEMENT:
        reason = f"train north-star delta {train_delta:+.3f} < {MIN_IMPROVEMENT}"
        promoted = False
    elif heldout_delta < REGRESSION_FLOOR:
        reason = f"held-out regressed {heldout_delta:+.3f} < {REGRESSION_FLOOR}"
        promoted = False
    elif catastrophe < CATASTROPHE_FLOOR:
        reason = f"a held-out criterion collapsed ({catastrophe:+.3f})"
        promoted = False
    else:
        reason = f"train {train_delta:+.3f}, held-out {heldout_delta:+.3f}, no regression"
        promoted = True

    return PromotionDecision(
        promoted=promoted, reason=reason,
        northstar_train_delta=train_delta, heldout_delta=heldout_delta, goodhart_gap=goodhart_gap,
    )


def _worst_base_criterion_drop(base: Scorecard, cand: Scorecard) -> float:
    worst = 0.0
    for name in NORTHSTAR_WEIGHTS:
        delta = cand.per_criterion_means.get(name, 0.0) - base.per_criterion_means.get(name, 0.0)
        worst = min(worst, delta)
    return round(worst, 4)


def _last_tool(scorecard: Scorecard) -> str:
    tool_names = [k for k in scorecard.per_criterion_means if k.startswith("tool_")]
    return tool_names[-1] if tool_names else ""

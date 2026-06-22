"""The centerpiece: the self-improvement loop must actually improve, audibly,
without regressing the held-out set — and the gate must reject non-improvements."""

import pytest

from lesson_loom.optimizer import promotion
from lesson_loom.optimizer.loop import OptimizerController
from lesson_loom.optimizer.tools import synthesize_for


@pytest.mark.parametrize("subject", ["science", "history"])
def test_loop_improves_on_unbiased_test_split(provider, subject):
    pack = __import__("lesson_loom.context_packs.loader", fromlist=["load_pack"]).load_pack(subject)
    result = OptimizerController(provider, pack, subject, max_rounds=8).run()
    # Substantial, real gain on the untouched TEST split (frozen metric).
    assert result.test_gain > 0.3
    assert result.best_test_northstar > result.baseline_test_northstar


def test_loop_is_auditable_and_gated(provider, science_pack):
    result = OptimizerController(provider, science_pack, "science", max_rounds=8).run()
    promoted = [r for r in result.rounds if r.decision.promoted]
    rejected = [r for r in result.rounds if not r.decision.promoted]
    tools = [r for r in result.rounds if r.axis == "tool" and r.decision.promoted]

    assert len(promoted) >= 3            # prompt self-edits landed
    assert len(rejected) >= 1            # the gate rejected a non-improvement
    assert len(tools) >= 1               # the agent designed + kept a tool
    # single-axis: every round changes exactly one generative axis
    assert all(r.axis in ("prompt", "tool") for r in result.rounds)
    # no promoted round regressed the held-out set past the floor
    assert all(r.decision.heldout_delta >= promotion.REGRESSION_FLOOR for r in promoted)
    # every round carries a human-readable rationale
    assert all(r.rationale for r in result.rounds)


def test_loop_is_reproducible(provider, science_pack):
    r1 = OptimizerController(provider, science_pack, "science", max_rounds=8).run()
    r2 = OptimizerController(provider, science_pack, "science", max_rounds=8).run()
    assert r1.best_test_northstar == r2.best_test_northstar
    assert r1.best_config_id == r2.best_config_id


def test_synthesize_for_engagement():
    from lesson_loom.core.schemas import Scorecard

    sc = Scorecard(system_config_id="c", split="train", items=[], northstar=0.5,
                   reward=0.5, per_criterion_means={"engagement": 0.25})
    tool = synthesize_for("engagement", sc)
    assert tool is not None and tool.template == "keyword_presence"

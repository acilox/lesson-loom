"""Promotion gate logic in isolation."""

from lesson_loom.core.schemas import Scorecard
from lesson_loom.optimizer import promotion


def _card(ns, crit=None):
    return Scorecard(system_config_id="c", split="x", items=[], northstar=ns, reward=ns,
                     per_criterion_means=crit or {})


def test_prompt_promoted_on_real_gain():
    d = promotion.evaluate("prompt", "readability", _card(0.5), _card(0.5), _card(0.6), _card(0.6))
    assert d.promoted and d.northstar_train_delta == 0.1


def test_prompt_rejected_on_no_gain():
    d = promotion.evaluate("prompt", "engagement", _card(0.9), _card(0.9), _card(0.9), _card(0.9))
    assert not d.promoted


def test_prompt_rejected_on_heldout_regression():
    d = promotion.evaluate("prompt", "x", _card(0.5), _card(0.6), _card(0.7), _card(0.4))
    assert not d.promoted  # big train gain but held-out collapsed


def test_tool_promoted_when_informative():
    base = _card(0.9)
    cand = _card(0.9, {"tool_x": 0.4})  # informative (fires) + no regression
    d = promotion.evaluate("tool", "engagement", base, base, cand, cand)
    assert d.promoted


def test_goodhart_gap_logged():
    d = promotion.evaluate("prompt", "x", _card(0.5), _card(0.5), _card(0.7), _card(0.6))
    assert d.goodhart_gap == round(0.2 - 0.1, 4)

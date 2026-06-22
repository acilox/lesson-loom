"""Schemas, north-star, and the structural-mock causal chain."""

from lesson_loom.core.northstar import NORTHSTAR_WEIGHTS, northstar_score
from lesson_loom.core.schemas import BASE_SCORERS, CriterionScore, ItemScorecard
from lesson_loom.generation.graph import generate


def test_northstar_weights_valid():
    assert abs(sum(NORTHSTAR_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(NORTHSTAR_WEIGHTS) == set(BASE_SCORERS)


def test_northstar_ignores_synthesized_tools():
    scores = {n: CriterionScore(name=n, raw_value=1, normalized=1.0) for n in BASE_SCORERS}
    scores["tool_made_up"] = CriterionScore(name="tool_made_up", raw_value=0, normalized=0.0)
    card = ItemScorecard(eval_item_id="x", artifact_id="y", scores=scores)
    # perfect base scores -> north-star 1.0, tool's 0.0 must not drag it down
    assert abs(northstar_score(card) - 1.0) < 1e-9


def test_structural_mock_responds_to_prompt_structure(provider, base_config, science_pack):
    """Weak prompt -> no citations / wrong reading level; rich prompt -> both fixed."""
    import textstat

    weak = generate(
        provider=provider, config=base_config, context_pack=science_pack, topic=science_pack.topic,
        grade_level="grade_5", content_type="explainer", reading_low=4.0, reading_high=6.5,
    )
    strong_cfg = base_config.evolve(
        generation_prompts={
            **base_config.generation_prompts,
            "draft": base_config.generation_prompts["draft"]
            + " Cite every fact with its [fact_id]. Write at the target reading level."
            + " Address every learning objective and tag it [obj_id].",
        }
    )
    strong = generate(
        provider=provider, config=strong_cfg, context_pack=science_pack, topic=science_pack.topic,
        grade_level="grade_5", content_type="explainer", reading_low=4.0, reading_high=6.5,
    )
    assert len(weak.citations) == 0 and len(strong.citations) >= 5
    assert len(strong.objectives_covered) >= 3
    assert textstat.flesch_kincaid_grade(strong.body) < textstat.flesch_kincaid_grade(weak.body)

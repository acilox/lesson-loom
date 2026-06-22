"""Scorers + synthesized-tool evaluation."""

from lesson_loom.core.schemas import (
    CitationLink,
    ContentArtifact,
    EvalItem,
    SynthesizedTool,
)
from lesson_loom.evals.scorers import grounding, hallucination, objective_coverage, readability
from lesson_loom.evals.scorers.synthesized import evaluate_tool


def _item(**kw):
    base = dict(
        id="t", subject="science", topic="Photosynthesis", grade_level="grade_5",
        content_type="explainer", split="train", reading_level_low=4.0, reading_level_high=6.5,
        required_objective_ids=["obj_001", "obj_002"], required_fact_ids=["fact_001", "fact_002"],
    )
    base.update(kw)
    return EvalItem(**base)


def _art(body="x", citations=None, objs=None):
    return ContentArtifact.build(
        system_config_id="c", topic="Photosynthesis", content_type="explainer",
        grade_level="grade_5", body=body, citations=citations or [], objectives_covered=objs or [],
    )


def test_grounding_coverage(science_pack):
    item = _item()
    art = _art(citations=[CitationLink(fact_id="fact_001", excerpt="e"),
                          CitationLink(fact_id="fact_002", excerpt="e")])
    assert grounding.score(art, item, science_pack, None).normalized == 1.0
    art0 = _art()
    assert grounding.score(art0, item, science_pack, None).normalized == 0.0


def test_objective_coverage(science_pack):
    item = _item()
    art = _art(objs=["obj_001", "obj_002"])
    assert objective_coverage.score(art, item, science_pack, None).normalized == 1.0


def test_hallucination_flags_unknown_ids(science_pack):
    item = _item()
    good = _art(citations=[CitationLink(fact_id="fact_001", excerpt="e")])
    assert hallucination.score(good, item, science_pack, None).normalized == 1.0
    bad = _art(citations=[CitationLink(fact_id="fact_999", excerpt="e")])
    assert hallucination.score(bad, item, science_pack, None).normalized == 0.0


def test_readability_band(science_pack):
    item = _item()
    simple = _art(body="The cat sat on the mat. It was a good day. We had fun today.")
    hard = _art(body="The multifaceted epistemological ramifications necessitate comprehensive "
                     "analytical reconsideration of foundational presuppositions throughout.")
    assert readability.score(simple, item, science_pack, None).normalized >= \
        readability.score(hard, item, science_pack, None).normalized


def test_synthesized_min_count_tool(science_pack):
    item = _item()
    tool = SynthesizedTool(name="tool_min_citations", template="min_count",
                           params={"target": "citations", "min": 3}, rationale="r")
    art = _art(citations=[CitationLink(fact_id=f"fact_00{i}", excerpt="e") for i in range(1, 4)])
    assert evaluate_tool(tool, art, item, science_pack).normalized == 1.0
    art1 = _art(citations=[CitationLink(fact_id="fact_001", excerpt="e")])
    assert evaluate_tool(tool, art1, item, science_pack).normalized < 1.0

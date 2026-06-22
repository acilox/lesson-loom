"""factual_grounding scorer — fraction of required pack facts actually cited."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem


def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
    cited = {c.fact_id for c in artifact.citations}
    required = set(item.required_fact_ids)
    if required:
        covered = len(required & cited)
        norm = covered / len(required)
        raw = float(covered)
    else:
        # no explicit requirement: reward having at least 3 grounded claims
        norm = min(1.0, len(cited) / 3.0)
        raw = float(len(cited))
    return CriterionScore(name="factual_grounding", raw_value=raw, normalized=round(norm, 4),
                          details={"cited": sorted(cited), "required": sorted(required)})

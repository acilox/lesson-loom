"""objective_coverage scorer — fraction of required learning objectives addressed."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem


def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
    covered = set(artifact.objectives_covered)
    required = set(item.required_objective_ids) or pack.objective_ids()
    hit = len(required & covered)
    norm = hit / len(required) if required else 0.0
    return CriterionScore(name="objective_coverage", raw_value=float(hit), normalized=round(norm, 4),
                          details={"covered": sorted(covered), "required": sorted(required)})

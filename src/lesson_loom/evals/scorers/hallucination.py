"""no_hallucination scorer — every cited id must exist in the context pack."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem


def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
    valid = pack.fact_ids()
    cited = [c.fact_id for c in artifact.citations]
    if not cited:
        return CriterionScore(name="no_hallucination", raw_value=0.0, normalized=1.0,
                              details={"hallucinated": []})
    bad = [fid for fid in cited if fid not in valid]
    norm = 1.0 - len(bad) / len(cited)
    return CriterionScore(name="no_hallucination", raw_value=float(len(bad)),
                          normalized=round(norm, 4), details={"hallucinated": bad})

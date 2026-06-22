"""readability scorer — Flesch-Kincaid grade vs the item's target band."""

from __future__ import annotations

import textstat

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem


def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
    fk = float(textstat.flesch_kincaid_grade(artifact.body or ""))
    low, high = item.reading_level_low, item.reading_level_high
    if low <= fk <= high:
        norm = 1.0
    else:
        # linear decay: 0.5 at 2 grades outside the band, 0.0 at 4+.
        dist = (low - fk) if fk < low else (fk - high)
        norm = max(0.0, 1.0 - dist / 4.0)
    return CriterionScore(name="readability", raw_value=fk, normalized=round(norm, 4),
                          details={"band": [low, high]})

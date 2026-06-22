"""format_validity scorer — does the artifact have the basic shape its type needs."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem


def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
    body = artifact.body or ""
    ok = len(body.strip()) >= 40
    if item.content_type == "quiz":
        ok = ok and "?" in body
    norm = 1.0 if ok else 0.5 if body.strip() else 0.0
    return CriterionScore(name="format_validity", raw_value=float(len(body)),
                          normalized=norm, details={"content_type": item.content_type})

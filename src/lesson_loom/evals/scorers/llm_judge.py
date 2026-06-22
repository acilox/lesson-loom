"""LLM-as-judge scorers (pedagogical_soundness, clarity, engagement).

Each calls provider.judge() with a rubric and normalizes the 1-5 score to [0,1].
Deterministic under the mock; real rubric grading under Claude (opus judge)."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem
from lesson_loom.generation.prompts import build_judge_user

_RUBRICS = {
    "pedagogical_soundness": (
        "Rate 1-5 how pedagogically sound this content is: does it build on prior "
        "knowledge, use examples, and scaffold complexity for the stated grade?"
    ),
    "clarity": "Rate 1-5 how clear and coherent this content is for the stated grade.",
    "engagement": "Rate 1-5 how engaging this content is likely to be for students.",
}


def make_judge(name: str):
    rubric = _RUBRICS[name]

    def score(artifact: ContentArtifact, item: EvalItem, pack, provider) -> CriterionScore:
        result = provider.judge(rubric, build_judge_user(rubric, artifact.body or ""))
        raw = float(result.get("score", 3))
        norm = max(0.0, min(1.0, (raw - 1.0) / 4.0))
        return CriterionScore(name=name, raw_value=raw, normalized=round(norm, 4),
                              details={"rationale": result.get("rationale", "")})

    return score

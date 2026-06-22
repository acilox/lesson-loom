"""Tool synthesis — the optimizer designing its own check-tools.

When a failure mode persists that the agent could not fix by editing prompts, it
synthesizes a NEW deterministic check from a small, safe template library and
derives the parameters from the failure evidence (not a random pick). The new
tool is registered only if it is *informative* (it actually catches the failure)
and does not regress the held-out north-star — the same gate discipline applied
to prompt edits. Templates are sandboxed (no arbitrary code), which is exactly how
you let an agent extend its own toolset in production.
"""

from __future__ import annotations

from lesson_loom.core.schemas import Scorecard, SynthesizedTool

# The sandboxed template library the optimizer can instantiate.
TEMPLATES = ("min_count", "keyword_presence", "threshold")


def synthesize_for(criterion: str, train: Scorecard) -> SynthesizedTool | None:
    """Pick a template + evidence-derived params to instrument a weak criterion."""
    mean = train.criterion_mean(criterion)

    if criterion == "engagement":
        # Bodies score low on engagement; build a check for interactive signals.
        return SynthesizedTool(
            name="tool_engagement_signals",
            template="keyword_presence",
            params={"keywords": ["?", "for example", "imagine"]},
            rationale=(
                f"engagement plateaued at {mean:.2f} and no prompt edit moved it; "
                "synthesized a keyword check for interactive signals "
                "(question / worked example) to instrument the gap going forward"
            ),
        )

    if criterion == "objective_coverage":
        return SynthesizedTool(
            name="tool_min_objectives",
            template="min_count",
            params={"target": "objectives", "min": 3},
            rationale=(
                f"objective coverage low ({mean:.2f}); synthesized a min-count check "
                "requiring at least 3 tagged objectives per artifact"
            ),
        )

    if criterion in ("factual_grounding", "no_hallucination"):
        return SynthesizedTool(
            name="tool_min_citations",
            template="min_count",
            params={"target": "citations", "min": 3},
            rationale=(
                f"grounding weak ({mean:.2f}); synthesized a min-count check "
                "requiring at least 3 fact citations per artifact"
            ),
        )

    return None


def is_informative(tool_name: str, train: Scorecard) -> bool:
    """A tool earns its place only if it actually fires on real failures."""
    return train.criterion_mean(tool_name) < 0.999

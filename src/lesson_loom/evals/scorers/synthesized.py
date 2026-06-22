"""Runtime evaluation of optimizer-synthesized check-tools.

The optimizer *designs* these (picks a template + derives params from failure
evidence — see optimizer/tools.py). Here we just *run* them, with the same
CriterionScore signature as any other scorer. Synthesized tools influence the
mutable reward and act as extra gates, but they never count toward the frozen
north-star (see core/northstar.py)."""

from __future__ import annotations

from lesson_loom.core.schemas import ContentArtifact, CriterionScore, EvalItem, SynthesizedTool


def evaluate_tool(
    tool: SynthesizedTool, artifact: ContentArtifact, item: EvalItem, pack
) -> CriterionScore:
    body = artifact.body or ""
    p = tool.params

    if tool.template == "min_count":
        target = p.get("target", "citations")
        count = len(artifact.citations) if target == "citations" else len(artifact.objectives_covered)
        minimum = max(1, int(p.get("min", 1)))
        norm = min(1.0, count / minimum)
        raw = float(count)

    elif tool.template == "keyword_presence":
        keywords = [k.lower() for k in p.get("keywords", [])]
        present = sum(1 for k in keywords if k in body.lower())
        norm = present / len(keywords) if keywords else 1.0
        raw = float(present)

    elif tool.template == "threshold":
        value = float(len(body))
        lo, hi = float(p.get("min", 0)), float(p.get("max", 1e9))
        norm = 1.0 if lo <= value <= hi else 0.5
        raw = value

    else:  # unknown template -> neutral
        norm, raw = 1.0, 0.0

    return CriterionScore(name=tool.name, raw_value=raw, normalized=round(norm, 4),
                          details={"template": tool.template, "params": p})

"""Render an ExperimentResult to a human-readable Markdown report (REPORT.md).

This is the "show your work" artifact — what you put in front of a reviewer: the
headline test-split gain, then the round-by-round attribution trace (one change
each, with the agent's rationale and the gate's verdict)."""

from __future__ import annotations

from lesson_loom.core.schemas import ExperimentResult


def render_report(result: ExperimentResult) -> str:
    promoted = sum(1 for r in result.rounds if r.decision.promoted)
    rejected = sum(1 for r in result.rounds if not r.decision.promoted)
    tools = sum(1 for r in result.rounds if r.axis == "tool" and r.decision.promoted)

    lines = [
        f"# lesson-loom — self-improvement report ({result.subject})",
        "",
        "## Headline (untouched TEST split, frozen north-star)",
        "",
        f"- Baseline north-star: **{result.baseline_test_northstar:.3f}**",
        f"- Best north-star: **{result.best_test_northstar:.3f}**",
        f"- **Gain: +{result.test_gain:.3f}** "
        f"({result.test_gain * 100:.1f} points) over {len(result.rounds)} rounds",
        f"- Promoted: {promoted} · Rejected by gate: {rejected} · Tools synthesized: {tools}",
        "",
        "> The promotion metric is the frozen north-star, which the agent cannot edit. "
        "The TEST split is never seen during search or gating, so this gain is unbiased "
        "(directional given the suite size).",
        "",
        "## Round-by-round attribution",
        "",
        "| # | axis | target | Δ train | Δ held-out | gate | verdict |",
        "|---|------|--------|---------|------------|------|---------|",
    ]
    for r in result.rounds:
        d = r.decision
        verdict = "✅ promoted" if d.promoted else "⛔ rejected"
        lines.append(
            f"| {r.round_num} | {r.axis} | {r.target} | "
            f"{d.northstar_train_delta:+.3f} | {d.heldout_delta:+.3f} | "
            f"{d.reason} | {verdict} |"
        )

    lines += ["", "## What the agent changed (lineage)", ""]
    for r in result.rounds:
        tag = "promoted" if r.decision.promoted else "rejected"
        lines.append(f"**Round {r.round_num}** ({tag}) — {r.rationale}")
        for m in r.mutations:
            lines.append(f"  - `{m.axis}`: {m.summary}")
        lines.append("")

    lines += [
        "## Known limitations",
        "",
        "- The frozen north-star prevents the agent from gaming its objective, but does "
        "not solve Goodhart in general (a rubric can be satisfied yet miss real pedagogy). "
        "A production north-star needs human raters and adversarial items.",
        "- The eval suite is small, so the test number is directional; the protocol "
        "(train / held-out / test + regression gate) is the contribution.",
        "- Offline runs use a deterministic structural mock — an honest simulation of the "
        "pipeline, not a claim about real-model quality. Run with `--provider claude` for that.",
    ]
    return "\n".join(lines) + "\n"

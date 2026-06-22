"""Proposal engine — the agent authoring one change to itself per round.

A proposal targets the top-ranked gap with exactly one *generative* axis:
  - prompt_edit: append a directive to the draft prompt (changes generation ->
    can move the frozen north-star);
  - tool_synthesis: design a new check for a gap prompts can't fix.

It also reshapes the agent's *mutable* reward to emphasize the criterion under
attack (refining its own reward function). Reward changes ride along but provably
cannot affect generation, so the north-star delta is still cleanly attributable to
the single generative axis.

Two strategies share one action schema:
  - heuristic: deterministic, template-authored (mock-safe, used everywhere);
  - llm: Claude authors the directive text (opt-in; falls back to heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass

from lesson_loom.core.schemas import (
    DETERMINISTIC_SCORERS,
    MutationRecord,
    RewardWeights,
    Scorecard,
    SystemConfig,
)
from lesson_loom.optimizer import tools
from lesson_loom.optimizer.failure_analyzer import Gap

# Directives are written so the structural mock (and a real model) both respond.
PROMPT_DIRECTIVE: dict[str, str] = {
    "factual_grounding": "Cite every fact you use with its [fact_id] tag so each claim is grounded.",
    "no_hallucination": "Cite every fact you use with its [fact_id] tag so each claim is grounded.",
    "readability": "Write at the specified target reading level for the grade band.",
    "objective_coverage": "Address every learning objective and tag each one with its [obj_id].",
    "pedagogical_soundness": "Scaffold from prior knowledge and build complexity gradually.",
    "clarity": "Use clear, well-structured explanations for the target grade.",
    "engagement": "Make it engaging with a guiding question and a worked example.",
}


@dataclass
class Proposal:
    config: SystemConfig
    axis: str  # "prompt" | "tool"
    target: str
    rationale: str
    mutations: list[MutationRecord]


def _refine_reward(config: SystemConfig, target: str) -> RewardWeights:
    """Up-weight the criterion under attack; keep deterministic floors + a cap."""
    w = dict(config.reward_weights.weights)
    if target in w:
        w[target] = min(0.35, w[target] * 1.5)
    for d in DETERMINISTIC_SCORERS:
        w[d] = max(w.get(d, 0.0), 0.10)
    return RewardWeights(weights=w).normalized()


def _author_directive(strategy: str, provider, criterion: str) -> str:
    if strategy == "llm" and provider is not None and provider.name == "claude":
        ask = (
            f"Write ONE concise instruction to add to a content-generation prompt so the "
            f"output improves on '{criterion}'. Reply with only the instruction sentence."
        )
        try:
            text = provider.complete("You improve prompts.", ask).text.strip()
            if text:
                return text
        except Exception:
            pass
    return PROMPT_DIRECTIVE[criterion]


class ProposalEngine:
    def __init__(self, strategy: str = "heuristic", provider=None):
        self.strategy = strategy
        self.provider = provider

    def propose(
        self, config: SystemConfig, gaps: list[Gap], train: Scorecard, attempts: set[tuple[str, str]]
    ) -> Proposal | None:
        for gap in gaps:
            c = gap.criterion

            # 1) Prefer a prompt edit (changes generation, can move the north-star).
            if ("prompt", c) not in attempts and c in PROMPT_DIRECTIVE:
                directive = _author_directive(self.strategy, self.provider, c)
                prompts = dict(config.generation_prompts)
                if directive not in prompts["draft"]:
                    prompts["draft"] = (prompts["draft"].rstrip() + " " + directive).strip()
                reward = _refine_reward(config, c)
                cand = config.evolve(generation_prompts=prompts, reward_weights=reward)
                muts = [MutationRecord(axis="prompt", summary=f"draft += “{directive}”")]
                if reward.weights != config.reward_weights.weights:
                    muts.append(MutationRecord(axis="reward", summary=f"up-weighted {c}"))
                return Proposal(
                    cand, "prompt", c,
                    rationale=(
                        f"{c} mean {gap.mean:.2f} < threshold; appended a draft directive "
                        f"targeting {c}"
                    ),
                    mutations=muts,
                )

            # 2) Otherwise synthesize a check-tool for the gap.
            if ("tool", c) not in attempts:
                tool = tools.synthesize_for(c, train)
                if tool and tool.name not in config.enabled_scorers:
                    reward = _refine_reward(config, c)
                    cand = config.evolve(
                        synthesized_tools=[*config.synthesized_tools, tool],
                        enabled_scorers=[*config.enabled_scorers, tool.name],
                        reward_weights=reward,
                    )
                    return Proposal(
                        cand, "tool", c,
                        rationale=tool.rationale,
                        mutations=[MutationRecord(axis="tool", summary=f"synthesized {tool.name}")],
                    )
        return None

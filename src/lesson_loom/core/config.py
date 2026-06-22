"""The baseline SystemConfig the optimizer starts from.

The seed prompts are deliberately MINIMAL — they contain none of the levers the
optimizer can add (reading-level targeting, objective enumeration, citation
directives). That headroom is what lets the self-improvement loop show a real,
attributable gain rather than starting at the ceiling.
"""

from __future__ import annotations

from lesson_loom.core.northstar import NORTHSTAR_WEIGHTS
from lesson_loom.core.schemas import BASE_SCORERS, RewardWeights, SystemConfig

# Intentionally weak baseline prompts (one bare instruction each, no placeholders
# — all variable data flows through the user block, so the optimizer can append
# directives freely without ever breaking string formatting).
DEFAULT_PROMPTS: dict[str, str] = {
    "plan": "Outline the requested educational content.",
    "draft": "Write the requested educational content for the student.",
    "critique": "Review the draft. Reply ACCEPT or REVISE.",
    "revise": "Improve the draft.",
}


def default_config(
    provider_name: str = "mock",
    generation_model: str = "claude-sonnet-4-6",
    judge_model: str = "claude-opus-4-8",
    seed: int = 42,
) -> SystemConfig:
    # Reward starts equal to the north-star, then the optimizer is free to reshape it.
    return SystemConfig(
        generation_prompts=dict(DEFAULT_PROMPTS),
        reward_weights=RewardWeights(weights=dict(NORTHSTAR_WEIGHTS)),
        enabled_scorers=list(BASE_SCORERS),
        provider_name=provider_name,
        generation_model=generation_model,
        judge_model=judge_model,
        seed=seed,
    )

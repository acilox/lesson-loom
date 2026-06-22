"""Eval harness: generate -> score -> aggregate into a Scorecard.

Two aggregates per scorecard:
  - northstar: the FROZEN promotion metric (base scorers only).
  - reward:    the MUTABLE objective the optimizer shapes (its weights, over all
               enabled scorers incl. synthesized tools).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import yaml

from lesson_loom.core.northstar import northstar_score
from lesson_loom.core.schemas import (
    BASE_SCORERS,
    ContentArtifact,
    ContextPack,
    EvalItem,
    ItemScorecard,
    Scorecard,
    SystemConfig,
)
from lesson_loom.evals.cache import ResultCache
from lesson_loom.evals.scorers import REGISTRY
from lesson_loom.evals.scorers.synthesized import evaluate_tool
from lesson_loom.generation.graph import generate
from lesson_loom.providers.base import LLMProvider

_DATASETS = {"science": "science.yaml", "history": "history.yaml"}


@cache
def _load_all(subject: str) -> tuple[EvalItem, ...]:
    text = (
        resources.files("lesson_loom.evals.datasets")
        .joinpath(_DATASETS[subject.lower()])
        .read_text(encoding="utf-8")
    )
    return tuple(EvalItem(**row) for row in yaml.safe_load(text))


def load_eval_items(subject: str, split: str | None = None) -> list[EvalItem]:
    items = list(_load_all(subject))
    return [i for i in items if split is None or i.split == split]


def _reward(scores: dict, config: SystemConfig) -> float:
    w = config.reward_weights
    num = sum(w.get(name) * cs.normalized for name, cs in scores.items())
    den = sum(w.get(name) for name in scores)
    if den <= 0:  # degenerate weighting -> plain mean
        return sum(cs.normalized for cs in scores.values()) / max(1, len(scores))
    return num / den


def score_item(
    provider: LLMProvider, config: SystemConfig, pack: ContextPack, item: EvalItem
) -> tuple[ItemScorecard, ContentArtifact]:
    artifact = generate(
        provider=provider,
        config=config,
        context_pack=pack,
        topic=item.topic,
        grade_level=item.grade_level,
        content_type=item.content_type,
        reading_low=item.reading_level_low,
        reading_high=item.reading_level_high,
    )
    scores = {}
    # base scorers always run (north-star needs them)
    for name in BASE_SCORERS:
        scores[name] = REGISTRY[name](artifact, item, pack, provider)
    # synthesized tools the optimizer has added
    for tool in config.synthesized_tools:
        scores[tool.name] = evaluate_tool(tool, artifact, item, pack)
    return ItemScorecard(eval_item_id=item.id, artifact_id=artifact.id, scores=scores), artifact


def run_eval(
    provider: LLMProvider,
    config: SystemConfig,
    pack: ContextPack,
    items: list[EvalItem],
    split: str = "train",
    cache: ResultCache | None = None,
) -> Scorecard:
    cache = cache or ResultCache()
    item_cards: list[ItemScorecard] = []
    ns_vals: list[float] = []
    reward_vals: list[float] = []
    crit_acc: dict[str, list[float]] = {}

    for item in items:
        k = cache.key(config, item, provider.name)
        cached = cache.get(k)
        if cached:
            _, card = cached
        else:
            card, artifact = score_item(provider, config, pack, item)
            cache.set(k, artifact, card)
        item_cards.append(card)
        ns_vals.append(northstar_score(card))
        reward_vals.append(_reward(card.scores, config))
        for name, cs in card.scores.items():
            crit_acc.setdefault(name, []).append(cs.normalized)

    n = max(1, len(item_cards))
    return Scorecard(
        system_config_id=config.id,
        split=split,
        items=item_cards,
        northstar=round(sum(ns_vals) / n, 4),
        reward=round(sum(reward_vals) / n, 4),
        per_criterion_means={k: round(sum(v) / len(v), 4) for k, v in crit_acc.items()},
    )

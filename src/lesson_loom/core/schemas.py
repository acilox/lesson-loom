"""All Pydantic v2 schemas for lesson-loom.

Design notes that matter:
- Generation + scoring outputs carry NO wall-clock timestamps and NO random
  uuids. Every id is a content hash, so the same inputs produce byte-identical
  output (the determinism guarantee). Timestamps live only in the lineage store.
- `RewardWeights` is the MUTABLE objective the optimizer edits to guide its
  search. The frozen promotion metric (the "north-star") lives in
  `core/northstar.py` and is never editable by the optimizer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from lesson_loom.core.hashing import content_hash

# --- Canonical base scorer names (the fixed evaluation surface) ----------------

DETERMINISTIC_SCORERS = [
    "readability",
    "factual_grounding",
    "objective_coverage",
    "format_validity",
    "no_hallucination",
]
JUDGE_SCORERS = ["pedagogical_soundness", "clarity", "engagement"]
BASE_SCORERS = DETERMINISTIC_SCORERS + JUDGE_SCORERS


# --- Context Pack -------------------------------------------------------------


class Fact(BaseModel):
    id: str  # e.g. "fact_001" — the citation key
    text: str
    source: str
    difficulty: int = Field(ge=1, le=5)


class ConceptNode(BaseModel):
    concept: str
    prerequisites: list[str] = []


class LearningObjective(BaseModel):
    id: str  # e.g. "obj_001"
    text: str
    bloom_level: str  # remember | understand | apply | analyze | evaluate | create


class Misconception(BaseModel):
    id: str
    text: str
    correction: str


class ContextPack(BaseModel):
    subject: str
    topic: str
    target_grade: str
    reading_level_low: float  # Flesch-Kincaid grade band, inclusive
    reading_level_high: float
    facts: list[Fact]
    concept_graph: list[ConceptNode] = []
    learning_objectives: list[LearningObjective]
    misconceptions: list[Misconception] = []
    difficulty_ladder: list[str] = []

    def fact_ids(self) -> set[str]:
        return {f.id for f in self.facts}

    def objective_ids(self) -> set[str]:
        return {o.id for o in self.learning_objectives}


# --- Content Artifact (the generated lesson) ----------------------------------


class CitationLink(BaseModel):
    fact_id: str
    excerpt: str


class ContentArtifact(BaseModel):
    id: str = ""  # filled deterministically in build()
    system_config_id: str
    topic: str
    content_type: str
    grade_level: str
    body: str
    citations: list[CitationLink] = []
    objectives_covered: list[str] = []

    @classmethod
    def build(cls, **kwargs) -> ContentArtifact:
        art = cls(**kwargs)
        art.id = content_hash(
            {
                "cfg": art.system_config_id,
                "topic": art.topic,
                "type": art.content_type,
                "grade": art.grade_level,
                "body": art.body,
            }
        )
        return art


# --- Eval items ---------------------------------------------------------------


class EvalCriterion(BaseModel):
    name: str
    description: str
    reference_answer: str | None = None


class EvalItem(BaseModel):
    id: str
    subject: str
    topic: str
    grade_level: str
    content_type: str
    split: str  # "train" | "heldout" | "test"
    criteria: list[EvalCriterion] = []
    reading_level_low: float
    reading_level_high: float
    required_objective_ids: list[str] = []
    required_fact_ids: list[str] = []


# --- Scorecards ---------------------------------------------------------------


class CriterionScore(BaseModel):
    name: str
    raw_value: float
    normalized: float = Field(ge=0.0, le=1.0)
    details: dict = {}


class ItemScorecard(BaseModel):
    eval_item_id: str
    artifact_id: str
    scores: dict[str, CriterionScore]  # scorer name -> score

    def normalized_map(self) -> dict[str, float]:
        return {k: v.normalized for k, v in self.scores.items()}


class Scorecard(BaseModel):
    """Aggregate over a split. `northstar` is the frozen promotion metric;
    `reward` is the mutable objective the optimizer is allowed to shape."""

    system_config_id: str
    split: str
    items: list[ItemScorecard]
    northstar: float
    reward: float
    per_criterion_means: dict[str, float]

    def criterion_mean(self, name: str) -> float:
        return self.per_criterion_means.get(name, 0.0)


# --- The mutable system the optimizer edits -----------------------------------


class SynthesizedTool(BaseModel):
    """A deterministic check the optimizer designed from a template + evidence."""

    name: str
    template: str  # "min_count" | "keyword_presence" | "threshold"
    params: dict
    rationale: str


class RewardWeights(BaseModel):
    """Mutable weighting over scorer names. The optimizer edits this to steer
    its own search — but it is NOT the promotion metric (see core/northstar)."""

    weights: dict[str, float]

    def normalized(self) -> RewardWeights:
        total = sum(self.weights.values()) or 1.0
        return RewardWeights(weights={k: v / total for k, v in self.weights.items()})

    def get(self, name: str) -> float:
        return self.weights.get(name, 0.0)


class SystemConfig(BaseModel):
    """The full mutable 'system'. Its id is a content hash, so identical configs
    share an id and any config can be reloaded and re-run exactly."""

    id: str = ""
    parent_id: str | None = None
    version: int = 0

    # Mutable by the optimizer
    generation_prompts: dict[str, str]  # node -> system prompt (plan/draft/critique/revise)
    reward_weights: RewardWeights
    enabled_scorers: list[str]
    synthesized_tools: list[SynthesizedTool] = []
    max_revisions: int = 2

    # Provider/runtime (NOT mutated by the optimizer)
    provider_name: str = "mock"  # "mock" | "claude"
    generation_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-opus-4-8"
    seed: int = 42

    @model_validator(mode="after")
    def _assign_id(self) -> SystemConfig:
        # id is a pure function of the mutable surface (+ models/seed), never of
        # parent_id/version/id themselves.
        object.__setattr__(
            self,
            "id",
            content_hash(
                {
                    "prompts": self.generation_prompts,
                    "reward": self.reward_weights.weights,
                    "scorers": sorted(self.enabled_scorers),
                    "tools": [t.model_dump() for t in self.synthesized_tools],
                    "max_revisions": self.max_revisions,
                    "gen_model": self.generation_model,
                    "judge_model": self.judge_model,
                    "seed": self.seed,
                }
            ),
        )
        return self

    def evolve(self, **changes) -> SystemConfig:
        """Return a child config with one set of changes applied, lineage wired."""
        data = self.model_dump()
        data.pop("id", None)
        data["parent_id"] = self.id
        data["version"] = self.version + 1
        data.update(changes)
        return SystemConfig(**data)

    def diff(self, other: SystemConfig) -> list[str]:
        out: list[str] = []
        sw, ow = self.reward_weights.weights, other.reward_weights.weights
        for k in sorted(set(sw) | set(ow)):
            a, b = sw.get(k, 0.0), ow.get(k, 0.0)
            if abs(a - b) > 1e-6:
                out.append(f"reward[{k}]: {a:.3f} -> {b:.3f}")
        for node in sorted(set(self.generation_prompts) | set(other.generation_prompts)):
            if self.generation_prompts.get(node) != other.generation_prompts.get(node):
                out.append(f"prompt[{node}] edited")
        if self.enabled_scorers != other.enabled_scorers:
            added = set(other.enabled_scorers) - set(self.enabled_scorers)
            if added:
                out.append(f"scorers += {sorted(added)}")
        if self.max_revisions != other.max_revisions:
            out.append(f"max_revisions: {self.max_revisions} -> {other.max_revisions}")
        return out


# --- Lineage records ----------------------------------------------------------


class MutationRecord(BaseModel):
    axis: str  # "prompt" | "reward" | "tool"
    summary: str  # human-readable diff line(s)


class PromotionDecision(BaseModel):
    promoted: bool
    reason: str
    northstar_train_delta: float
    heldout_delta: float
    goodhart_gap: float  # train_delta - heldout_delta


class ExperimentRound(BaseModel):
    round_num: int
    parent_config_id: str
    candidate_config_id: str
    axis: str
    target: str = ""
    rationale: str
    mutations: list[MutationRecord]
    train_northstar: float
    heldout_northstar: float
    decision: PromotionDecision


class ExperimentResult(BaseModel):
    experiment_id: str
    subject: str
    rounds: list[ExperimentRound]
    baseline_config_id: str
    best_config_id: str
    baseline_test_northstar: float
    best_test_northstar: float

    @property
    def test_gain(self) -> float:
        return self.best_test_northstar - self.baseline_test_northstar

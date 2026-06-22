# lesson-loom

**A self-improving educational-content engine.** An agent generates lesson
content grounded in structured **context packs**, a hybrid **eval harness** scores
it, and an **optimizer loop rewrites the system itself** — its own prompts, its
own reward, and even new check-tools it designs — keeping only the changes that
beat a **frozen north-star** without regressing a held-out set.

It runs **end-to-end offline in seconds with zero API keys** (deterministic
structural mock), and swaps to **Claude** (`claude-sonnet-4-6` generation,
`claude-opus-4-8` judge) behind the same interface when `ANTHROPIC_API_KEY` is set.

```
$ lesson-loom optimize --subject science --rounds 8

 #  axis    target               Δtrain   Δheld-out   verdict
 1  prompt  factual_grounding     +0.280   +0.280     ✅ promoted
 2  prompt  readability           +0.180   +0.180     ✅ promoted
 3  prompt  objective_coverage    +0.194   +0.196     ✅ promoted
 4  prompt  engagement            +0.000   +0.000     ⛔ rejected   (gate caught a no-op)
 5  tool    engagement            +0.000   +0.000     ✅ promoted   (agent designed a check)

TEST north-star: baseline 0.326 → best 0.978  (+0.652, frozen metric, unbiased split)
```

The agent reasoned about its own failures, made **one attributable change per
round**, got **rejected by its own regression gate** when a change didn't help,
and **synthesized a new check-tool** for a weakness prompts couldn't fix — all on
a metric it is **not allowed to edit**.

---

## Why this exists

Built as a reference implementation of the hardest parts of an AI-content system:

1. **Self-improving content generation** — an agent loop that generates, evaluates,
   and iterates on *itself*.
2. **Evaluation sets + experiment harness** — curated `train / held-out / test`
   datasets and a hybrid scorer suite with measurable success criteria.
3. **Context packs** — structured, validated domain knowledge that grounds
   generation and is the source of truth for citation checking.

Demo subjects are **Science (Photosynthesis)** and **History (Industrial
Revolution)** to show grounding across very different domains and reading levels.

## Architecture

```
context pack ─▶ LangGraph generation graph ─▶ content artifact (with citations)
 (facts,            plan → retrieve → draft           │
  objectives,        → self-critique ⇄ revise         ▼
  misconceptions,        → format            hybrid eval harness
  reading band)                              5 deterministic + 3 LLM-judge scorers
                                                       │
                                          ┌────────────┴─────────────┐
                                   FROZEN north-star          MUTABLE reward
                                   (promotion gate, KPI)      (guides the search)
                                                       │
                          optimizer loop:  failure analysis ─▶ propose ONE change
                          (prompt edit | reward reshape | TOOL SYNTHESIS)
                            ─▶ re-eval on train + held-out ─▶ promotion gate
                            ─▶ promote / reject ─▶ SQLite lineage  ─▶ REPORT.md
```

### The design decisions that matter

- **Frozen north-star vs. mutable reward.** The agent freely reshapes its own
  `RewardWeights` to steer its search, but the **promotion gate and the KPI use a
  frozen north-star it cannot touch** — so it can't win by moving the goalposts
  (the RLHF "don't let the policy edit the benchmark" principle). Synthesized tools
  never count toward the north-star either.
- **Single change per round + a stored rationale.** Every candidate differs from
  its parent on exactly one generative axis and carries a one-line evidence-based
  reason. The lineage is therefore a clean **attribution tree**, not a black box.
- **Regression-gated promotion.** A change is promoted only if it raises the
  train north-star by ≥ 0.02, doesn't drop the **held-out** set past a floor, and
  doesn't let any single criterion collapse. The **test** split is never seen
  during search or gating and is reported once as the headline number.
- **Designs its own tools.** When a weakness can't be fixed by prompting, the
  agent synthesizes a new deterministic check from a sandboxed template library,
  **deriving its parameters from the failure evidence**, and keeps it only if it's
  informative and harmless — the same gate discipline as a prompt edit.
- **Honest offline mode.** The mock provider responds to the *structure* of a
  prompt (does it ask for a reading level? citations? objective coverage?), not to
  magic strings — so prompt self-edits genuinely move the deterministic scores and
  the result generalizes across items. It's an honest simulation of the pipeline,
  not a claim about real-model pedagogy (that's the `--provider claude` path).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

lesson-loom optimize --subject science --rounds 8   # full loop, offline, < 5s
lesson-loom optimize --subject history --rounds 8
lesson-loom lineage                                  # query the attribution tree
lesson-loom eval --subject science --split test      # score the baseline
lesson-loom generate --subject science               # one grounded artifact
lesson-loom serve                                    # operator console at :8000

pytest --cov=lesson_loom                             # 27 tests, ~90% coverage
```

Run against real Claude (optional):

```bash
pip install -e ".[claude]"
export ANTHROPIC_API_KEY=sk-...
lesson-loom optimize --subject science --provider claude --strategy llm \
  --gen-model claude-sonnet-4-6 --judge-model claude-opus-4-8
```

## Project layout

```
src/lesson_loom/
  core/         schemas, frozen north-star, baseline config, content hashing
  providers/    LLMProvider interface · structural mock · Claude
  context_packs/ validated knowledge artifacts (science, history)
  generation/   LangGraph graph + pure nodes + prompt builders
  evals/        scorers (5 deterministic + 3 judge + synthesized) · harness · cache
  optimizer/    failure analysis · proposal engine · tool synthesis ·
                promotion gate · SQLite lineage store · controller · report
  app/          thin FastAPI operator console
tests/          determinism, scorers, graph, gate, tool synthesis, loop, cli, api
```

## Eval scorers

Deterministic: `readability` (Flesch-Kincaid vs target band), `factual_grounding`
(required pack facts cited), `objective_coverage`, `format_validity`,
`no_hallucination` (cited ids must exist in the pack). LLM-judge:
`pedagogical_soundness`, `clarity`, `engagement`. Each normalizes to `[0,1]`; the
north-star is a fixed weighting over the base set.

## Known limitations (by design, stated honestly)

- The frozen north-star stops the agent gaming its objective but does **not** solve
  Goodhart in general — a rubric can be satisfied yet miss real pedagogy. A
  production north-star needs human raters and adversarial items.
- The eval suite is small, so the test number is **directional**; the contribution
  is the protocol (train / held-out / test + regression-gated promotion + auditable
  lineage), which scales to larger suites unchanged.
- Offline runs use a deterministic structural mock — honest about the *pipeline*,
  not about real-model quality. Use `--provider claude` to judge real content.

## License

MIT.

"""The headline guarantee: identical inputs -> byte-identical outputs."""

from lesson_loom.evals.harness import load_eval_items, run_eval
from lesson_loom.generation.graph import generate


def _gen(provider, config, pack):
    return generate(
        provider=provider, config=config, context_pack=pack, topic=pack.topic,
        grade_level=pack.target_grade, content_type="explainer",
        reading_low=pack.reading_level_low, reading_high=pack.reading_level_high,
    )


def test_generation_is_byte_identical(provider, base_config, science_pack):
    a = _gen(provider, base_config, science_pack)
    b = _gen(provider, base_config, science_pack)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.id == b.id  # id is content-derived


def test_eval_is_deterministic(provider, base_config, science_pack):
    items = load_eval_items("science", "train")
    s1 = run_eval(provider, base_config, science_pack, items, split="train")
    s2 = run_eval(provider, base_config, science_pack, items, split="train")
    assert s1.northstar == s2.northstar
    assert s1.per_criterion_means == s2.per_criterion_means


def test_config_id_is_stable_and_content_addressed(base_config):
    from lesson_loom.core.config import default_config

    assert base_config.id == default_config().id  # same content -> same id
    evolved = base_config.evolve(max_revisions=5)
    assert evolved.id != base_config.id
    assert evolved.parent_id == base_config.id
    assert evolved.version == base_config.version + 1

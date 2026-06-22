"""Builders for the user-message blocks the generation nodes send to the provider.

All variable data (topic, grade, target reading level, retrieved facts and
objectives) travels in these blocks. The (optimizer-mutable) system prompts stay
free of placeholders, so a mutated prompt can never break formatting.
"""

from __future__ import annotations

from lesson_loom.core.schemas import Fact, LearningObjective


def build_plan_user(topic: str, content_type: str, grade: str) -> str:
    return f"TOPIC: {topic}\nCONTENT_TYPE: {content_type}\nGRADE: {grade}"


def build_draft_user(
    topic: str,
    content_type: str,
    grade: str,
    reading_low: float,
    reading_high: float,
    facts: list[Fact],
    objectives: list[LearningObjective],
) -> str:
    fact_lines = "\n".join(f"- {f.id}: {f.text}" for f in facts)
    obj_lines = "\n".join(f"- {o.id}: {o.text}" for o in objectives)
    return (
        f"TOPIC: {topic}\n"
        f"CONTENT_TYPE: {content_type}\n"
        f"GRADE: {grade}\n"
        f"READING_TARGET: {reading_low}-{reading_high}\n"
        f"FACTS:\n{fact_lines}\n"
        f"OBJECTIVES:\n{obj_lines}"
    )


def build_judge_user(rubric: str, body: str) -> str:
    # BODY goes last so the mock judge can read to end-of-message.
    return f"RUBRIC: {rubric}\nBODY: {body}"

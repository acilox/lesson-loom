"""Load + validate Context Packs from YAML fixtures."""

from __future__ import annotations

from functools import cache
from importlib import resources

import yaml

from lesson_loom.core.schemas import ContextPack

_FIXTURES = {
    "science": "science_photosynthesis.yaml",
    "history": "history_industrial_revolution.yaml",
}


@cache
def load_pack(subject: str) -> ContextPack:
    subject = subject.lower()
    if subject not in _FIXTURES:
        raise ValueError(f"unknown subject {subject!r}; have {sorted(_FIXTURES)}")
    text = (
        resources.files("lesson_loom.context_packs.fixtures")
        .joinpath(_FIXTURES[subject])
        .read_text(encoding="utf-8")
    )
    return ContextPack(**yaml.safe_load(text))


def available_subjects() -> list[str]:
    return sorted(_FIXTURES)

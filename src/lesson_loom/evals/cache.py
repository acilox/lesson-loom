"""In-memory result cache.

Keyed on content hashes — config id (already a content hash), the eval item's
content, and the provider name — so reloading an identical config hits the same
entry and a mock result never collides with a Claude result. Caching whole
ItemScorecards (with the artifact) makes the optimizer loop fast and lets you
audit exactly what was produced for any (config, item) pair without re-running.
"""

from __future__ import annotations

from lesson_loom.core.hashing import content_hash
from lesson_loom.core.schemas import ContentArtifact, EvalItem, ItemScorecard, SystemConfig


class ResultCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[ContentArtifact, ItemScorecard]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(config: SystemConfig, item: EvalItem, provider_name: str) -> str:
        return f"{config.id}:{content_hash(item)}:{provider_name}"

    def get(self, k: str):
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def set(self, k: str, artifact: ContentArtifact, scorecard: ItemScorecard) -> None:
        self._store[k] = (artifact, scorecard)

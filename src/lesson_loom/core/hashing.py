"""Content-addressable hashing.

Cache keys and config identities are derived from content, never from
auto-increment ids — so reloading a config with the same content always hits
the same cache entry, and two different configs can never collide.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace jitter, pydantic-aware."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(obj: Any, *, length: int = 12) -> str:
    """Short, stable sha256 of any JSON-serializable (or pydantic) object."""
    digest = hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()
    return digest[:length]


def text_hash(text: str, *, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

"""Shared cache helpers.

In-memory cache for ontology lookups / FAISS indices, plus small file-layout
helpers (cache-dir convention, metadata sidecar). Stage 4 will add a SQLite
cache for LLM responses keyed by (model, prompt_hash); for now this module is
the in-memory side only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable
from pathlib import Path


def ontology_cache_dir(cache_root: Path, name: str) -> Path:
    return cache_root / "ontologies" / name


def write_metadata_sidecar(
    directory: Path, payload: dict, *, filename: str = "metadata.json"
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(payload, indent=2) + "\n")


class KeyedCache[K: Hashable, V]:
    """Thread-unsafe in-memory cache with explicit invalidation.

    Used to amortize expensive `load_*` calls (unpickling a 27 MB Cellosaurus
    lookup, reading a 245 MB FAISS index, loading a sentence-transformer
    model) across repeat `map_chipatlas` calls in the same process.
    """

    def __init__(self) -> None:
        self._store: dict[K, V] = {}

    def get_or_load(self, key: K, loader: Callable[[], V]) -> V:
        cached = self._store.get(key)
        if cached is not None:
            return cached
        value = loader()
        self._store[key] = value
        return value

    def invalidate(self, key: K) -> None:
        self._store.pop(key, None)

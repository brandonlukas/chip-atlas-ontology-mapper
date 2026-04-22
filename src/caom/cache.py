"""Shared cache helpers.

In-memory cache for ontology lookups / FAISS indices, plus a SQLite cache for
LLM responses keyed by `(model, prompt_hash)`. Small file-layout helpers
(cache-dir convention, metadata sidecar) also live here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Hashable
from datetime import UTC, datetime
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


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite-backed cache of LLM responses keyed by `(model, prompt_hash)`.

    The cache stores the raw JSON produced by `LLMPick.model_dump()`. On a hit,
    callers revalidate through pydantic so a schema bump still rejects stale
    rows instead of silently returning mismatched shapes.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS llm_cache ("
        "  model TEXT NOT NULL,"
        "  prompt_hash TEXT NOT NULL,"
        "  response TEXT NOT NULL,"
        "  created_at TEXT NOT NULL,"
        "  PRIMARY KEY (model, prompt_hash)"
        ")"
    )

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        # WAL + synchronous=NORMAL: `put` is called once per LLM miss, so
        # per-commit fsync dominates write cost without these pragmas.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    def get(self, model: str, prompt: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT response FROM llm_cache WHERE model = ? AND prompt_hash = ?",
            (model, _prompt_hash(prompt)),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row is not None else None

    def put(self, model: str, prompt: str, response: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (model, prompt_hash, response, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                model,
                _prompt_hash(prompt),
                json.dumps(response, sort_keys=True),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LLMCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def llm_cache_path(cache_root: Path) -> Path:
    return cache_root / "llm" / "llm_cache.sqlite"


_LLM_CACHE_CACHE: KeyedCache[Path, LLMCache] = KeyedCache()


def get_cached_llm_cache(cache_root: Path) -> LLMCache:
    """Reuse a single open SQLite connection per cache_root across calls.

    Opening + PRAGMA + schema check adds up across many `map_chipatlas` calls,
    and orphaned connections can leave WAL pages pending on exit.
    """
    return _LLM_CACHE_CACHE.get_or_load(
        cache_root, lambda: LLMCache(llm_cache_path(cache_root))
    )

"""Orchestrator for ontology + embedding index setup."""

from __future__ import annotations

from pathlib import Path

from caom.config import Config, load_config
from caom.ontologies import cellosaurus


def update_ontologies(
    *,
    force: bool = False,
    cache_dir: str | Path | None = None,
    config: Config | None = None,
) -> None:
    """Download Cellosaurus (and later EFO + FAISS) into the cache.

    Parameters
    ----------
    force
        If True, re-download and rebuild even when a cached copy exists.
    cache_dir
        Override the cache directory. Defaults to `$CAOM_CACHE_DIR` or `./.cache`.
    """
    cfg = config or load_config(cache_dir=cache_dir)
    cellosaurus.refresh_cache(cfg.cache_dir, force=force)
    # Stage 3 will add EFO download + FAISS build here.

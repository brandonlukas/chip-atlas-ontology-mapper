"""Orchestrator for ontology + embedding index setup."""

from __future__ import annotations

from pathlib import Path

from caom.config import Config, load_config
from caom.ontologies import cellosaurus, efo
from caom.retrieval import index as index_mod
from caom.retrieval.embedder import SentenceTransformerEmbedder


def update_ontologies(
    *,
    force: bool = False,
    cache_dir: str | Path | None = None,
    config: Config | None = None,
) -> None:
    """Download Cellosaurus + EFO and build the EFO FAISS index into the cache.

    Parameters
    ----------
    force
        If True, re-download and rebuild even when cached copies exist.
    cache_dir
        Override the cache directory. Defaults to `$CAOM_CACHE_DIR` or `./.cache`.
    """
    cfg = config or load_config(cache_dir=cache_dir)

    cellosaurus.refresh_cache(cfg.cache_dir, force=force)
    efo_terms = efo.refresh_cache(cfg.cache_dir, force=force)

    if not force and index_mod.is_cached(cfg.cache_dir):
        return

    # Narrow the FAISS corpus to legitimate ChIP-Atlas context prefixes before
    # embedding. This both avoids ~70 MB of wasted embedding compute on PR /
    # OBA / HGNC terms and prevents them from contaminating retrieval.
    filtered_terms = index_mod.filter_corpus(efo_terms.terms)

    # Instantiate (not cache) the embedder: it's only needed for this one-shot
    # build, and the sentence-transformer eats ~500 MB of VRAM.
    emb = SentenceTransformerEmbedder(cfg.embedding_model)
    # itertuples avoids the per-row pd.Series allocation that iterrows does.
    corpus_texts = [
        index_mod.build_corpus_text(
            {"label": r.label, "synonyms": r.synonyms, "definition": r.definition}
        )
        for r in filtered_terms.itertuples(index=False)
    ]
    vectors = emb.encode(corpus_texts, show_progress=True)
    efo_index = index_mod.build_index(
        terms_df=filtered_terms,
        embeddings=vectors,
        embedding_model=cfg.embedding_model,
        efo_version=efo_terms.version,
    )
    index_mod.save_index(cfg.cache_dir, efo_index)
    index_mod.invalidate_cache(cfg.cache_dir)

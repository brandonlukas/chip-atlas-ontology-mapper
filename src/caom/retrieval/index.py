"""FAISS index build + top-K query with row-aligned metadata parquet. Stage 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from caom.cache import KeyedCache, write_metadata_sidecar
from caom.retrieval.embedder import EmbedderProtocol
from caom.types import Candidate

INDEX_FILENAME = "efo.faiss"
TERMS_FILENAME = "efo_terms.parquet"
META_FILENAME = "efo.metadata.json"


def _cache_dir(cache_root: Path) -> Path:
    return cache_root / "embeddings"


def _coerce_synonyms(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def build_corpus_text(row: pd.Series | dict) -> str:
    """Concatenate label + synonyms + definition for embedding.

    Pipe-separated so the tokenizer sees clear segment boundaries.
    """
    parts: list[str] = []
    label = row["label"]
    if label:
        parts.append(str(label))
    syns = _coerce_synonyms(row["synonyms"])
    if syns:
        parts.append("; ".join(syns))
    definition = row["definition"]
    if definition:
        parts.append(str(definition))
    return " | ".join(parts)


@dataclass
class EFOIndex:
    """FAISS cosine-similarity index + row-aligned EFO terms DataFrame."""

    faiss_index: Any  # faiss.Index; typed as Any to avoid import at module load.
    terms: pd.DataFrame
    embedding_model: str
    efo_version: str
    built_at: str

    def search_vectors(
        self, query_vecs: np.ndarray, top_k: int
    ) -> list[list[Candidate]]:
        """Search with pre-computed unit-norm query vectors."""
        if query_vecs.size == 0:
            return []
        if query_vecs.ndim == 1:
            query_vecs = query_vecs[None, :]
        if query_vecs.dtype != np.float32:
            query_vecs = query_vecs.astype(np.float32)
        if not query_vecs.flags["C_CONTIGUOUS"]:
            query_vecs = np.ascontiguousarray(query_vecs)

        k = min(top_k, self.faiss_index.ntotal)
        if k <= 0:
            return [[] for _ in range(query_vecs.shape[0])]
        scores, indices = self.faiss_index.search(query_vecs, k)

        results: list[list[Candidate]] = []
        for q_scores, q_idx in zip(scores, indices, strict=True):
            cands: list[Candidate] = []
            for s, i in zip(q_scores, q_idx, strict=True):
                if i < 0:
                    continue
                row = self.terms.iloc[int(i)]
                definition = row["definition"] if row["definition"] else None
                cands.append(
                    Candidate(
                        ontology_id=str(row["ontology_id"]),
                        ontology_label=str(row["label"]),
                        ontology_source="efo",
                        synonyms=_coerce_synonyms(row["synonyms"]),
                        definition=str(definition) if definition is not None else None,
                        retrieval_score=float(s),
                    )
                )
            results.append(cands)
        return results

    def search_texts(
        self,
        texts: list[str],
        embedder: EmbedderProtocol,
        top_k: int,
    ) -> list[list[Candidate]]:
        """Encode `texts` via `embedder`, then search_vectors."""
        if not texts:
            return []
        vecs = embedder.encode(texts)
        return self.search_vectors(vecs, top_k=top_k)


def build_index(
    terms_df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_model: str,
    efo_version: str,
) -> EFOIndex:
    """Build an IndexFlatIP index over L2-normalized `embeddings`.

    Row i in `embeddings` must correspond to row i in `terms_df`.
    """
    import faiss

    if len(terms_df) != len(embeddings):
        raise ValueError(
            f"terms_df rows ({len(terms_df)}) != embeddings rows ({len(embeddings)})"
        )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if not embeddings.flags["C_CONTIGUOUS"]:
        embeddings = np.ascontiguousarray(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return EFOIndex(
        faiss_index=index,
        terms=terms_df.reset_index(drop=True),
        embedding_model=embedding_model,
        efo_version=efo_version,
        built_at=datetime.now(UTC).isoformat(),
    )


def save_index(cache_root: Path, index: EFOIndex) -> None:
    import faiss

    d = _cache_dir(cache_root)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.faiss_index, str(d / INDEX_FILENAME))
    index.terms.to_parquet(d / TERMS_FILENAME, index=False)
    write_metadata_sidecar(
        d,
        {
            "embedding_model": index.embedding_model,
            "efo_version": index.efo_version,
            "built_at": index.built_at,
            "term_count": len(index.terms),
            "dim": int(index.faiss_index.d),
        },
        filename=META_FILENAME,
    )


def load_index(cache_root: Path) -> EFOIndex:
    import faiss

    d = _cache_dir(cache_root)
    idx_path = d / INDEX_FILENAME
    if not idx_path.exists():
        raise FileNotFoundError(
            f"EFO FAISS index not found at {idx_path}. "
            "Run `caom.update_ontologies()` first."
        )
    faiss_index = faiss.read_index(str(idx_path))
    terms = pd.read_parquet(d / TERMS_FILENAME)
    meta = json.loads((d / META_FILENAME).read_text())
    return EFOIndex(
        faiss_index=faiss_index,
        terms=terms,
        embedding_model=meta.get("embedding_model", ""),
        efo_version=meta.get("efo_version", ""),
        built_at=meta.get("built_at", ""),
    )


def is_cached(cache_root: Path) -> bool:
    d = _cache_dir(cache_root)
    return (
        (d / INDEX_FILENAME).exists()
        and (d / TERMS_FILENAME).exists()
        and (d / META_FILENAME).exists()
    )


_INDEX_CACHE: KeyedCache[Path, EFOIndex] = KeyedCache()


def get_cached_index(cache_root: Path) -> EFOIndex:
    """Load the EFO FAISS index, reusing an in-memory copy across calls."""
    return _INDEX_CACHE.get_or_load(cache_root, lambda: load_index(cache_root))


def invalidate_cache(cache_root: Path) -> None:
    """Drop the in-memory index for `cache_root` (after a rebuild)."""
    _INDEX_CACHE.invalidate(cache_root)

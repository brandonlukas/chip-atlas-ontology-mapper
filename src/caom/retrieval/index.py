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

# Allow-list of CURIE prefixes that are legitimate answers to "what cell /
# tissue / cell-line / disease context was this ChIP-Atlas assay run in".
#
# EFO imports large slices of unrelated ontologies (PR ~17k proteins, OBA
# ~17k biological attributes, HGNC ~6k genes, HP phenotypes, CHEBI chemicals,
# NCBITaxon, GO, dbpedia geographic terms, etc.). Their labels lexically
# overlap common cell-type queries (e.g. "CD4" surfaces as `PR:*` CD4 protein
# variants rather than `CL:0000624` CD4-positive T cell), so a PubMedBert
# cosine search over the full corpus reliably buries the correct UBERON / CL
# / EFO answer beneath non-context noise.
#
# Allow-list (not deny-list) so future ontology refreshes can't silently
# re-introduce contamination — a new import is excluded by default until a
# human adds it here. Anything filtered out here still lives in the raw
# `terms.parquet`; only the FAISS corpus is narrowed.
_ALLOWED_PREFIXES: frozenset[str] = frozenset({
    # Core context ontologies.
    "CL",        # Cell Ontology — primary cell types
    "UBERON",    # cross-species anatomy
    "EFO",       # Experimental Factor Ontology (cell lines, contexts, diseases)
    "CLO",       # Cell Line Ontology
    "BTO",       # BRENDA Tissue Ontology
    "MONDO",     # disease ontology
    "Orphanet",  # rare-disease IDs surfaced via MONDO / EFO
    "NCIT",      # cancer-type entries occasionally used as contexts
    # Organism-specific anatomy / life-stage ontologies.
    "FBbt",      # Drosophila anatomy
    "FBdv",      # Drosophila development
    "ZFA",       # zebrafish anatomy
    "MA",        # mouse anatomy
    "FMA",       # Foundational Model of Anatomy (human)
    "PO",        # Plant Ontology
    "WBls",      # C. elegans life stage
})


def filter_corpus(terms_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict `terms_df` to rows whose `ontology_id` prefix is allow-listed.

    EFO imports unrelated ontologies (PR, OBA, HGNC, HP, CHEBI, GO, dbpedia,
    etc.) whose labels lexically overlap cell-type queries and bury the
    correct UBERON / CL / EFO answer in retrieval. Allow-listing (not
    deny-list) ensures future refreshes exclude unvetted imports by default.
    Run before embedding to skip wasted compute and prevent contamination.
    """
    prefixes = terms_df["ontology_id"].astype(str).str.split(":", n=1).str[0]
    mask = prefixes.isin(_ALLOWED_PREFIXES)
    return terms_df.loc[mask].reset_index(drop=True)


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

"""Unit tests for the retrieval layer: corpus text, FAISS index (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caom.retrieval.index import (
    EFOIndex,
    build_corpus_text,
    build_index,
    get_cached_index,
    is_cached,
    load_index,
    save_index,
)


def _mini_terms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ontology_id": "EFO:0001187",
                "label": "HEK293",
                "synonyms": ["HEK-293", "HEK 293"],
                "definition": "Human embryonic kidney cell line.",
                "parents": ["CL:0000000"],
            },
            {
                "ontology_id": "CL:0000182",
                "label": "hepatocyte",
                "synonyms": ["liver cell"],
                "definition": "A cell of the liver.",
                "parents": ["CL:0000000"],
            },
            {
                "ontology_id": "EFO:K562",
                "label": "K562",
                "synonyms": [],
                "definition": "Chronic myeloid leukemia cell line.",
                "parents": [],
            },
        ]
    )


# -- build_corpus_text -------------------------------------------------------


def test_build_corpus_text_joins_label_synonyms_definition():
    terms = _mini_terms()
    text = build_corpus_text(terms.iloc[0])
    assert text == "HEK293 | HEK-293; HEK 293 | Human embryonic kidney cell line."


def test_build_corpus_text_drops_empty_sections():
    row = {"label": "Foo", "synonyms": [], "definition": None}
    assert build_corpus_text(row) == "Foo"

    row = {"label": "", "synonyms": ["bar"], "definition": None}
    assert build_corpus_text(row) == "bar"


# -- build / save / load index ----------------------------------------------


def _orthonormal_embeddings(n: int, dim: int) -> np.ndarray:
    """n unit vectors, one hot in distinct dims so search ordering is exact."""
    assert n <= dim
    m = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        m[i, i] = 1.0
    return m


def test_build_index_rejects_mismatched_shapes():
    terms = _mini_terms()
    with pytest.raises(ValueError, match="embeddings rows"):
        build_index(
            terms_df=terms,
            embeddings=np.zeros((2, 4), dtype=np.float32),
            embedding_model="test",
            efo_version="99.9",
        )


def test_build_save_load_roundtrip(tmp_path: Path):
    terms = _mini_terms()
    embs = _orthonormal_embeddings(len(terms), dim=4)

    idx = build_index(terms, embs, embedding_model="test-model", efo_version="99.9")
    save_index(tmp_path, idx)

    assert (tmp_path / "embeddings" / "efo.faiss").exists()
    assert (tmp_path / "embeddings" / "efo_terms.parquet").exists()
    assert (tmp_path / "embeddings" / "efo.metadata.json").exists()
    assert is_cached(tmp_path)

    reloaded = load_index(tmp_path)
    assert reloaded.embedding_model == "test-model"
    assert reloaded.efo_version == "99.9"
    assert reloaded.faiss_index.ntotal == 3
    assert list(reloaded.terms["ontology_id"]) == list(terms["ontology_id"])


def test_search_returns_top_k_sorted_candidates():
    terms = _mini_terms()
    embs = _orthonormal_embeddings(len(terms), dim=4)
    idx = build_index(terms, embs, embedding_model="test", efo_version="99.9")

    # Query aligned with hepatocyte's embedding (row 1).
    q = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    result = idx.search_vectors(q, top_k=2)
    assert len(result) == 1
    cands = result[0]
    assert len(cands) == 2
    assert cands[0].ontology_id == "CL:0000182"
    assert cands[0].ontology_label == "hepatocyte"
    assert cands[0].ontology_source == "efo"
    assert cands[0].retrieval_score == pytest.approx(1.0)
    assert "liver cell" in cands[0].synonyms
    # Second-best should be one of the orthogonal rows with score 0.
    assert cands[1].retrieval_score == pytest.approx(0.0)


def test_search_clamps_top_k_to_corpus_size():
    terms = _mini_terms()
    embs = _orthonormal_embeddings(len(terms), dim=4)
    idx = build_index(terms, embs, embedding_model="t", efo_version="v")

    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = idx.search_vectors(q, top_k=999)
    assert len(result[0]) == 3  # clamped to corpus size


def test_get_cached_index_reuses_in_memory_copy(tmp_path: Path):
    terms = _mini_terms()
    embs = _orthonormal_embeddings(len(terms), dim=4)
    save_index(
        tmp_path,
        build_index(terms, embs, embedding_model="t", efo_version="v"),
    )
    a = get_cached_index(tmp_path)
    b = get_cached_index(tmp_path)
    assert a is b
    assert isinstance(a, EFOIndex)

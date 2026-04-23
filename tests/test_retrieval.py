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
    filter_corpus,
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


# -- filter_corpus (Stage 7) -------------------------------------------------


def _row(ontology_id: str) -> dict:
    return {
        "ontology_id": ontology_id,
        "label": ontology_id,
        "synonyms": [],
        "definition": None,
        "parents": [],
    }


@pytest.mark.parametrize(
    "allowed_id",
    [
        "CL:0000624",       # T cell
        "UBERON:0002048",   # lung
        "EFO:0001187",      # HEK293
        "CLO:0001230",      # cell line entry
        "BTO:0000001",      # BRENDA tissue
        "MONDO:0004992",    # cancer
        "Orphanet:586",     # rare disease
        "NCIT:C12439",      # NCI thesaurus
        "FBbt:00005106",    # Drosophila anatomy
        "FBdv:00005334",    # Drosophila development
        "ZFA:0000001",      # zebrafish anatomy
        "MA:0000001",       # mouse anatomy
        "FMA:7195",         # FMA
        "PO:0025001",       # Plant Ontology
        "WBls:0000001",     # C. elegans life stage
    ],
)
def test_filter_corpus_keeps_allowed_prefixes(allowed_id: str):
    df = pd.DataFrame([_row(allowed_id)])
    out = filter_corpus(df)
    assert list(out["ontology_id"]) == [allowed_id]


@pytest.mark.parametrize(
    "excluded_id",
    [
        "PR:000001",           # protein
        "HGNC:5",              # gene symbol
        "OBA:0000001",         # biological attribute
        "GO:0008150",          # biological process
        "NCBITaxon:9606",      # taxon
        "HP:0000001",          # human phenotype
        "CHEBI:15377",         # chemical
        "dbpedia:Paris",       # geographic
        "UO:0000001",          # unit
        "HANCESTRO:0004",      # ancestry
        "PATO:0000001",        # phenotypic attribute
        "SO:0000001",          # sequence
        "GSSO:000001",         # sex/gender
    ],
)
def test_filter_corpus_drops_excluded_prefixes(excluded_id: str):
    df = pd.DataFrame([_row(excluded_id)])
    out = filter_corpus(df)
    assert len(out) == 0


def test_filter_corpus_preserves_row_order_and_resets_index():
    df = pd.DataFrame([
        _row("PR:000001"),
        _row("CL:0000001"),
        _row("OBA:0000001"),
        _row("UBERON:0000001"),
        _row("HGNC:5"),
    ])
    out = filter_corpus(df)
    assert list(out["ontology_id"]) == ["CL:0000001", "UBERON:0000001"]
    assert list(out.index) == [0, 1]


def test_filter_corpus_drops_rows_with_no_prefix():
    df = pd.DataFrame([_row("CL:0000001"), _row("no_colon_here"), _row("")])
    out = filter_corpus(df)
    assert list(out["ontology_id"]) == ["CL:0000001"]

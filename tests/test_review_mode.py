"""End-to-end tests for `map_chipatlas(review=True)` (Stage 3).

These wire together a fixture Cellosaurus cache, a hand-built EFO FAISS index
with orthonormal embeddings, and a fake embedder that returns those same
vectors for specific query keywords. All fixtures live in `tests/conftest.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import (
    FakeEmbedder,
    install_cellosaurus_cache,
    install_full_cache,
)


def test_review_mode_cellosaurus_hit_returns_single_row(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas
    from caom.api import REVIEW_COLUMNS

    df = pd.DataFrame({"cell_type": ["K-562"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, review=True, top_k=3, cache_dir=tmp_path, embedder=FakeEmbedder()
    )

    assert list(result.columns) == list(REVIEW_COLUMNS)
    assert len(result) == 1
    assert result.loc[0, "rank"] == 1
    assert result.loc[0, "ontology_id"] == "CVCL_0004"
    assert result.loc[0, "ontology_source"] == "cellosaurus"
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9"
    assert pd.isna(result.loc[0, "retrieval_score"])


def test_review_mode_cellosaurus_miss_returns_efo_top_k(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["hepatocyte"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, review=True, top_k=3, cache_dir=tmp_path, embedder=FakeEmbedder()
    )

    assert len(result) == 3
    assert list(result["rank"]) == [1, 2, 3]
    assert result.loc[0, "ontology_id"] == "CL:0000182"
    assert result.loc[0, "ontology_source"] == "efo"
    assert result.loc[0, "ontology_version"] == "efo:99.9"
    assert result.loc[0, "retrieval_score"] == pytest.approx(1.0)
    assert {result.loc[1, "ontology_id"], result.loc[2, "ontology_id"]} == {
        "EFO:0001187",
        "CL:0000236",
    }


def test_review_mode_ambiguous_cellosaurus_keeps_candidates_plus_efo(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    # "Raw" without an assembly hint is ambiguous (matches CVCL_0493 + CVCL_XXXX).
    df = pd.DataFrame({"cell_type": ["Raw"]})
    result = map_chipatlas(
        df, review=True, top_k=2, cache_dir=tmp_path, embedder=FakeEmbedder()
    )

    assert len(result) == 4
    assert list(result["rank"]) == [1, 2, 3, 4]
    sources = result["ontology_source"].tolist()
    assert sources[:2] == ["cellosaurus", "cellosaurus"]
    assert sources[2:] == ["efo", "efo"]
    assert set(result.loc[:1, "ontology_id"]) == {"CVCL_0493", "CVCL_XXXX"}


def test_review_mode_batches_efo_queries(tmp_path: Path):
    """Review should encode rows once per call, not once per row."""
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    calls: list[int] = []
    real_embedder = FakeEmbedder()

    class CountingEmbedder(FakeEmbedder):
        def encode(self, texts, **kw):
            calls.append(len(texts))
            return real_embedder.encode(texts, **kw)

    df = pd.DataFrame(
        {
            "cell_type": ["hepatocyte", "B cell", "K-562"],
            "assembly": ["hg38", "hg38", "hg38"],
        }
    )
    _ = map_chipatlas(
        df, review=True, top_k=1, cache_dir=tmp_path, embedder=CountingEmbedder()
    )
    # Two rows miss Cellosaurus and need retrieval; K-562 is a single hit.
    assert calls == [2]


def test_review_mode_without_efo_cache_raises(tmp_path: Path):
    install_cellosaurus_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        map_chipatlas(df, review=True, cache_dir=tmp_path, embedder=FakeEmbedder())

"""End-to-end tests for `map_chipatlas(review=True)` (Stage 3).

These wire together:
- a fixture Cellosaurus cache (reused from test_cellosaurus),
- a hand-built EFO FAISS index with orthonormal embeddings,
- a fake embedder that returns those same vectors for specific query keywords.
This avoids network and sentence-transformers model loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caom.ontologies.cellosaurus import (
    build_lookup,
    parse_cellosaurus,
    save_lookup,
)
from caom.retrieval.index import build_index, save_index
from tests.test_cellosaurus import FIXTURE as CELLOSAURUS_FIXTURE

# --- fixtures ---------------------------------------------------------------


EFO_TERMS = pd.DataFrame(
    [
        {
            "ontology_id": "CL:0000182",
            "label": "hepatocyte",
            "synonyms": ["liver cell"],
            "definition": "A cell of the liver.",
            "parents": ["CL:0000000"],
        },
        {
            "ontology_id": "EFO:0001187",
            "label": "HEK293",
            "synonyms": ["HEK-293", "HEK 293"],
            "definition": "Human embryonic kidney cell line.",
            "parents": ["CL:0000000"],
        },
        {
            "ontology_id": "CL:0000236",
            "label": "B cell",
            "synonyms": ["B lymphocyte"],
            "definition": "A lymphocyte of B lineage.",
            "parents": ["CL:0000000"],
        },
    ]
)

# One-hot embeddings in R^4. Row i aligns with EFO_TERMS row i.
_EMBS = np.eye(len(EFO_TERMS), 4, dtype=np.float32)


class _FakeEmbedder:
    """Deterministic embedder: maps query keywords → pre-wired vectors.

    The rule: for each query string, find the first keyword in `_map` that
    appears (case-insensitive) and return its vector. Queries with no match
    map to the zero vector (cosine score 0 against every corpus row).
    """

    def __init__(self, dim: int = 4):
        self._dim = dim
        self._map: dict[str, np.ndarray] = {
            "hepatocyte": _EMBS[0],
            "liver cell": _EMBS[0],
            "hek": _EMBS[1],
            "b cell": _EMBS[2],
            "b lymphocyte": _EMBS[2],
        }

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts, *, batch_size=32, show_progress=False):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = t.lower()
            for key, v in self._map.items():
                if key in tl:
                    out[i] = v
                    break
        return out


def _install_caches(tmp_path: Path) -> None:
    """Pre-populate both Cellosaurus and EFO+FAISS caches in `tmp_path`."""
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(CELLOSAURUS_FIXTURE)
    entries, version = parse_cellosaurus(flat)
    save_lookup(
        tmp_path,
        build_lookup(entries, version=version, downloaded_at="2099-01-01T00:00:00+00:00"),
    )
    idx = build_index(
        EFO_TERMS.copy(),
        _EMBS.copy(),
        embedding_model="fake-embedder",
        efo_version="99.9",
    )
    save_index(tmp_path, idx)


# --- review mode tests ------------------------------------------------------


def test_review_mode_cellosaurus_hit_returns_single_row(tmp_path: Path):
    _install_caches(tmp_path)
    from caom import map_chipatlas
    from caom.api import REVIEW_COLUMNS

    df = pd.DataFrame({"cell_type": ["K-562"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, review=True, top_k=3, cache_dir=tmp_path, embedder=_FakeEmbedder()
    )

    assert list(result.columns) == list(REVIEW_COLUMNS)
    assert len(result) == 1
    assert result.loc[0, "rank"] == 1
    assert result.loc[0, "ontology_id"] == "CVCL_0004"
    assert result.loc[0, "ontology_source"] == "cellosaurus"
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9"
    assert pd.isna(result.loc[0, "retrieval_score"])


def test_review_mode_cellosaurus_miss_returns_efo_top_k(tmp_path: Path):
    _install_caches(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["hepatocyte"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, review=True, top_k=3, cache_dir=tmp_path, embedder=_FakeEmbedder()
    )

    assert len(result) == 3
    assert list(result["rank"]) == [1, 2, 3]
    assert result.loc[0, "ontology_id"] == "CL:0000182"
    assert result.loc[0, "ontology_source"] == "efo"
    assert result.loc[0, "ontology_version"] == "efo:99.9"
    assert result.loc[0, "retrieval_score"] == pytest.approx(1.0)
    # Ranks 2 and 3 are the orthogonal rows with score 0.
    assert {result.loc[1, "ontology_id"], result.loc[2, "ontology_id"]} == {
        "EFO:0001187",
        "CL:0000236",
    }


def test_review_mode_ambiguous_cellosaurus_keeps_candidates_plus_efo(tmp_path: Path):
    _install_caches(tmp_path)
    from caom import map_chipatlas

    # "Raw" without an assembly hint is ambiguous (matches CVCL_0493 + CVCL_XXXX).
    df = pd.DataFrame({"cell_type": ["Raw"]})
    result = map_chipatlas(
        df, review=True, top_k=2, cache_dir=tmp_path, embedder=_FakeEmbedder()
    )

    # 2 ambiguous Cellosaurus candidates + 2 EFO top-K = 4 rows
    assert len(result) == 4
    assert list(result["rank"]) == [1, 2, 3, 4]
    sources = result["ontology_source"].tolist()
    assert sources[:2] == ["cellosaurus", "cellosaurus"]
    assert sources[2:] == ["efo", "efo"]
    assert set(result.loc[:1, "ontology_id"]) == {"CVCL_0493", "CVCL_XXXX"}


def test_review_mode_batches_efo_queries(tmp_path: Path, monkeypatch):
    """Review should encode rows once per call, not once per row."""
    _install_caches(tmp_path)
    from caom import map_chipatlas

    calls: list[int] = []
    real_embedder = _FakeEmbedder()

    class CountingEmbedder(_FakeEmbedder):
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
    # Two rows miss Cellosaurus and need retrieval; K-562 is a single hit and is skipped.
    assert calls == [2]


def test_review_mode_without_efo_cache_raises(tmp_path: Path):
    # Cellosaurus only; no FAISS index.
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(CELLOSAURUS_FIXTURE)
    entries, version = parse_cellosaurus(flat)
    save_lookup(
        tmp_path,
        build_lookup(entries, version=version, downloaded_at="2099-01-01T00:00:00+00:00"),
    )
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        map_chipatlas(df, review=True, cache_dir=tmp_path, embedder=_FakeEmbedder())

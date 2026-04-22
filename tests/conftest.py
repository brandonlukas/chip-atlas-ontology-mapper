"""Shared test fixtures: cache installers and LLM / embedder fakes.

Tests avoid real network calls (Cellosaurus / EFO downloads), real
sentence-transformers loads, and real Ollama traffic by installing
pre-built caches into `tmp_path` and injecting the fakes below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from caom.ontologies.cellosaurus import (
    build_lookup,
    parse_cellosaurus,
    save_lookup,
)
from caom.retrieval.index import build_index, save_index
from caom.types import LLMPick

# Minimal synthetic Cellosaurus flat file covering:
#   - K-562 with punctuation-variant synonyms (the canonical test case)
#   - MCF-7 (simple cell line)
#   - NIH-3T3 (mouse, exercises species filtering)
#   - Raw 264.7 / RAW (same-name-different-species pair → ambiguity + taxon filter)
CELLOSAURUS_FIXTURE = """\
----------------------------------------------------------------------------
        CALIPHO group at the SIB - Swiss Institute of Bioinformatics
----------------------------------------------------------------------------

 Description: Cellosaurus: a knowledge resource on cell lines
 Version: 99.9
 Last update: 01-Jan-2026

----------------------------------------------------------------------------
ID   K-562
AC   CVCL_0004
SY   K562; K.562; K 562; GM05372; GM05372E
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Cancer cell line
DT   Created: 04-04-12; Last updated: 22-06-23; Version: 45
//
ID   MCF-7
AC   CVCL_0031
SY   MCF7; MCF 7
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Cancer cell line
//
ID   NIH-3T3
AC   CVCL_0594
SY   NIH/3T3; NIH 3T3; 3T3
OX   NCBI_TaxID=10090; ! Mus musculus
CA   Spontaneously immortalized cell line
//
ID   Raw 264.7
AC   CVCL_0493
SY   RAW264.7; RAW
OX   NCBI_TaxID=10090; ! Mus musculus
CA   Spontaneously immortalized cell line
//
ID   RAW
AC   CVCL_XXXX
SY   Raw
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Undefined cell line type
//
"""


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

EFO_EMBS = np.eye(len(EFO_TERMS), 4, dtype=np.float32)


class FakeEmbedder:
    """Deterministic embedder for tests: maps query keywords → pre-wired vectors.

    The rule: for each query string, find the first keyword in `_map` that
    appears (case-insensitive) and return its vector. Queries with no keyword
    match produce the zero vector (cosine score 0 against every corpus row).
    """

    def __init__(self, dim: int = 4):
        self._dim = dim
        self._map: dict[str, np.ndarray] = {
            "hepatocyte": EFO_EMBS[0],
            "liver cell": EFO_EMBS[0],
            "hek": EFO_EMBS[1],
            "b cell": EFO_EMBS[2],
            "b lymphocyte": EFO_EMBS[2],
        }

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self, texts: list[str], *, batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = t.lower()
            for key, v in self._map.items():
                if key in tl:
                    out[i] = v
                    break
        return out


class FakeLLMClient:
    """Scripted LLM client for tests.

    - `rules` maps substrings to `LLMPick`s. The first rule whose key appears
      (case-insensitive) in the prompt wins. A `None` value signals the
      unmappable pick.
    - `default_pick` is returned when no rule matches (defaults to null pick).
    - `calls` records every prompt passed to `pick` so tests can assert call
      counts and prompt shape.
    """

    def __init__(
        self,
        *,
        model: str = "fake-model",
        rules: dict[str, LLMPick] | None = None,
        default_pick: LLMPick | None = None,
    ):
        self.model = model
        self._rules = rules or {}
        self._default = default_pick or LLMPick(
            ontology_id=None, confidence=0.0, rationale="no match"
        )
        self.calls: list[str] = []

    def pick(self, prompt: str) -> LLMPick:
        self.calls.append(prompt)
        lowered = prompt.lower()
        for key, value in self._rules.items():
            if key.lower() in lowered:
                return value
        return self._default


def install_cellosaurus_cache(tmp_path: Path) -> None:
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(CELLOSAURUS_FIXTURE)
    entries, version = parse_cellosaurus(flat)
    save_lookup(
        tmp_path,
        build_lookup(entries, version=version, downloaded_at="2099-01-01T00:00:00+00:00"),
    )


def install_efo_index(tmp_path: Path) -> None:
    idx = build_index(
        EFO_TERMS.copy(),
        EFO_EMBS.copy(),
        embedding_model="fake-embedder",
        efo_version="99.9",
    )
    save_index(tmp_path, idx)


def install_full_cache(tmp_path: Path) -> None:
    """Install both Cellosaurus lookup and EFO FAISS index."""
    install_cellosaurus_cache(tmp_path)
    install_efo_index(tmp_path)


def invalidate_all_caches(tmp_path: Path) -> None:
    """Drop in-memory caches for this path so each test sees a fresh load."""
    from caom.ontologies import cellosaurus as _cs
    from caom.retrieval import index as _idx

    _cs._LOOKUP_CACHE.invalidate(tmp_path)
    _idx.invalidate_cache(tmp_path)


def read_llm_cache_rows(db_path: Path) -> list[dict[str, Any]]:
    """Return every row from the SQLite LLM cache for inspection."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT model, prompt_hash, response, created_at FROM llm_cache"
        )
        return [
            {"model": m, "prompt_hash": h, "response": r, "created_at": t}
            for (m, h, r, t) in cur.fetchall()
        ]
    finally:
        conn.close()

"""Harness runner: compose gold-standard input into `map_chipatlas` calls.

`Mode.CELLOSAURUS_ONLY` injects stubs so only the Stage 2 fast-path contributes
mappings; any row that would have gone to retrieval + LLM ends up unmappable.
This measures "how far does Cellosaurus get us?" without requiring Ollama.

`Mode.FULL` runs the real pipeline (real embedder + real LLM). It requires
Ollama to be reachable at the configured host and a pre-built EFO FAISS index.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from caom.api import map_chipatlas
from caom.types import LLMPick
from tests.conftest import FakeLLMClient


class Mode(StrEnum):
    CELLOSAURUS_ONLY = "cellosaurus_only"
    FULL = "full"


class _NullEmbedder:
    """Zero-vector embedder of a given dim — for Stage 2 gate only.

    `FakeEmbedder` in conftest.py isn't reusable here: its keyword-match map is
    hard-coded to length-4 vectors, so passing `dim=768` to match the real
    FAISS index would raise a shape error on any query that trips a keyword.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(
        self, texts: list[str], *, batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


def _efo_index_dim(cache_root: Path) -> int:
    meta_path = cache_root / "embeddings" / "efo.metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"EFO index metadata not found at {meta_path}. "
            "Run `caom.update_ontologies()` first."
        )
    return int(json.loads(meta_path.read_text())["dim"])


_NULL_PICK = LLMPick(
    ontology_id=None,
    confidence=0.0,
    rationale="Stage 2 validation gate — LLM disabled.",
)


def run_prediction(gold: pd.DataFrame, *, mode: Mode, cache_root: Path) -> pd.DataFrame:
    """Run `map_chipatlas` on `gold` in the specified mode and return its output."""
    # Only cell_type is available on Ikeda rows; we don't fabricate an assembly,
    # so cross-species Cellosaurus hits hit the ambiguity path rather than being
    # species-filtered — which is the realistic shape for this gold standard.
    df = gold[["cell_type"]]

    if mode is Mode.CELLOSAURUS_ONLY:
        return map_chipatlas(
            df,
            cache_dir=cache_root,
            embedder=_NullEmbedder(dim=_efo_index_dim(cache_root)),
            llm_client=FakeLLMClient(default_pick=_NULL_PICK),
        )

    return map_chipatlas(df, cache_dir=cache_root)

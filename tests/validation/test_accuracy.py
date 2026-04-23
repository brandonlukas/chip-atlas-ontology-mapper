"""Accuracy gates on the Ikeda 2025 gold standard.

These tests are opt-in because they require:

- a populated Cellosaurus cache (Stage 2 gate)
- a populated EFO FAISS index (both gates)
- network access or a pre-cached gold-standard TSV (both gates)
- Ollama running with the configured model (full-pipeline gate only)

Set ``CAOM_RUN_VALIDATION=1`` to run the Stage 2 fast-path gate.
Set ``CAOM_RUN_LLM_VALIDATION=1`` *in addition* to also run the full-pipeline gate.

Calibration notes (Cellosaurus v49, EFO v3.89, qwen2.5:7b-instruct):

    2026-04-22, Stage 2 fast-path only:
      n=600  acc@1=0.903  pick_precision=1.000  unmap_recall=0.997  coverage=0.450

    2026-04-22, Stage 8 full pipeline (Cellosaurus → retrieval → LLM re-rank):
      n=600  acc@1=0.933  pick_precision=0.989  unmap_recall=0.977  coverage=0.480

The gate floors below are set a few points below the observed numbers to
absorb Cellosaurus / EFO / model-version churn. Tighten when numbers stabilize
across a couple of stage runs; loosen if a legitimate version bump trips them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from caom.config import load_config
from caom.ontologies.cellosaurus import is_cached as cellosaurus_is_cached
from caom.retrieval.index import is_cached as efo_index_is_cached
from tests.validation.ikeda_gold_standard import load_gold_standard, summarize
from tests.validation.metrics import compute_accuracy
from tests.validation.runner import Mode, run_prediction

_RUN_VALIDATION = os.environ.get("CAOM_RUN_VALIDATION") == "1"
_RUN_LLM_VALIDATION = os.environ.get("CAOM_RUN_LLM_VALIDATION") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_VALIDATION,
    reason="Set CAOM_RUN_VALIDATION=1 to run the Ikeda accuracy gates.",
)


@pytest.fixture(scope="module")
def cache_root() -> Path:
    root = load_config().cache_dir
    if not cellosaurus_is_cached(root):
        pytest.skip(f"Cellosaurus cache missing at {root}. Run caom.update_ontologies() first.")
    if not efo_index_is_cached(root):
        pytest.skip(f"EFO FAISS index missing at {root}. Run caom.update_ontologies() first.")
    return root


@pytest.fixture(scope="module")
def gold(cache_root: Path):
    df = load_gold_standard(cache_root)
    stats = summarize(df)
    assert stats.total > 0
    assert stats.mapped > 0
    assert stats.unmappable > 0
    return df


def test_cellosaurus_fast_path_accuracy_gate(gold, cache_root: Path) -> None:
    """Stage 2 alone should resolve the majority of cell-line rows."""
    pred = run_prediction(gold, mode=Mode.CELLOSAURUS_ONLY, cache_root=cache_root)
    report = compute_accuracy(pred, gold)
    print(f"\n[cellosaurus fast-path] {report.format()}")

    assert report.accuracy_at_1 >= 0.85, report.format()
    assert report.pick_precision >= 0.98, report.format()
    assert report.unmappable_recall >= 0.95, report.format()


@pytest.mark.skipif(
    not _RUN_LLM_VALIDATION,
    reason="Set CAOM_RUN_LLM_VALIDATION=1 to run the full-pipeline gate (needs Ollama).",
)
def test_full_pipeline_accuracy_gate(gold, cache_root: Path) -> None:
    """Full pipeline (Cellosaurus → retrieval → LLM re-rank) must beat Stage 2.

    `pick_precision` is gated separately so we can detect a regression where
    the LLM starts committing to wrong ids without the overall acc@1 number
    moving — the after-Stage-8 prompt is deliberately tuned for high precision
    over coverage so a slip there is the more likely failure mode.
    """
    pred = run_prediction(gold, mode=Mode.FULL, cache_root=cache_root)
    report = compute_accuracy(pred, gold)
    print(f"\n[full pipeline] {report.format()}")

    assert report.accuracy_at_1 >= 0.91, report.format()
    assert report.pick_precision >= 0.97, report.format()
    assert report.unmappable_recall >= 0.95, report.format()

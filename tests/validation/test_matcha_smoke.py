"""Matcha-shape regression gate. Stage 9.

Promoted from `tests/validation/run_matcha_smoke.py` so the per-cell-type
expectations hold across LLM / ontology version drift instead of needing a
human to eyeball the smoke output. The driver script is still useful for
ad-hoc runs that print full rationales; this module is the pass/fail gate.

Why both the Ikeda gate AND a matcha gate:
- The Ikeda gold standard is cell-line focused (its TSV's `extraction
  answer` column names cell lines). A regression on tissues / primary cell
  types / disease-as-context would not move Ikeda's `acc@1`.
- The matcha shape (full ChIP-Atlas metadata: `assembly`,
  `cell_type_class`, `title`) exposes failure modes the Ikeda harness
  cannot — the Stage 9 wins on `Lung`, `Brain`, `Acute myeloid leukemia`
  are visible only here.

Skips by default; run with `CAOM_RUN_LLM_VALIDATION=1`. Skips with a
reason when Ollama / matcha parquet / EFO cache are missing rather than
failing — the gate is opt-in for environments where the dependencies
exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from caom import map_chipatlas
from caom.config import load_config
from caom.ontologies.cellosaurus import is_cached as cellosaurus_is_cached
from caom.retrieval.index import is_cached as efo_index_is_cached

_RUN_LLM_VALIDATION = os.environ.get("CAOM_RUN_LLM_VALIDATION") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_LLM_VALIDATION,
    reason="Set CAOM_RUN_LLM_VALIDATION=1 to run the matcha smoke gate (needs Ollama).",
)

MATCHA_PARQUET = Path("/home/brandon/code/matcha/data/metadata/curated_metadata.parquet")

# Locked-in picks observed against Cellosaurus v49 + EFO v3.89 with the
# Stage-10 substring marker on qwen2.5:7b-instruct. Tagged by category so a
# future failure points at *which kind* of regression occurred.
#
#   FAST_PATH (Cellosaurus): deterministic; should never regress.
#   EXACT_HIT (Stage 9 win): label / synonym match into UBERON / CL / MONDO
#     that previously lost rank-1 to a noisier subtype.
#   SUBSTRING_HIT (Stage 10 win): normalized-substring match that fires the
#     `[substring]` marker, pulling a canonical parent ahead of a subtype or
#     unrelated anatomical container the cosine layer ranked higher.
#   LLM_PICK (best the corpus offers): canonical term not in narrowed corpus
#     (e.g. CL:2000001 / CL:0011020 absent), so the LLM picks the closest
#     parent CL/UBERON term. Locked in to detect drift; defer absolute fix
#     to Stage 11's corpus refresh / BM25 / embedder upgrade.
EXPECTED: dict[str, str] = {
    # FAST_PATH
    "K-562":                     "CVCL_0004",
    "MCF-7":                     "CVCL_0031",
    "293":                       "CVCL_0045",   # HEK293
    "HCT 116":                   "CVCL_0291",
    "GM12878":                   "CVCL_7526",
    "HAP1":                      "CVCL_Y019",
    # EXACT_HIT — Stage 9 wins (Lung/Brain/AML previously picked subtypes).
    "Brain":                     "UBERON:0000955",
    "Lung":                      "UBERON:0002048",
    "Acute myeloid leukemia":    "MONDO:0018874",
    # SUBSTRING_HIT — Stage 10 win: plural query 'Pancreatic islets' now
    # substring-matches the synonym 'pancreatic islet' on UBERON:0000006
    # (canonical islet of Langerhans). Previously lost to UBERON:0000016.
    "Pancreatic islets":         "UBERON:0000006",
    # LLM_PICK — corpus-best parent terms for queries whose canonical CL
    # term is not in the Stage 9 narrowed allow-list.
    "iPS cells":                 "CL:0002248",   # pluripotent stem cell
    "PBMC":                      "CL:0000842",   # mononuclear leukocyte (PBMC synonym)
    # LLM_PICK — known-imperfect, qwen2.5:7b-instruct-specific picks. Locked
    # in to surface regressions; a better pick flipping the expected here is
    # the intended "you improved something" signal.
    #   `iPSC derived neural cells`: 7b picks CL:0002351 (pancreatic endocrine
    #     progenitor — wrong). 14b previously picked CL:0002248; neural stem
    #     cell CL:0000047 is also corpus-available and arguably best, but 7b
    #     is unstable on this query. Stage 11 corpus refresh adding CL:0011020
    #     (neural progenitor) would be the durable fix.
    #   `CD4+ T cells`: title says "CD4 activation ATAC-seq", which leads 7b
    #     to the activated subtype CL:0000896 over the canonical CL:0000624.
    #     The `+` vs `-positive` mismatch means the substring rule doesn't
    #     fire — a punctuation-normalization preprocessor would fix this but
    #     veers into the "asymmetric query rewriting" anti-goal (CLAUDE.md).
    "iPSC derived neural cells": "CL:0002351",
    "CD4+ T cells":              "CL:0000896",
}


@pytest.fixture(scope="module")
def cache_root() -> Path:
    root = load_config().cache_dir
    if not cellosaurus_is_cached(root):
        pytest.skip(
            f"Cellosaurus cache missing at {root}. Run caom.update_ontologies() first."
        )
    if not efo_index_is_cached(root):
        pytest.skip(
            f"EFO FAISS index missing at {root}. Run caom.update_ontologies() first."
        )
    return root


@pytest.fixture(scope="module")
def matcha_sample() -> pd.DataFrame:
    if not MATCHA_PARQUET.exists():
        pytest.skip(
            f"matcha parquet not found at {MATCHA_PARQUET}; the gate needs the "
            "downstream consumer's curated metadata. See INSTRUCTIONS.md → Pointers."
        )
    full = pd.read_parquet(MATCHA_PARQUET)
    targets = list(EXPECTED.keys())
    sampled = (
        full[full["cell_type"].isin(targets)]
        .drop_duplicates("cell_type")
        .set_index("cell_type")
        .loc[targets]
        .reset_index()
    )
    assert len(sampled) == len(targets), (
        f"matcha parquet missing rows for: "
        f"{sorted(set(targets) - set(sampled['cell_type']))}"
    )
    return sampled


def test_matcha_smoke_gate(matcha_sample: pd.DataFrame, cache_root: Path) -> None:
    """Every cell_type in EXPECTED must map to its locked-in ontology id.

    A failure here is one of three things:
      (1) Stage 9 regression — exact-match layer broke or allow-list drifted.
      (2) Ontology version bump — re-anchor the EXPECTED entry to the new id
          after confirming it's an equivalent or improved pick.
      (3) LLM-version bump — re-anchor after confirming the new pick is at
          least as good as the old.
    """
    pred = map_chipatlas(matcha_sample, cache_dir=cache_root)
    actual: dict[str, str | None] = dict(zip(pred["cell_type"], pred["ontology_id"]))

    mismatches = {
        ct: (EXPECTED[ct], actual.get(ct))
        for ct in EXPECTED
        if actual.get(ct) != EXPECTED[ct]
    }
    if mismatches:
        lines = [f"  {ct!r}: expected {exp!r}, got {got!r}"
                 for ct, (exp, got) in mismatches.items()]
        pytest.fail("matcha smoke regressions:\n" + "\n".join(lines))

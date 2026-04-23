"""Loader for the Ikeda et al. 2025 BioSample → Cellosaurus gold standard.

Source: Zenodo DOI 10.5281/zenodo.14881142
File: biosample_cellosaurus_mapping_gold_standard.tsv (600 rows)

The TSV columns are:
- ``BioSample ID``             — NCBI BioSample accession (SAMD*, SAMEA*, SAMN*)
- ``Experiment type``          — ChIP-Atlas antigen class (ATAC-Seq, Histone, TFs and others, RNA polymerase)
- ``extraction answer``        — free-text cell-line name extracted from BioSample metadata
- ``mapping answer ID``        — gold Cellosaurus accession, formatted ``CVCL:NNNN`` (empty = not a cell line)
- ``mapping answer label``     — gold Cellosaurus primary name

We treat ``extraction answer`` as the ``cell_type`` input: it is a messy free-text
cell-line string with the same shape as ChIP-Atlas's ``cell_type`` column, and
the LLM-extracted string is what Ikeda's own gold standard was built against.
Joining BioSample IDs back to ChIP-Atlas's ``cell_type`` column would require a
second large data source (SRA metadata to bridge BioSample ↔ SRX) for no new signal.

Rows with a blank ``extraction answer`` stand in for "no cell line mentioned" cases;
rows with a blank ``mapping answer ID`` are the gold-standard "unmappable" set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

GOLD_URL = (
    "https://zenodo.org/records/14881142/files/"
    "biosample_cellosaurus_mapping_gold_standard.tsv?download=1"
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "biosample_id",
    "experiment_type",
    "cell_type",
    "gold_ontology_id",
    "gold_label",
)


@dataclass(frozen=True)
class GoldStandardStats:
    total: int
    mapped: int
    unmappable: int
    with_cell_type: int


def _cache_path(cache_root: Path) -> Path:
    return cache_root / "validation" / "biosample_cellosaurus_mapping_gold_standard.tsv"


def download_gold_standard(dest: Path, *, url: str = GOLD_URL) -> None:
    """Stream the Ikeda gold-standard TSV to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _normalize_cvcl(raw: object) -> str | None:
    """Turn ``CVCL:1234`` (gold TSV) into ``CVCL_1234`` (Cellosaurus parser)."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    return s.replace(":", "_", 1)


def _coerce_str(raw: object) -> str:
    """Return a stripped string; treat NaN / None as ``""``."""
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def parse_gold_standard(tsv_path: Path) -> pd.DataFrame:
    """Parse the Ikeda gold-standard TSV into caom's input shape.

    Returns a DataFrame with columns:
        biosample_id, experiment_type, cell_type, gold_ontology_id, gold_label.

    ``cell_type`` is the ``extraction answer`` column; ``gold_ontology_id`` is
    the normalized Cellosaurus ID (or None for unmappable rows).
    """
    raw = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    expected = {
        "BioSample ID",
        "Experiment type",
        "extraction answer",
        "mapping answer ID",
        "mapping answer label",
    }
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(
            f"Gold standard TSV missing expected columns: {sorted(missing)}. "
            f"Got: {list(raw.columns)}"
        )
    return pd.DataFrame(
        {
            "biosample_id": raw["BioSample ID"].map(_coerce_str),
            "experiment_type": raw["Experiment type"].map(_coerce_str),
            "cell_type": raw["extraction answer"].map(_coerce_str),
            "gold_ontology_id": raw["mapping answer ID"].map(_normalize_cvcl),
            "gold_label": raw["mapping answer label"].map(
                lambda v: _coerce_str(v) or None
            ),
        },
        columns=list(OUTPUT_COLUMNS),
    )


def load_gold_standard(
    cache_root: Path,
    *,
    force: bool = False,
    url: str = GOLD_URL,
) -> pd.DataFrame:
    """Load the gold-standard TSV, downloading + caching on first use."""
    dest = _cache_path(cache_root)
    if force or not dest.exists():
        download_gold_standard(dest, url=url)
    return parse_gold_standard(dest)


def summarize(df: pd.DataFrame) -> GoldStandardStats:
    """Count totals for display in the validation report header."""
    return GoldStandardStats(
        total=len(df),
        mapped=int(df["gold_ontology_id"].notna().sum()),
        unmappable=int(df["gold_ontology_id"].isna().sum()),
        with_cell_type=int((df["cell_type"] != "").sum()),
    )

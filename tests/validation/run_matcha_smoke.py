"""Ad-hoc driver: run the full pipeline on a curated handful of real matcha rows.

Usage::

    python -m tests.validation.run_matcha_smoke

Why a separate driver from `run_full_pipeline.py`:
- `run_full_pipeline.py` measures against the Ikeda gold standard (cell-line
  focused, no `assembly` column) — answers "how does the pipeline do on
  cell-line extraction?".
- This driver measures against the matcha shape (full ChIP-Atlas metadata
  including assembly / cell_type_class / title) — answers "how does the
  pipeline do on the actual downstream consumer's input?". The two surface
  different failure modes; Stage 9 was scoped from this driver's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from caom import map_chipatlas

MATCHA_PARQUET = Path("/home/brandon/code/matcha/data/metadata/curated_metadata.parquet")

# Diverse real cell_type values exercising:
#   - cell-line fast-path (Cellosaurus exact / synonym match)
#   - retrieval + LLM (primary cell types, tissues)
#   - edge cases (numeric codes, punctuation variants, disease-as-context)
#   - Stage 7 failure-mode queries that motivated the allow-list filter
#   - Stage 9 subtype-overshoot probes (Brain, Lung, Pancreatic islets)
#   - Stage 9 head-to-head additions vs mochi (`iPSC derived neural cells`)
TARGET_CELL_TYPES: list[str] = [
    "K-562",
    "MCF-7",
    "293",
    "HCT 116",
    "GM12878",
    "HAP1",
    "CD4+ T cells",
    "iPS cells",
    "iPSC derived neural cells",
    "Pancreatic islets",
    "Brain",
    "Lung",
    "PBMC",
    "Acute myeloid leukemia",
]


def main() -> int:
    try:
        full = pd.read_parquet(MATCHA_PARQUET)
    except FileNotFoundError:
        print(
            f"matcha parquet not found at {MATCHA_PARQUET}. "
            "See INSTRUCTIONS.md → Pointers for the canonical input path.",
            file=sys.stderr,
        )
        return 1

    sampled = (
        full[full["cell_type"].isin(TARGET_CELL_TYPES)]
        .drop_duplicates("cell_type")
        .set_index("cell_type")
        .loc[TARGET_CELL_TYPES]
        .reset_index()
    )

    print(f"sampled {len(sampled)} rows from {len(full):,} total matcha rows\n")

    pred = map_chipatlas(sampled)

    side = pd.concat(
        [
            sampled[["cell_type", "cell_type_class", "assembly"]].reset_index(drop=True),
            pred[
                [
                    "status",
                    "ontology_id",
                    "ontology_label",
                    "ontology_source",
                    "confidence",
                    "rationale",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    print(side.to_string(index=False, max_colwidth=80, line_width=240))

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Ad-hoc driver: run the full pipeline on the Ikeda gold standard, print
the accuracy report, and dump predictions + a failure breakdown to disk.

Usage::

    python -m tests.validation.run_full_pipeline

Writes:
    .cache/validation/full_pipeline_predictions.parquet
    .cache/validation/full_pipeline_failures.tsv

The parquet has one row per gold row with both prediction and gold columns
joined, so failure analysis can iterate without re-running the LLM.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from caom.config import load_config
from tests.validation.ikeda_gold_standard import load_gold_standard, summarize
from tests.validation.metrics import compute_accuracy
from tests.validation.runner import Mode, run_prediction


def main() -> int:
    cache_root = load_config().cache_dir

    gold = load_gold_standard(cache_root)
    stats = summarize(gold)
    print(f"gold: total={stats.total} mapped={stats.mapped} "
          f"unmappable={stats.unmappable} with_cell_type={stats.with_cell_type}")

    print(f"\n[full pipeline] running map_chipatlas on {stats.total} rows ...")
    pred = run_prediction(gold, mode=Mode.FULL, cache_root=cache_root)
    report = compute_accuracy(pred, gold)
    print(f"[full pipeline] {report.format()}")

    joined = pd.concat(
        [
            gold.reset_index(drop=True),
            pred.reset_index(drop=True).rename(
                columns={c: f"pred_{c}" for c in pred.columns}
            ),
        ],
        axis=1,
    )

    gold_na = joined["gold_ontology_id"].isna()
    pred_na = joined["pred_ontology_id"].isna()
    mismatch = joined["gold_ontology_id"] != joined["pred_ontology_id"]
    joined["failure_kind"] = np.select(
        [
            gold_na & ~pred_na,
            ~gold_na & pred_na,
            ~gold_na & ~pred_na & mismatch,
        ],
        [
            "false_positive_id_on_unmappable",
            "abstained_on_mappable",
            "wrong_id_on_mappable",
        ],
        default="ok",
    )

    out_dir = cache_root / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "full_pipeline_predictions.parquet"
    joined.to_parquet(pred_path, index=False)
    print(f"\nwrote predictions: {pred_path}")

    failures = joined[joined["failure_kind"] != "ok"]
    fail_cols = [
        "biosample_id",
        "experiment_type",
        "cell_type",
        "gold_ontology_id",
        "gold_label",
        "pred_ontology_id",
        "pred_ontology_label",
        "pred_ontology_source",
        "pred_status",
        "pred_confidence",
        "pred_rationale",
        "failure_kind",
    ]
    fail_path = out_dir / "full_pipeline_failures.tsv"
    failures[fail_cols].to_csv(fail_path, sep="\t", index=False)
    print(f"wrote failures ({len(failures)}): {fail_path}")

    print("\nfailure-kind breakdown:")
    print(failures["failure_kind"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())

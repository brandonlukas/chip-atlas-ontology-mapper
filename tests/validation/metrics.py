"""Accuracy metrics for the Ikeda gold-standard harness.

The gold standard has two partitions:

- **mapped rows**: ``gold_ontology_id`` is a Cellosaurus accession. Scored on
  accuracy@1 (did we emit the correct id?) and accuracy@1_strict (same, but
  counting unmappable predictions as wrong instead of as abstentions).
- **unmappable rows**: ``gold_ontology_id`` is None. Scored on unmappable
  recall (fraction we correctly abstained on).

We also report overall coverage (fraction of rows we mapped at all), which
reads together with unmappable-recall to catch the degenerate cases of
"everything mapped" and "everything unmappable".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AccuracyReport:
    total: int
    mapped_total: int
    mapped_correct: int
    mapped_wrong_id: int
    mapped_abstained: int
    unmappable_total: int
    unmappable_correct: int
    unmappable_wrong: int

    @property
    def accuracy_at_1(self) -> float:
        """Correct / mapped_total, with abstentions counted as wrong.

        Undefined (returned as 0.0) when mapped_total == 0.
        """
        if self.mapped_total == 0:
            return 0.0
        return self.mapped_correct / self.mapped_total

    @property
    def pick_precision(self) -> float:
        """Correct / (correct + wrong_id) among rows where we emitted any id.

        Complements accuracy_at_1: a high pick_precision with low accuracy_at_1
        means the pipeline is accurate when it commits but abstains too often.
        """
        committed = self.mapped_correct + self.mapped_wrong_id
        if committed == 0:
            return 0.0
        return self.mapped_correct / committed

    @property
    def unmappable_recall(self) -> float:
        """Correctly abstained / unmappable_total. Undefined when 0."""
        if self.unmappable_total == 0:
            return 0.0
        return self.unmappable_correct / self.unmappable_total

    @property
    def coverage(self) -> float:
        """Fraction of rows the pipeline emitted a non-null id for."""
        if self.total == 0:
            return 0.0
        predicted = (
            self.mapped_correct
            + self.mapped_wrong_id
            + self.unmappable_wrong
        )
        return predicted / self.total

    def format(self) -> str:
        """One-liner-ish summary for pytest failure messages / stdout."""
        return (
            f"n={self.total}  "
            f"acc@1={self.accuracy_at_1:.3f} "
            f"(correct={self.mapped_correct}/{self.mapped_total}, "
            f"wrong_id={self.mapped_wrong_id}, abstained={self.mapped_abstained})  "
            f"pick_precision={self.pick_precision:.3f}  "
            f"unmap_recall={self.unmappable_recall:.3f} "
            f"({self.unmappable_correct}/{self.unmappable_total})  "
            f"coverage={self.coverage:.3f}"
        )


def compute_accuracy(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    pred_id_col: str = "ontology_id",
    gold_id_col: str = "gold_ontology_id",
) -> AccuracyReport:
    """Join predictions to gold row-wise and tally outcomes.

    Both DataFrames must be the same length and in the same row order. We do
    not join on any key — the harness constructs ``predictions`` by calling
    ``map_chipatlas(gold[['cell_type', ...]])``, which preserves row order.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"row count mismatch: predictions={len(predictions)}, gold={len(gold)}"
        )

    pred_ids = predictions[pred_id_col].reset_index(drop=True)
    gold_ids = gold[gold_id_col].reset_index(drop=True)

    gold_has = gold_ids.notna()
    pred_has = pred_ids.notna()

    mapped_rows = gold_has
    unmappable_rows = ~gold_has

    mapped_correct = int(((mapped_rows) & (pred_ids == gold_ids)).sum())
    mapped_wrong_id = int(
        (mapped_rows & pred_has & (pred_ids != gold_ids)).sum()
    )
    mapped_abstained = int((mapped_rows & ~pred_has).sum())

    unmappable_correct = int((unmappable_rows & ~pred_has).sum())
    unmappable_wrong = int((unmappable_rows & pred_has).sum())

    return AccuracyReport(
        total=len(gold),
        mapped_total=int(mapped_rows.sum()),
        mapped_correct=mapped_correct,
        mapped_wrong_id=mapped_wrong_id,
        mapped_abstained=mapped_abstained,
        unmappable_total=int(unmappable_rows.sum()),
        unmappable_correct=unmappable_correct,
        unmappable_wrong=unmappable_wrong,
    )

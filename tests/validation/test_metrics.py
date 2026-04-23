"""Unit tests for the accuracy metric helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.validation.metrics import AccuracyReport, compute_accuracy


def _gold(rows: list[tuple[str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cell_type", "gold_ontology_id"])


def _pred(ids: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame({"ontology_id": ids})


def test_all_correct() -> None:
    gold = _gold([("K562", "CVCL_0004"), ("MCF7", "CVCL_0031")])
    pred = _pred(["CVCL_0004", "CVCL_0031"])
    r = compute_accuracy(pred, gold)
    assert r.mapped_total == 2
    assert r.mapped_correct == 2
    assert r.accuracy_at_1 == 1.0
    assert r.pick_precision == 1.0
    assert r.coverage == 1.0


def test_wrong_id_counts_as_wrong_not_abstained() -> None:
    gold = _gold([("K562", "CVCL_0004")])
    pred = _pred(["CVCL_9999"])
    r = compute_accuracy(pred, gold)
    assert r.mapped_correct == 0
    assert r.mapped_wrong_id == 1
    assert r.mapped_abstained == 0
    assert r.accuracy_at_1 == 0.0
    assert r.pick_precision == 0.0


def test_abstention_lowers_accuracy_but_not_pick_precision() -> None:
    gold = _gold([("K562", "CVCL_0004"), ("MCF7", "CVCL_0031")])
    pred = _pred(["CVCL_0004", None])
    r = compute_accuracy(pred, gold)
    assert r.mapped_correct == 1
    assert r.mapped_abstained == 1
    assert r.accuracy_at_1 == 0.5
    # Only one row committed to a pick; that pick was right.
    assert r.pick_precision == 1.0


def test_unmappable_recall() -> None:
    gold = _gold([
        ("liver tissue", None),
        ("tumor", None),
        ("K562", "CVCL_0004"),
    ])
    pred = _pred([None, "CVCL_1234", "CVCL_0004"])
    r = compute_accuracy(pred, gold)
    assert r.unmappable_total == 2
    assert r.unmappable_correct == 1
    assert r.unmappable_wrong == 1
    assert r.unmappable_recall == 0.5


def test_coverage_is_fraction_with_any_prediction() -> None:
    gold = _gold([
        ("K562", "CVCL_0004"),
        ("MCF7", "CVCL_0031"),
        ("tissue", None),
        ("other tissue", None),
    ])
    pred = _pred(["CVCL_0004", None, "CVCL_9999", None])
    r = compute_accuracy(pred, gold)
    # Two predictions emitted out of four rows.
    assert r.coverage == 0.5


def test_row_count_mismatch_raises() -> None:
    gold = _gold([("K562", "CVCL_0004")])
    pred = _pred(["CVCL_0004", "CVCL_0031"])
    with pytest.raises(ValueError, match="row count mismatch"):
        compute_accuracy(pred, gold)


def test_empty_inputs_do_not_crash() -> None:
    gold = pd.DataFrame({"cell_type": [], "gold_ontology_id": []})
    pred = pd.DataFrame({"ontology_id": []})
    r = compute_accuracy(pred, gold)
    assert r.total == 0
    assert r.accuracy_at_1 == 0.0
    assert r.unmappable_recall == 0.0
    assert r.coverage == 0.0


def test_format_includes_all_headline_numbers() -> None:
    r = AccuracyReport(
        total=4,
        mapped_total=2,
        mapped_correct=1,
        mapped_wrong_id=0,
        mapped_abstained=1,
        unmappable_total=2,
        unmappable_correct=2,
        unmappable_wrong=0,
    )
    s = r.format()
    assert "acc@1=0.500" in s
    assert "unmap_recall=1.000" in s
    assert "coverage=" in s

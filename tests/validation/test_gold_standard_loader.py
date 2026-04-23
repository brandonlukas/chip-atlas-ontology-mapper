"""Unit tests for the Ikeda gold-standard loader.

Offline — writes a tiny synthetic TSV to tmp_path and exercises the parsing
and normalization paths without touching Zenodo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.validation.ikeda_gold_standard import (
    OUTPUT_COLUMNS,
    load_gold_standard,
    parse_gold_standard,
    summarize,
)

# Header + four rows spanning the real-world shapes:
# - mapped cell line with punctuation in the label
# - mapped cell line where extraction answer and label don't match literally
# - unmappable: both extraction answer and mapping id blank (primary tissue)
# - partial: extraction answer present but mapping id blank (cell line not in Cellosaurus)
_FIXTURE_TSV = (
    "BioSample ID\tExperiment type\textraction answer\tmapping answer ID\tmapping answer label\n"
    "SAMD00011704\tHistone\tH2126\tCVCL:1532\tNCI-H2126\n"
    "SAMEA6479266\tATAC-Seq\tRS4;11\tCVCL:0093\tRS4;11\n"
    "SAMEA103884999\tATAC-Seq\t\t\t\n"
    "SAMD00144947\tTFs and others\tCAL1\t\t\n"
)


@pytest.fixture
def gold_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "gold.tsv"
    p.write_text(_FIXTURE_TSV)
    return p


def test_parse_gold_standard_columns(gold_tsv: Path) -> None:
    df = parse_gold_standard(gold_tsv)
    assert list(df.columns) == list(OUTPUT_COLUMNS)
    assert len(df) == 4


def test_parse_gold_standard_normalizes_cvcl_colon_to_underscore(gold_tsv: Path) -> None:
    df = parse_gold_standard(gold_tsv)
    assert df.loc[0, "gold_ontology_id"] == "CVCL_1532"
    assert df.loc[1, "gold_ontology_id"] == "CVCL_0093"


def test_parse_gold_standard_blank_ids_become_null(gold_tsv: Path) -> None:
    # pandas normalizes None to NaN in object columns; notna() treats both as missing.
    df = parse_gold_standard(gold_tsv)
    assert pd.isna(df.loc[2, "gold_ontology_id"])
    assert pd.isna(df.loc[3, "gold_ontology_id"])
    assert pd.isna(df.loc[2, "gold_label"])


def test_parse_gold_standard_blank_extraction_answer_empty_string(gold_tsv: Path) -> None:
    """Empty extraction answer is kept as '' — map_chipatlas treats it as unmappable."""
    df = parse_gold_standard(gold_tsv)
    assert df.loc[2, "cell_type"] == ""
    assert df.loc[3, "cell_type"] == "CAL1"


def test_parse_gold_standard_preserves_punctuation_in_cell_type(gold_tsv: Path) -> None:
    df = parse_gold_standard(gold_tsv)
    assert df.loc[1, "cell_type"] == "RS4;11"


def test_parse_gold_standard_raises_on_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("foo\tbar\n1\t2\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        parse_gold_standard(bad)


def test_load_gold_standard_uses_cache(tmp_path: Path, gold_tsv: Path) -> None:
    """A pre-populated cache short-circuits the download entirely."""
    cache_root = tmp_path / "cache"
    cache_dest = cache_root / "validation" / "biosample_cellosaurus_mapping_gold_standard.tsv"
    cache_dest.parent.mkdir(parents=True)
    cache_dest.write_text(_FIXTURE_TSV)

    df = load_gold_standard(
        cache_root,
        url="http://invalid.example/should-not-be-fetched",
    )
    assert len(df) == 4


def test_summarize_counts(gold_tsv: Path) -> None:
    df = parse_gold_standard(gold_tsv)
    stats = summarize(df)
    assert stats.total == 4
    assert stats.mapped == 2
    assert stats.unmappable == 2
    assert stats.with_cell_type == 3


def test_summarize_on_empty_frame() -> None:
    df = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    stats = summarize(df)
    assert stats.total == 0
    assert stats.mapped == 0
    assert stats.unmappable == 0
    assert stats.with_cell_type == 0

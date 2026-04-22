"""Unit tests for the EFO ontology loader (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from caom.ontologies.efo import (
    EFOTerms,
    is_cached,
    load_terms,
    parse_efo,
    refresh_cache,
    save_terms,
)

# A tiny OBO fixture. Pronto accepts .obo natively, which is far cheaper to
# parse in tests than the real ~100 MB EFO OWL.
OBO_FIXTURE = """\
format-version: 1.2
data-version: efo/releases/2099-01-01/efo.obo
ontology: efo

[Term]
id: CL:0000000
name: cell
def: "Smallest unit of an organism." []

[Term]
id: EFO:0001187
name: HEK293
def: "Human embryonic kidney cell line." []
synonym: "HEK-293" EXACT []
synonym: "HEK 293" RELATED []
is_a: CL:0000000 ! cell

[Term]
id: CL:0000182
name: hepatocyte
def: "A cell of the liver." []
synonym: "liver cell" EXACT []
is_a: CL:0000000 ! cell

[Term]
id: EFO:OBSOLETE
name: old thing
is_obsolete: true
"""


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "efo.obo"
    p.write_text(OBO_FIXTURE)
    return p


# -- parse_efo ---------------------------------------------------------------


def test_parse_efo_extracts_terms_and_skips_obsolete(tmp_path: Path):
    df, version = parse_efo(_write_fixture(tmp_path))
    assert "efo/releases/2099-01-01/efo.obo" in version

    ids = df["ontology_id"].tolist()
    assert set(ids) == {"CL:0000000", "EFO:0001187", "CL:0000182"}
    assert "EFO:OBSOLETE" not in ids

    hek = df.set_index("ontology_id").loc["EFO:0001187"]
    assert hek["label"] == "HEK293"
    assert set(hek["synonyms"]) == {"HEK-293", "HEK 293"}
    assert hek["definition"] == "Human embryonic kidney cell line."
    assert hek["parents"] == ["CL:0000000"]


def test_parse_efo_has_expected_columns(tmp_path: Path):
    df, _ = parse_efo(_write_fixture(tmp_path))
    assert list(df.columns) == [
        "ontology_id",
        "label",
        "synonyms",
        "definition",
        "parents",
    ]


# -- save / load roundtrip ---------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path):
    df, version = parse_efo(_write_fixture(tmp_path))
    terms = EFOTerms(terms=df, version=version, downloaded_at="2099-01-01T00:00:00+00:00")
    save_terms(tmp_path, terms)

    assert (tmp_path / "ontologies" / "efo" / "terms.parquet").exists()
    assert (tmp_path / "ontologies" / "efo" / "metadata.json").exists()
    assert is_cached(tmp_path)

    reloaded = load_terms(tmp_path)
    assert reloaded.version == version
    assert reloaded.downloaded_at == "2099-01-01T00:00:00+00:00"
    assert len(reloaded.terms) == 3


def test_load_without_cache_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        load_terms(tmp_path)


# -- refresh_cache honors existing cache -------------------------------------


def test_refresh_cache_skips_download_when_cached(tmp_path: Path, monkeypatch):
    df, version = parse_efo(_write_fixture(tmp_path))
    save_terms(
        tmp_path,
        EFOTerms(terms=df, version=version, downloaded_at="2099-01-01T00:00:00+00:00"),
    )

    # If refresh_cache tries to download, the monkey-patched function will fail
    # the test. It shouldn't be invoked because is_cached() returns True.
    def _fail(*a, **kw):
        raise AssertionError("download_efo should not be called when cached")

    monkeypatch.setattr("caom.ontologies.efo.download_efo", _fail)
    terms = refresh_cache(tmp_path)
    assert isinstance(terms.terms, pd.DataFrame)
    assert len(terms.terms) == 3

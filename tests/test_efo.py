"""Unit tests for the EFO ontology loader (Stage 3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from caom.ontologies.efo import (
    EFOTerms,
    is_cached,
    load_terms,
    normalize_ontology_id,
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


# -- normalize_ontology_id (Stage 6) -----------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Already-CURIE forms pass through untouched.
        ("CL:0000000", "CL:0000000"),
        ("EFO:0001187", "EFO:0001187"),
        ("UBERON:0002174", "UBERON:0002174"),
        # EFO-native URIs → EFO CURIE (the dominant failure mode from the
        # post-Stage-5 smoke test).
        ("http://www.ebi.ac.uk/efo/EFO_0022456", "EFO:0022456"),
        ("http://www.ebi.ac.uk/efo/EFO_0000001", "EFO:0000001"),
        # Orphanet disease terms imported into EFO.
        ("http://www.orpha.net/ORDO/Orphanet_100", "Orphanet:100"),
        ("http://www.orpha.net/ORDO/Orphanet_100006", "Orphanet:100006"),
        # BAO uses a `#` fragment.
        ("http://www.bioassayontology.org/bao#BAO_0000875", "BAO:0000875"),
        # DBpedia resources: tail is a label with underscores; keep it verbatim
        # after the prefix so we don't corrupt multi-word names.
        ("http://dbpedia.org/resource/Albania", "dbpedia:Albania"),
        ("http://dbpedia.org/resource/Burkina_Faso", "dbpedia:Burkina_Faso"),
    ],
)
def test_normalize_ontology_id(raw: str, expected: str):
    assert normalize_ontology_id(raw) == expected


def test_normalize_ontology_id_leaves_unknown_uri_unchanged():
    # Unrecognized hosts pass through so callers can surface them rather
    # than silently producing a wrong CURIE.
    weird = "http://example.org/whatever/Foo_42"
    assert normalize_ontology_id(weird) == weird


def test_normalize_ontology_id_handles_empty():
    assert normalize_ontology_id("") == ""


def test_parse_efo_normalizes_uri_ontology_ids(tmp_path: Path):
    """Full URI term IDs (pronto outputs these for hosts outside its idspace
    map) should land in the parsed DataFrame as canonical CURIEs — for both
    the term itself and every parent reference."""
    obo = """\
format-version: 1.2
data-version: efo/releases/2099-01-01/efo.obo
ontology: efo

[Term]
id: http://www.ebi.ac.uk/efo/EFO_0000001
name: experimental factor

[Term]
id: http://www.ebi.ac.uk/efo/EFO_0022456
name: example downstream term
is_a: http://www.ebi.ac.uk/efo/EFO_0000001 ! experimental factor
"""
    p = tmp_path / "uri.obo"
    p.write_text(obo)
    df, _ = parse_efo(p)

    ids = set(df["ontology_id"])
    assert ids == {"EFO:0000001", "EFO:0022456"}
    child = df.set_index("ontology_id").loc["EFO:0022456"]
    assert child["parents"] == ["EFO:0000001"]


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

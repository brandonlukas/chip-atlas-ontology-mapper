"""Unit tests for the Cellosaurus fast-path (Stage 2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from caom.ontologies.cellosaurus import (
    build_lookup,
    load_lookup,
    normalize_name,
    parse_cellosaurus,
    save_lookup,
)

# A minimal synthetic Cellosaurus flat file covering:
#   - K-562 with punctuation-variant synonyms (the canonical test case)
#   - MCF-7 (simple cell line)
#   - A mouse cell line to exercise species filtering
#   - A same-name-different-species pair to exercise ambiguity / taxon filter
FIXTURE = """\
----------------------------------------------------------------------------
        CALIPHO group at the SIB - Swiss Institute of Bioinformatics
----------------------------------------------------------------------------

 Description: Cellosaurus: a knowledge resource on cell lines
 Version: 99.9
 Last update: 01-Jan-2026

----------------------------------------------------------------------------
ID   K-562
AC   CVCL_0004
SY   K562; K.562; K 562; GM05372; GM05372E
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Cancer cell line
DT   Created: 04-04-12; Last updated: 22-06-23; Version: 45
//
ID   MCF-7
AC   CVCL_0031
SY   MCF7; MCF 7
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Cancer cell line
//
ID   NIH-3T3
AC   CVCL_0594
SY   NIH/3T3; NIH 3T3; 3T3
OX   NCBI_TaxID=10090; ! Mus musculus
CA   Spontaneously immortalized cell line
//
ID   Raw 264.7
AC   CVCL_0493
SY   RAW264.7; RAW
OX   NCBI_TaxID=10090; ! Mus musculus
CA   Spontaneously immortalized cell line
//
ID   RAW
AC   CVCL_XXXX
SY   Raw
OX   NCBI_TaxID=9606; ! Homo sapiens
CA   Undefined cell line type
//
"""


@pytest.fixture
def lookup(tmp_path: Path):
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(FIXTURE)
    entries, version = parse_cellosaurus(flat)
    return build_lookup(entries, version=version, downloaded_at="2026-01-01T00:00:00+00:00")


# -- normalize_name ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("K-562", "k562"),
        ("K562", "k562"),
        ("K 562", "k562"),
        ("K.562", "k562"),
        ("MCF-7", "mcf7"),
        ("Hep G2", "hepg2"),
        ("HepG2", "hepg2"),
        ("HEP-G2", "hepg2"),
        ("   K-562   ", "k562"),
        ("", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


# -- parse_cellosaurus -------------------------------------------------------


def test_parse_extracts_entries_and_version(tmp_path: Path):
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(FIXTURE)
    entries, version = parse_cellosaurus(flat)

    assert version == "99.9"
    assert set(entries) == {"CVCL_0004", "CVCL_0031", "CVCL_0594", "CVCL_0493", "CVCL_XXXX"}

    k562 = entries["CVCL_0004"]
    assert k562.primary_name == "K-562"
    assert k562.synonyms == ["K562", "K.562", "K 562", "GM05372", "GM05372E"]
    assert k562.taxon_ids == ["9606"]
    assert k562.category == "Cancer cell line"
    assert "Homo sapiens" in k562.species[0]


# -- lookup ------------------------------------------------------------------


@pytest.mark.parametrize("query", ["K-562", "K562", "K 562", "K.562", "  k562  "])
def test_k562_aliases_all_resolve_to_cvcl_0004(lookup, query):
    matches = lookup.lookup(query)
    assert [m.accession for m in matches] == ["CVCL_0004"]


def test_lookup_miss_returns_empty(lookup):
    assert lookup.lookup("NoSuchCellLine") == []


def test_lookup_empty_query_returns_empty(lookup):
    assert lookup.lookup("") == []
    assert lookup.lookup("   ") == []


def test_species_filter_resolves_cross_species_name_collision(lookup):
    # "Raw" appears in a human entry (CVCL_XXXX) and matches "RAW" synonym of
    # the mouse Raw 264.7 entry — both normalize to "raw". Taxon filter picks
    # the right one.
    human_hits = [m.accession for m in lookup.lookup("raw", taxon_id="9606")]
    mouse_hits = [m.accession for m in lookup.lookup("raw", taxon_id="10090")]
    assert human_hits == ["CVCL_XXXX"]
    assert mouse_hits == ["CVCL_0493"]


def test_taxon_filter_dropping_sole_candidate_returns_empty(lookup):
    # MCF-7 is human-only; asking for mouse should return nothing.
    assert lookup.lookup("MCF-7", taxon_id="10090") == []


# -- save / load round-trip --------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path, lookup):
    save_lookup(tmp_path, lookup)
    assert (tmp_path / "ontologies" / "cellosaurus" / "lookup.pkl").exists()
    assert (tmp_path / "ontologies" / "cellosaurus" / "metadata.json").exists()

    reloaded = load_lookup(tmp_path)
    assert reloaded.version == "99.9"
    assert reloaded.lookup("K-562")[0].accession == "CVCL_0004"


def test_load_without_cache_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        load_lookup(tmp_path)


# -- map_chipatlas end-to-end ------------------------------------------------


def _install_fixture_cache(tmp_path: Path) -> None:
    """Pre-populate a tmp cache dir with the fixture lookup."""
    flat = tmp_path / "cellosaurus.txt"
    flat.write_text(FIXTURE)
    entries, version = parse_cellosaurus(flat)
    save_lookup(
        tmp_path,
        build_lookup(entries, version=version, downloaded_at="2026-01-01T00:00:00+00:00"),
    )


def test_map_chipatlas_returns_new_dataframe_with_expected_columns(tmp_path: Path):
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas
    from caom.api import OUTPUT_COLUMNS

    df = pd.DataFrame({"cell_type": ["K-562", "MCF-7", "UnknownCell"]})
    result = map_chipatlas(df, cache_dir=tmp_path)

    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert len(result) == len(df)
    assert result is not df
    assert list(df.columns) == ["cell_type"]


def test_map_chipatlas_cellosaurus_hits(tmp_path: Path):
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame(
        {
            "cell_type": ["K-562", "k 562", "MCF7", "UnknownCell"],
            "assembly": ["hg38", "hg19", "hg38", "hg38"],
        }
    )
    result = map_chipatlas(df, cache_dir=tmp_path)

    assert result.loc[0, "ontology_id"] == "CVCL_0004"
    assert result.loc[0, "status"] == "mapped"
    assert result.loc[0, "ontology_source"] == "cellosaurus"
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9"
    assert result.loc[0, "confidence"] == 1.0

    assert result.loc[1, "ontology_id"] == "CVCL_0004"
    assert result.loc[2, "ontology_id"] == "CVCL_0031"

    assert result.loc[3, "status"] == "unmappable"
    assert pd.isna(result.loc[3, "ontology_id"])
    assert pd.isna(result.loc[3, "ontology_source"])
    assert pd.isna(result.loc[3, "confidence"])


def test_map_chipatlas_species_filter_via_assembly(tmp_path: Path):
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame(
        {
            "cell_type": ["Raw", "Raw", "NIH-3T3", "NIH-3T3"],
            "assembly": ["mm10", "hg38", "mm10", "hg38"],
        }
    )
    result = map_chipatlas(df, cache_dir=tmp_path)

    # Raw + mouse → mouse Raw 264.7
    assert result.loc[0, "ontology_id"] == "CVCL_0493"
    # Raw + human → human RAW entry
    assert result.loc[1, "ontology_id"] == "CVCL_XXXX"
    # NIH-3T3 is mouse-only; correct assembly maps, wrong assembly defers.
    assert result.loc[2, "ontology_id"] == "CVCL_0594"
    assert result.loc[3, "status"] == "unmappable"
    assert "taxon 9606" in result.loc[3, "rationale"]


def test_map_chipatlas_without_assembly_column(tmp_path: Path):
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562", "MCF-7"]})
    result = map_chipatlas(df, cache_dir=tmp_path)
    assert result["status"].tolist() == ["mapped", "mapped"]


def test_map_chipatlas_handles_missing_cell_type(tmp_path: Path):
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562", None, "", "   "]})
    result = map_chipatlas(df, cache_dir=tmp_path)
    assert result.loc[0, "status"] == "mapped"
    assert result.loc[1, "status"] == "unmappable"
    assert result.loc[2, "status"] == "unmappable"
    assert result.loc[3, "status"] == "unmappable"


def test_map_chipatlas_without_cache_raises(tmp_path: Path):
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        map_chipatlas(df, cache_dir=tmp_path)


def test_map_chipatlas_review_mode_requires_efo_cache(tmp_path: Path):
    # Cellosaurus cache alone is not sufficient for review mode; the EFO
    # FAISS index must also be present.
    _install_fixture_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        map_chipatlas(df, cache_dir=tmp_path, review=True)

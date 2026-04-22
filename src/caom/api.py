"""Public API entry point: `map_chipatlas`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from caom.config import Config, load_config
from caom.ontologies.cellosaurus import (
    CellosaurusEntry,
    CellosaurusLookup,
    get_cached_lookup,
)
from caom.schema import validate_input
from caom.types import Mapping
from caom.version import __version__

_ASSEMBLY_PREFIX_TO_TAXON: tuple[tuple[str, str], ...] = (
    ("hg", "9606"),
    ("grch", "9606"),
    ("t2t", "9606"),
    ("chm13", "9606"),
    ("mm", "10090"),
    ("grcm", "10090"),
    ("rn", "10116"),
    ("grcr", "10116"),
    ("dm", "7227"),
    ("ce", "6239"),
    ("saccer", "559292"),
    ("danrer", "7955"),
    ("grcz", "7955"),
)

OUTPUT_COLUMNS: tuple[str, ...] = tuple(Mapping.model_fields)


def _taxon_id_from_assembly(assembly: object) -> str | None:
    if not isinstance(assembly, str) or not assembly:
        return None
    a = assembly.lower()
    for prefix, taxon in _ASSEMBLY_PREFIX_TO_TAXON:
        if a.startswith(prefix):
            return taxon
    return None


def _mapped(
    cell_type: str,
    entry: CellosaurusEntry,
    ontology_version: str,
    *,
    rationale: str,
) -> Mapping:
    return Mapping(
        cell_type=cell_type,
        status="mapped",
        ontology_id=entry.accession,
        ontology_label=entry.primary_name,
        confidence=1.0,
        rationale=rationale,
        ontology_source="cellosaurus",
        ontology_version=ontology_version,
        caom_version=__version__,
    )


def _unmappable(
    cell_type: str,
    ontology_version: str,
    *,
    rationale: str,
) -> Mapping:
    return Mapping(
        cell_type=cell_type,
        status="unmappable",
        ontology_id=None,
        ontology_label=None,
        confidence=None,
        rationale=rationale,
        ontology_source=None,
        ontology_version=ontology_version,
        caom_version=__version__,
    )


def _map_row(
    cell_type_value: object,
    assembly_value: object,
    lookup: CellosaurusLookup,
    ontology_version: str,
) -> Mapping:
    if not isinstance(cell_type_value, str) or not cell_type_value.strip():
        return _unmappable(
            cell_type=cell_type_value if isinstance(cell_type_value, str) else "",
            ontology_version=ontology_version,
            rationale="Missing or empty cell_type.",
        )

    cell_type = cell_type_value
    taxon_id = _taxon_id_from_assembly(assembly_value)
    candidates = lookup.lookup(cell_type, taxon_id=taxon_id)

    if not candidates and taxon_id is not None and lookup.lookup(cell_type):
        # Don't promote a cross-species match: symmetric normalization means
        # the hit could be coincidental. Defer to the Stage 4 LLM tier.
        return _unmappable(
            cell_type=cell_type,
            ontology_version=ontology_version,
            rationale=(
                f"Cellosaurus match(es) found but none for taxon {taxon_id} "
                f"(assembly={assembly_value!r}); deferred to EFO/LLM tier."
            ),
        )

    if len(candidates) == 1:
        return _mapped(
            cell_type=cell_type,
            entry=candidates[0],
            ontology_version=ontology_version,
            rationale="Exact match via Cellosaurus normalized-name lookup.",
        )

    if len(candidates) > 1:
        accs = ", ".join(c.accession for c in candidates)
        return _unmappable(
            cell_type=cell_type,
            ontology_version=ontology_version,
            rationale=(
                f"Ambiguous Cellosaurus match ({len(candidates)} candidates: {accs}); "
                "deferred to EFO/LLM tier."
            ),
        )

    return _unmappable(
        cell_type=cell_type,
        ontology_version=ontology_version,
        rationale="No Cellosaurus match; deferred to EFO/LLM tier.",
    )


def map_chipatlas(
    df: pd.DataFrame,
    *,
    review: bool = False,
    top_k: int = 10,
    cache_dir: str | Path | None = None,
    ollama_host: str | None = None,
    llm_model: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Map ChIP-Atlas rows to standardized ontology IDs (Cellosaurus + EFO).

    Parameters
    ----------
    df
        ChIP-Atlas metadata. Must contain a `cell_type` column. Any of
        `cell_type_class`, `cell_type_description`, `assembly`, `title`,
        `antigen`, `tf_name` are used if present.
    review
        If True, return top-K retrieval candidates per input row with scores
        and LLM rationale. If False (default), return one row per input row
        with the best pick or `status="unmappable"`.
    top_k
        Number of retrieval candidates surfaced to the LLM re-ranker, and the
        number of rows emitted per input in `review` mode.
    cache_dir, ollama_host, llm_model
        Per-call overrides. If `config` is passed, it takes precedence.

    Returns
    -------
    pd.DataFrame
        Always a new DataFrame; the input is never mutated.
    """
    validate_input(df)
    cfg = config or load_config(
        cache_dir=cache_dir, ollama_host=ollama_host, llm_model=llm_model
    )

    if review:
        raise NotImplementedError(
            "review=True surfaces top-K EFO candidates, which lands in Stage 3."
        )

    lookup = get_cached_lookup(cfg.cache_dir)
    ontology_version = f"cellosaurus:{lookup.version}" if lookup.version else "cellosaurus"

    cell_types = df["cell_type"].tolist()
    assemblies = (
        df["assembly"].tolist() if "assembly" in df.columns else [None] * len(df)
    )
    rows = [
        _map_row(ct, asm, lookup, ontology_version).model_dump()
        for ct, asm in zip(cell_types, assemblies, strict=True)
    ]
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))

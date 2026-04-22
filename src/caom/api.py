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
from caom.retrieval.embedder import EmbedderProtocol
from caom.retrieval.embedder import get_cached_embedder as _get_cached_embedder
from caom.retrieval.index import EFOIndex
from caom.retrieval.index import get_cached_index as _get_cached_index
from caom.schema import validate_input
from caom.types import Candidate, Mapping, ReviewRow
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
REVIEW_COLUMNS: tuple[str, ...] = tuple(ReviewRow.model_fields)


def _taxon_id_from_assembly(assembly: object) -> str | None:
    if not isinstance(assembly, str) or not assembly:
        return None
    a = assembly.lower()
    for prefix, taxon in _ASSEMBLY_PREFIX_TO_TAXON:
        if a.startswith(prefix):
            return taxon
    return None


def _assembly_list(df: pd.DataFrame) -> list[object]:
    if "assembly" in df.columns:
        return df["assembly"].tolist()
    return [None] * len(df)


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


def _cellosaurus_candidates(
    cell_type_value: object,
    assembly_value: object,
    lookup: CellosaurusLookup,
) -> tuple[list[CellosaurusEntry], str | None]:
    """Return (candidates, unmappable_rationale).

    If `unmappable_rationale` is not None, the caller should skip EFO retrieval
    in best-pick mode; it explains why.
    """
    if not isinstance(cell_type_value, str) or not cell_type_value.strip():
        return [], "Missing or empty cell_type."

    taxon_id = _taxon_id_from_assembly(assembly_value)
    candidates = lookup.lookup(cell_type_value, taxon_id=taxon_id)

    if not candidates and taxon_id is not None and lookup.lookup(cell_type_value):
        # Don't promote a cross-species match: symmetric normalization means
        # the hit could be coincidental. Defer to the Stage 4 LLM tier.
        return (
            [],
            f"Cellosaurus match(es) found but none for taxon {taxon_id} "
            f"(assembly={assembly_value!r}); deferred to EFO/LLM tier.",
        )
    return candidates, None


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
    candidates, deferred_reason = _cellosaurus_candidates(
        cell_type_value, assembly_value, lookup
    )

    if deferred_reason is not None:
        return _unmappable(
            cell_type=cell_type,
            ontology_version=ontology_version,
            rationale=deferred_reason,
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


def _review_rows_for(
    cell_type: str,
    cellosaurus_candidates: list[CellosaurusEntry],
    efo_candidates: list[Candidate],
    cellosaurus_version: str,
    efo_version: str,
) -> list[ReviewRow]:
    """Compose ReviewRows for one input row.

    - Cellosaurus single hit → 1 Cellosaurus row (no EFO retrieval was run).
    - Ambiguous Cellosaurus (>1) → all Cellosaurus candidates, then EFO top-K.
    - No Cellosaurus match → EFO top-K only.
    """
    rows: list[ReviewRow] = []
    for c in cellosaurus_candidates:
        rows.append(
            ReviewRow(
                cell_type=cell_type,
                rank=len(rows) + 1,
                ontology_id=c.accession,
                ontology_label=c.primary_name,
                retrieval_score=None,
                llm_confidence=None,
                ontology_source="cellosaurus",
                ontology_version=cellosaurus_version,
                caom_version=__version__,
            )
        )
    for cand in efo_candidates:
        rows.append(
            ReviewRow(
                cell_type=cell_type,
                rank=len(rows) + 1,
                ontology_id=cand.ontology_id,
                ontology_label=cand.ontology_label,
                retrieval_score=cand.retrieval_score,
                llm_confidence=None,
                ontology_source="efo",
                ontology_version=efo_version,
                caom_version=__version__,
            )
        )
    return rows


def _efo_query_text(cell_type: str, row: pd.Series) -> str:
    """Build the retrieval query string from the cell_type + optional metadata."""
    parts: list[str] = [cell_type]
    for col in ("cell_type_class", "cell_type_description", "title"):
        if col in row.index:
            v = row[col]
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    return " | ".join(parts)


def _run_review(
    df: pd.DataFrame,
    lookup: CellosaurusLookup,
    efo_index: EFOIndex,
    embedder: EmbedderProtocol,
    top_k: int,
    cellosaurus_version: str,
    efo_version: str,
) -> pd.DataFrame:
    cell_types_raw = df["cell_type"].tolist()
    assemblies = _assembly_list(df)

    need_retrieval_idx: list[int] = []
    per_row_cand: list[list[CellosaurusEntry]] = []
    for i, (ct, asm) in enumerate(zip(cell_types_raw, assemblies, strict=True)):
        cands, _deferred = _cellosaurus_candidates(ct, asm, lookup)
        per_row_cand.append(cands)
        if len(cands) != 1:
            need_retrieval_idx.append(i)

    efo_results: dict[int, list[Candidate]] = {}
    if need_retrieval_idx:
        queries = [
            _efo_query_text(str(cell_types_raw[i] or ""), df.iloc[i])
            for i in need_retrieval_idx
        ]
        batches = efo_index.search_texts(queries, embedder=embedder, top_k=top_k)
        for i, cands in zip(need_retrieval_idx, batches, strict=True):
            efo_results[i] = cands

    out_rows: list[dict] = []
    for i, ct in enumerate(cell_types_raw):
        cell_type_str = ct if isinstance(ct, str) else ""
        rev_rows = _review_rows_for(
            cell_type=cell_type_str,
            cellosaurus_candidates=per_row_cand[i],
            efo_candidates=efo_results.get(i, []),
            cellosaurus_version=cellosaurus_version,
            efo_version=efo_version,
        )
        out_rows.extend(r.model_dump() for r in rev_rows)

    return pd.DataFrame(out_rows, columns=list(REVIEW_COLUMNS))


def map_chipatlas(
    df: pd.DataFrame,
    *,
    review: bool = False,
    top_k: int | None = None,
    cache_dir: str | Path | None = None,
    ollama_host: str | None = None,
    llm_model: str | None = None,
    embedder: EmbedderProtocol | None = None,
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
        If True, return top-K retrieval candidates per input row with scores.
        If False (default), return one row per input row with the best pick
        or `status="unmappable"`.
    top_k
        Number of retrieval candidates surfaced per input row in review mode
        (and, once Stage 4 lands, to the LLM re-ranker). Defaults to
        `config.retrieval_top_k`.
    cache_dir, ollama_host, llm_model
        Per-call overrides. If `config` is passed, it takes precedence.
    embedder
        Optional injected embedder (mainly for tests). Defaults to the
        sentence-transformers model named in `config.embedding_model`.

    Returns
    -------
    pd.DataFrame
        Always a new DataFrame; the input is never mutated. In review mode
        there may be multiple output rows per input row.
    """
    validate_input(df)
    cfg = config or load_config(
        cache_dir=cache_dir, ollama_host=ollama_host, llm_model=llm_model
    )
    effective_top_k = top_k if top_k is not None else cfg.retrieval_top_k

    lookup = get_cached_lookup(cfg.cache_dir)
    cellosaurus_version = f"cellosaurus:{lookup.version}" if lookup.version else "cellosaurus"

    if not review:
        cell_types = df["cell_type"].tolist()
        assemblies = _assembly_list(df)
        rows = [
            _map_row(ct, asm, lookup, cellosaurus_version).model_dump()
            for ct, asm in zip(cell_types, assemblies, strict=True)
        ]
        return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))

    efo_index = _get_cached_index(cfg.cache_dir)
    efo_version = f"efo:{efo_index.efo_version}" if efo_index.efo_version else "efo"
    emb = embedder if embedder is not None else _get_cached_embedder(cfg.embedding_model)

    return _run_review(
        df=df,
        lookup=lookup,
        efo_index=efo_index,
        embedder=emb,
        top_k=effective_top_k,
        cellosaurus_version=cellosaurus_version,
        efo_version=efo_version,
    )

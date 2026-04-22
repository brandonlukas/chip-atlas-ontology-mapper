"""Public API entry point: `map_chipatlas`."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from caom.cache import get_cached_llm_cache
from caom.config import Config, load_config
from caom.llm.client import LLMClient, OllamaClient
from caom.llm.prompts import build_rerank_prompt
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
from caom.types import Candidate, LLMPick, Mapping, ReviewRow
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

_EFO_QUERY_COLUMNS: tuple[str, ...] = (
    "cell_type_class",
    "cell_type_description",
    "title",
)

_LLM_METADATA_COLUMNS: tuple[str, ...] = (
    "cell_type_class",
    "cell_type_description",
    "assembly",
    "title",
    "antigen",
    "tf_name",
)

OUTPUT_COLUMNS: tuple[str, ...] = tuple(Mapping.model_fields)
REVIEW_COLUMNS: tuple[str, ...] = tuple(ReviewRow.model_fields)


def _nonempty_str(v: object) -> str | None:
    """Return a stripped string if `v` is a non-empty string, else None."""
    if isinstance(v, str):
        s = v.strip()
        if s:
            return s
    return None


def _row_str_fields(row: pd.Series, cols: Iterable[str]) -> dict[str, str]:
    """Collect stripped non-empty string values from `row` for the given `cols`."""
    out: dict[str, str] = {}
    for c in cols:
        if c in row.index:
            s = _nonempty_str(row[c])
            if s is not None:
                out[c] = s
    return out


def _taxon_id_from_assembly(assembly: object) -> str | None:
    s = _nonempty_str(assembly)
    if s is None:
        return None
    a = s.lower()
    for prefix, taxon in _ASSEMBLY_PREFIX_TO_TAXON:
        if a.startswith(prefix):
            return taxon
    return None


def _assembly_list(df: pd.DataFrame) -> list[object]:
    if "assembly" in df.columns:
        return df["assembly"].tolist()
    return [None] * len(df)


def _mapped_cellosaurus(
    cell_type: str,
    entry: CellosaurusEntry,
    ontology_version: str,
) -> Mapping:
    return Mapping(
        cell_type=cell_type,
        status="mapped",
        ontology_id=entry.accession,
        ontology_label=entry.primary_name,
        confidence=1.0,
        rationale="Exact match via Cellosaurus normalized-name lookup.",
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
) -> list[CellosaurusEntry]:
    """Return Cellosaurus candidates for the row, or `[]` if none survive.

    Cross-species hits (the cell_type matches at some taxon but not the
    taxon implied by `assembly`) are dropped rather than promoted: the
    retrieval + LLM tier is the right place to adjudicate them.
    """
    if _nonempty_str(cell_type_value) is None:
        return []
    taxon_id = _taxon_id_from_assembly(assembly_value)
    return lookup.lookup(cell_type_value, taxon_id=taxon_id)  # type: ignore[arg-type]


def _efo_query_text(cell_type: str, row: pd.Series) -> str:
    """Build the retrieval query string from `cell_type` + optional metadata."""
    parts = [cell_type, *_row_str_fields(row, _EFO_QUERY_COLUMNS).values()]
    return " | ".join(parts)


def _llm_metadata(row: pd.Series) -> dict[str, str]:
    return _row_str_fields(row, _LLM_METADATA_COLUMNS)


def _cellosaurus_entry_to_candidate(entry: CellosaurusEntry) -> Candidate:
    """Project a Cellosaurus entry into the unified Candidate shape for the LLM."""
    meta_parts: list[str] = []
    if entry.species:
        meta_parts.append(f"organism: {', '.join(entry.species)}")
    if entry.category:
        meta_parts.append(f"category: {entry.category}")
    definition = "; ".join(meta_parts) if meta_parts else None
    return Candidate(
        ontology_id=entry.accession,
        ontology_label=entry.primary_name,
        ontology_source="cellosaurus",
        synonyms=list(entry.synonyms),
        definition=definition,
    )


def _build_llm_candidates(
    cellosaurus_entries: list[CellosaurusEntry],
    efo_candidates: list[Candidate],
) -> list[Candidate]:
    """Unify Cellosaurus (first) + EFO candidates into one ordered list."""
    return [_cellosaurus_entry_to_candidate(e) for e in cellosaurus_entries] + list(
        efo_candidates
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


def _classify_rows(
    df: pd.DataFrame,
    lookup: CellosaurusLookup,
) -> tuple[list[str], list[list[CellosaurusEntry]]]:
    """Return `(normalized_cell_types, per_row_candidates)` aligned with `df`.

    `normalized_cell_types[i]` is the stripped string, or `""` when the input
    cell_type is missing/blank. `per_row_candidates[i]` is `[]` in that case
    (short-circuiting Cellosaurus lookup for invalid rows).
    """
    assemblies = _assembly_list(df)
    normalized = [_nonempty_str(ct) or "" for ct in df["cell_type"].tolist()]
    per_row_cand = [
        _cellosaurus_candidates(nt, asm, lookup) if nt else []
        for nt, asm in zip(normalized, assemblies, strict=True)
    ]
    return normalized, per_row_cand


def _retrieve_efo(
    df: pd.DataFrame,
    indices: list[int],
    normalized: list[str],
    efo_index: EFOIndex,
    embedder: EmbedderProtocol,
    top_k: int,
) -> dict[int, list[Candidate]]:
    if not indices:
        return {}
    queries = [_efo_query_text(normalized[i], df.iloc[i]) for i in indices]
    batches = efo_index.search_texts(queries, embedder=embedder, top_k=top_k)
    return dict(zip(indices, batches, strict=True))


def _run_review(
    df: pd.DataFrame,
    lookup: CellosaurusLookup,
    efo_index: EFOIndex,
    embedder: EmbedderProtocol,
    top_k: int,
    cellosaurus_version: str,
    efo_version: str,
) -> pd.DataFrame:
    normalized, per_row_cand = _classify_rows(df, lookup)
    need_retrieval = [i for i, cands in enumerate(per_row_cand) if len(cands) != 1]
    efo_results = _retrieve_efo(
        df, need_retrieval, normalized, efo_index, embedder, top_k
    )

    out_rows: list[dict] = []
    for i, ct in enumerate(normalized):
        rev_rows = _review_rows_for(
            cell_type=ct,
            cellosaurus_candidates=per_row_cand[i],
            efo_candidates=efo_results.get(i, []),
            cellosaurus_version=cellosaurus_version,
            efo_version=efo_version,
        )
        out_rows.extend(r.model_dump() for r in rev_rows)

    return pd.DataFrame(out_rows, columns=list(REVIEW_COLUMNS))


def _pick_to_mapping(
    cell_type: str,
    pick: LLMPick,
    candidates: list[Candidate],
    *,
    cellosaurus_version: str,
    efo_version: str,
) -> Mapping:
    combined_version = f"{cellosaurus_version};{efo_version}"
    if pick.ontology_id is None:
        return _unmappable(
            cell_type=cell_type,
            ontology_version=combined_version,
            rationale=(
                f"LLM: {pick.rationale}" if pick.rationale else "LLM returned no match."
            ),
        )
    cand = next((c for c in candidates if c.ontology_id == pick.ontology_id), None)
    if cand is None:
        return _unmappable(
            cell_type=cell_type,
            ontology_version=combined_version,
            rationale=(
                f"LLM returned non-candidate id {pick.ontology_id!r} "
                f"(discarded as hallucination). Original rationale: {pick.rationale}"
            ),
        )
    version = cellosaurus_version if cand.ontology_source == "cellosaurus" else efo_version
    return Mapping(
        cell_type=cell_type,
        status="mapped",
        ontology_id=cand.ontology_id,
        ontology_label=cand.ontology_label,
        confidence=pick.confidence,
        rationale=pick.rationale,
        ontology_source=cand.ontology_source,
        ontology_version=version,
        caom_version=__version__,
    )


def _default_llm_client(cfg: Config) -> LLMClient:
    cache = get_cached_llm_cache(cfg.cache_dir)
    return OllamaClient(model=cfg.llm_model, host=cfg.ollama_host, cache=cache)


def _run_best_pick(
    df: pd.DataFrame,
    lookup: CellosaurusLookup,
    cfg: Config,
    top_k: int,
    embedder: EmbedderProtocol | None,
    llm_client: LLMClient | None,
    cellosaurus_version: str,
) -> pd.DataFrame:
    normalized, per_row_cand = _classify_rows(df, lookup)
    needs_llm = [
        i
        for i, (nt, cands) in enumerate(zip(normalized, per_row_cand, strict=True))
        if nt and len(cands) != 1
    ]

    efo_results: dict[int, list[Candidate]] = {}
    efo_version = ""
    llm: LLMClient | None = None
    if needs_llm:
        efo_index = _get_cached_index(cfg.cache_dir)
        efo_version = f"efo:{efo_index.efo_version}" if efo_index.efo_version else "efo"
        emb = (
            embedder
            if embedder is not None
            else _get_cached_embedder(cfg.embedding_model)
        )
        llm = llm_client if llm_client is not None else _default_llm_client(cfg)
        efo_results = _retrieve_efo(
            df, needs_llm, normalized, efo_index, emb, top_k
        )

    out_rows: list[dict] = []
    for i, (ct, cands) in enumerate(zip(normalized, per_row_cand, strict=True)):
        if not ct:
            out_rows.append(
                _unmappable(
                    cell_type="",
                    ontology_version=cellosaurus_version,
                    rationale="Missing or empty cell_type.",
                ).model_dump()
            )
            continue
        if len(cands) == 1:
            out_rows.append(
                _mapped_cellosaurus(
                    cell_type=ct,
                    entry=cands[0],
                    ontology_version=cellosaurus_version,
                ).model_dump()
            )
            continue

        assert llm is not None
        row = df.iloc[i]
        candidates = _build_llm_candidates(cands, efo_results.get(i, []))
        prompt = build_rerank_prompt(
            cell_type=ct,
            metadata=_llm_metadata(row),
            candidates=candidates,
        )
        pick = llm.pick(prompt)
        out_rows.append(
            _pick_to_mapping(
                cell_type=ct,
                pick=pick,
                candidates=candidates,
                cellosaurus_version=cellosaurus_version,
                efo_version=efo_version,
            ).model_dump()
        )

    return pd.DataFrame(out_rows, columns=list(OUTPUT_COLUMNS))


def map_chipatlas(
    df: pd.DataFrame,
    *,
    review: bool = False,
    top_k: int | None = None,
    cache_dir: str | Path | None = None,
    ollama_host: str | None = None,
    llm_model: str | None = None,
    embedder: EmbedderProtocol | None = None,
    llm_client: LLMClient | None = None,
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
        (via LLM re-rank on non-trivial cases) or `status="unmappable"`.
    top_k
        Number of retrieval candidates surfaced per input row to the LLM
        re-ranker (and in review mode). Defaults to `config.retrieval_top_k`.
    cache_dir, ollama_host, llm_model
        Per-call overrides. If `config` is passed, it takes precedence.
    embedder
        Optional injected embedder (mainly for tests). Defaults to the
        sentence-transformers model named in `config.embedding_model`.
    llm_client
        Optional injected LLM client (mainly for tests). Defaults to an
        `OllamaClient` pointed at `config.ollama_host` with a SQLite-backed
        response cache under `<cache_dir>/llm/`.

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
    cellosaurus_version = (
        f"cellosaurus:{lookup.version}" if lookup.version else "cellosaurus"
    )

    if review:
        efo_index = _get_cached_index(cfg.cache_dir)
        efo_version = (
            f"efo:{efo_index.efo_version}" if efo_index.efo_version else "efo"
        )
        emb = (
            embedder
            if embedder is not None
            else _get_cached_embedder(cfg.embedding_model)
        )
        return _run_review(
            df=df,
            lookup=lookup,
            efo_index=efo_index,
            embedder=emb,
            top_k=effective_top_k,
            cellosaurus_version=cellosaurus_version,
            efo_version=efo_version,
        )

    return _run_best_pick(
        df=df,
        lookup=lookup,
        cfg=cfg,
        top_k=effective_top_k,
        embedder=embedder,
        llm_client=llm_client,
        cellosaurus_version=cellosaurus_version,
    )

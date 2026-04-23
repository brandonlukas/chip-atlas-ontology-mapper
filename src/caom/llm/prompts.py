"""Prompt template for the LLM re-rank step.

Input: a free-text `cell_type`, a dict of disambiguation fields from the
surrounding ChIP-Atlas row, and a unified list of candidate ontology terms
(Cellosaurus + EFO). Output: a prompt string that asks the LLM to pick one
candidate's id (verbatim) or null. The caller is responsible for invoking the
LLM with a schema-constrained JSON format (see `caom.types.LLMPick`).
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC

from caom.types import Candidate

_SYSTEM_INSTRUCTIONS = """\
You are a biomedical ontology curator. You are given a short cell-type
annotation from the ChIP-Atlas database (sometimes with noisy punctuation or
abbreviations) together with metadata from the surrounding experimental row
(assembly organism, assay title, description, antigen / transcription factor).
You are also given a list of candidate ontology terms drawn from Cellosaurus
(cell lines, prefix CVCL_) and EFO (cell types, tissues, diseases; prefixes
CL, EFO, UBERON, MONDO, etc.).

Your task is to pick the candidate whose id best matches the query, or return
null if none of the candidates is a correct match.

Rules:
- The `ontology_id` in your response MUST be copied verbatim from one of the
  candidate ids listed below. If no candidate is correct, return `null`.
- Do NOT invent ontology ids. Do NOT guess an id that is not in the list.
- Prefer a Cellosaurus (CVCL_) candidate for a named cell line; prefer an EFO
  / CL / UBERON / MONDO candidate for a primary cell type, tissue, or disease.
- When two candidates clearly refer to the same cell line (same primary label,
  or one is an obvious variant of the other — e.g. EFO mirroring a Cellosaurus
  entry), prefer the Cellosaurus (CVCL_) id. Cellosaurus is canonical for cell
  lines; do not let a higher EFO retrieval_score override this.
- Use the organism implied by `assembly` (hg* = human, mm* = mouse, rn* = rat,
  etc.) to disambiguate same-name cross-species candidates.
- If the query is a very short abbreviation (≤3 characters, e.g. "ED", "H1")
  with no disambiguating metadata, return `null` unless a candidate's primary
  label or one of its synonyms is an exact, case-insensitive match for the
  query. Do not stretch a short query into a partial / disease-name match.
- Candidates marked `[exact]` are label or exact-synonym matches for the
  query (after lowercase + non-alphanumeric stripping). Prefer them over
  cosine-ranked candidates unless the disambiguation context clearly
  contradicts (wrong organism, wrong disease context, wrong cell-line series).
- `confidence` is a float in [0, 1] reflecting how certain you are.
- `rationale` is one short sentence explaining the choice.
"""


def _truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _format_candidate(index: int, c: Candidate) -> str:
    marker = " [exact]" if c.exact else ""
    parts = [
        f"[{index}]{marker} id={c.ontology_id} source={c.ontology_source} "
        f"label={c.ontology_label!r}"
    ]
    if c.synonyms:
        shown = "; ".join(c.synonyms[:6])
        if len(c.synonyms) > 6:
            shown += f"; … (+{len(c.synonyms) - 6} more)"
        parts.append(f"synonyms: {shown}")
    if c.definition:
        parts.append(f"definition: {_truncate(c.definition, 240)}")
    if c.retrieval_score is not None:
        parts.append(f"retrieval_score: {c.retrieval_score:.3f}")
    return "\n    ".join(parts)


def _format_metadata(metadata: MappingABC[str, str]) -> str:
    kept = [(k, v.strip()) for k, v in metadata.items() if isinstance(v, str) and v.strip()]
    if not kept:
        return "  (none provided)"
    return "\n".join(f"  {k}: {v!r}" for k, v in kept)


def build_rerank_prompt(
    cell_type: str,
    metadata: MappingABC[str, str],
    candidates: list[Candidate],
) -> str:
    """Build the LLM re-rank prompt for one input row.

    Parameters
    ----------
    cell_type
        The raw ChIP-Atlas `cell_type` string.
    metadata
        Disambiguation context (e.g. `cell_type_class`, `cell_type_description`,
        `assembly`, `title`, `antigen`, `tf_name`). Missing / empty keys are
        silently dropped.
    candidates
        Unified candidate list. Cellosaurus candidates should come first so the
        LLM sees cell-line matches before EFO; the caller controls ordering.
    """
    if candidates:
        candidates_block = "\n\n".join(
            _format_candidate(i, c) for i, c in enumerate(candidates)
        )
    else:
        candidates_block = "(no candidates provided — return null)"

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n"
        "--- Query ---\n"
        f"cell_type: {cell_type!r}\n"
        f"Disambiguation context:\n{_format_metadata(metadata)}\n\n"
        "--- Candidates ---\n"
        f"{candidates_block}\n\n"
        "--- Task ---\n"
        "Return the JSON pick now."
    )

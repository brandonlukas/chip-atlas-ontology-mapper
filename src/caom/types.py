"""Shared dataclasses / pydantic models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["mapped", "unmappable", "error"]
OntologySource = Literal["cellosaurus", "efo"]


class Candidate(BaseModel):
    """A retrieval candidate surfaced to the LLM for re-ranking."""

    ontology_id: str
    ontology_label: str
    ontology_source: OntologySource
    synonyms: list[str] = Field(default_factory=list)
    definition: str | None = None
    retrieval_score: float | None = None
    exact: bool = False


class LLMPick(BaseModel):
    """Schema-constrained LLM output. `ontology_id=None` means unmappable."""

    ontology_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class Mapping(BaseModel):
    """One output row in best-pick mode."""

    cell_type: str
    status: Status
    ontology_id: str | None
    ontology_label: str | None
    confidence: float | None
    rationale: str | None
    ontology_source: OntologySource | None
    ontology_version: str | None
    caom_version: str


class ReviewRow(BaseModel):
    """One output row in review mode (multiple per input row, ranked)."""

    cell_type: str
    rank: int
    ontology_id: str
    ontology_label: str
    retrieval_score: float | None
    llm_confidence: float | None
    ontology_source: OntologySource
    ontology_version: str | None
    caom_version: str

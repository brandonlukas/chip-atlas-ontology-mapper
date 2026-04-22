"""Runtime configuration and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    cache_dir: Path
    ollama_host: str
    llm_model: str
    embedding_model: str
    retrieval_top_k: int


def load_config(
    cache_dir: str | Path | None = None,
    ollama_host: str | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    retrieval_top_k: int | None = None,
) -> Config:
    return Config(
        cache_dir=Path(cache_dir or os.environ.get("CAOM_CACHE_DIR", ".cache")).resolve(),
        ollama_host=ollama_host or os.environ.get("CAOM_OLLAMA_HOST", "http://localhost:11434"),
        llm_model=llm_model or os.environ.get("CAOM_LLM_MODEL", "qwen2.5:7b-instruct"),
        embedding_model=embedding_model
        or os.environ.get("CAOM_EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"),
        retrieval_top_k=retrieval_top_k
        or int(os.environ.get("CAOM_RETRIEVAL_TOP_K", "20")),
    )

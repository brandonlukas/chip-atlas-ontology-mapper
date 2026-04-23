"""sentence-transformers wrapper for query + corpus encoding. Stage 3."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from caom.cache import KeyedCache


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface: encode list[str] → L2-normalized float32 [N, dim]."""

    @property
    def dim(self) -> int: ...

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper producing L2-normalized float32 vectors.

    The default model is `pritamdeka/S-PubMedBert-MS-MARCO` (see config); the
    same model is used for corpus and query encoding so cosine similarity over
    the two sides is directly comparable.
    """

    def __init__(self, model_name: str, device: str | None = None):
        from huggingface_hub import try_to_load_from_cache
        from sentence_transformers import SentenceTransformer

        # If the model is already in the HF cache, load from disk without
        # hitting the Hub. `map_chipatlas()` is advertised as offline (the
        # only network-active entry point is `update_ontologies()`), and a
        # per-call Hub HEAD request both violates that and emits an
        # unauthenticated-rate-limit warning. `config.json` is a reliable
        # cache probe because every sentence-transformers model ships one.
        cached = try_to_load_from_cache(model_name, "config.json") is not None
        self._model = SentenceTransformer(
            model_name, device=device, local_files_only=cached
        )
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim = get_dim() or 0
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr


_EMBEDDER_CACHE: KeyedCache[
    tuple[str, str | None], SentenceTransformerEmbedder
] = KeyedCache()


def get_cached_embedder(
    model_name: str, *, device: str | None = None
) -> SentenceTransformerEmbedder:
    """Load (or reuse) a sentence-transformer embedder for `model_name`."""
    return _EMBEDDER_CACHE.get_or_load(
        (model_name, device),
        lambda: SentenceTransformerEmbedder(model_name, device=device),
    )

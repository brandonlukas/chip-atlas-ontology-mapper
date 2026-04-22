"""LLM client interface + Ollama implementation with schema-constrained JSON.

The client's job is narrow: take a rendered prompt, return a validated
`LLMPick`. ID hallucination is defended against in two layers:
1. Ollama is called with `format=LLMPick.model_json_schema()` so the raw
   content parses as the right shape.
2. The caller (`caom.api`) validates the picked id against the candidate set
   and discards non-matches.

Callers that want to avoid real inference (tests, batch replays) inject any
object satisfying `LLMClient`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from caom.cache import LLMCache
from caom.types import LLMPick

# Walking the pydantic model on every request is pure waste — the schema
# never changes for a given process.
_LLM_PICK_SCHEMA = LLMPick.model_json_schema()


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface used by `caom.api` for LLM re-rank."""

    model: str

    def pick(self, prompt: str) -> LLMPick: ...


class OllamaClient:
    """Ollama chat client with pydantic-schema-constrained JSON output.

    Uses `format=LLMPick.model_json_schema()` so the server constrains output
    to the expected shape. Temperature is pinned to 0 for determinism so the
    SQLite cache is meaningful across re-runs of the same input.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        cache: LLMCache | None = None,
        inner_client: Any | None = None,
    ):
        self.model = model
        self._host = host
        self._cache = cache
        self._client: Any | None = inner_client

    def _get_client(self) -> Any:
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._host) if self._host else Client()
        return self._client

    def pick(self, prompt: str) -> LLMPick:
        if self._cache is not None:
            cached = self._cache.get(self.model, prompt)
            if cached is not None:
                return LLMPick.model_validate(cached)

        client = self._get_client()
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=_LLM_PICK_SCHEMA,
            options={"temperature": 0.0},
        )
        content = response.message.content or "{}"
        pick = LLMPick.model_validate_json(content)

        if self._cache is not None:
            self._cache.put(self.model, prompt, pick.model_dump())
        return pick

"""Stage 4 unit tests: SQLite LLM cache, prompt shape, Ollama client (mocked)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from caom.cache import LLMCache
from caom.llm.client import LLMClient, OllamaClient
from caom.llm.prompts import build_rerank_prompt
from caom.types import Candidate, LLMPick

# --- SQLite LLM cache -------------------------------------------------------


def test_llm_cache_miss_then_hit_roundtrip(tmp_path: Path):
    with LLMCache(tmp_path / "llm.sqlite") as cache:
        assert cache.get("modelA", "prompt-1") is None

        cache.put(
            "modelA",
            "prompt-1",
            {"ontology_id": "CL:0000182", "confidence": 0.91, "rationale": "hepatocyte"},
        )
        assert cache.get("modelA", "prompt-1") == {
            "ontology_id": "CL:0000182",
            "confidence": 0.91,
            "rationale": "hepatocyte",
        }


def test_llm_cache_keys_on_model_and_prompt(tmp_path: Path):
    with LLMCache(tmp_path / "llm.sqlite") as cache:
        cache.put("modelA", "prompt-1", {"ontology_id": "A"})
        cache.put("modelB", "prompt-1", {"ontology_id": "B"})
        cache.put("modelA", "prompt-2", {"ontology_id": "C"})

        assert cache.get("modelA", "prompt-1") == {"ontology_id": "A"}
        assert cache.get("modelB", "prompt-1") == {"ontology_id": "B"}
        assert cache.get("modelA", "prompt-2") == {"ontology_id": "C"}
        assert cache.get("modelA", "prompt-3") is None


def test_llm_cache_put_is_idempotent(tmp_path: Path):
    with LLMCache(tmp_path / "llm.sqlite") as cache:
        cache.put("m", "p", {"ontology_id": "X", "confidence": 0.1, "rationale": "old"})
        cache.put("m", "p", {"ontology_id": "X", "confidence": 0.9, "rationale": "new"})
        assert cache.get("m", "p")["rationale"] == "new"


def test_llm_cache_persists_across_connections(tmp_path: Path):
    db = tmp_path / "llm.sqlite"
    with LLMCache(db) as cache:
        cache.put("m", "p", {"ontology_id": "X"})
    with LLMCache(db) as reopened:
        assert reopened.get("m", "p") == {"ontology_id": "X"}


# --- prompt builder ---------------------------------------------------------


def _candidate(
    *,
    ontology_id: str,
    ontology_label: str,
    source: str = "efo",
    synonyms: list[str] | None = None,
    definition: str | None = None,
    score: float | None = None,
) -> Candidate:
    return Candidate(
        ontology_id=ontology_id,
        ontology_label=ontology_label,
        ontology_source=source,  # type: ignore[arg-type]
        synonyms=synonyms or [],
        definition=definition,
        retrieval_score=score,
    )


def test_prompt_contains_cell_type_metadata_and_every_candidate_id():
    candidates = [
        _candidate(
            ontology_id="CVCL_0004",
            ontology_label="K-562",
            source="cellosaurus",
            synonyms=["K562", "K 562"],
            definition="organism: Homo sapiens; category: Cancer cell line",
        ),
        _candidate(
            ontology_id="CL:0000236",
            ontology_label="B cell",
            synonyms=["B lymphocyte"],
            definition="A lymphocyte of B lineage.",
            score=0.82,
        ),
    ]
    prompt = build_rerank_prompt(
        cell_type="K-562",
        metadata={"assembly": "hg38", "title": "ChIP-seq"},
        candidates=candidates,
    )

    assert "'K-562'" in prompt
    assert "assembly" in prompt and "'hg38'" in prompt
    assert "title" in prompt and "'ChIP-seq'" in prompt
    for cand in candidates:
        assert cand.ontology_id in prompt
        assert cand.ontology_label in prompt
    assert "0.820" in prompt  # retrieval_score formatting
    # Unmappable escape hatch is wired into the instructions.
    assert "null" in prompt.lower()


def test_prompt_empty_metadata_and_no_candidates_has_sensible_defaults():
    prompt = build_rerank_prompt(cell_type="???", metadata={}, candidates=[])
    assert "none provided" in prompt
    assert "no candidates provided" in prompt


def test_prompt_truncates_long_definitions():
    long_def = "x" * 500
    candidates = [
        _candidate(ontology_id="ID:1", ontology_label="lbl", definition=long_def)
    ]
    prompt = build_rerank_prompt("q", {}, candidates)
    # The full 500-char definition should NOT be inlined verbatim.
    assert long_def not in prompt
    # But a truncated prefix + ellipsis should be present.
    assert "x" * 200 in prompt
    assert "…" in prompt


def test_prompt_caps_synonym_list():
    long_syns = [f"alias_{i}" for i in range(12)]
    candidates = [_candidate(ontology_id="ID:1", ontology_label="lbl", synonyms=long_syns)]
    prompt = build_rerank_prompt("q", {}, candidates)
    assert "alias_0" in prompt
    assert "alias_5" in prompt
    # Beyond the first 6, the prompt notes additional aliases by count.
    assert "alias_11" not in prompt
    assert "+6 more" in prompt


def test_prompt_carries_stage8_disambiguation_rules():
    """Stage 8 added two principled rules — they must stay in the system block.

    1. Cellosaurus-prefer when CVCL_ and EFO candidates describe the same entity.
    2. Caution on ≤3-char queries with no disambiguating context.

    Concretely worded so a future prompt rewrite that drops them will fail loudly
    rather than silently reintroduce the BLaER / P493 / "ED" failure modes.
    """
    # Whitespace-flatten so soft-wrapped lines in the system block don't break
    # phrase matching.
    flat = " ".join(build_rerank_prompt("q", {}, []).lower().split())
    assert "prefer the cellosaurus (cvcl_) id" in flat
    assert "do not let a higher efo retrieval_score override this" in flat
    assert "≤3 characters" in flat
    assert "do not stretch a short query" in flat


# --- OllamaClient (with mocked inner client) --------------------------------


class _StubChatClient:
    """Fake ollama.Client with a programmable response and call log."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def chat(self, *, model, messages, format, options, **kwargs):
        self.calls.append(
            {"model": model, "messages": messages, "format": format, "options": options}
        )
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def _make_ollama_with_stub(stub: _StubChatClient, cache: LLMCache | None = None):
    return OllamaClient(model="test-model", cache=cache, inner_client=stub)


def test_ollama_client_implements_llmclient_protocol():
    assert isinstance(OllamaClient(model="m"), LLMClient)


def test_ollama_client_returns_parsed_pick_and_sends_schema():
    stub = _StubChatClient(
        '{"ontology_id": "CL:0000182", "confidence": 0.9, "rationale": "hepatocyte match"}'
    )
    client = _make_ollama_with_stub(stub)

    pick = client.pick("pretend prompt")

    assert isinstance(pick, LLMPick)
    assert pick.ontology_id == "CL:0000182"
    assert pick.confidence == pytest.approx(0.9)
    assert stub.calls[0]["model"] == "test-model"
    assert stub.calls[0]["options"]["temperature"] == 0.0
    # The schema-constrained format must be populated (pydantic JSON schema dict).
    schema = stub.calls[0]["format"]
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert set(schema["properties"]) >= {"ontology_id", "confidence", "rationale"}


def test_ollama_client_uses_cache_on_second_call(tmp_path: Path):
    stub = _StubChatClient(
        '{"ontology_id": "CL:0000236", "confidence": 0.5, "rationale": "B cell"}'
    )
    with LLMCache(tmp_path / "llm.sqlite") as cache:
        client = _make_ollama_with_stub(stub, cache=cache)

        first = client.pick("prompt-A")
        second = client.pick("prompt-A")  # should hit cache, not the stub

        assert first == second
        assert len(stub.calls) == 1  # only the first call reached Ollama


def test_ollama_client_handles_null_ontology_id():
    stub = _StubChatClient(
        '{"ontology_id": null, "confidence": 0.0, "rationale": "nothing matches"}'
    )
    client = _make_ollama_with_stub(stub)
    pick = client.pick("prompt")
    assert pick.ontology_id is None
    assert pick.rationale == "nothing matches"

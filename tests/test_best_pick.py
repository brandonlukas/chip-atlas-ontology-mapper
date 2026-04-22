"""Stage 4 end-to-end tests for `map_chipatlas(review=False)` with LLM re-rank.

The LLM is stubbed via `tests.conftest.FakeLLMClient`; the embedder via
`FakeEmbedder`. No network or model loads occur.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from caom.types import LLMPick
from tests.conftest import (
    FakeEmbedder,
    FakeLLMClient,
    install_cellosaurus_cache,
    install_full_cache,
)

# --- happy paths ------------------------------------------------------------


def test_single_cellosaurus_hit_bypasses_llm(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas
    from caom.api import OUTPUT_COLUMNS

    llm = FakeLLMClient()
    df = pd.DataFrame({"cell_type": ["K-562"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
    )

    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert result.loc[0, "ontology_id"] == "CVCL_0004"
    assert result.loc[0, "status"] == "mapped"
    assert result.loc[0, "ontology_source"] == "cellosaurus"
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9"
    # LLM was never consulted — fast-path bypasses retrieval + re-rank entirely.
    assert llm.calls == []


def test_cellosaurus_miss_routes_to_llm_and_emits_efo_pick(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    llm = FakeLLMClient(
        rules={
            "hepatocyte": LLMPick(
                ontology_id="CL:0000182", confidence=0.92, rationale="liver cell"
            )
        }
    )
    df = pd.DataFrame({"cell_type": ["hepatocyte"], "assembly": ["hg38"]})
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm, top_k=3
    )

    assert len(result) == 1
    assert result.loc[0, "status"] == "mapped"
    assert result.loc[0, "ontology_id"] == "CL:0000182"
    assert result.loc[0, "ontology_label"] == "hepatocyte"
    assert result.loc[0, "ontology_source"] == "efo"
    assert result.loc[0, "ontology_version"] == "efo:99.9"
    assert result.loc[0, "confidence"] == pytest.approx(0.92)
    assert result.loc[0, "rationale"] == "liver cell"
    assert len(llm.calls) == 1


def test_ambiguous_cellosaurus_passes_both_sources_to_llm(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    # LLM picks the mouse Cellosaurus entry; both candidates must have been
    # offered in the prompt.
    llm = FakeLLMClient(
        rules={
            "cvcl_0493": LLMPick(
                ontology_id="CVCL_0493",
                confidence=0.8,
                rationale="mouse raw macrophage",
            )
        }
    )
    df = pd.DataFrame({"cell_type": ["Raw"]})  # ambiguous without assembly
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
    )

    assert result.loc[0, "status"] == "mapped"
    assert result.loc[0, "ontology_id"] == "CVCL_0493"
    assert result.loc[0, "ontology_source"] == "cellosaurus"
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9"

    (prompt,) = llm.calls
    # Both ambiguous Cellosaurus candidates appear; LLM had a real choice.
    assert "CVCL_0493" in prompt
    assert "CVCL_XXXX" in prompt
    # EFO top-K also appended.
    assert "CL:0000182" in prompt or "CL:0000236" in prompt


# --- failure / safety paths -------------------------------------------------


def test_llm_null_pick_produces_unmappable_with_llm_rationale(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    llm = FakeLLMClient(
        default_pick=LLMPick(
            ontology_id=None, confidence=0.1, rationale="no candidate fits"
        )
    )
    df = pd.DataFrame({"cell_type": ["gibberish xyz"]})
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
    )

    assert result.loc[0, "status"] == "unmappable"
    assert pd.isna(result.loc[0, "ontology_id"])
    assert pd.isna(result.loc[0, "ontology_source"])
    assert "no candidate fits" in result.loc[0, "rationale"]
    assert result.loc[0, "ontology_version"] == "cellosaurus:99.9;efo:99.9"


def test_llm_hallucinated_id_is_discarded_as_unmappable(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    llm = FakeLLMClient(
        default_pick=LLMPick(
            ontology_id="EFO:9999999",  # not among the offered candidates
            confidence=0.99,
            rationale="hallucinated",
        )
    )
    df = pd.DataFrame({"cell_type": ["hepatocyte"]})
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
    )

    assert result.loc[0, "status"] == "unmappable"
    assert pd.isna(result.loc[0, "ontology_id"])
    assert "hallucination" in result.loc[0, "rationale"]


def test_llm_called_once_per_needing_row_not_per_single_hit(tmp_path: Path):
    install_full_cache(tmp_path)
    from caom import map_chipatlas

    llm = FakeLLMClient(
        rules={
            "hepatocyte": LLMPick(
                ontology_id="CL:0000182", confidence=0.9, rationale="x"
            ),
            "b cell": LLMPick(
                ontology_id="CL:0000236", confidence=0.9, rationale="x"
            ),
        }
    )
    df = pd.DataFrame(
        {
            "cell_type": ["K-562", "hepatocyte", "B cell"],
            "assembly": ["hg38"] * 3,
        }
    )
    _ = map_chipatlas(df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm)

    # K-562 is a single Cellosaurus hit → no LLM call.
    assert len(llm.calls) == 2


def test_best_pick_without_efo_cache_raises_when_llm_needed(tmp_path: Path):
    # Cellosaurus only. The one row is a single hit, so best-pick should NOT
    # need EFO or the LLM.
    install_cellosaurus_cache(tmp_path)
    from caom import map_chipatlas

    df = pd.DataFrame({"cell_type": ["K-562"], "assembly": ["hg38"]})
    llm = FakeLLMClient()
    result = map_chipatlas(
        df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
    )
    assert result.loc[0, "status"] == "mapped"
    assert llm.calls == []

    # But a row that misses Cellosaurus must trigger loading of the EFO index
    # and should raise when the index isn't cached.
    df_miss = pd.DataFrame({"cell_type": ["UnknownCell"], "assembly": ["hg38"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        map_chipatlas(
            df_miss, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=llm
        )


def test_best_pick_llm_cache_suppresses_repeated_calls(tmp_path: Path):
    """Two map_chipatlas calls with the same (query, candidates) → one LLM call.

    This wires the on-disk SQLite cache through the real OllamaClient shell
    with a programmable inner client, proving the cache key spans invocations.
    """
    install_full_cache(tmp_path)
    from types import SimpleNamespace

    from caom import map_chipatlas
    from caom.cache import LLMCache, llm_cache_path
    from caom.llm.client import OllamaClient

    class StubInner:
        def __init__(self):
            self.calls = 0

        def chat(self, **kw):
            self.calls += 1
            return SimpleNamespace(
                message=SimpleNamespace(
                    content='{"ontology_id": "CL:0000182", "confidence": 0.9, "rationale": "x"}'
                )
            )

    stub = StubInner()
    cache = LLMCache(llm_cache_path(tmp_path))
    client = OllamaClient(model="fake-model", cache=cache, inner_client=stub)

    df = pd.DataFrame({"cell_type": ["hepatocyte"], "assembly": ["hg38"]})
    _ = map_chipatlas(df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=client)
    _ = map_chipatlas(df, cache_dir=tmp_path, embedder=FakeEmbedder(), llm_client=client)

    assert stub.calls == 1
    cache.close()

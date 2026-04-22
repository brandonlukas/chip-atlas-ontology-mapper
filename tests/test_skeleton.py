"""Smoke tests: package imports cleanly and public API is reachable."""

import pandas as pd
import pytest


def test_package_imports():
    import caom

    assert caom.__version__
    assert callable(caom.map_chipatlas)
    assert callable(caom.update_ontologies)


def test_config_respects_env(monkeypatch, tmp_path):
    from caom.config import load_config

    monkeypatch.setenv("CAOM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CAOM_OLLAMA_HOST", "http://test:11434")
    monkeypatch.setenv("CAOM_LLM_MODEL", "test-model")

    cfg = load_config()
    assert cfg.cache_dir == tmp_path.resolve()
    assert cfg.ollama_host == "http://test:11434"
    assert cfg.llm_model == "test-model"


def test_map_chipatlas_validates_required_columns():
    import caom

    df = pd.DataFrame({"not_cell_type": ["x"]})
    with pytest.raises(ValueError, match="cell_type"):
        caom.map_chipatlas(df)


def test_map_chipatlas_requires_ontology_cache(tmp_path):
    """Without an ontology cache, map_chipatlas should tell the user to update."""
    import caom

    df = pd.DataFrame({"cell_type": ["K-562", "MCF-7"]})
    with pytest.raises(FileNotFoundError, match="update_ontologies"):
        caom.map_chipatlas(df, cache_dir=tmp_path)

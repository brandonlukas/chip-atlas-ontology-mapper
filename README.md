# chip-atlas-ontology-mapper

Map [ChIP-Atlas](https://chip-atlas.org/) `cell_type` annotations to standardized biomedical ontology IDs (Cellosaurus + EFO).

ChIP-Atlas's controlled vocabulary is curated but not ontology-encoded. This tool closes that gap for downstream reproducibility and cross-dataset joins. It is deliberately scoped to ChIP-Atlas metadata — not a general-purpose entity linker.

## Approach

A deterministic fast-path for cell lines plus an LLM-assisted retrieval step for everything else:

1. **Cellosaurus fast-path.** Normalized-name lookup against Cellosaurus resolves the cell-line majority (K-562, MCF-7, HeLa, ...) with near-100% accuracy and zero LLM cost.
2. **Retrieval.** For cells/tissues/diseases not in Cellosaurus, embed the query against a FAISS index of EFO labels + synonyms and pull top-K candidates.
3. **LLM re-rank (Ollama).** A local Ollama model picks the best candidate from the top-K, using `cell_type_class`, `cell_type_description`, `assembly`, and `title` as disambiguation context. Output is schema-constrained — the model cannot invent IDs outside the candidate set. Output includes explicit `"unmappable"` status when no candidate fits.

All inference runs locally; the only network calls are the initial ontology downloads via `caom.update_ontologies()`.

## Installation

Create a dedicated conda env and install in editable mode:

```bash
conda create -n caom python=3.11 -y
conda activate caom
pip install -e '.[dev]'
```

Start a local Ollama instance and pull the default model:

```bash
ollama pull qwen2.5:7b-instruct
```

## Usage

```python
import caom
import pandas as pd

# One-time (or on-demand) ontology download + embedding index build.
caom.update_ontologies()

# Input: a ChIP-Atlas metadata DataFrame with `cell_type` (required) plus any of
# `cell_type_class`, `cell_type_description`, `assembly`, `title`, `antigen`, `tf_name`.
df = pd.read_parquet("chipatlas_metadata.parquet")

mapped = caom.map_chipatlas(df)                       # best-pick mode, 1 row in -> 1 row out
review = caom.map_chipatlas(df, review=True, top_k=10)  # top-K candidates per row
```

### Output (best-pick mode)

| column | type | description |
| --- | --- | --- |
| `cell_type` | str | original ChIP-Atlas label |
| `status` | `"mapped" \| "unmappable" \| "error"` | outcome |
| `ontology_id` | str \| None | e.g. `CVCL_0004`, `EFO:0001203` |
| `ontology_label` | str \| None | canonical term label |
| `confidence` | float \| None | 0-1 |
| `rationale` | str \| None | short model/rule explanation |
| `ontology_source` | `"cellosaurus" \| "efo"` | which ontology produced the hit |
| `ontology_version` | str | for reproducibility |
| `caom_version` | str | tool version stamp |

## Configuration

Defaults can be overridden per call or via environment variables:

| option | env var | default |
| --- | --- | --- |
| Ollama host | `CAOM_OLLAMA_HOST` | `http://localhost:11434` |
| LLM model | `CAOM_LLM_MODEL` | `qwen2.5:7b-instruct` |
| Cache dir | `CAOM_CACHE_DIR` | `./.cache` (project-local) |

## Updating ontologies

```python
caom.update_ontologies()             # download if missing or a new version is pinned
caom.update_ontologies(force=True)   # force re-download and rebuild FAISS index
```

The ontology version used for each mapping is stamped on every output row (`ontology_version`) for reproducibility.

## Validation

The test suite includes an accuracy gate against the [Ikeda et al. 2025 gold standard](https://doi.org/10.5281/zenodo.14881142) (~322 manually curated BioSample → Cellosaurus mappings). Run:

```bash
pytest tests/validation
```

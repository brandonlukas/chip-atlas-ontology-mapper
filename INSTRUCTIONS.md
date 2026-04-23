# Instructions for future sessions

Read this first when picking up work on this repo. It's the "what are we building and why" brief; `CLAUDE.md` is narrowly about environment policy.

## Mission

`caom` maps ChIP-Atlas `cell_type` annotations (messy free-text: `K-562`, `MCF-7`, `CD4+ T cells`, `Hep G2`, `iPSC derived neural cells`, ...) to standardized biomedical ontology IDs (Cellosaurus + EFO). It is **deliberately ChIP-Atlas-specific** — resist generalizing it. Its advantage over general-purpose tools (e.g. `text2term`) is that it exploits ChIP-Atlas's structured metadata (`cell_type_class`, `cell_type_description`, `assembly`, `title`) as disambiguation signal.

Downstream consumer: the `matcha` project (`~/code/matcha`). The parquet at `~/code/matcha/data/metadata/curated_metadata.parquet` is the canonical input shape — though note `matcha` renames `cell_type` to `context_id` internally; `caom` keys on the ChIP-Atlas-native name.

## Current status

- **Stage 1: skeleton + pyproject + env** — DONE
  - conda env `caom` exists with Python 3.12 and all deps installed editable
  - Public API stubs (`map_chipatlas`, `update_ontologies`) import and raise `NotImplementedError`
- **Stage 2: Cellosaurus fast-path** — DONE
  - `caom.ontologies.cellosaurus`: download + parse flat file, normalized-name lookup, pickled cache with `metadata.json` sidecar, in-memory cache for repeat calls
  - `caom.update_ontologies()` orchestrates the download; `caom.map_chipatlas()` runs the fast-path per row with species filter via `assembly`, emits `Mapping` rows (schema-validated) with `ontology_source="cellosaurus"` on hits and `status="unmappable"` on misses / intra-species ambiguity
  - 33 tests passing; live end-to-end on Cellosaurus v55.0
- **Stage 3: EFO download + FAISS retrieval** — DONE
  - `caom.ontologies.efo`: pronto-based download + parse of EFO OWL/OBO → `terms.parquet` (columns: `ontology_id`, `label`, `synonyms`, `definition`, `parents`) + `metadata.json` sidecar; in-memory cache mirroring Cellosaurus
  - `caom.retrieval.embedder`: `SentenceTransformerEmbedder` wrapper producing L2-normalized float32 vectors with an in-memory model cache (default `pritamdeka/S-PubMedBert-MS-MARCO`)
  - `caom.retrieval.index`: FAISS `IndexFlatIP` over label+synonyms+definition concatenations, row-aligned with the terms parquet and persisted at `.cache/embeddings/efo.{faiss,metadata.json}` + `efo_terms.parquet`
  - `caom.update_ontologies()` now downloads EFO and builds/saves the FAISS index after Cellosaurus
  - `map_chipatlas(review=True)` returns `ReviewRow`s: Cellosaurus single hit → 1 row; ambiguous → all Cellosaurus candidates + EFO top-K; miss → EFO top-K. EFO queries are batched across rows in a single embedder call.
  - 50 tests passing (17 new). Tests avoid real model loads via an injected `embedder=` parameter on `map_chipatlas` + a `_FakeEmbedder` with one-hot vectors aligned to a hand-built terms DataFrame.
- **Stage 4: Ollama LLM re-rank** — DONE
  - `caom.cache.LLMCache`: SQLite cache keyed by `(model, prompt_hash)` persisted at `.cache/llm/llm_cache.sqlite`; round-trip verified across connection reopen
  - `caom.llm.client`: `LLMClient` Protocol + `OllamaClient` implementation with `format=LLMPick.model_json_schema()` and `temperature=0` for deterministic / cacheable output; inner ollama client is lazy-imported
  - `caom.llm.prompts.build_rerank_prompt`: unified candidate list (Cellosaurus first, then EFO top-K), truncates long definitions + caps synonyms, includes organism/category hints for Cellosaurus candidates, mandates verbatim-id pick or null
  - `caom.api.map_chipatlas` best-pick wired end-to-end: single Cellosaurus hit still bypasses the LLM; ambiguous Cellosaurus + Cellosaurus miss + cross-species defer all route through retrieval → LLM; hallucinated ids (not in the offered candidate set) are discarded as unmappable
  - 70 tests passing (20 new, split across `tests/test_llm.py` unit tests and `tests/test_best_pick.py` end-to-end tests with a `FakeLLMClient`); `tests/conftest.py` hosts shared fixtures
- **Stage 5: Ikeda validation harness + accuracy gates** — DONE
  - `tests/validation/ikeda_gold_standard.py`: downloads Zenodo TSV (600 rows), caches to `.cache/validation/`, normalizes `CVCL:NNNN` → `CVCL_NNNN`, projects into a DataFrame with `cell_type` (from `extraction answer`) + `gold_ontology_id`. The BioSample → ChIP-Atlas `cell_type` join was sidestepped: the TSV's extraction-answer column already has the free-text cell-line name, so we skip the SRA bridge.
  - `tests/validation/metrics.py`: `AccuracyReport` with `accuracy_at_1`, `pick_precision` (correct / committed), `unmappable_recall`, `coverage`; counts abstentions separately from wrong IDs so a high-abstention tier is distinguishable from a low-accuracy one
  - `tests/validation/runner.py`: `Mode.CELLOSAURUS_ONLY` (injects null embedder + null LLM → measures Stage 2 alone) and `Mode.FULL` (real pipeline) drivers on top of `map_chipatlas`
  - `tests/validation/test_accuracy.py`: gated on `CAOM_RUN_VALIDATION=1`; full-pipeline sub-gate additionally needs `CAOM_RUN_LLM_VALIDATION=1`. Stage 2 baseline observed 2026-04-22: `acc@1=0.903, pick_precision=1.000, unmap_recall=0.997, coverage=0.450` (Cellosaurus v49, EFO v3.89). Gate floors set below observed to absorb version drift.
  - 87 tests passing (17 new offline: loader + metrics unit tests); 2 validation-gate tests skip by default
- **Stage 6: EFO ontology-ID normalization** — NEXT
  - Surfaced during a post-Stage-5 smoke test on 24 real ChIP-Atlas rows: the EFO candidate pipeline emits a mix of CURIE form (`UBERON:0002174`, `CL:0000623`) and full-URI form (`http://www.ebi.ac.uk/efo/EFO_0022456`) depending on the source term's `id` in the OWL. The LLM returns whichever form it saw, so downstream output is heterogeneous and will trip consumers.
  - Scope: normalize to a single CURIE form at ingest time (`caom.ontologies.efo.parse_efo` or immediately downstream), so `terms.parquet` and the FAISS-aligned candidate list only ever carry `PREFIX:LOCAL`. Prevents the issue at the root rather than post-processing `map_chipatlas` output.
  - Must land before Stage 7 so the full-pipeline baseline measures retrieval/LLM quality, not an ID-format artifact.
- **Stage 7: full-pipeline threshold tuning** — after Stage 6
  - Run `CAOM_RUN_LLM_VALIDATION=1 pytest tests/validation` to get a baseline with clean IDs, then iterate on prompt + retrieval + top-K to close the gap between Stage 2 (acc@1=0.903) and the full pipeline. Known failure mode from the smoke test: retrieval surfaces over-narrow candidates for broad queries (`Lung` → `middle lobe of right lung`, `Brain` → `diencephalon`), which prompt / parent-term augmentation should address.

Each stage is independently useful. Stage 2 alone handles ~75% of real ChIP-Atlas contexts (cell lines).

## Architecture (hybrid retrieval + LLM re-rank)

```
Input: DataFrame row with cell_type [+ metadata]
  │
  ▼
┌─────────────────────────────┐
│ Cellosaurus fast-path       │  Deterministic, no LLM.
│ normalized-name lookup      │  Handles cell-line majority.
└─────────────────────────────┘
  │ miss
  ▼
┌─────────────────────────────┐
│ Retrieval                   │  Embed query via sentence-transformers,
│ FAISS top-K over EFO        │  top-K (default 20) cosine-nearest terms.
│ labels + synonyms + defs    │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ LLM re-rank (Ollama)        │  Pick best candidate (or null = unmappable).
│ schema-constrained JSON     │  Prompt includes cell_type_class,
│ output via pydantic         │  description, assembly, title.
└─────────────────────────────┘
  │
  ▼
Output: Mapping row (best-pick) or ReviewRow (review=True)
```

**Why this shape:**
- Cellosaurus is authoritative for cell lines and ships with dense synonyms; exact-match after light normalization is near-100% accurate with zero LLM cost. No reason to use an LLM on the easy majority.
- TF-IDF character overlap (`text2term`'s approach) fails on short codes, punctuation variants, and semantic mismatches. Embeddings + LLM re-rank fixes both failure modes.
- LLM's job is "pick 1 of 20", not "know biology". Structured JSON output prevents ID hallucination.
- Ontology scope is intentionally **lumped** (EFO re-uses CL / UBERON / MONDO terms internally). Revisit only if coverage is insufficient on real data.

## Locked-in decisions (and why)

| Decision | Rationale |
| --- | --- |
| **Python library only, no CLI (yet)** | API is the source of truth; a CLI is a thin shim added later if needed. |
| **Input = `pd.DataFrame`, key column = `cell_type`** | ChIP-Atlas native name. `matcha` users can rename `context_id → cell_type` upstream. |
| **Per-row mapping granularity** | `title`/`description` are row-specific. Users dedup upstream if they want per-`cell_type` behavior. |
| **Best-pick default; `review=True` returns top-K** | Best-pick is production; review is validation and human/LLM audit. |
| **Cellosaurus + EFO, lumped** | Minimal moving parts. Expand only if real coverage gaps force it. |
| **Ollama with `qwen2.5:7b-instruct` default; swappable** | Strong at structured pick-one tasks, runs anywhere with reasonable VRAM. |
| **Project-local `./.cache/`** | Reproducible per-repo; ontology artifacts are hundreds of MB but don't change per project. |
| **Explicit `caom.update_ontologies()`** | Ontology refresh is a decision, not a side effect. Every output row is stamped with the ontology version used. |
| **Explicit `status="unmappable"` + null `ontology_id`** | Silent drops hide coverage problems; explicit nulls make them measurable. |
| **Always return new DataFrame, never mutate input** | Mutating input is surprising. |
| **Validation against Ikeda gold standard from day 1** | Without a target metric, model/prompt tweaks become vibes. |

## Build stages (detail)

### Stage 2: Cellosaurus fast-path
1. `caom.ontologies.cellosaurus`: download Cellosaurus flat file (`cellosaurus.txt` from https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt, ~30 MB), parse into a dict of `{normalized_name: CellosaurusEntry}` where normalization = lowercase + strip non-alphanumerics. Include all synonyms (`SY` fields). Pickle to `.cache/ontologies/cellosaurus/lookup.pkl` with a `metadata.json` sidecar containing version + download timestamp.
2. `caom.ontologies.update.update_ontologies()`: orchestrate download + parse.
3. `caom.api.map_chipatlas()`: for each row, normalize `cell_type`, look up in Cellosaurus. Emit `ontology_source="cellosaurus"` for hits, `status="unmappable"` for misses (temporary — Stage 3 will route misses to EFO retrieval).
4. Optionally filter by species using `assembly` (`hg*` → human, `mm*` → mouse, ...). Cellosaurus entries have `OX` (organism) fields.
5. Tests: unit tests for normalization edge cases, round-trip parse, lookup hits for known aliases (`K-562`/`K562`/`K 562`/`K.562` all → `CVCL_0004`).

### Stage 3: EFO + FAISS retrieval
1. `caom.ontologies.efo`: download `efo.owl` via `pronto` (lighter than `owlready2`, no Java dep), extract labels + synonyms + definitions + immediate parent classes. Write to `.cache/ontologies/efo/terms.parquet`.
2. `caom.retrieval.embedder`: wrap sentence-transformers (default `pritamdeka/S-PubMedBert-MS-MARCO`). Encode queries and corpus with the same model.
3. `caom.retrieval.index`: build FAISS index over corpus embeddings, persist to `.cache/embeddings/efo.faiss` with a row-aligned sidecar parquet. Query API: `(query_str, top_k) -> list[Candidate]`.
4. Extend `map_chipatlas(review=True)` to return top-K EFO candidates with retrieval scores (no LLM yet).

### Stage 4: Ollama LLM re-rank
1. `caom.llm.client`: Ollama client using the `ollama` Python package. Use `format=LLMPick.model_json_schema()` for structured output (prevents ID hallucination).
2. `caom.llm.prompts`: prompt template that includes `cell_type`, aggregated disambiguation context (`cell_type_class`, `description`, `assembly`, `title`), and the top-K candidates (id, label, synonyms, one-line def, parent class). Require an "unmappable" escape hatch.
3. `caom.cache`: SQLite cache keyed by `(model, prompt_hash)` so re-runs with identical input don't re-query the LLM.
4. Wire re-rank into `map_chipatlas()`: retrieval → LLM pick → emit `Mapping` row with confidence + rationale.

### Stage 5: Validation harness
1. `tests/validation/ikeda_gold_standard.py`: download `biosample_cellosaurus_mapping_gold_standard.tsv` from Zenodo DOI 10.5281/zenodo.14881142. The TSV's `extraction answer` column is the free-text cell-line name already (it's what Ikeda's LLM extracted from BioSample metadata); we use it directly as `cell_type` input rather than joining through SRA to ChIP-Atlas, which would need a second large data source for no new signal. Gold Cellosaurus IDs are normalized from `CVCL:NNNN` to `CVCL_NNNN` to match Cellosaurus parser output.
2. `tests/validation/metrics.py`: `AccuracyReport` with `accuracy_at_1`, `pick_precision`, `unmappable_recall`, `coverage`. Abstentions counted separately from wrong-ID picks so a cautious tier reads differently from an inaccurate one.
3. `tests/validation/runner.py`: modes for running just the Cellosaurus fast-path (null embedder + null LLM) vs. full pipeline. Both reuse `caom.api.map_chipatlas` rather than duplicating orchestration.
4. `tests/validation/test_accuracy.py`: gated on `CAOM_RUN_VALIDATION=1` (Stage 2) and `CAOM_RUN_LLM_VALIDATION=1` (full pipeline). Thresholds set below observed baseline to absorb version drift; tighten after each pipeline change.

## Anti-goals (resist these)

- **Don't generalize.** No "biomedical entity linker" refactor. The scope is ChIP-Atlas.
- **Don't mutate the input DataFrame.** Always return a new one.
- **Don't add fuzzy matching to the Cellosaurus fast-path.** Exact match after normalization; anything fuzzy belongs in the retrieval + LLM tier so the LLM can reason about it.
- **Don't bolt on rule-based regex preprocessors for source strings.** This was tried in the `text2term` exploration and the user explicitly pushed back — overfits on a tiny sample, maintenance burden scales badly. Normalization for Cellosaurus (lowercase + strip non-alphanumerics) is fine because it's symmetric (applied to both query and corpus). Surface-form edits that rewrite the query to look more like the target are not.
- **Don't let the LLM invent IDs.** `format=LLMPick.model_json_schema()` is non-negotiable. If you relax it, cache false-positive IDs forever.
- **Don't silently drop unmappable rows.** They must surface as `status="unmappable"` with a reason.
- **Don't download ontologies eagerly on import.** `update_ontologies()` is explicit.

## History / context worth knowing

- The user explored `text2term` first (see `~/code/playground/text2term` and its `examples/`). It handled clean biomedical terms well but failed on ChIP-Atlas-style data: short codes (`293`), punctuation variants (`K-562`/`MCF-7`/`HCT 116`), and semantic gaps (`iPS cells` ≠ `induced pluripotent stem cell` under char-TFIDF). We also hit three real bugs in its syntactic-mapper path; monkey-patching through them didn't inspire confidence.
- The conclusion driving `caom`: **TF-IDF character overlap is the wrong similarity for this task.** Embeddings handle semantic matches; the LLM resolves the ambiguous residual with structured metadata.
- The Ikeda et al. 2025 paper (GigaScience, bsllmner) used an LLM-assisted approach on ~322 ChIP-Atlas BioSamples. Their gold standard TSV is published (Zenodo 10.5281/zenodo.14881142) and is what we validate against. Their code (github.com/sh-ikeda/bsllmner) is worth skimming for prompt ideas.

## Pointers

| What | Where |
| --- | --- |
| ChIP-Atlas metadata (example) | `~/code/matcha/data/metadata/curated_metadata.parquet` |
| `matcha` curation logic | `~/code/matcha/src/matcha/data/curation.py` (shows how `context_id` is derived from `cell_type`) |
| `text2term` exploration | `~/code/playground/text2term/examples/map_cell_types.py`, `map_matcha_contexts.py` |
| Cellosaurus flat file | `https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt` |
| EFO OWL | `http://www.ebi.ac.uk/efo/efo.owl` |
| Ikeda gold standard | Zenodo DOI `10.5281/zenodo.14881142` → `biosample_cellosaurus_mapping_gold_standard.tsv` |
| Ikeda code | `https://github.com/sh-ikeda/bsllmner` |
| SeMRA SSSOM (cell-line cross-references, adjacent) | `https://zenodo.org/records/15164183` |

## Dev workflow

```bash
conda activate caom
pip install -e '.[dev]'      # editable install
pytest                       # run tests
pytest tests/validation      # run accuracy gates (once Stage 5 lands)
ruff check src tests         # lint
mypy src                     # type-check
```

Never use the conda `base` env (see `CLAUDE.md`). Transient SSL errors on `pip` / `conda` calls are expected on this machine — retry the same command.

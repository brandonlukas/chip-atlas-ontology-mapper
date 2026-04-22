# Instructions for future sessions

Read this first when picking up work on this repo. It's the "what are we building and why" brief; `CLAUDE.md` is narrowly about environment policy.

## Mission

`caom` maps ChIP-Atlas `cell_type` annotations (messy free-text: `K-562`, `MCF-7`, `CD4+ T cells`, `Hep G2`, `iPSC derived neural cells`, ...) to standardized biomedical ontology IDs (Cellosaurus + EFO). It is **deliberately ChIP-Atlas-specific** — resist generalizing it. Its advantage over general-purpose tools (e.g. `text2term`) is that it exploits ChIP-Atlas's structured metadata (`cell_type_class`, `cell_type_description`, `assembly`, `title`) as disambiguation signal.

Downstream consumer: the `matcha` project (`~/code/matcha`). The parquet at `~/code/matcha/data/metadata/curated_metadata.parquet` is the canonical input shape — though note `matcha` renames `cell_type` to `context_id` internally; `caom` keys on the ChIP-Atlas-native name.

## Current status

- **Stage 1: skeleton + pyproject + env** — DONE (commit pending)
  - conda env `caom` exists with Python 3.11 and all deps installed editable
  - Public API stubs (`map_chipatlas`, `update_ontologies`) import and raise `NotImplementedError`
  - `pytest tests/test_skeleton.py` passes (4 tests)
- **Stage 2: Cellosaurus fast-path** — NEXT
- **Stage 3: EFO download + FAISS retrieval** — pending
- **Stage 4: Ollama LLM re-rank** — pending
- **Stage 5: Ikeda validation harness + accuracy gates in CI** — pending

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
1. `tests/validation/ikeda_gold_standard.py`: download `biosample_cellosaurus_mapping_gold_standard.tsv` from Zenodo DOI 10.5281/zenodo.14881142. Join to ChIP-Atlas metadata to extract the `cell_type` string for each BioSample ID (this is the actual input shape for `caom`).
2. `tests/validation/test_accuracy.py`: run `caom.map_chipatlas()` on the joined DataFrame, assert accuracy@1 ≥ threshold. Gate should be tightened as each stage lands (Stage 2 alone should hit a high number on cell-line rows).

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

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
- **Stage 6: EFO ontology-ID normalization** — DONE
  - `caom.ontologies.efo.normalize_ontology_id`: URI → CURIE rewriter applied in `parse_efo` to both `ontology_id` and each `parents` entry. Per-host rules (EFO / Orphanet / BAO / dbpedia) handle the four URI families pronto emits for terms outside its built-in OBO idspace map; unknown URIs pass through unchanged so they surface rather than silently corrupt.
  - Cache rebuilt against the already-downloaded OWL (no re-download): 0 URI-form `ontology_id`s remain in `terms.parquet` and `efo_terms.parquet` (from 15,599), 0 URI-form parent refs remain (from 16,948), EFO-prefixed rows 1 → 13,452.
  - 100 tests passing (13 new: parametrized unit tests for the 4 URI families + already-CURIE passthrough + unknown-URI fallthrough + an end-to-end `parse_efo` round-trip over an OBO fixture with URI-form ids and an `is_a` URI parent).
- **Stage 7: retrieval quality fix (allow-list)** — DONE
  - `caom.retrieval.index.filter_corpus`: CURIE-prefix allow-list applied before embedding. Kept prefixes: `CL`, `UBERON`, `EFO`, `CLO`, `BTO`, `MONDO`, `Orphanet`, `NCIT`, and organism anatomy (`FBbt`, `FBdv`, `ZFA`, `MA`, `FMA`, `PO`, `WBls`). Allow-list (not deny-list) so a future EFO refresh that imports a new ontology cannot silently re-introduce contamination. Rationale documented inline at the constant.
  - Corpus narrowed 83,418 → 34,079 rows (≈59% reduction): 17,542 `PR` + 17,441 `OBA` + 6,413 `HGNC` + 2,481 `HP` + 2,072 `CHEBI` + 1,507 `NCBITaxon` + 703 `GO` + 110 `dbpedia` + smaller utility ontologies excluded. FAISS file 256 MB → 105 MB, `efo_terms.parquet` 8.5 MB → 4.5 MB.
  - Post-rebuild smoke test on the 5 Stage-7 failure-mode queries (`top_k=10`): `Pancreatic islets` → **UBERON:0000006 rank 1** (was off top-10); `iPS cells` → **EFO:0004905 rank 1** (was buried beneath 10× iPS-Nn subtypes); `CD4+ T cells` → CL:0000492 (helper) rank 1 + CL:0000624 rank 4 (both defensible, was entirely outside top-10); `Lung` → UBERON:0002048 rank 2; `Brain` → UBERON:0000955 rank 3. Failure mode A is fully resolved. Mode B (subtype dominance) is now tight enough that the LLM has the canonical parent inside top-K and can adjudicate from disambiguation context — query augmentation deferred pending Stage 8's full-pipeline number.
  - 130 tests passing (30 new: parametrized allow-list / deny-list coverage + order-preservation + empty-prefix handling). Cellosaurus validation gate unchanged.
- **Stage 8: full-pipeline threshold tuning** — DONE
  - Established full-pipeline baseline against Ikeda 600-row gold standard (Cellosaurus v49, EFO v3.89, qwen2.5:7b-instruct): `acc@1=0.940, pick_precision=0.969, unmap_recall=0.970, coverage=0.497`. Already beats Stage 2's `acc@1=0.903`.
  - Failure analysis (27 misses) split evenly across 9 wrong-id, 9 false-positive on unmappable, 9 abstained on mappable. Two principled prompt rules added to `caom.llm.prompts`: (a) **Cellosaurus-prefer** when CVCL_ and EFO candidates clearly describe the same cell line — kills the BLaER / P493 wrong-id failures where EFO mirrors a Cellosaurus entry and a higher EFO retrieval_score was overriding source-canonicality; (b) **caution on ≤3-char queries** with no disambiguating metadata — kills the `ED` → MONDO disease overreach.
  - Post-prompt full pipeline (same fixtures): `acc@1=0.933, pick_precision=0.989, unmap_recall=0.977, coverage=0.480`. Trade-off: −2 correct picks (JK1, C4-2 — both 3-char with dash variants in Cellosaurus) in exchange for −6 wrong-ids (BLaER ×2, HAP1, H1 ×3 → cautious abstentions). Net reduction in wrong-id rate from 9 → 3 (≈65% drop). Chosen operating point favors precision: a wrong CVCL_X is silent contamination downstream; an abstention is reviewable.
  - Top-K (default 20) and abstention threshold left at defaults — neither was the bottleneck on the residual failures (which are mostly H9/H1 stem-cell ambiguity solvable only with extra context, gold-standard noise on cell-line-named-but-not-used rows, and naming variants like `WI-38hTERT/GFP-RAF1-ER` → `WI-38/hTERT/GFP-RAF-ER` that need fuzzy matching deliberately kept out per CLAUDE.md).
  - `tests/validation/test_accuracy.py` full-pipeline gate floors raised: `acc@1 >= 0.91`, `pick_precision >= 0.97` (new), `unmap_recall >= 0.95`, all a few points below observed to absorb model / ontology version drift.
  - `tests/validation/run_full_pipeline.py` ad-hoc driver added — dumps `full_pipeline_predictions.parquet` + `full_pipeline_failures.tsv` so future failure analysis can iterate without re-running the LLM.
  - 131 tests passing (1 new prompt-shape lock-in test); both validation gates skip by default.
- **Stage 9: exact-match retrieval layer + allow-list narrowing** — DONE
  - **Exact-match layer in `caom/retrieval/index.py`.** New `build_exact_index(terms_df)` → `dict[str, list[int]]` keyed by `normalize_name(label_or_synonym)` (shared with the Cellosaurus fast-path). Built inside `build_index`, persisted at `.cache/embeddings/efo.exact.pkl`, loaded in `load_index` (falls back to rebuilding from the parquet if the pkl is missing, for legacy caches). `EFOIndex.exact_lookup(query)` returns `Candidate`s with `retrieval_score=1.0` and a new `exact=True` field (added to `caom.types.Candidate`). `api._retrieve_efo` prepends exact hits to the FAISS top-K and de-dupes by `ontology_id` so the LLM never sees the same id twice.
  - **Allow-list narrowed** in `caom/retrieval/index.py`. Dropped `EFO`, `CLO`, `BTO`, `NCIT`, `Orphanet`; kept `CL`, `UBERON`, `MONDO` + organism anatomy (`FBbt`, `FBdv`, `ZFA`, `MA`, `FMA`, `PO`, `WBls`). Inline rationale at `_ALLOWED_PREFIXES` documents why (cell-line mirrors duplicate Cellosaurus; NCIT / Orphanet overlap MONDO and overrank subtypes). Corpus narrowed 34,079 → 18,091 rows (≈47% reduction, 15,988 rows dropped). Filtered terms parquet 4.5 MB → 3.4 MB; FAISS file 105 MB → 56 MB; exact-pkl adds 3.3 MB over 90,160 normalized keys.
  - **Prompt rule** appended to `_SYSTEM_INSTRUCTIONS` in `caom/llm/prompts.py`: *"Candidates marked `[exact]` are label or exact-synonym matches for the query (after lowercase + non-alphanumeric stripping). Prefer them over cosine-ranked candidates unless the disambiguation context clearly contradicts (wrong organism, wrong disease context, wrong cell-line series)."* `_format_candidate` renders `[N] [exact]` for `Candidate.exact=True`. Stage 8's Cellosaurus-prefer and short-abbreviation rules kept intact — the exact layer complements them.
  - **Matcha smoke pytest gate** at `tests/validation/test_matcha_smoke.py`, gated on `CAOM_RUN_LLM_VALIDATION=1` + matcha parquet presence. Asserts a hardcoded `{cell_type → expected_id}` dict over 14 rows (the existing 13 + `iPSC derived neural cells`). Stage 9 wins locked in: `Brain → UBERON:0000955`, `Lung → UBERON:0002048`, `Acute myeloid leukemia → MONDO:0018874` — all previously picked subtypes, now canonical parents via exact-match hits. `PBMC → CL:0000842 mononuclear leukocyte` (the `peripheral blood mononuclear cell` synonym) and `iPS cells → CL:0002248 pluripotent stem cell` also locked in — corpus-best parents, since `CL:2000001` / `CL:0011020` / `CL:0008007` don't exist in EFO v3.89's download. Two residuals locked in too, marked as known-imperfect so regressions surface: `Pancreatic islets → UBERON:0000016 endocrine pancreas` (plural mismatch against synonym `pancreatic islet`; canonical is `UBERON:0000006`) and `CD4+ T cells → CL:0000896 activated subtype` (plural mismatch kills exact hit; canonical is `CL:0000624`). The substring-preference fallback (Stage 9 step 4, carried into Stage 10) is the planned fix.
  - **Ikeda full-pipeline gate** (Cellosaurus v49, EFO v3.89, `qwen2.5:14b`, 2026-04-23): `n=600 acc@1=0.950 (correct=283/298, wrong_id=3, abstained=12) pick_precision=0.990 unmap_recall=0.964 (291/302) coverage=0.495`. Vs Stage 8's 7b baseline: `acc@1` **+1.7pp** (0.933 → 0.950), `pick_precision` +0.1pp (0.989 → 0.990), `unmap_recall` **−1.3pp** (0.977 → 0.964), `coverage` +1.5pp (0.480 → 0.495). The unmap_recall dip (2 more false-positive ids on should-abstain rows) is plausibly the 14b model being less cautious than 7b rather than a Stage-9-intrinsic regression — Stage 10's 7b baseline reset will confirm. All three gate floors pass with margin: `acc@1 >= 0.91` ✓, `pick_precision >= 0.97` ✓, `unmap_recall >= 0.95` ✓.
  - **Reproduce.** `CAOM_RUN_LLM_VALIDATION=1 pytest tests/validation/test_matcha_smoke.py` (14 rows, <10s). `CAOM_RUN_VALIDATION=1 CAOM_RUN_LLM_VALIDATION=1 pytest tests/validation/test_accuracy.py` (Ikeda 600 rows; ~minutes with a cold LLM cache).
  - **Note on LLM model used for Stage 9 validation.** Stage 8 was calibrated on `qwen2.5:7b-instruct`. Stage 9's measurements were taken with `qwen2.5:14b` because 7b was not locally installed at the time. The gate floors absorb the variance, but returning to 7b for the next calibration is advisable — the published baseline in `tests/validation/test_accuracy.py` still references 7b.
  - **145 tests collected** (142 offline passing + 3 validation gates that skip by default: Ikeda Stage 2, Ikeda full-pipeline, matcha smoke). 12 new tests vs Stage 8: exact-index build + lookup (7 — keys, lookup-unit-score, punct-symmetry, collisions, empty, save/load, legacy-rebuild), prompt marker rendering + rule lock-in (2), end-to-end exact-prepend + dedup (2), matcha smoke (1). `filter_corpus` parametrized tests were rebalanced — 5 allowed params dropped, 5 new excluded params added — but test count on that file unchanged.
- **Stage 10: substring marker + LLM-model-baseline reset** — DONE
  - **What shipped.**
    1. **`[substring]` marker** in `caom/llm/prompts.py`, deterministic parallel to Stage 9's `[exact]`. For each candidate, if `normalize_name(query)` (lowercase + strip non-alphanumerics, shared with the Cellosaurus fast-path and the exact layer) is a substring of any candidate label/synonym — or vice-versa — the candidate is rendered as `[N] [substring] id=...` in the prompt. A system-prompt rule tells the LLM to prefer `[substring]` candidates over unmarked cosine candidates unless Cellosaurus-prefer or disambiguation context overrides; `[exact]` is strictly stronger. The marker is computed in `_format_candidate(index, c, query_key)` so nothing persists on the `Candidate` model and the decision stays a presentation-layer concern.
    2. **Baseline reset to `qwen2.5:7b-instruct`.** Model pulled locally; Stage 9 measurements had used 14b only because 7b was absent. The matcha smoke EXPECTED dict is re-anchored to the 7b + `[substring]` pick distribution, tagged by category (`FAST_PATH` / `EXACT_HIT` / `SUBSTRING_HIT` / `LLM_PICK`). Gate floors unchanged.
  - **Matcha smoke (Cellosaurus v49, EFO v3.89, `qwen2.5:7b-instruct`, 2026-04-23).** Stage 10 WIN: `Pancreatic islets → UBERON:0000006 islet of Langerhans` (previously UBERON:0000016 endocrine pancreas; fires because the synonym `pancreatic islet` is a normalized substring of the query `pancreatic islets`). Two residuals re-locked as known-imperfect:
    - `CD4+ T cells → CL:0000896` (activated subtype). `+` vs `-positive` mismatch means `cd4tcells` does not substring-match `cd4positivealphabetatcell` either direction; the matcha title field says `CD4 activation ATAC-seq`, pushing 7b to the activated subtype. A punctuation-normalization preprocessor would fix this but veers into the asymmetric-query-rewriting anti-goal.
    - `iPSC derived neural cells → CL:0002351 progenitor cell of endocrine pancreas` (7b-specific noise; 14b picked `CL:0002248 pluripotent stem cell`, rank-1 cosine is `CL:0000047 neural stem cell` — all three are corpus-available because the canonical `CL:0011020 neural progenitor` is not in EFO v3.89). Stage 11 corpus refresh is the durable fix.
  - **Ikeda full-pipeline gate** (Cellosaurus v49, EFO v3.89, `qwen2.5:7b-instruct`, 2026-04-23): `n=600 acc@1=0.940 (correct=280/298, wrong_id=5, abstained=13) pick_precision=0.982 unmap_recall=0.983 (297/302) coverage=0.483`. Vs Stage 8 (same 7b model, Stage 7 allow-list, no exact, no substring): `acc@1` **+0.7pp** (0.933 → 0.940), `pick_precision` −0.7pp (0.989 → 0.982, two extra false-positive-ids among the 302 unmappable rows), `unmap_recall` **+0.6pp** (0.977 → 0.983), `coverage` +0.3pp (0.480 → 0.483). Vs the 14b Stage 9 score: `acc@1` −1.0pp (0.950 → 0.940) — the 14b gain was largely model-strength-driven, not Stage 9 feature-driven, so it does not carry to 7b. All three gate floors still pass with margin.
  - **Residual failure profile (23 rows).** 13 abstain-on-mappable (mostly short cell-line names `P493`, `HAP1`, `BLaER`×2, `RD cells`, `JHRCC12`×2, `HPNE`, `SUDH4`, `OS-LM1`, `22PC`, `WI-38hTERT/GFP-RAF1-ER`, one empty cell_type — the ≤3-char caution rule from Stage 8 is doing what it was tuned for), 5 wrong-id-on-mappable (H1×3 picked `CVCL_N541 H-1` over `CVCL_9771`, H9×2 picked `CVCL_1240` over `CVCL_9773` — a stem-cell vs unrelated-cell-line collision that a Cellosaurus-side organism filter cannot catch because both are human), 5 false-positive-id-on-unmappable (`SCC25`, `SCC-25`, `CAL1`, `FS4`, `GBM8` — short cell-line names where the LLM commits where it should abstain). None are substring-rule regressions.
  - **Reproduce.** `CAOM_LLM_MODEL=qwen2.5:7b-instruct CAOM_RUN_LLM_VALIDATION=1 pytest tests/validation/test_matcha_smoke.py` (14 rows, ~10s cached). `CAOM_LLM_MODEL=qwen2.5:7b-instruct CAOM_RUN_VALIDATION=1 CAOM_RUN_LLM_VALIDATION=1 pytest tests/validation/test_accuracy.py` (Ikeda 600 rows; cold cache ~5min, warm ~5s).
  - **147 tests collected** (144 offline passing + 3 validation gates that skip by default). Two new prompt-shape lock-ins on the substring marker: `test_prompt_carries_stage10_substring_rule` asserts the rule text is present and bidirectional, `test_prompt_renders_substring_marker_symmetrically` exercises the `_format_candidate` output including the `[exact]` > `[substring]` precedence.

- **Stage 11: corpus refresh + stem-cell disambiguation** — NEXT
  - **Goal.** Close out the three known-imperfect matcha picks (`iPSC derived neural cells`, `PBMC`, `iPS cells`) by getting the canonical CL terms into the corpus, and reduce the H1 / H9 stem-cell wrong-id rate that dominates Ikeda's residual failure set.
  - **Approach — in expected value-to-risk order.**
    1. **Corpus refresh.** Retry `caom.update_ontologies(force=True)` against the latest EFO. If `CL:2000001` (PBMC), `CL:0011020` (neural progenitor), `CL:0008007` (induced pluripotent stem cell) land in the refresh, the matcha `LLM_PICK` → canonical tightening is automatic and gate floors can move up. If the refresh doesn't include them, punt on a direct-CL-pull step — it veers into the "generalizing outside ChIP-Atlas" anti-goal and the coverage delta is small (3 matcha rows).
    2. **Stem-cell cell-line prompt hint.** For queries like `H1`, `H9`, `H-1`, when Cellosaurus returns multiple candidates across unrelated cell-line series (stem-cell `CVCL_9771 H1` vs the unrelated `CVCL_N541 H-1`), the current ≤3-char caution rule picks wrong rather than abstaining. A small prompt rule — *"For ambiguous short-cell-line queries where candidates disagree on cell-line series (stem cell vs. cancer vs. primary), prefer abstention unless the title or description pins down the series"* — would shift these 5 wrong-ids to 5 abstentions (net pick_precision gain, small acc@1 cost). Risk: could increase abstention on legitimate short-cell-line queries (`CAL1`, `FS4`) where abstention is already the current behavior, so near-zero downside.
    3. **BM25 / FTS second layer.** Still the biggest-impact retrieval change — SQLite FTS5 persisted alongside FAISS, queried as tier 2 between exact (tier 1) and vector (tier 3), marker `[bm25]` parallel to `[exact]` / `[substring]`. Only attempt if (1) and (2) don't move the Ikeda / matcha numbers enough to justify the Stage 11 closeout; the exact + substring layers already catch the low-hanging lexical signal.
  - **Deliberately out of scope (deferred to Stage 12+).** Embedder upgrade (`pritamdeka/S-PubMedBert-MS-MARCO` → E5 / nemotron family with asymmetric query/passage prefixes — own FAISS rebuild), vector-store swap (FAISS → LanceDB — architecture, not quality), ontology-scope expansion beyond `CL + UBERON + MONDO` + organism-anatomy (let real-data coverage gaps drive it).
  - **Anti-goals (carry over + reinforced).** No fuzzy matching in the Cellosaurus fast-path. No asymmetric query rewriting — the Stage 10 substring rule matches symmetrically via `in` on `normalize_name`'d strings and that contract must not be broken (in particular, no stripping `+` → `-positive` in the query to force a CD4+ match). No relaxing the LLM-id allow-list (the `Candidate`-set check in `_pick_to_mapping` is what keeps hallucinated ids out of the output). No silently dropping unmappable rows. No generalizing `caom` outside ChIP-Atlas scope — in particular, no introducing a direct-CL pull step unless the EFO import is demonstrably lagging on terms we need. No abandoning the Cellosaurus separate-source split.

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

# Environment policy

**Never install Python packages into the conda `base` environment.** Use the dedicated env for this repo:

```bash
conda activate caom
```

If the env does not exist yet, create it before installing anything:

```bash
conda create -n caom python=3.11 -y
conda activate caom
pip install -e '.[dev]'
```

For non-interactive shells, prefer the explicit interpreter path so activation state is irrelevant:

```bash
/home/brandon/miniconda3/envs/caom/bin/python -c "import caom; ..."
```

## Transient SSL errors with conda / pip

On this machine `conda create`, `conda install`, and `pip install` occasionally fail mid-download with errors like:

```
CondaSSLError: Encountered an SSL error. Most likely a certificate verification issue.
Exception: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] ...
```

These are transient. **Just retry the same command** — it typically succeeds within a few attempts. Do not disable SSL verification, switch channels, or swap package managers as a workaround.

# Architectural notes

- `caom` is deliberately ChIP-Atlas-specific. Resist refactors that generalize it into a "biomedical entity mapper" — the value of this tool is exploiting ChIP-Atlas metadata structure.
- Canonical input is a `pd.DataFrame` with a `cell_type` column. `cell_type_class`, `cell_type_description`, `assembly`, `title`, `antigen`, `tf_name` are used if present and silently skipped if missing.
- Per-row mapping granularity (title/description are row-specific). Users dedup upstream if they want per-`cell_type` behavior.
- Output is always a new DataFrame — never mutate the input.
- Ontology scope is Cellosaurus + EFO, lumped. Revisit only if coverage proves insufficient.
- LLM default is Ollama `qwen2.5:7b-instruct`; model is swappable via config / `CAOM_LLM_MODEL`.
- Cache is project-local (`./.cache/`) by default. Override with `cache_dir=` or `CAOM_CACHE_DIR`.
- Ontology updates are explicit (`caom.update_ontologies()`) and version-stamped on every output row for reproducibility.
- "Unmappable" is an explicit status with `ontology_id=None`, not a dropped row.

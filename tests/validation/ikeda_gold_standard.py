"""Loader for the Ikeda et al. 2025 BioSample -> Cellosaurus gold standard.

Source: Zenodo DOI 10.5281/zenodo.14881142
File: biosample_cellosaurus_mapping_gold_standard.tsv (~322 rows)

Stages:
- Stage 2+: used to score accuracy@1 of the Cellosaurus fast-path, joined to
  ChIP-Atlas metadata so BioSample IDs map to cell_type strings.
"""

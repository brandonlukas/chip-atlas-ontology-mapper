"""EFO download + term extraction (labels, synonyms, defs, parents). Stage 3."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from caom.cache import ontology_cache_dir, write_metadata_sidecar

_EFO_RELEASE_RE = re.compile(r"/releases/(v[^/]+)/")

EFO_URL = "http://www.ebi.ac.uk/efo/efo.owl"

# The ebi.ac.uk → GitHub release asset → Azure Blob redirect chain is flaky
# on some networks: TLS records corrupt mid-stream. These constants tune
# `download_efo`'s retry-with-Range-resume loop.
_DOWNLOAD_MAX_ATTEMPTS = 60
_DOWNLOAD_BACKOFF_BASE = 1.5
_DOWNLOAD_BACKOFF_CAP = 60.0
_STAGNATION_LIMIT = 8


@dataclass
class EFOTerms:
    """In-memory EFO term table."""

    terms: pd.DataFrame  # columns: ontology_id, label, synonyms, definition, parents
    version: str
    downloaded_at: str


def download_efo(dest: Path, *, url: str = EFO_URL) -> None:
    """Stream the EFO OWL file to `dest`, resuming across SSL / connection flakes.

    Mid-stream errors leave a `.part` file in place; subsequent attempts issue
    a Range request from its current offset. Consecutive attempts that make
    no forward progress count against `_STAGNATION_LIMIT` so a real outage
    isn't masked.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_exc: Exception | None = None
    stagnation = 0
    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        written_this_try = 0
        try:
            with requests.get(url, stream=True, timeout=600, headers=headers) as r:
                if offset > 0 and r.status_code == 200:
                    # A redirect hop rewrote the Range header; start over.
                    tmp.unlink(missing_ok=True)
                    offset = 0
                r.raise_for_status()
                mode = "ab" if offset > 0 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        if chunk:
                            f.write(chunk)
                            written_this_try += len(chunk)
            tmp.replace(dest)
            return
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.SSLError,
                requests.exceptions.ReadTimeout) as exc:
            last_exc = exc
            if written_this_try > 0:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= _STAGNATION_LIMIT:
                    break
            if attempt >= _DOWNLOAD_MAX_ATTEMPTS:
                break
            sleep_for = min(_DOWNLOAD_BACKOFF_BASE ** attempt, _DOWNLOAD_BACKOFF_CAP)
            sleep_for *= 1 + random.uniform(-0.25, 0.25)
            time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


def parse_efo(path: Path) -> tuple[pd.DataFrame, str]:
    """Parse an EFO OWL/OBO file via pronto.

    Returns
    -------
    tuple[pd.DataFrame, str]
        (terms_df, version). Columns: ontology_id, label, synonyms, definition, parents.
        Obsolete and un-named terms are skipped.
    """
    import pronto

    onto = pronto.Ontology(str(path))
    raw_version = str(
        onto.metadata.data_version or onto.metadata.format_version or ""
    ).strip()
    # EFO's data_version is the release URL; extract "v3.89.0" so the version
    # stamped on each output row stays compact.
    m = _EFO_RELEASE_RE.search(raw_version)
    version = m.group(1) if m else raw_version

    rows: list[dict] = []
    for term in onto.terms():
        if term.obsolete or not term.name:
            continue
        synonyms = sorted({s.description for s in term.synonyms if s.description})
        parents = sorted(
            {p.id for p in term.superclasses(distance=1, with_self=False) if p.id}
        )
        definition = str(term.definition) if term.definition else None
        rows.append(
            {
                "ontology_id": term.id,
                "label": term.name,
                "synonyms": synonyms,
                "definition": definition,
                "parents": parents,
            }
        )
    df = pd.DataFrame(
        rows, columns=["ontology_id", "label", "synonyms", "definition", "parents"]
    )
    return df, version


def _cache_dir(cache_root: Path) -> Path:
    return ontology_cache_dir(cache_root, "efo")


def save_terms(cache_root: Path, terms: EFOTerms) -> None:
    d = _cache_dir(cache_root)
    d.mkdir(parents=True, exist_ok=True)
    terms.terms.to_parquet(d / "terms.parquet", index=False)
    write_metadata_sidecar(
        d,
        {
            "source": "efo",
            "version": terms.version,
            "downloaded_at": terms.downloaded_at,
            "term_count": len(terms.terms),
        },
    )


def load_terms(cache_root: Path) -> EFOTerms:
    d = _cache_dir(cache_root)
    parq = d / "terms.parquet"
    if not parq.exists():
        raise FileNotFoundError(
            f"EFO cache not found at {parq}. Run `caom.update_ontologies()` first."
        )
    meta_path = d / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return EFOTerms(
        terms=pd.read_parquet(parq),
        version=meta.get("version", ""),
        downloaded_at=meta.get("downloaded_at", ""),
    )


def is_cached(cache_root: Path) -> bool:
    d = _cache_dir(cache_root)
    return (d / "terms.parquet").exists() and (d / "metadata.json").exists()


def refresh_cache(
    cache_root: Path,
    *,
    force: bool = False,
    url: str = EFO_URL,
) -> EFOTerms:
    """Download + parse + cache EFO. Returns the loaded terms.

    If the cache already exists and `force` is False, loads from cache without
    re-downloading.
    """
    if not force and is_cached(cache_root):
        return load_terms(cache_root)

    d = _cache_dir(cache_root)
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "efo.owl"
    download_efo(raw, url=url)
    df, version = parse_efo(raw)
    terms = EFOTerms(
        terms=df,
        version=version,
        downloaded_at=datetime.now(UTC).isoformat(),
    )
    save_terms(cache_root, terms)
    return terms

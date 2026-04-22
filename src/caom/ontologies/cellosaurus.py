"""Cellosaurus download, parse, and normalized-name lookup (Stage 2)."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import requests

from caom.cache import KeyedCache, ontology_cache_dir, write_metadata_sidecar

CELLOSAURUS_URL = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt"

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    """Lowercase and strip non-alphanumerics. Symmetric on query and corpus."""
    if not name:
        return ""
    return _NON_ALNUM_RE.sub("", name.lower())


@dataclass
class CellosaurusEntry:
    accession: str
    primary_name: str
    synonyms: list[str] = field(default_factory=list)
    taxon_ids: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    category: str | None = None


@dataclass
class CellosaurusLookup:
    """Normalized-name → accession(s) index, plus accession → entry map."""

    entries: dict[str, CellosaurusEntry]
    name_index: dict[str, list[str]]
    version: str
    downloaded_at: str

    def lookup(
        self, cell_type: str, *, taxon_id: str | None = None
    ) -> list[CellosaurusEntry]:
        key = normalize_name(cell_type)
        if not key:
            return []
        accs = self.name_index.get(key, [])
        matches = [self.entries[a] for a in accs]
        if taxon_id is not None:
            matches = [m for m in matches if taxon_id in m.taxon_ids]
        return matches


def download_cellosaurus(dest: Path, *, url: str = CELLOSAURUS_URL) -> None:
    """Stream the Cellosaurus flat file to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def _parse_stream(stream: IO[str]) -> tuple[dict[str, CellosaurusEntry], str]:
    entries: dict[str, CellosaurusEntry] = {}
    version = ""
    current: dict | None = None

    for raw in stream:
        line = raw.rstrip("\n\r")

        if current is None and not version and "Version:" in line:
            version = line.split("Version:", 1)[1].strip()

        if line == "//":
            if current and "accession" in current:
                entries[current["accession"]] = CellosaurusEntry(
                    accession=current["accession"],
                    primary_name=current.get("id", current["accession"]),
                    synonyms=current.get("synonyms", []),
                    taxon_ids=current.get("taxon_ids", []),
                    species=current.get("species", []),
                    category=current.get("category"),
                )
            current = None
            continue

        if len(line) < 5 or line[2:5] != "   ":
            continue

        code = line[:2]
        value = line[5:].strip()

        if code == "ID":
            current = {
                "id": value,
                "synonyms": [],
                "taxon_ids": [],
                "species": [],
            }
            continue

        if current is None:
            continue

        if code == "AC":
            current["accession"] = value
        elif code == "SY":
            current["synonyms"].extend(
                s.strip() for s in value.split(";") if s.strip()
            )
        elif code == "OX":
            m = re.search(r"NCBI_TaxID=(\d+)", value)
            if m:
                current["taxon_ids"].append(m.group(1))
            if "!" in value:
                current["species"].append(value.split("!", 1)[1].strip())
        elif code == "CA":
            current["category"] = value

    return entries, version


def parse_cellosaurus(path: Path) -> tuple[dict[str, CellosaurusEntry], str]:
    """Parse a Cellosaurus flat file. Returns (entries, version_string)."""
    with open(path, encoding="utf-8") as f:
        return _parse_stream(f)


def build_lookup(
    entries: dict[str, CellosaurusEntry],
    version: str,
    downloaded_at: str,
) -> CellosaurusLookup:
    name_index: dict[str, list[str]] = {}
    for acc, entry in entries.items():
        keys: set[str] = set()
        for name in (entry.primary_name, *entry.synonyms):
            k = normalize_name(name)
            if k:
                keys.add(k)
        for k in keys:
            name_index.setdefault(k, []).append(acc)
    return CellosaurusLookup(
        entries=entries,
        name_index=name_index,
        version=version,
        downloaded_at=downloaded_at,
    )


def _cache_dir(cache_root: Path) -> Path:
    return ontology_cache_dir(cache_root, "cellosaurus")


def save_lookup(cache_root: Path, lookup: CellosaurusLookup) -> None:
    d = _cache_dir(cache_root)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "lookup.pkl", "wb") as f:
        pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_metadata_sidecar(
        d,
        {
            "source": "cellosaurus",
            "version": lookup.version,
            "downloaded_at": lookup.downloaded_at,
            "entry_count": len(lookup.entries),
            "name_key_count": len(lookup.name_index),
        },
    )


def load_lookup(cache_root: Path) -> CellosaurusLookup:
    pkl = _cache_dir(cache_root) / "lookup.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"Cellosaurus cache not found at {pkl}. "
            "Run `caom.update_ontologies()` first."
        )
    with open(pkl, "rb") as f:
        return pickle.load(f)


_LOOKUP_CACHE: KeyedCache[Path, CellosaurusLookup] = KeyedCache()


def get_cached_lookup(cache_root: Path) -> CellosaurusLookup:
    """Load the Cellosaurus lookup, reusing an in-memory copy across calls.

    The unpickled lookup is ~27 MB; callers typically invoke `map_chipatlas`
    many times per process. `refresh_cache` clears this when it re-downloads.
    """
    return _LOOKUP_CACHE.get_or_load(cache_root, lambda: load_lookup(cache_root))


def is_cached(cache_root: Path) -> bool:
    d = _cache_dir(cache_root)
    return (d / "lookup.pkl").exists() and (d / "metadata.json").exists()


def refresh_cache(
    cache_root: Path,
    *,
    force: bool = False,
    url: str = CELLOSAURUS_URL,
) -> CellosaurusLookup:
    """Download + parse + cache Cellosaurus. Returns the loaded lookup.

    If the cache already exists and `force` is False, loads from cache without
    re-downloading.
    """
    if not force and is_cached(cache_root):
        return get_cached_lookup(cache_root)

    d = _cache_dir(cache_root)
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "cellosaurus.txt"
    download_cellosaurus(raw, url=url)
    entries, version = parse_cellosaurus(raw)
    downloaded_at = datetime.now(UTC).isoformat()
    lookup = build_lookup(entries, version, downloaded_at)
    save_lookup(cache_root, lookup)
    _LOOKUP_CACHE.invalidate(cache_root)
    return lookup

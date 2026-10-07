# torchcell/artifacts/tiers.py
# [[torchcell.artifacts.tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/tiers.py
# Test file: tests/torchcell/artifacts/test_artifact_tiers.py

"""Where the four artifact tiers live under ``DATA_ROOT`` and how their manifests read.

- ``raw``: ``$DATA_ROOT/torchcell-raw/<key>/manifest.json`` (literature ``Manifest``).
- ``library``: ``$DATA_ROOT/torchcell-library/<key>/manifest.json`` (``Manifest``).
- ``objects``: ``$DATA_ROOT/torchcell-objects/<key>/manifest.json`` (``Manifest``).
- ``genomes``: the genomes registry, ``$DATA_ROOT/torchcell-genomes/<assembly_set>/``,
  whose ``GenomeManifest`` reuses the same per-file ``ArtifactRecord``.

A literature-shape manifest must name its own key (``citation_key == key``); the genomes
registry applies the same rule to ``assembly_set``. Either disagreement raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from torchcell.artifacts.ref import ArtifactRef, Tier
from torchcell.literature.manifest import MANIFEST_FILENAME, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import (
    GenomeManifest,
    assembly_set_dir,
    load_genome_manifest,
)

RAW_DIR = "torchcell-raw"
LIBRARY_DIR = "torchcell-library"
OBJECTS_DIR = "torchcell-objects"
#: The literature-shape tiers and their directory under ``DATA_ROOT``.
MANIFEST_TIER_DIRS: dict[str, str] = {
    "raw": RAW_DIR,
    "library": LIBRARY_DIR,
    "objects": OBJECTS_DIR,
}

#: The tiers whose manifest is the literature ``Manifest``.
ManifestTier = Literal["raw", "library", "objects"]

#: A tier manifest: the literature ``Manifest`` or, for ``genomes``, ``GenomeManifest``.
TierManifest = Manifest | GenomeManifest


class TierManifestKeyError(RuntimeError):
    """A tier manifest names a different key than the directory it sits in."""


def resolve_data_root(data_root: str | Path | None) -> Path:
    """``data_root`` as a ``Path``; when None, ``DATA_ROOT`` after ``load_dotenv()``.

    An unset ``DATA_ROOT`` raises ``KeyError('DATA_ROOT')``; there is no default root.
    """
    if data_root is None:
        from dotenv import load_dotenv

        load_dotenv()
        return Path(os.environ["DATA_ROOT"])
    return Path(data_root)


def key_dir(tier: Tier, key: str, data_root: str | Path | None = None) -> Path:
    """The directory one key occupies in one tier (not checked for existence)."""
    root = resolve_data_root(data_root)
    if tier == "genomes":
        return Path(assembly_set_dir(key, str(root)))
    return root / MANIFEST_TIER_DIRS[tier] / key


def manifest_path(tier: Tier, key: str, data_root: str | Path | None = None) -> Path:
    """``<key_dir>/manifest.json`` (not checked for existence)."""
    return key_dir(tier, key, data_root) / MANIFEST_FILENAME


def load_tier_manifest(
    tier: Tier, key: str, data_root: str | Path | None = None
) -> TierManifest:
    """Validate and return one key's manifest; an absent manifest raises.

    Raises ``FileNotFoundError`` when the manifest is absent and
    ``TierManifestKeyError`` (``GenomeIntegrityError`` for genomes) when it names
    another key.
    """
    if tier == "genomes":
        return load_genome_manifest(key, str(resolve_data_root(data_root)))
    return load_manifest(tier, key, data_root)


def load_manifest(
    tier: ManifestTier, key: str, data_root: str | Path | None = None
) -> Manifest:
    """One literature-shape tier manifest (``raw``, ``library``, ``objects``)."""
    path = manifest_path(tier, key, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"no {tier} manifest for key {key!r} at {path}")
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.citation_key != key:
        raise TierManifestKeyError(
            f"{path} names citation_key {manifest.citation_key!r}, not {key!r}"
        )
    return manifest


def load_tier_records(
    tier: Tier, key: str, data_root: str | Path | None = None
) -> list[ArtifactRecord]:
    """Every ``ArtifactRecord`` one key's manifest lists (raises like the loader)."""
    return load_tier_manifest(tier, key, data_root).files


def find_record(records: list[ArtifactRecord], path: str) -> ArtifactRecord | None:
    """The record whose ``path`` equals ``path``, or None when none does."""
    return next((rec for rec in records if rec.path == path), None)


def manifest_record(
    ref: ArtifactRef, data_root: str | Path | None = None
) -> ArtifactRecord | None:
    """The local manifest record for ``ref.path``, or None when there is none.

    None means the key has no manifest in this tier on this machine, or the manifest
    does not list the path. Whether the record's sha256 agrees with ``ref.sha256`` is
    the caller's question; this function only locates the record.
    """
    if not manifest_path(ref.tier, ref.key, data_root).is_file():
        return None
    return find_record(load_tier_records(ref.tier, ref.key, data_root), ref.path)

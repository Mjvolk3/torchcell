# tests/torchcell/artifacts/_fakes.py
# [[tests.torchcell.artifacts._fakes]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/_fakes.py
"""Hermetic tier builders and a fake ``RemoteSource`` for the artifact-tier tests.

Tier roots are built under ``tmp_path`` with real ``manifest.json`` files written by the
literature manifest helpers (``build_manifest`` + ``write_manifest``) or, for genomes, by
``registry.deposit_assembly_set``. ``FakeRemote`` serves manifests and bytes from dicts
and records every call, so a test can assert that nothing was downloaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from torchcell.artifacts.ref import ArtifactRef, Tier
from torchcell.artifacts.resolve import RemoteMissError
from torchcell.artifacts.tiers import MANIFEST_TIER_DIRS, TierManifest
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    build_manifest,
    write_manifest,
)
from torchcell.sequence.genome.registry import (
    ROLE_CONTAINER,
    GenomeManifest,
    deposit_assembly_set,
)

CREATED_AT = "2026-10-07T00:00:00+00:00"


def sha(data: bytes) -> str:
    """sha256 hex of ``data``."""
    return hashlib.sha256(data).hexdigest()


def write_tier(
    data_root: Path, tier: Tier, key: str, files: dict[str, bytes]
) -> Manifest:
    """Write ``files`` under a literature-shape tier key and its built manifest."""
    key_dir = data_root / MANIFEST_TIER_DIRS[tier] / key
    for rel, data in files.items():
        (key_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (key_dir / rel).write_bytes(data)
    manifest = build_manifest(key_dir, citation_key=key, created_at=CREATED_AT)
    write_manifest(key_dir, manifest)
    return manifest


def write_genomes(data_root: Path, assembly_set: str, files: dict[str, bytes]) -> None:
    """Write ``files`` into a genomes assembly set and deposit its registry manifest."""
    set_dir = data_root / "torchcell-genomes" / assembly_set
    set_dir.mkdir(parents=True)
    for rel, data in files.items():
        (set_dir / rel).write_bytes(data)
    deposit_assembly_set(
        GenomeManifest(
            assembly_set=assembly_set,
            organism="Saccharomyces cerevisiae",
            strain_or_population="1,011 isolates",
            source="test",
            release="v1",
            files=[
                ArtifactRecord(
                    path=rel, role=ROLE_CONTAINER, bytes=len(data), sha256=sha(data)
                )
                for rel, data in files.items()
            ],
            provenance_complete=False,
            created_at=CREATED_AT,
        ),
        data_root=str(data_root),
    )


def ref_for(
    tier: Tier, key: str, path: str, data: bytes, member: str | None = None
) -> ArtifactRef:
    """A ref pinning ``data`` (by sha256 and size) at ``tier/key/path``."""
    return ArtifactRef(
        tier=tier, key=key, path=path, member=member, sha256=sha(data), bytes=len(data)
    )


def literature_manifest(key: str, files: dict[str, bytes]) -> Manifest:
    """A literature-shape manifest listing ``files`` (no directory needed)."""
    return Manifest(
        citation_key=key,
        files=[
            ArtifactRecord(path=rel, role="object", bytes=len(data), sha256=sha(data))
            for rel, data in files.items()
        ],
        created_at=CREATED_AT,
    )


class FakeRemote:
    """A ``RemoteSource`` serving manifests and bytes from dicts; records every call."""

    def __init__(
        self,
        manifests: dict[tuple[str, str], TierManifest],
        blobs: dict[tuple[str, str, str], bytes],
    ) -> None:
        """``manifests`` by ``(tier, key)``; ``blobs`` by ``(tier, key, path)``."""
        self.manifests = manifests
        self.blobs = blobs
        self.manifest_calls: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str, str, Path]] = []

    def __repr__(self) -> str:
        """The name error messages use."""
        return "fake-remote"

    def manifest(self, tier: Tier, key: str) -> TierManifest:
        """The stored manifest, or ``RemoteMissError``."""
        self.manifest_calls.append((tier, key))
        if (tier, key) not in self.manifests:
            raise RemoteMissError(f"no key {tier}/{key}")
        return self.manifests[(tier, key)]

    def download(self, tier: Tier, key: str, path: str, dest: Path) -> None:
        """Write the stored bytes to ``dest``, or ``RemoteMissError``."""
        self.downloads.append((tier, key, path, dest))
        if (tier, key, path) not in self.blobs:
            raise RemoteMissError(f"no file {tier}/{key}/{path}")
        dest.write_bytes(self.blobs[(tier, key, path)])

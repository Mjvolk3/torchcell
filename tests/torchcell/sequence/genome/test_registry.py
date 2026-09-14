# tests/torchcell/sequence/genome/test_registry.py
"""Pure tests for the genomes tier registry against a tmp_path tier."""

from __future__ import annotations

import hashlib
import os.path as osp
from pathlib import Path

import pytest

from torchcell.literature.manifest import ArtifactRecord
from torchcell.sequence.genome.registry import (
    ROLE_SEQUENCE,
    SENTINEL_ASSEMBLY_SETS,
    SGD_S288C_R64,
    GenomeIntegrityError,
    GenomeManifest,
    deposit_assembly_set,
    genomes_root,
    load_genome_manifest,
    resolve,
    verify_assembly_set,
)

SET = "test_set_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tier(tmp_path: Path) -> tuple[str, GenomeManifest]:
    """A tier with one set holding two files; returns (data_root, manifest)."""
    data_root = str(tmp_path)
    d = tmp_path / "torchcell-genomes" / SET
    d.mkdir(parents=True)
    a = b">chrI\nACGT\n"
    b = b">YAL001C\nATG\n"
    (d / "genome.fsa").write_bytes(a)
    (d / "orfs.fasta").write_bytes(b)
    manifest = GenomeManifest(
        assembly_set=SET,
        organism="Saccharomyces cerevisiae",
        strain_or_population="S288C",
        source="test",
        release="v1",
        files=[
            ArtifactRecord(
                path="genome.fsa", role=ROLE_SEQUENCE, bytes=len(a), sha256=_sha(a)
            ),
            ArtifactRecord(
                path="orfs.fasta", role=ROLE_SEQUENCE, bytes=len(b), sha256=_sha(b)
            ),
        ],
        provenance_complete=False,
        created_at="2026-09-14T00:00:00+00:00",
        notes="test fixture; no retrieval",
    )
    return data_root, manifest


def test_deposit_then_round_trip(tmp_path: Path) -> None:
    data_root, manifest = _tier(tmp_path)
    path = deposit_assembly_set(manifest, data_root=data_root)
    assert osp.isfile(path)
    loaded = load_genome_manifest(SET, data_root=data_root)
    assert loaded == manifest
    assert genomes_root(data_root) == osp.join(data_root, "torchcell-genomes")


def test_resolve_returns_absolute_verified_path(tmp_path: Path) -> None:
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    p = resolve(SET, "genome.fsa", data_root=data_root)
    assert osp.isabs(p)
    assert p == osp.join(data_root, "torchcell-genomes", SET, "genome.fsa")
    assert verify_assembly_set(SET, data_root=data_root) == {
        "genome.fsa": manifest.files[0].sha256,
        "orfs.fasta": manifest.files[1].sha256,
    }


def test_flipped_byte_raises_integrity_error(tmp_path: Path) -> None:
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    target = Path(data_root) / "torchcell-genomes" / SET / "orfs.fasta"
    target.write_bytes(b">YAL001C\nATC\n")
    with pytest.raises(GenomeIntegrityError):
        resolve(SET, "orfs.fasta", data_root=data_root)
    assert resolve(SET, "orfs.fasta", verify=False, data_root=data_root) == str(target)
    with pytest.raises(GenomeIntegrityError):
        verify_assembly_set(SET, data_root=data_root)


def test_missing_set_names_the_rsync(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rsync -a"):
        resolve("absent_set", "x", data_root=str(tmp_path))


def test_unlisted_file_raises_key_error(tmp_path: Path) -> None:
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    with pytest.raises(KeyError):
        resolve(SET, "not_listed.fa", data_root=data_root)


def test_deposit_refuses_overwrite_and_wrong_hashes(tmp_path: Path) -> None:
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    with pytest.raises(FileExistsError):
        deposit_assembly_set(manifest, data_root=data_root)
    data_root2, manifest2 = _tier(tmp_path / "second")
    wrong = manifest2.model_copy(
        update={
            "files": [
                manifest2.files[0].model_copy(update={"sha256": "0" * 64}),
                manifest2.files[1],
            ]
        }
    )
    with pytest.raises(GenomeIntegrityError):
        deposit_assembly_set(wrong, data_root=data_root2)
    assert not osp.exists(
        osp.join(data_root2, "torchcell-genomes", SET, "manifest.json")
    )


def test_bloom_sentinel_maps_to_the_sgd_set() -> None:
    sentinel = "S288C_reference_genome_R64-4-1_20230830 (SGD; torchcell reference)"
    assert SENTINEL_ASSEMBLY_SETS[sentinel] == SGD_S288C_R64

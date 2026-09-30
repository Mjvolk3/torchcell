# tests/torchcell/sequence/genome/test_registry.py
# [[tests.torchcell.sequence.genome.test_registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_registry.py
"""Pure tests for the genomes tier registry against a tmp_path tier.

2026.09.30 (Phase 16). The fixture is one assembly set ``test_set_v1`` under
``<tmp_path>/torchcell-genomes/`` holding ``genome.fsa`` (11 bytes: ``>chrI``, ``ACGT``, each line newline-terminated)
and ``orfs.fasta`` (13 bytes: ``>YAL001C``, ``ATG``), each pinned by the sha256 of those
bytes. Expected values, derived from the source: every resolved path is
``<data_root>/torchcell-genomes/<set>/<file>``; with no ``data_root`` the root comes from
``os.environ["DATA_ROOT"]`` after ``load_dotenv()`` (stubbed here) and a missing
``DATA_ROOT`` is a ``KeyError``, never a default; each refusal carries the seed hint
``rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/<set>/
<root>/<set>/`` or names the file, the pinned value and the value on disk.
"""

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
    assembly_set_dir,
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


def _hint(root: str, assembly_set: str) -> str:
    return (
        f"assembly set {assembly_set!r} is not present under {root}; seed it with: "
        "rsync -a gilahyper:/scratch/projects/torchcell-scratch/torchcell-genomes/"
        f"{assembly_set}/ {root}/{assembly_set}/"
    )


def test_genomes_root_reads_data_root_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No explicit root: ``load_dotenv()`` runs once, then ``DATA_ROOT`` is read."""
    calls: list[int] = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: calls.append(1))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert genomes_root() == f"{tmp_path}/torchcell-genomes"
    assert calls == [1]
    assert assembly_set_dir(SET) == f"{tmp_path}/torchcell-genomes/{SET}"


def test_genomes_root_without_data_root_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset ``DATA_ROOT`` raises ``KeyError('DATA_ROOT')``; no legacy path is tried."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(KeyError) as err:
        genomes_root()
    assert err.value.args == ("DATA_ROOT",)


def test_resolve_refusals_carry_exact_messages(tmp_path: Path) -> None:
    """Absent set: the seed hint. Unlisted file: set and name. Deleted file: both."""
    data_root, manifest = _tier(tmp_path)
    root = f"{data_root}/torchcell-genomes"
    with pytest.raises(FileNotFoundError) as absent:
        resolve("absent_set", "x", data_root=data_root)
    assert str(absent.value) == _hint(root, "absent_set")
    deposit_assembly_set(manifest, data_root=data_root)
    with pytest.raises(KeyError) as unlisted:
        resolve(SET, "not_listed.fa", data_root=data_root)
    assert unlisted.value.args == (f"{SET}: 'not_listed.fa' is not in the manifest",)
    (tmp_path / "torchcell-genomes" / SET / "orfs.fasta").unlink()
    with pytest.raises(FileNotFoundError) as gone:
        resolve(SET, "orfs.fasta", data_root=data_root)
    assert str(gone.value) == (
        f"{SET}/orfs.fasta is in the manifest but not on disk; " + _hint(root, SET)
    )
    assert resolve(SET, "genome.fsa", data_root=data_root) == f"{root}/{SET}/genome.fsa"


def test_resolve_integrity_error_names_both_hashes(tmp_path: Path) -> None:
    """A flipped base: the message states the disk sha256 and the pinned one."""
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    flipped = b">YAL001C\nATC\n"
    (tmp_path / "torchcell-genomes" / SET / "orfs.fasta").write_bytes(flipped)
    with pytest.raises(GenomeIntegrityError) as err:
        resolve(SET, "orfs.fasta", data_root=data_root)
    assert str(err.value) == (
        f"{SET}/orfs.fasta: sha256 {_sha(flipped)} on disk, manifest pins "
        f"{_sha(b'>YAL001C\nATG\n')}"
    )


def test_load_manifest_refuses_a_manifest_naming_another_set(tmp_path: Path) -> None:
    """A manifest copied into the wrong directory is refused, naming both sets."""
    data_root, manifest = _tier(tmp_path)
    deposit_assembly_set(manifest, data_root=data_root)
    moved = tmp_path / "torchcell-genomes" / "other_set"
    (tmp_path / "torchcell-genomes" / SET).rename(moved)
    with pytest.raises(GenomeIntegrityError) as err:
        load_genome_manifest("other_set", data_root=data_root)
    assert str(err.value) == (
        f"{moved}/manifest.json names assembly set {SET!r}, not 'other_set'"
    )


def test_record_selects_by_path_and_names_a_missing_file() -> None:
    """``record`` returns the record whose ``path`` matches, not the first one, and a
    name the manifest does not list raises ``KeyError`` whose message names the
    assembly set and the file (registry.py line 93).
    """
    manifest = GenomeManifest(
        assembly_set="s",
        organism="o",
        strain_or_population="p",
        source="src",
        release="r",
        files=[
            ArtifactRecord(path="a.fa", role=ROLE_SEQUENCE, bytes=3, sha256="a" * 64),
            ArtifactRecord(path="b.fa", role=ROLE_SEQUENCE, bytes=4, sha256="b" * 64),
        ],
        provenance_complete=True,
        created_at="t",
    )
    assert manifest.record("b.fa").bytes == 4
    with pytest.raises(KeyError) as excinfo:
        manifest.record("c.fa")
    assert excinfo.value.args == ("s: 'c.fa' is not in the manifest",)


def test_deposit_refuses_empty_missing_and_wrong_size(tmp_path: Path) -> None:
    """Each deposit refusal is exact, and none of them writes a manifest.

    A record whose sha256 matches but whose ``bytes`` is 12 for the 11-byte
    ``genome.fsa`` is refused on size.
    """
    data_root, manifest = _tier(tmp_path)
    manifest_path = tmp_path / "torchcell-genomes" / SET / "manifest.json"
    with pytest.raises(ValueError) as empty:
        deposit_assembly_set(manifest.model_copy(update={"files": []}), data_root)
    assert str(empty.value) == f"{SET}: a manifest must list its files"
    ghost = manifest.files[0].model_copy(update={"path": "ghost.fa"})
    with pytest.raises(FileNotFoundError) as missing:
        deposit_assembly_set(manifest.model_copy(update={"files": [ghost]}), data_root)
    assert str(missing.value) == f"{SET}/ghost.fa is not on disk"
    wrong_size = manifest.files[0].model_copy(update={"bytes": 12})
    with pytest.raises(GenomeIntegrityError) as size:
        deposit_assembly_set(
            manifest.model_copy(update={"files": [wrong_size]}), data_root
        )
    assert str(size.value) == f"{SET}/genome.fsa: record says 12 bytes, disk has 11"
    wrong_hash = manifest.files[1].model_copy(update={"sha256": "0" * 64})
    with pytest.raises(GenomeIntegrityError) as bad_hash:
        deposit_assembly_set(
            manifest.model_copy(update={"files": [wrong_hash]}), data_root
        )
    assert str(bad_hash.value) == (
        f"{SET}/orfs.fasta: record pins {'0' * 64}, disk has {_sha(b'>YAL001C\nATG\n')}"
    )
    assert not manifest_path.exists()
    written = deposit_assembly_set(manifest, data_root)
    assert written == str(manifest_path)
    with pytest.raises(FileExistsError) as again:
        deposit_assembly_set(manifest, data_root)
    assert str(again.value) == (
        f"{manifest_path} exists; a deposited manifest is never rewritten"
    )

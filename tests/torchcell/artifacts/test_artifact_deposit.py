# tests/torchcell/artifacts/test_artifact_deposit.py
# [[tests.torchcell.artifacts.test_artifact_deposit]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_deposit.py
"""``deposit``: a byte-stable manifest, and a changed sha256 refused without consent."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.torchcell.artifacts._fakes import CREATED_AT, ref_for, sha
from torchcell.artifacts import ResolvedArtifact, resolve
from torchcell.artifacts.deposit import DepositChangeError, deposit
from torchcell.literature.manifest import ArtifactRecord, Manifest, ProcessingRecord

EMB = b"\x93NUMPY embedding bytes"
IDX = b"isolate\trow\nAAA\t0\n"
NEW = b"\x93NUMPY re-embedded bytes"

PROCESSING = ProcessingRecord(
    processor="experiments.035.scripts.embed_isolates.main",
    tool="esm2",
    version="t33_650M_UR50D",
    params={"layer": 33},
    input_sha256=["a" * 64],
)


def _staging(tmp_path: Path, files: dict[str, bytes]) -> Path:
    staging = tmp_path / "staging"
    for rel, data in files.items():
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        (staging / rel).write_bytes(data)
    return staging


def _manifest_file(root: Path, key: str = "esm2") -> Path:
    return root / "torchcell-objects" / key / "manifest.json"


def test_deposit_copies_hashes_and_writes_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"emb/iso.npy": EMB, "index.tsv": IDX})
    manifest = deposit(
        staging,
        key="esm2",
        processing=PROCESSING,
        roles={"index.tsv": "index"},
        data_root=root,
        created_at=CREATED_AT,
    )
    assert manifest == Manifest(
        citation_key="esm2",
        files=[
            ArtifactRecord(
                path="emb/iso.npy",
                role="object",
                bytes=len(EMB),
                sha256=sha(EMB),
                processing=PROCESSING,
            ),
            ArtifactRecord(
                path="index.tsv",
                role="index",
                bytes=len(IDX),
                sha256=sha(IDX),
                processing=PROCESSING,
            ),
        ],
        provenance_complete=True,
        created_at=CREATED_AT,
    )
    key_dir = root / "torchcell-objects" / "esm2"
    assert (key_dir / "emb" / "iso.npy").read_bytes() == EMB
    assert Manifest.model_validate_json(_manifest_file(root).read_text()) == manifest
    # The deposited file resolves through the objects tier.
    ref = ref_for("objects", "esm2", "emb/iso.npy", EMB)
    assert resolve(ref, data_root=root) == ResolvedArtifact(
        path=key_dir / "emb" / "iso.npy", source="local", verified=True
    )


def test_depositing_the_same_bytes_twice_writes_the_same_manifest_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"emb/iso.npy": EMB, "index.tsv": IDX})
    deposit(staging, key="esm2", processing=PROCESSING, data_root=root)
    first = _manifest_file(root).read_bytes()
    deposit(staging, key="esm2", processing=PROCESSING, data_root=root)
    assert _manifest_file(root).read_bytes() == first
    # In place (the key directory itself) is byte-stable too.
    deposit(root / "torchcell-objects" / "esm2", key="esm2", data_root=root)
    assert _manifest_file(root).read_bytes() == first


def test_a_changed_sha256_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"emb/iso.npy": EMB})
    deposit(staging, key="esm2", data_root=root)
    before = _manifest_file(root).read_bytes()
    (staging / "emb" / "iso.npy").write_bytes(NEW)
    with pytest.raises(DepositChangeError) as err:
        deposit(staging, key="esm2", data_root=root)
    assert str(err.value) == (
        f"objects/esm2: listed files changed sha256 emb/iso.npy ({sha(EMB)} -> "
        f"{sha(NEW)}); pass allow_change=True to replace them"
    )
    assert _manifest_file(root).read_bytes() == before
    assert (root / "torchcell-objects" / "esm2" / "emb" / "iso.npy").read_bytes() == EMB


def test_allow_change_replaces_and_records_the_superseded_sha256(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"emb/iso.npy": EMB, "index.tsv": IDX})
    first = deposit(staging, key="esm2", data_root=root, created_at=CREATED_AT)
    (staging / "emb" / "iso.npy").write_bytes(NEW)
    second = deposit(
        staging, key="esm2", processing=PROCESSING, data_root=root, allow_change=True
    )
    assert second.created_at == CREATED_AT
    assert second.files == [
        ArtifactRecord(
            path="emb/iso.npy",
            role="object",
            bytes=len(NEW),
            sha256=sha(NEW),
            source=f"supersedes sha256:{sha(EMB)}",
            processing=PROCESSING,
        ),
        first.files[1],
    ]
    # The unchanged index keeps its record with no processing, so the set is incomplete.
    assert second.provenance_complete is False
    assert (root / "torchcell-objects" / "esm2" / "emb" / "iso.npy").read_bytes() == NEW


def test_an_update_keeps_records_the_deposit_does_not_touch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    deposit(_staging(tmp_path, {"index.tsv": IDX}), key="esm2", data_root=root)
    second_stage = tmp_path / "second"
    (second_stage / "emb").mkdir(parents=True)
    (second_stage / "emb" / "iso.npy").write_bytes(EMB)
    manifest = deposit(second_stage, key="esm2", data_root=root)
    assert [r.path for r in manifest.files] == ["emb/iso.npy", "index.tsv"]


def test_a_listed_file_missing_from_the_tier_is_copied_again(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"index.tsv": IDX})
    deposit(staging, key="esm2", data_root=root)
    placed = root / "torchcell-objects" / "esm2" / "index.tsv"
    placed.unlink()
    deposit(staging, key="esm2", data_root=root)
    assert placed.read_bytes() == IDX


def test_the_raw_tier_takes_the_raw_data_role(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = deposit(
        _staging(tmp_path, {"data/expr.tsv": IDX}), tier="raw", key="k", data_root=root
    )
    assert [(r.path, r.role) for r in manifest.files] == [("data/expr.tsv", "raw_data")]
    assert (root / "torchcell-raw" / "k" / "manifest.json").is_file()


def test_deposit_refusals(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"index.tsv": IDX})
    with pytest.raises(ValueError, match="not 'genomes'; the genomes tier uses"):
        deposit(staging, tier="genomes", key="k", data_root=root)  # type: ignore[arg-type]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="holds no files to deposit"):
        deposit(empty, key="k", data_root=root)
    with pytest.raises(
        ValueError, match=r"roles name paths not in .*: \['absent.npy'\]"
    ):
        deposit(staging, key="k", roles={"absent.npy": "object"}, data_root=root)
    assert not root.exists()


def test_a_staged_manifest_json_is_not_deposited(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _staging(tmp_path, {"index.tsv": IDX, "manifest.json": b"{}"})
    manifest = deposit(staging, key="esm2", data_root=root)
    assert [r.path for r in manifest.files] == ["index.tsv"]

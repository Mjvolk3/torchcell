"""Tests for the one-dataset dev-tree LMDB build CLI."""

from pathlib import Path

import pytest

from torchcell.database.build_dataset_lmdb import (
    build_dataset,
    dataset_default_root,
    resolve_dataset_class,
)


def test_resolve_dataset_class_uses_the_registry() -> None:
    cls = resolve_dataset_class("NadalRibellesPerturbSeq2025Dataset")
    assert cls.__name__ == "NadalRibellesPerturbSeq2025Dataset"
    assert dataset_default_root(cls) == "data/torchcell/nadal_ribelles_perturbseq2025"
    with pytest.raises(KeyError, match="not a registered dataset"):
        resolve_dataset_class("NoSuchDataset")


def test_build_refuses_to_reuse_an_existing_lmdb(tmp_path: Path) -> None:
    cls = resolve_dataset_class("NadalRibellesPerturbSeq2025Dataset")
    lmdb_dir = tmp_path / dataset_default_root(cls) / "processed" / "lmdb"
    lmdb_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="deprecate.sh"):
        build_dataset(cls, str(tmp_path), io_workers=0)

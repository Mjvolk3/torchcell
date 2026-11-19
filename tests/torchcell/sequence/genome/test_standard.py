# tests/torchcell/sequence/genome/test_standard.py
# [[tests.torchcell.sequence.genome.test_standard]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_standard.py
"""Tests for the standard-genome lookup against a tmp tier and an in-process tc-data.

No test touches the network or the deposited tier. The server fixture is the real
``tc-data`` app over a tmp genomes root holding one set ``fake_set_v1`` with two files
(``a.fna``, a one-record FASTA, and ``sub/b.gff``, a GFF header line); the
client fetches into a second, empty tmp ``DATA_ROOT``. ``load_genome`` is exercised with
a stand-in genome class that records its ``genome_root``, registered under a tmp row, so
the expected values are the paths the row and ``DATA_ROOT`` spell out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from torchcell.datasets.client import DatasetClient
from torchcell.datasets.server import DataServerConfig, create_app
from torchcell.literature.manifest import ArtifactRecord
from torchcell.sequence.genome import standard
from torchcell.sequence.genome.registry import (
    GO_RELEASE_20260805,
    GenomeManifest,
    load_genome_manifest,
    resolve,
)
from torchcell.sequence.genome.standard import (
    STANDARD_GENOMES,
    StandardGenome,
    fetch_assembly_set,
    genome_for,
    is_deposited,
    load_genome,
    reference_assembly_sets,
    standard_genome,
    standard_genomes,
)

SET = "fake_set_v1"
FILES = {"a.fna": b">chr\nACGT\n", "sub/b.gff": b"##gff-version 3\n"}


def _served(root: Path) -> DatasetClient:
    """A client on the real tc-data app whose genomes tier holds ``SET``."""
    genomes = root / "served" / "genomes"
    for name, data in FILES.items():
        path = genomes / SET / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = GenomeManifest(
        assembly_set=SET,
        organism="Escherichia coli",
        strain_or_population="fake",
        source="test",
        release="v1",
        files=[
            ArtifactRecord(
                path=name,
                role="sequence",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            for name, data in FILES.items()
        ],
        provenance_complete=True,
        created_at="2026-11-18T00:00:00+00:00",
    )
    (genomes / SET / "manifest.json").write_text(manifest.model_dump_json())
    empty = {}
    for tier in ("store", "raw", "objects", "library"):
        (root / "served" / tier).mkdir(parents=True)
        empty[tier] = root / "served" / tier
    config = DataServerConfig(
        store_root=empty["store"],
        raw_root=empty["raw"],
        genomes_root=genomes,
        objects_root=empty["objects"],
        library_root=empty["library"],
    )
    return DatasetClient("http://testserver", http=TestClient(create_app(config)))


class _FakeGenome:
    """Stands in for a genome class: keeps the ``genome_root`` it was built on."""

    def __init__(self, genome_root: str) -> None:
        self.genome_root = genome_root


def _row(requires: tuple[str, ...] = (SET,)) -> StandardGenome:
    return StandardGenome(
        assembly_set=SET,
        organism="Escherichia coli",
        strain="fake",
        module=__name__,
        class_name="_FakeGenome",
        genome_root="data/fake/genome",
        requires=requires,
    )


def _dataset(*sets: str) -> Any:
    """An object with the one attribute the lookup reads, one index row per set."""
    return SimpleNamespace(
        experiment_reference_index=[
            SimpleNamespace(
                reference=SimpleNamespace(
                    genome_reference=SimpleNamespace(assembly_set=name)
                )
            )
            for name in sets
        ]
    )


def test_table_rows_agree_with_the_classes_they_name() -> None:
    """Each row repeats three class-level facts; they must equal the class's own."""
    assert [row.assembly_set for row in standard_genomes()] == sorted(STANDARD_GENOMES)
    assert len(STANDARD_GENOMES) == 5
    for key, row in STANDARD_GENOMES.items():
        cls = row.genome_class()
        assert key == row.assembly_set == cls.ASSEMBLY_SET
        assert row.genome_root == cls.ASSEMBLY.default_genome_root
        assert row.requires[0] == cls.ASSEMBLY_SET
        assert set(row.requires) == {
            cls.ASSEMBLY_SET,
            cls.ASSEMBLY.go_source.assembly_set,
            GO_RELEASE_20260805,
        }
        assert (row.organism, row.strain) == (
            cls.ASSEMBLY.organism,
            cls.ASSEMBLY.strain,
        )


def test_unknown_assembly_set_is_refused_with_the_known_ids() -> None:
    with pytest.raises(KeyError, match="has no standard genome class"):
        standard_genome("sgd_S288C_R64-4-1_20230830")


def test_fetch_writes_every_file_then_the_manifest_and_resolves(tmp_path: Path) -> None:
    client = _served(tmp_path)
    data_root = str(tmp_path / "local")
    assert not is_deposited(SET, data_root)
    directory = fetch_assembly_set(SET, data_root, client)
    assert directory == str(tmp_path / "local" / "torchcell-genomes" / SET)
    assert is_deposited(SET, data_root)
    for name, data in FILES.items():
        assert Path(resolve(SET, name, data_root=data_root)).read_bytes() == data
    assert [f.path for f in load_genome_manifest(SET, data_root).files] == list(FILES)


def test_fetch_leaves_a_deposited_set_alone(tmp_path: Path) -> None:
    client = _served(tmp_path)
    data_root = str(tmp_path / "local")
    fetch_assembly_set(SET, data_root, client)
    kept = Path(data_root) / "torchcell-genomes" / SET / "a.fna"
    kept.write_bytes(b"edited")
    fetch_assembly_set(SET, data_root, client=None)  # no client: no request is made
    assert kept.read_bytes() == b"edited"


def test_load_genome_refuses_an_absent_set_without_the_endpoint_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(STANDARD_GENOMES, SET, _row())
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    with pytest.raises(FileNotFoundError, match="Set TC_DATA_URL to download"):
        load_genome(SET)


def test_load_genome_fetches_each_required_set_and_builds_on_the_row_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _served(tmp_path)
    local = tmp_path / "local"
    monkeypatch.setitem(STANDARD_GENOMES, SET, _row())
    monkeypatch.setenv("DATA_ROOT", str(local))
    monkeypatch.setenv("TC_DATA_URL", "http://testserver")
    fetched: list[str] = []

    def fetch(assembly_set: str) -> str:
        fetched.append(assembly_set)
        return fetch_assembly_set(assembly_set, str(local), client)

    monkeypatch.setattr(standard, "fetch_assembly_set", fetch)
    genome = load_genome(SET)
    assert fetched == [SET]
    assert isinstance(genome, _FakeGenome)
    assert genome.genome_root == str(local / "data" / "fake" / "genome")
    assert Path(genome.genome_root).is_dir()
    # A second load finds the set deposited and asks for nothing.
    assert load_genome(SET, str(tmp_path / "elsewhere")).genome_root == str(
        tmp_path / "elsewhere"
    )
    assert fetched == [SET]


def test_reference_assembly_sets_are_the_distinct_ids_sorted() -> None:
    assert reference_assembly_sets(_dataset("b_set", "a_set", "b_set")) == [
        "a_set",
        "b_set",
    ]


def test_genome_for_loads_the_single_set_and_refuses_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _served(tmp_path)
    local = tmp_path / "local"
    fetch_assembly_set(SET, str(local), client)
    monkeypatch.setitem(STANDARD_GENOMES, SET, _row())
    monkeypatch.setenv("DATA_ROOT", str(local))
    genome = genome_for(_dataset(SET, SET))
    assert genome.genome_root == str(local / "data" / "fake" / "genome")
    with pytest.raises(ValueError, match="name 2 assembly sets"):
        genome_for(_dataset(SET, "other_set"))

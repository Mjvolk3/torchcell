# tests/torchcell/sequence/genome/test_standard.py
# [[tests.torchcell.sequence.genome.test_standard]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_standard.py
"""Tests for the standard-genome lookup against a tmp tier and an in-process tc-data.

No test touches the network or the deposited tier. The server fixture is the real
``tc-data`` app over a tmp genomes root holding one set ``fake_set_v1`` with two files
(``a.fna``, a one-record FASTA, and ``sub/b.gff``, a GFF header line); the
client fetches into a second, empty tmp ``DATA_ROOT``; a file's recorded source is an
``httpx.MockTransport``. ``load_genome`` is exercised with
a stand-in genome class that records its ``genome_root``, registered under a tmp row, so
the expected values are the paths the row and ``DATA_ROOT`` spell out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import attrs
import httpx
import pytest
from fastapi.testclient import TestClient

from torchcell.datasets.client import DatasetClient
from torchcell.datasets.server import DataServerConfig, create_app
from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome import standard
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    ECOLI_K12_W3110,
    GO_RELEASE_20240117,
    GO_RELEASE_20260805,
    PPUTIDA_KT2440,
    SCEREVISIAE_CENPK113_7D,
    SGD_S288C_R64,
    GenomeManifest,
    load_genome_manifest,
    resolve,
)
from torchcell.sequence.genome.standard import (
    STANDARD_GENOMES,
    AssemblySetFetchReport,
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


def _served(root: Path, sources: dict[str, str] | None = None) -> DatasetClient:
    """A client on the real tc-data app whose genomes tier holds ``SET``.

    ``sources`` gives a file a recorded source URL (a ``RetrievalRecord``).
    """
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
                retrieval=(
                    RetrievalRecord(
                        method=RetrievalMethod.direct_url,
                        source_url=sources[name],
                        retriever="torchcell.literature.retrieve.direct_url",
                        sha256=hashlib.sha256(data).hexdigest(),
                        retrieved_at="2026-11-18",
                    )
                    if sources and name in sources
                    else None
                ),
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
    """Each row repeats class-level facts; they must equal the class's own."""
    assert [row.assembly_set for row in standard_genomes()] == sorted(STANDARD_GENOMES)
    assert len(STANDARD_GENOMES) == 7
    for key, row in STANDARD_GENOMES.items():
        cls = row.genome_class()
        assert key == row.assembly_set == cls.ASSEMBLY_SET == row.requires[0]
        assert row.genome_root == attrs.fields(cls).genome_root.default
        for keyword, path in row.extra_roots.items():
            assert path == getattr(attrs.fields(cls), keyword).default


def test_bacterial_rows_require_their_go_source_and_the_go_release() -> None:
    bacterial = [
        ECOLI_B_REL606,
        ECOLI_K12_BW25113,
        ECOLI_K12_MG1655,
        ECOLI_K12_W3110,
        PPUTIDA_KT2440,
    ]
    for key in bacterial:
        row = STANDARD_GENOMES[key]
        cls = row.genome_class()
        assert set(row.requires) == {
            cls.ASSEMBLY_SET,
            cls.ASSEMBLY.go_source.assembly_set,
            GO_RELEASE_20260805,
        }
        assert (row.organism, row.strain) == (
            cls.ASSEMBLY.organism,
            cls.ASSEMBLY.strain,
        )
        assert (row.extra_roots, row.placed_files) == ({}, ())


def test_yeast_rows_pin_their_ontology_files() -> None:
    s288c = STANDARD_GENOMES[SGD_S288C_R64]
    assert s288c.requires == (SGD_S288C_R64, GO_RELEASE_20240117)
    assert s288c.extra_roots == {"go_root": "data/go"}
    assert s288c.placed_files == ((GO_RELEASE_20240117, "go.obo", "data/go/go.obo"),)
    cenpk = STANDARD_GENOMES[SCEREVISIAE_CENPK113_7D]
    assert cenpk.requires == (SCEREVISIAE_CENPK113_7D, GO_RELEASE_20260805)


def test_unknown_assembly_set_is_refused_with_the_known_ids() -> None:
    with pytest.raises(KeyError, match="has no standard genome class"):
        standard_genome("peter2018_1011_assemblies")


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
    report = AssemblySetFetchReport.model_validate_json(
        (Path(directory) / "fetch_report.json").read_text()
    )
    assert (report.assembly_set, report.mirror_url) == (SET, "http://testserver")
    # The fixture manifest records no source URL, so both files are the mirror's.
    assert [(f.path, f.origin, f.source_failure) for f in report.files] == [
        (name, "mirror", "no source URL is recorded for this file") for name in FILES
    ]


def test_fetch_leaves_a_deposited_set_alone(tmp_path: Path) -> None:
    client = _served(tmp_path)
    data_root = str(tmp_path / "local")
    fetch_assembly_set(SET, data_root, client)
    kept = Path(data_root) / "torchcell-genomes" / SET / "a.fna"
    kept.write_bytes(b"edited")
    fetch_assembly_set(SET, data_root, client=None)  # no client: no request is made
    assert kept.read_bytes() == b"edited"


def test_load_genome_fetches_each_required_set_and_builds_on_the_row_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _served(tmp_path)
    local = tmp_path / "local"
    monkeypatch.setitem(STANDARD_GENOMES, SET, _row())
    monkeypatch.setenv("DATA_ROOT", str(local))
    monkeypatch.delenv("TC_DATA_URL", raising=False)  # no switch turns fetching on
    fetched: list[str] = []

    def fetch(assembly_set: str) -> str:
        if not is_deposited(assembly_set):
            fetched.append(assembly_set)
        return fetch_assembly_set(assembly_set, str(local), client)

    monkeypatch.setattr(standard, "fetch_assembly_set", fetch)
    genome = load_genome(SET)
    assert fetched == [SET]
    assert isinstance(genome, _FakeGenome)
    assert genome.genome_root == str(local / "data" / "fake" / "genome")
    assert Path(genome.genome_root).is_dir()
    # A second load finds the set deposited and downloads nothing.
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


def test_fetch_takes_the_source_when_its_bytes_match_and_the_mirror_when_not(
    tmp_path: Path,
) -> None:
    """One file's source serves the pinned bytes, the other's serves changed bytes."""
    client = _served(
        tmp_path,
        sources={"a.fna": "https://up.test/a.fna", "sub/b.gff": "https://up.test/b"},
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a.fna":
            return httpx.Response(200, content=FILES["a.fna"])
        return httpx.Response(200, content=b"##gff-version 3.1\n")

    data_root = str(tmp_path / "local")
    directory = fetch_assembly_set(
        SET,
        data_root,
        client,
        source_http=httpx.Client(transport=httpx.MockTransport(upstream)),
    )
    report = AssemblySetFetchReport.model_validate_json(
        (Path(directory) / "fetch_report.json").read_text()
    )
    assert [(f.path, f.origin) for f in report.files] == [
        ("a.fna", "source"),
        ("sub/b.gff", "mirror"),
    ]
    assert report.files[1].source_failure is not None
    assert report.files[1].source_failure.startswith(
        "source bytes differ from the pin: 18 bytes"
    )
    for name, data in FILES.items():
        assert Path(resolve(SET, name, data_root=data_root)).read_bytes() == data


def test_load_genome_places_a_pinned_file_where_the_class_expects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _served(tmp_path)
    local = tmp_path / "local"
    fetch_assembly_set(SET, str(local), client)
    row = _row().model_copy(
        update={"placed_files": ((SET, "a.fna", "data/placed/a.fna"),)}
    )
    monkeypatch.setitem(STANDARD_GENOMES, SET, row)
    monkeypatch.setenv("DATA_ROOT", str(local))
    load_genome(SET)
    placed = local / "data" / "placed" / "a.fna"
    assert placed.read_bytes() == FILES["a.fna"]
    placed.write_bytes(b"kept")  # a file already there is never replaced
    load_genome(SET)
    assert placed.read_bytes() == b"kept"

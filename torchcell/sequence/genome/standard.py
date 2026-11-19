# torchcell/sequence/genome/standard
# [[torchcell.sequence.genome.standard]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/genome/standard
# Test file: tests/torchcell/sequence/genome/test_standard.py

"""Standard genomes by assembly set: look one up, fetch its files, construct it.

Every record of a dataset names the deposited reference its gene identifiers are
written against, ``reference.genome_reference.assembly_set``. This module turns that
string into a genome object::

    from torchcell.sequence.genome.standard import genome_for, load_genome

    genome = genome_for(dataset)  # the genome the dataset's records are written against
    genome = load_genome("ecoli_K12_MG1655_ASM584v2")  # or any standard one, by id

:data:`STANDARD_GENOMES` is the lookup table: one :class:`StandardGenome` per assembly
set that has a genome class, naming the class, the cache directory it builds its
``data.db`` in, and every assembly set that class reads (its own, the set its GO
annotation comes from, and the GO release). :func:`load_genome` makes sure each of
those sets is in the genomes tier (``$DATA_ROOT/torchcell-genomes/<assembly_set>/``)
and then constructs the class.

A set that is absent is fetched automatically, with no switch to set. The manifest of
the ``tc-data`` mirror pins every file by sha256 and size; each file is downloaded from
its recorded source URL first and accepted only when its bytes equal the pin, and it is
taken from the mirror when the source fails or has changed (the reason is logged and
written to ``fetch_report.json`` beside the files). The genome class then re-checks
every file it opens through :func:`torchcell.sequence.genome.registry.resolve`, so a
fetched tier is held to the same pins as a seeded one. ``TC_DATA_URL`` only chooses which
mirror is asked (``DatasetClient.from_env``).

Constructing a genome class directly does not fetch: the classes resolve through the
registry, which stays a pure path authority. Call :func:`load_genome`, or
:func:`fetch_assembly_set` first.
"""

from __future__ import annotations

import importlib
import os
import os.path as osp
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from torchcell.provenance.pinned_fetch import FetchOutcome, fetch_pinned
from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    ECOLI_K12_W3110,
    GO_RELEASE_20240117,
    GO_RELEASE_20260805,
    MANIFEST_FILENAME,
    PPUTIDA_KT2440,
    SCEREVISIAE_CENPK113_7D,
    SGD_S288C_R64,
    assembly_set_dir,
    resolve,
)

if TYPE_CHECKING:
    from torchcell.datasets.client import DatasetClient

FETCH_REPORT_FILENAME = "fetch_report.json"


class StandardGenome(BaseModel):
    """One row of the lookup: an assembly set and the genome class that reads it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str
    organism: str
    strain: str
    module: str
    class_name: str
    #: The class's ``data.db`` cache directory, relative to ``DATA_ROOT``.
    genome_root: str
    #: Every assembly set the class reads, its own first.
    requires: tuple[str, ...]
    #: Other directories the constructor takes, keyword to path relative to
    #: ``DATA_ROOT`` (the S288C class keeps GO's ``go.obo`` under ``go_root``).
    extra_roots: dict[str, str] = Field(default_factory=dict)
    #: Pinned tier files the class expects at a path of its own, as
    #: ``(assembly_set, member, path relative to DATA_ROOT)``; placed when absent.
    placed_files: tuple[tuple[str, str, str], ...] = ()

    def genome_class(self) -> type[Any]:
        """Import and return the genome class (imported on demand, not at lookup)."""
        cls: type[Any] = getattr(importlib.import_module(self.module), self.class_name)
        return cls


_ECOLI = "torchcell.sequence.genome.ecoli"

STANDARD_GENOMES: dict[str, StandardGenome] = {
    row.assembly_set: row
    for row in (
        StandardGenome(
            assembly_set=ECOLI_K12_BW25113,
            organism="Escherichia coli",
            strain="K-12 BW25113",
            module=_ECOLI,
            class_name="EcoliK12BW25113Genome",
            genome_root="data/ecoli/bw25113/genome",
            requires=(ECOLI_K12_BW25113, GO_RELEASE_20260805),
        ),
        StandardGenome(
            assembly_set=ECOLI_K12_MG1655,
            organism="Escherichia coli",
            strain="K-12 MG1655",
            module=_ECOLI,
            class_name="EcoliK12MG1655Genome",
            genome_root="data/ecoli/mg1655/genome",
            requires=(ECOLI_K12_MG1655, GO_RELEASE_20260805),
        ),
        StandardGenome(
            assembly_set=ECOLI_K12_W3110,
            organism="Escherichia coli",
            strain="K-12 W3110",
            module=_ECOLI,
            class_name="EcoliK12W3110Genome",
            genome_root="data/ecoli/w3110/genome",
            requires=(ECOLI_K12_W3110, GO_RELEASE_20260805),
        ),
        StandardGenome(
            assembly_set=ECOLI_B_REL606,
            organism="Escherichia coli",
            strain="B REL606",
            module=_ECOLI,
            class_name="EcoliBREL606Genome",
            genome_root="data/ecoli/rel606/genome",
            requires=(ECOLI_B_REL606, GO_RELEASE_20260805),
        ),
        StandardGenome(
            assembly_set=SGD_S288C_R64,
            organism="Saccharomyces cerevisiae",
            strain="S288C",
            module="torchcell.sequence.genome.scerevisiae.s288c",
            class_name="SCerevisiaeGenome",
            genome_root="data/sgd/genome",
            requires=(SGD_S288C_R64, GO_RELEASE_20240117),
            extra_roots={"go_root": "data/go"},
            # Without this the class downloads the CURRENT go.obo when none is cached.
            placed_files=((GO_RELEASE_20240117, "go.obo", "data/go/go.obo"),),
        ),
        StandardGenome(
            assembly_set=SCEREVISIAE_CENPK113_7D,
            organism="Saccharomyces cerevisiae",
            strain="CEN.PK113-7D",
            module="torchcell.sequence.genome.scerevisiae.ncbi",
            class_name="CenPk1137DGenome",
            genome_root="data/ncbi/cenpk113_7d",
            requires=(SCEREVISIAE_CENPK113_7D, GO_RELEASE_20260805),
        ),
        StandardGenome(
            assembly_set=PPUTIDA_KT2440,
            organism="Pseudomonas putida",
            strain="KT2440",
            module="torchcell.sequence.genome.pputida",
            class_name="PPutidaKT2440Genome",
            genome_root="data/pputida/kt2440/genome",
            requires=(PPUTIDA_KT2440, GO_RELEASE_20260805),
        ),
    )
}


def standard_genomes() -> list[StandardGenome]:
    """Every standard genome, ordered by assembly set id."""
    return [STANDARD_GENOMES[key] for key in sorted(STANDARD_GENOMES)]


def standard_genome(assembly_set: str) -> StandardGenome:
    """The lookup row of ``assembly_set``; an id with no genome class is refused."""
    if assembly_set not in STANDARD_GENOMES:
        raise KeyError(
            f"{assembly_set!r} has no standard genome class; known: "
            f"{sorted(STANDARD_GENOMES)}"
        )
    return STANDARD_GENOMES[assembly_set]


def is_deposited(assembly_set: str, data_root: str | None = None) -> bool:
    """Whether the genomes tier holds this set's ``manifest.json``."""
    return osp.isfile(
        osp.join(assembly_set_dir(assembly_set, data_root), MANIFEST_FILENAME)
    )


class AssemblySetFetchReport(BaseModel):
    """Written as ``fetch_report.json`` beside a fetched set: the origin of every file."""

    model_config = ConfigDict(extra="forbid")

    assembly_set: str
    mirror_url: str
    fetched_at: str
    files: list[FetchOutcome]


def fetch_assembly_set(
    assembly_set: str,
    data_root: str | None = None,
    client: DatasetClient | None = None,
    source_http: Any | None = None,
) -> str:
    """Fetch one assembly set into the genomes tier; return its directory.

    The manifest comes from the ``tc-data`` mirror and is the pin authority. Each file
    it lists is taken from its recorded source URL when the bytes there still equal the
    pin, and from the mirror otherwise
    (:func:`torchcell.provenance.pinned_fetch.fetch_pinned`); a file with no recorded
    source URL comes from the mirror. ``fetch_report.json`` records the origin of every
    file and why a source was passed over. ``manifest.json`` is written last, so a set is
    "deposited" only once all of its files are in place. A set that already has its
    manifest is left alone.
    """
    from torchcell.datasets.client import DatasetClient

    directory = Path(assembly_set_dir(assembly_set, data_root))
    if is_deposited(assembly_set, data_root):
        return str(directory)
    mirror_client = client if client is not None else DatasetClient.from_env()
    manifest = mirror_client.genome_manifest(assembly_set)
    outcomes: list[FetchOutcome] = []
    for record in manifest.files:
        rel_path = record.path

        def from_mirror(dest: Path, rel_path: str = rel_path) -> Path:
            return mirror_client.download_genome_file(assembly_set, rel_path, dest)

        outcome = fetch_pinned(
            directory / rel_path,
            sha256=record.sha256,
            nbytes=record.bytes,
            source_url=record.retrieval.source_url if record.retrieval else None,
            mirror=from_mirror,
            label=f"{assembly_set}/{rel_path}",
            http=source_http,
        )
        outcomes.append(outcome.model_copy(update={"path": rel_path}))
    report = AssemblySetFetchReport(
        assembly_set=assembly_set,
        mirror_url=mirror_client.url,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        files=outcomes,
    )
    (directory / FETCH_REPORT_FILENAME).write_text(report.model_dump_json(indent=2))
    (directory / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
    return str(directory)


def load_genome(assembly_set: str, genome_root: str | None = None) -> Any:
    """The genome object of a standard assembly set.

    Each set the class reads is fetched into the genomes tier under ``DATA_ROOT`` when
    it is not there (:func:`fetch_assembly_set`). ``genome_root`` is where the class
    keeps its ``data.db`` cache (built on first use); by default
    ``$DATA_ROOT/<StandardGenome.genome_root>``. Any other directory the class keeps
    (``StandardGenome.extra_roots``) is placed under ``DATA_ROOT`` the same way.
    """
    entry = standard_genome(assembly_set)
    for required in entry.requires:
        fetch_assembly_set(required)
    data_root = os.environ["DATA_ROOT"]
    if genome_root is None:
        genome_root = osp.join(data_root, entry.genome_root)
    os.makedirs(genome_root, exist_ok=True)
    for placed_set, member, rel_path in entry.placed_files:
        target = osp.join(data_root, rel_path)
        if not osp.exists(target):
            os.makedirs(osp.dirname(target), exist_ok=True)
            shutil.copyfile(resolve(placed_set, member), target)
    extra = {
        name: osp.join(data_root, path) for name, path in entry.extra_roots.items()
    }
    return entry.genome_class()(genome_root=genome_root, **extra)


def reference_assembly_sets(dataset: Any) -> list[str]:
    """The assembly sets a dataset's records are written against, sorted.

    Read from the dataset's experiment reference index, so no record is opened.
    """
    return sorted(
        {
            row.reference.genome_reference.assembly_set
            for row in dataset.experiment_reference_index
        }
    )


def genome_for(dataset: Any, genome_root: str | None = None) -> Any:
    """The genome a dataset's records are written against (:func:`load_genome`).

    A dataset whose records name more than one assembly set is refused: pick one with
    :func:`reference_assembly_sets` and :func:`load_genome`.
    """
    sets = reference_assembly_sets(dataset)
    if len(sets) != 1:
        raise ValueError(
            f"{type(dataset).__name__} records name {len(sets)} assembly sets "
            f"({sets}); load one explicitly with load_genome"
        )
    return load_genome(sets[0], genome_root)

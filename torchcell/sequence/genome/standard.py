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

A set that is absent is downloaded from the ``tc-data`` endpoint only when
``TC_DATA_URL`` is set, the same switch the dataset loaders use; with the variable unset
an absent set is an error that names both remedies. A downloaded file is checked against
the sha256 of its manifest row before it is renamed into place, and the genome class
re-checks every file it opens through
:func:`torchcell.sequence.genome.registry.resolve`, so a downloaded tier is held to the
same pins as a seeded one.

The table covers the bacterial genome classes, whose constructors share one signature.
"""

from __future__ import annotations

import importlib
import os
import os.path as osp
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from torchcell.sequence.genome.registry import (
    ECOLI_B_REL606,
    ECOLI_K12_BW25113,
    ECOLI_K12_MG1655,
    ECOLI_K12_W3110,
    GO_RELEASE_20260805,
    MANIFEST_FILENAME,
    PPUTIDA_KT2440,
    assembly_set_dir,
    genomes_root,
)

if TYPE_CHECKING:
    from torchcell.datasets.client import DatasetClient

#: The variable that turns downloading on, shared with the dataset loaders.
TC_DATA_URL_VAR = "TC_DATA_URL"


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


def fetch_assembly_set(
    assembly_set: str, data_root: str | None = None, client: DatasetClient | None = None
) -> str:
    """Download one assembly set from ``tc-data`` into the genomes tier; return its directory.

    Every file of the served manifest is streamed and sha256-checked against its
    manifest row (:meth:`~torchcell.datasets.client.DatasetClient.download_genome_file`).
    ``manifest.json`` is written last, so a set is "deposited" only once all of its
    files are in place, and an interrupted fetch resumes. A set that already has its
    manifest is left alone.
    """
    from torchcell.datasets.client import DatasetClient

    directory = Path(assembly_set_dir(assembly_set, data_root))
    if is_deposited(assembly_set, data_root):
        return str(directory)
    if client is None:
        client = DatasetClient.from_env()
    manifest = client.genome_manifest(assembly_set)
    for record in manifest.files:
        client.download_genome_file(assembly_set, record.path, directory / record.path)
    (directory / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
    return str(directory)


def load_genome(assembly_set: str, genome_root: str | None = None) -> Any:
    """The genome object of a standard assembly set.

    Each set the class reads must be in the genomes tier under ``DATA_ROOT``; an absent
    one is fetched from ``tc-data`` when ``TC_DATA_URL`` is set and refused otherwise.
    ``genome_root`` is where the class keeps its ``data.db`` cache (built on first use);
    by default ``$DATA_ROOT/<StandardGenome.genome_root>``.
    """
    entry = standard_genome(assembly_set)
    for required in entry.requires:
        if is_deposited(required):
            continue
        if not os.environ.get(TC_DATA_URL_VAR):
            raise FileNotFoundError(
                f"assembly set {required!r} is not under {genomes_root()}. Set "
                f"{TC_DATA_URL_VAR} to download it from the tc-data endpoint, or seed "
                "the genomes tier by hand."
            )
        fetch_assembly_set(required)
    if genome_root is None:
        genome_root = osp.join(os.environ["DATA_ROOT"], entry.genome_root)
    os.makedirs(genome_root, exist_ok=True)
    return entry.genome_class()(genome_root=genome_root)


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

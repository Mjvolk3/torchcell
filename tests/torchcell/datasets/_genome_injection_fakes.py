# tests/torchcell/datasets/_genome_injection_fakes.py
# [[tests.torchcell.datasets._genome_injection_fakes]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/_genome_injection_fakes.py
"""Recording stand-ins for the genomes the build entry points inject, and toy loaders.

Each fake genome SUBCLASSES the real genome class and replaces ``__init__`` with one that
only records its keyword arguments, so an injected object passes ``isinstance`` against
the real class (``SCerevisiaeGenome``, ``EcoliK12Genome``, ``PPutidaKT2440Genome``) while
no genome is built and no tier file is read. :func:`install_bacterial_fakes` puts the
three bacterial fakes into ``bacteria_common.BACTERIAL_GENOME_CLASSES``, the one map every
entry point reaches through ``BacterialGenomeInjector``; the yeast fake is patched at each
entry point's own import site by the test using it. The toy loaders declare exactly the
parameters the injection rule reads.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

import torchcell.datasets.bacteria_common as bacteria_common
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

#: Every fake genome construction, in order: (fake class name, kwargs).
BUILD_LOG: list[tuple[str, dict[str, Any]]] = []


class _Records:
    """Mixin: ``__init__`` records the kwargs on the instance and in :data:`BUILD_LOG`."""

    recorded: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        self.recorded = kwargs
        BUILD_LOG.append((type(self).__name__, kwargs))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.recorded})"


class FakeYeastGenome(_Records, SCerevisiaeGenome):
    """An ``SCerevisiaeGenome`` that only records its kwargs."""


class FakeMG1655Genome(_Records, EcoliK12MG1655Genome):
    """An ``EcoliK12MG1655Genome`` that only records its kwargs."""


class FakeBW25113Genome(_Records, EcoliK12BW25113Genome):
    """An ``EcoliK12BW25113Genome`` that only records its kwargs."""


class FakeKT2440Genome(_Records, PPutidaKT2440Genome):
    """A ``PPutidaKT2440Genome`` that only records its kwargs."""


def install_bacterial_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, dict[str, Any]]]:
    """Route every bacterial genome construction to a fake; return the cleared log."""
    BUILD_LOG.clear()
    monkeypatch.setitem(
        bacteria_common.BACTERIAL_GENOME_CLASSES, "MG1655", FakeMG1655Genome
    )
    monkeypatch.setitem(
        bacteria_common.BACTERIAL_GENOME_CLASSES, "BW25113", FakeBW25113Genome
    )
    monkeypatch.setitem(
        bacteria_common.BACTERIAL_GENOME_CLASSES, "KT2440", FakeKT2440Genome
    )
    return BUILD_LOG


class EcoliMG1655Loader:
    """A loader written against MG1655: names ``ecoli_genome``."""

    REFERENCE_STRAIN: ClassVar[str] = "MG1655"

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_mg1655_toy",
        io_workers: int = 0,
        ecoli_genome: Any = None,
    ) -> None:
        self.kwargs = {
            "root": root,
            "io_workers": io_workers,
            "ecoli_genome": ecoli_genome,
        }


class EcoliBW25113Loader:
    """A loader written against BW25113 (the Keio background): names ``ecoli_genome``."""

    REFERENCE_STRAIN: ClassVar[str] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_bw25113_toy",
        io_workers: int = 0,
        ecoli_genome: Any = None,
    ) -> None:
        self.kwargs = {
            "root": root,
            "io_workers": io_workers,
            "ecoli_genome": ecoli_genome,
        }


class PputidaLoader:
    """A loader written against KT2440: names ``pputida_genome``."""

    REFERENCE_STRAIN: ClassVar[str] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/pputida_toy",
        io_workers: int = 0,
        pputida_genome: Any = None,
    ) -> None:
        self.kwargs = {
            "root": root,
            "io_workers": io_workers,
            "pputida_genome": pputida_genome,
        }


class YeastLoader:
    """A yeast loader: names ``genome``."""

    def __init__(
        self,
        root: str = "data/torchcell/yeast_toy",
        io_workers: int = 0,
        genome: Any = None,
    ) -> None:
        self.kwargs = {"root": root, "io_workers": io_workers, "genome": genome}


class BacterialLoaderNamingYeastGenome:
    """A KT2440 loader that names the yeast ``genome``: refused before any build."""

    REFERENCE_STRAIN: ClassVar[str] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/misdeclared_toy",
        io_workers: int = 0,
        genome: Any = None,
    ) -> None:
        self.kwargs = {"root": root, "io_workers": io_workers, "genome": genome}

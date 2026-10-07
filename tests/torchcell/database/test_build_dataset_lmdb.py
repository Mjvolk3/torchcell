# tests/torchcell/database/test_build_dataset_lmdb.py
# [[tests.torchcell.database.test_build_dataset_lmdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_build_dataset_lmdb.py
"""Tests for the one-dataset dev-tree LMDB build CLI.

2026.09.30 (Phase 17): the CLI end to end on a toy loader registered for the test.
``ToyBuildDataset`` is a concrete ``ExperimentDataset`` whose default root is
``data/torchcell/toy_build`` and whose ``process`` (under ``post_process``, as every
real loader) writes three fitness records: deletions of YAL001C (reference fitness
1.0), YAL002W (reference 1.0) and YBR001C (reference 0.9). So the build has 3 records,
the gene set has 3 genes and the reference index has 2 references (1.0 -> [0, 1],
0.9 -> [2]). ``time.time`` is replaced on the module by a clock that answers 100.0 then
142.4, so the elapsed time prints as ``42s``; git is pinned for the build manifest and
``load_dotenv`` is stubbed. ``ToyNoManifestDataset`` is the same loader without
``post_process``, so no ``build_manifest.json`` is written and ``main`` returns 1.

What is pinned: the exact stdout and stderr, the exit codes (0, 1, and argparse's 2),
that ``--data-root`` wins over ``$DATA_ROOT`` and ``$DATA_ROOT`` is read after
``load_dotenv``, that nothing is ever created under ``<data_root>/database`` (the KG
build tree), the exact refusals for an unregistered class and an existing
``processed/lmdb``, and the genome and graph injection rule on recorder classes (built
only when the loader's ``__init__`` names ``genome`` or ``scerevisiae_graph``, each
passed only when named, the genome with ``overwrite=False``).

2026.10.07 (bacterial loader skeleton): injection is by parameter NAME. A loader naming
``ecoli_genome`` or ``pputida_genome`` receives the genome of its ``REFERENCE_STRAIN``
from that strain's default cache root (``overwrite=False``) and no S288C genome is built;
one naming ``genome`` still receives ``SCerevisiaeGenome`` and no bacterial genome is
built; a bacterial loader naming ``genome`` is refused before anything is built. The
genome classes are recording subclasses of the real ones
(``tests/torchcell/datasets/_genome_injection_fakes.py``), so ``isinstance`` holds and
nothing is constructed.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import pandas as pd
import pytest

import torchcell.database.build_dataset_lmdb as m
import torchcell.graph
import torchcell.provenance.build_manifest as build_manifest
import torchcell.sequence.genome.scerevisiae.s288c as s288c
from tests.torchcell.datasets._genome_injection_fakes import (
    BacterialLoaderNamingYeastGenome,
    EcoliBW25113Loader,
    FakeYeastGenome,
    PputidaLoader,
    YeastLoader,
    install_bacterial_fakes,
)
from torchcell.data.experiment_dataset import ExperimentDataset, post_process
from torchcell.database.build_dataset_lmdb import (
    build_dataset,
    dataset_default_root,
    resolve_dataset_class,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.sequence.genome.ecoli.k12 import EcoliK12BW25113Genome, EcoliK12Genome
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

_ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
_PUBLICATION = Publication(pubmed_id="1", pubmed_url="u", doi="d", doi_url="du")
# Printed to stdout by the base class's reference-index pass
# (experiment_dataset.py line 136), so it is part of the CLI's output.
_STREAMING = "Computing experiment_reference_index (streaming)..."
_ROWS = [("YAL001C", 0.5, 1.0), ("YAL002W", 0.25, 1.0), ("YBR001C", 0.75, 0.9)]


def _write_records(dataset: ExperimentDataset) -> None:
    Path(dataset.preprocess_dir).mkdir(parents=True, exist_ok=True)
    env = lmdb.open(os.path.join(dataset.processed_dir, "lmdb"), map_size=10**8)
    with env.begin(write=True) as txn:
        for i, (gene, fitness, reference_fitness) in enumerate(_ROWS):
            experiment = FitnessExperiment(
                dataset_name=dataset.name,
                genotype=Genotype(
                    perturbations=[
                        KanMxDeletionPerturbation(
                            systematic_gene_name=gene, perturbed_gene_name=gene
                        )
                    ]
                ),
                environment=_ENVIRONMENT,
                phenotype=FitnessPhenotype(fitness=fitness),
            )
            reference = FitnessExperimentReference(
                dataset_name=dataset.name,
                genome_reference=ReferenceGenome(
                    species="Saccharomyces cerevisiae", strain="S288C"
                ),
                environment_reference=_ENVIRONMENT,
                phenotype_reference=FitnessPhenotype(fitness=reference_fitness),
            )
            txn.put(
                f"{i}".encode(),
                pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": _PUBLICATION.model_dump(),
                    }
                ),
            )
    env.close()


class ToyBuildDataset(ExperimentDataset):
    """Three fitness records under the default root ``data/torchcell/toy_build``."""

    def __init__(
        self, root: str = "data/torchcell/toy_build", io_workers: int = 0
    ) -> None:
        """Only ``root`` and ``io_workers``: no genome or graph is injected."""
        super().__init__(root, io_workers)

    @property
    def experiment_class(self) -> type[FitnessExperiment]:
        """Fitness records."""
        return FitnessExperiment

    @property
    def reference_class(self) -> type[FitnessExperimentReference]:
        """Fitness references."""
        return FitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """One marker file written by ``download``."""
        return ["raw.txt"]

    def download(self) -> None:
        """Write the marker file."""
        Path(self.raw_dir, "raw.txt").write_text("toy\n")

    @post_process
    def process(self) -> None:
        """Write the three records."""
        _write_records(self)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Unused."""
        raise NotImplementedError

    def create_experiment(self) -> None:
        """Unused."""
        raise NotImplementedError


class ToyNoManifestDataset(ToyBuildDataset):
    """The same records without ``post_process``: no build manifest is written."""

    def __init__(
        self, root: str = "data/torchcell/toy_no_manifest", io_workers: int = 0
    ) -> None:
        """Same signature, its own default root."""
        super().__init__(root, io_workers)

    def process(self) -> None:
        """Write the three records and nothing else."""
        _write_records(self)


class _Clock:
    """``time.time`` answering 100.0 then 142.4."""

    def __init__(self) -> None:
        self.values = iter([100.0, 142.4])

    def time(self) -> float:
        return next(self.values)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Register the toys, pin git and the clock, stub ``load_dotenv``; yields its calls."""
    dotenv_calls: list[str] = []
    monkeypatch.setitem(dataset_registry, "ToyBuildDataset", ToyBuildDataset)
    monkeypatch.setitem(dataset_registry, "ToyNoManifestDataset", ToyNoManifestDataset)
    monkeypatch.setattr(build_manifest, "_git_info", lambda _: ("c0ffee", False))
    monkeypatch.setattr(m, "time", SimpleNamespace(time=_Clock().time))
    monkeypatch.setattr(m, "load_dotenv", lambda: dotenv_calls.append("load_dotenv"))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    yield dotenv_calls


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


def test_main_builds_the_toy_in_the_dev_tree_and_prints_its_manifest(
    tmp_path: Path, cli: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--data-root`` is used as given: the LMDB lands under
    ``<data_root>/data/torchcell/toy_build``, stdout is the base class's streaming
    line, the BUILT line (3 records, 42 s,
    3 genes, 2 references) then the manifest path, the exit code is 0, and the only
    top-level entry of the data root is ``data`` (no ``database/`` tree).
    """
    root = tmp_path / "data" / "torchcell" / "toy_build"
    assert m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        f"{_STREAMING}\n"
        f"BUILT ToyBuildDataset: 3 records at {root} in 42s; gene_set size 3; "
        "references 2\n"
        f"build manifest: {root}/preprocess/build_manifest.json\n"
    )
    # stderr carries only tqdm progress bars from the base class, no ERROR line
    assert [x for x in captured.err.splitlines() if x.startswith("ERROR")] == []
    assert cli == ["load_dotenv"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]
    assert (root / "processed" / "lmdb" / "data.mdb").is_file()


def test_main_reads_data_root_from_the_environment_after_load_dotenv(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--data-root`` the root comes from ``$DATA_ROOT`` as ``load_dotenv``
    leaves it: the variable is unset before the call and the stubbed ``load_dotenv`` sets
    it to ``<tmp>/from_dotenv``, so the build lands there only if the environment is read
    after ``load_dotenv``. ``--io-workers 0`` is accepted.
    """
    monkeypatch.delenv("DATA_ROOT", raising=False)
    from_dotenv = tmp_path / "from_dotenv"

    def load_dotenv() -> None:
        cli.append("load_dotenv")
        os.environ["DATA_ROOT"] = str(from_dotenv)

    monkeypatch.setattr(m, "load_dotenv", load_dotenv)
    assert m.main(["--dataset", "ToyBuildDataset", "--io-workers", "0"]) == 0
    root = from_dotenv / "data" / "torchcell" / "toy_build"
    assert capsys.readouterr().out.splitlines()[:2] == [
        _STREAMING,
        f"BUILT ToyBuildDataset: 3 records at {root} in 42s; gene_set size 3; "
        "references 2",
    ]
    assert cli == ["load_dotenv"]
    assert sorted(p.name for p in from_dotenv.iterdir()) == ["data"]


def test_main_returns_one_when_the_loader_writes_no_build_manifest(
    tmp_path: Path, cli: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A loader whose ``process`` skips ``post_process`` still builds, but the CLI
    reports the missing manifest on stderr and exits 1. The reference index and gene
    set are computed lazily from the store (2 references, 3 genes).
    """
    root = tmp_path / "data" / "torchcell" / "toy_no_manifest"
    code = m.main(["--dataset", "ToyNoManifestDataset", "--data-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == (
        f"{_STREAMING}\n"
        f"BUILT ToyNoManifestDataset: 3 records at {root} in 42s; gene_set size 3; "
        "references 2\n"
    )
    error = f"ERROR: build manifest missing at {root}/preprocess/build_manifest.json"
    assert [x for x in captured.err.splitlines() if x.startswith("ERROR")] == [error]
    assert captured.err.endswith(f"{error}\n")


def test_main_refuses_an_unregistered_class_before_touching_the_data_root(
    tmp_path: Path, cli: list[str]
) -> None:
    """The ``KeyError`` names the class and lists every registered class, sorted and
    comma-separated, the two toys among them; nothing is created under the root.
    """
    with pytest.raises(KeyError) as excinfo:
        m.main(["--dataset", "NoSuchDataset", "--data-root", str(tmp_path / "dr")])
    message = excinfo.value.args[0]
    prefix = "'NoSuchDataset' is not a registered dataset; known: "
    assert message.startswith(prefix)
    known = message[len(prefix) :].split(", ")
    assert known == sorted(known)
    assert {"ToyBuildDataset", "ToyNoManifestDataset"} <= set(known)
    assert not (tmp_path / "dr").exists()


def test_main_refuses_an_existing_lmdb_with_the_retirement_recipe(
    tmp_path: Path, cli: list[str]
) -> None:
    """An existing ``processed/lmdb`` is never reused: the refusal names the store and
    the exact ``deprecate.sh`` command for its ``processed`` directory, and the loader
    never runs (no ``raw/`` is created).
    """
    root = tmp_path / "data" / "torchcell" / "toy_build"
    (root / "processed" / "lmdb").mkdir(parents=True)
    with pytest.raises(FileExistsError) as excinfo:
        m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)])
    assert str(excinfo.value) == (
        f"{root}/processed/lmdb already exists; the loader would reuse it instead of "
        "rebuilding. Retire it first: DEPRECATED_DIR=<graveyard> scripts/deprecate.sh "
        f"{root}/processed '<reason>' (and the sibling preprocess/)."
    )
    assert sorted(p.name for p in root.iterdir()) == ["processed"]


def test_main_requires_the_dataset_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse exits 2 with the usage line and the missing-argument error."""
    with pytest.raises(SystemExit) as excinfo:
        m.main([])
    assert excinfo.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1] == (
        "python -m torchcell.database.build_dataset_lmdb: error: the following "
        "arguments are required: --dataset"
    )


class _Recorder:
    """Records the keyword arguments of every construction."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]], name: str) -> None:
        self.calls = calls
        self.name = name

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((self.name, kwargs))
        return SimpleNamespace(built=self.name)


class _GenomeLoader:
    def __init__(
        self, root: str = "data/torchcell/g", io_workers: int = 0, genome: Any = None
    ) -> None:
        self.kwargs = {"root": root, "io_workers": io_workers, "genome": genome}


class _GraphLoader:
    def __init__(
        self,
        root: str = "data/torchcell/s",
        io_workers: int = 0,
        scerevisiae_graph: Any = None,
    ) -> None:
        self.kwargs = {
            "root": root,
            "io_workers": io_workers,
            "scerevisiae_graph": scerevisiae_graph,
        }


class _PlainLoader:
    def __init__(self, root: str = "data/torchcell/p", io_workers: int = 0) -> None:
        self.kwargs = {"root": root, "io_workers": io_workers}


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace the genome and graph classes ``build_dataset`` imports with recorders."""
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(s288c, "SCerevisiaeGenome", _Recorder(calls, "genome"))
    monkeypatch.setattr(torchcell.graph, "SCerevisiaeGraph", _Recorder(calls, "graph"))
    return calls


def test_a_loader_naming_genome_gets_a_read_only_genome_and_no_graph(
    recorders: list[tuple[str, dict[str, Any]]],
) -> None:
    dataset = build_dataset(_GenomeLoader, "/dr", io_workers=3)
    assert recorders == [
        (
            "genome",
            {
                "genome_root": "/dr/data/sgd/genome",
                "go_root": "/dr/data/go",
                "overwrite": False,
            },
        )
    ]
    assert dataset.kwargs == {
        "root": "/dr/data/torchcell/g",
        "io_workers": 3,
        "genome": SimpleNamespace(built="genome"),
    }


def test_a_loader_naming_only_the_graph_gets_the_graph_built_on_the_genome(
    recorders: list[tuple[str, dict[str, Any]]],
) -> None:
    """The genome is still built (the graph needs it) but is not passed to the loader."""
    dataset = build_dataset(_GraphLoader, "/dr", io_workers=0)
    assert [name for name, _ in recorders] == ["genome", "graph"]
    assert recorders[1][1] == {
        "sgd_root": "/dr/data/sgd/genome",
        "string_root": "/dr/data/string",
        "tflink_root": "/dr/data/tflink",
        "genome": SimpleNamespace(built="genome"),
    }
    assert dataset.kwargs == {
        "root": "/dr/data/torchcell/s",
        "io_workers": 0,
        "scerevisiae_graph": SimpleNamespace(built="graph"),
    }


def test_a_loader_naming_neither_builds_no_genome(
    recorders: list[tuple[str, dict[str, Any]]],
) -> None:
    dataset = build_dataset(_PlainLoader, "/dr", io_workers=0)
    assert recorders == []
    assert dataset.kwargs == {"root": "/dr/data/torchcell/p", "io_workers": 0}


# --------------------------------------------------------------------------- #
# Host-aware injection: by parameter NAME, bacterial genomes built on request
# --------------------------------------------------------------------------- #
@pytest.fixture
def genome_fakes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Every genome class ``build_dataset`` reaches is a recording fake of the real one."""
    monkeypatch.setattr(s288c, "SCerevisiaeGenome", FakeYeastGenome)
    monkeypatch.setattr(torchcell.graph, "SCerevisiaeGraph", _Recorder([], "graph"))
    return install_bacterial_fakes(monkeypatch)


def test_a_loader_naming_ecoli_genome_gets_its_strain_s_genome_and_no_s288c(
    genome_fakes: list[tuple[str, dict[str, Any]]],
) -> None:
    """``ecoli_genome`` receives an ``EcoliK12Genome`` of the loader's
    ``REFERENCE_STRAIN`` (BW25113) on its default cache root, read-only; no S288C
    genome is built.
    """
    dataset = build_dataset(EcoliBW25113Loader, "/dr", io_workers=2)
    injected = dataset.kwargs["ecoli_genome"]
    assert isinstance(injected, EcoliK12Genome)
    assert isinstance(injected, EcoliK12BW25113Genome)
    assert genome_fakes == [
        (
            "FakeBW25113Genome",
            {"genome_root": "/dr/data/ecoli/bw25113/genome", "overwrite": False},
        )
    ]
    assert dataset.kwargs == {
        "root": "/dr/data/torchcell/ecoli_bw25113_toy",
        "io_workers": 2,
        "ecoli_genome": injected,
    }


def test_a_loader_naming_pputida_genome_gets_kt2440(
    genome_fakes: list[tuple[str, dict[str, Any]]],
) -> None:
    dataset = build_dataset(PputidaLoader, "/dr", io_workers=0)
    assert isinstance(dataset.kwargs["pputida_genome"], PPutidaKT2440Genome)
    assert [name for name, _ in genome_fakes] == ["FakeKT2440Genome"]


def test_a_loader_naming_genome_still_gets_s288c_and_no_bacterial_genome(
    genome_fakes: list[tuple[str, dict[str, Any]]],
) -> None:
    """The yeast rule is unchanged, and a yeast build never builds a bacterial genome."""
    dataset = build_dataset(YeastLoader, "/dr", io_workers=0)
    assert isinstance(dataset.kwargs["genome"], SCerevisiaeGenome)
    assert genome_fakes == [
        (
            "FakeYeastGenome",
            {
                "genome_root": "/dr/data/sgd/genome",
                "go_root": "/dr/data/go",
                "overwrite": False,
            },
        )
    ]


def test_a_bacterial_loader_naming_genome_is_refused_before_any_genome_is_built(
    genome_fakes: list[tuple[str, dict[str, Any]]],
) -> None:
    with pytest.raises(TypeError, match="is a bacterial loader that names 'genome'"):
        build_dataset(BacterialLoaderNamingYeastGenome, "/dr", io_workers=0)
    assert genome_fakes == []

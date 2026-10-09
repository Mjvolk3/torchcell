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

2026.10.09 (#833): the unreadable state and ``--verify``. A toy store whose record is
re-pickled with an instance of a class from a throwaway module reads ``unreadable`` with
``ModuleNotFoundError`` although every fingerprint matches, is named by ``--list-stale``
on stdout, and carries its reason on stderr. ``--verify`` resolves the entry point that
covers the dataset (a bioproduction registry first, then the loader module's
``run_verification`` or ``verify_build``), writes a ``verify_build`` report into
``preprocess/``, and on a raised exception or a failed report RENAMES the build manifest
to ``build_manifest.json.unverified.<stamp>``, so the store reads ``no_manifest``
instead of ``fresh``; an entry point needing a ``family``, ``arm`` or ``name`` is refused
rather than guessed at, and a dataset with none says so and keeps its manifest.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import types
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
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

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
        "python -m torchcell.database.build_dataset_lmdb: error: --dataset is required "
        "unless --list-stale is given"
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


# --------------------------------------------------------------------------- #
# 2026.10.08: --list-stale and --retire-existing, the bulk-rebuild half the array
# slurm script (gilahyper_build_dataset_lmdbs_array.slurm) runs per task.
# --------------------------------------------------------------------------- #
def test_retire_existing_renames_both_directories_and_refuses_a_taken_stamp(
    tmp_path: Path,
) -> None:
    """``processed`` and ``preprocess`` become ``.superseded.<stamp>`` siblings (a rename,
    their contents intact); a missing directory is skipped; a sibling that already carries
    the stamp is refused before anything moves.
    """
    root = tmp_path / "store"
    (root / "processed" / "lmdb").mkdir(parents=True)
    (root / "processed" / "lmdb" / "data.mdb").write_bytes(b"x")
    (root / "preprocess").mkdir()
    moved = m.retire_existing(str(root), stamp="20261008-000000")
    assert moved == [
        f"{root}/processed.superseded.20261008-000000",
        f"{root}/preprocess.superseded.20261008-000000",
    ]
    assert (
        root / "processed.superseded.20261008-000000" / "lmdb" / "data.mdb"
    ).exists()
    assert sorted(p.name for p in root.iterdir()) == [
        "preprocess.superseded.20261008-000000",
        "processed.superseded.20261008-000000",
    ]
    (root / "processed").mkdir()
    with pytest.raises(FileExistsError, match="refusing to retire over it"):
        m.retire_existing(str(root), stamp="20261008-000000")
    assert (root / "processed").is_dir()


def test_main_retire_existing_rebuilds_over_an_old_store(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second build with ``--retire-existing`` prints the two retired paths, leaves the
    old store as superseded siblings, and builds a fresh one in place.
    """
    import time as real_time

    root = tmp_path / "data" / "torchcell" / "toy_build"
    monkeypatch.setattr(m, "time", real_time)  # two builds outrun the two-tick clock
    assert m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (
        m.main(
            [
                "--dataset",
                "ToyBuildDataset",
                "--data-root",
                str(tmp_path),
                "--retire-existing",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out.splitlines()
    retired = [line for line in out if line.startswith("retired ")]
    assert len(retired) == 2
    assert retired[0].startswith(f"retired {root}/processed.superseded.")
    assert retired[1].startswith(f"retired {root}/preprocess.superseded.")
    assert (root / "processed" / "lmdb" / "data.mdb").is_file()
    assert (root / "preprocess" / "build_manifest.json").is_file()
    names = sorted(p.name for p in root.iterdir())
    assert names[:2] == ["preprocess", names[1]] and names[1].startswith(
        "preprocess.superseded."
    )


def test_list_stale_names_every_mapped_store_that_is_not_fresh(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Over a map of three toys: a just-built store is fresh (not listed), a store
    without a build manifest and an unbuilt one are listed, in map order; the status
    objects carry the reason.
    """
    import time as real_time

    import torchcell.knowledge_graphs.dataset_adapter_map as adapter_map

    monkeypatch.setattr(m, "time", real_time)  # two builds outrun the two-tick clock

    class ToyUnbuiltDataset(ToyBuildDataset):
        def __init__(self, root: str = "data/torchcell/toy_unbuilt", **kw: Any) -> None:
            super().__init__(root=root, **kw)

    calls: list[bool] = []

    def build_adapter_map(include_private: bool = False) -> dict[type, type]:
        calls.append(include_private)
        return {
            ToyBuildDataset: object,
            ToyNoManifestDataset: object,
            ToyUnbuiltDataset: object,
        }

    monkeypatch.setattr(adapter_map, "build_adapter_map", build_adapter_map)
    assert m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)]) == 0
    assert (
        m.main(["--dataset", "ToyNoManifestDataset", "--data-root", str(tmp_path)]) == 1
    )
    capsys.readouterr()
    assert (
        m.main(["--list-stale", "--include-private", "--data-root", str(tmp_path)]) == 0
    )
    assert capsys.readouterr().out == "ToyNoManifestDataset\nToyUnbuiltDataset\n"
    assert calls == [True]
    statuses = m.mapped_store_status(str(tmp_path), include_private=False)
    assert [(s.dataset_class, s.state, s.drift) for s in statuses] == [
        ("ToyBuildDataset", "fresh", []),
        ("ToyNoManifestDataset", "no_manifest", []),
        ("ToyUnbuiltDataset", "no_lmdb", []),
    ]
    assert [s.needs_rebuild for s in statuses] == [False, True, True]


def test_list_stale_reports_a_store_whose_closure_drifted(
    tmp_path: Path, cli: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing one stored fingerprint makes the store stale on that symbol alone."""
    import json as _json

    import torchcell.knowledge_graphs.dataset_adapter_map as adapter_map

    monkeypatch.setattr(
        adapter_map,
        "build_adapter_map",
        lambda include_private=False: {ToyBuildDataset: object},
    )
    assert m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)]) == 0
    manifest = (
        tmp_path
        / "data"
        / "torchcell"
        / "toy_build"
        / "preprocess"
        / "build_manifest.json"
    )
    doc = _json.loads(manifest.read_text())
    symbol = sorted(doc["closure"])[0]
    doc["closure"][symbol] = "0" * 16
    manifest.write_text(_json.dumps(doc))
    (status,) = m.mapped_store_status(str(tmp_path), include_private=False)
    assert (status.state, status.drift) == ("stale", [symbol])


# --------------------------------------------------------------------------- #
# 2026.10.09 (#833): the unreadable state, and --verify after a build.
#
# ``ToyBuildDataset``'s records are real fitness records, so the toy store is readable
# and reads ``fresh``. A toy whose module is a throwaway ``types.ModuleType`` is how a
# verification entry point is given to the resolver without a second loader file: the
# class's ``__module__`` names the fake module, which is exactly what
# ``resolve_verifier`` imports.
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri="https://example.invalid/toy",
    citation_key="toy",
    method="synthetic",
    page="n/a",
    sha256="0" * 64,
)


def _report(name: str, passed: bool) -> VerificationReport:
    report = VerificationReport(dataset_name=name, provenance=VERIFIER_PROVENANCE)
    report.add(
        LevelResult(level=Level.L0, name="structural", passed=passed, message="toy")
    )
    return report


def _loader_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, **members: Any
) -> types.ModuleType:
    """A throwaway loader module carrying the given verification entry points.

    It gets a real ``__file__`` on disk importing one schema symbol, because the build
    manifest is computed from the loader file's own import closure.
    """
    path = tmp_path / f"{name}.py"
    path.write_text("from torchcell.datamodels.schema import FitnessExperiment\n")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    for member, value in members.items():
        setattr(module, member, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _toy_in_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slug: str,
    module_name: str,
    **members: Any,
) -> type:
    """A ``ToyBuildDataset`` under its own slug whose module carries the verifier."""
    _loader_module(monkeypatch, tmp_path, module_name, **members)

    class Toy(ToyBuildDataset):
        def __init__(self, root: str = f"data/torchcell/{slug}", **kw: Any) -> None:
            super().__init__(root=root, **kw)

    Toy.__name__ = f"Toy{slug.title().replace('_', '')}Dataset"
    Toy.__module__ = module_name
    monkeypatch.setitem(dataset_registry, Toy.__name__, Toy)
    return Toy


def test_list_stale_names_an_unreadable_store_and_prints_its_reason_on_stderr(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A store whose records name a class this code lacks is listed, with the exception.

    The #833 failure: the manifest's fingerprints all match (a class the schema does not
    declare is in no closure), so the store read ``fresh`` and the array rebuild skipped
    it. stdout stays the bare class list an array job reads line by line; the reason
    goes to stderr.
    """
    import torchcell.knowledge_graphs.dataset_adapter_map as adapter_map

    monkeypatch.setattr(
        adapter_map,
        "build_adapter_map",
        lambda include_private=False: {ToyBuildDataset: object},
    )
    assert m.main(["--dataset", "ToyBuildDataset", "--data-root", str(tmp_path)]) == 0
    capsys.readouterr()
    root = tmp_path / "data" / "torchcell" / "toy_build"

    lost = types.ModuleType("tc_gone")

    class Censoring:
        pass

    Censoring.__module__ = "tc_gone"
    Censoring.__qualname__ = "Censoring"
    lost.Censoring = Censoring  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tc_gone", lost)
    env = lmdb.open(str(root / "processed" / "lmdb"), map_size=10**8)
    with env.begin(write=True) as txn:
        record = pickle.loads(txn.get(b"0"))
        record["experiment"]["phenotype"]["censoring"] = Censoring()
        txn.put(b"0", pickle.dumps(record))
    env.close()
    monkeypatch.delitem(sys.modules, "tc_gone")

    (status,) = m.mapped_store_status(str(tmp_path), include_private=False)
    assert (status.state, status.needs_rebuild) == ("unreadable", True)
    assert status.reason == "ModuleNotFoundError: No module named 'tc_gone'"
    assert m.main(["--list-stale", "--data-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "ToyBuildDataset\n"
    assert captured.err.splitlines() == [
        "ToyBuildDataset: unreadable: ModuleNotFoundError: No module named 'tc_gone'"
    ]


def test_verify_runs_the_modules_run_verification_and_keeps_the_manifest(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--verify`` calls ``run_verification(data_root)`` and the store stays fresh."""
    calls: list[str] = []

    def run_verification(data_root: str | None = None) -> VerificationReport:
        calls.append(str(data_root))
        return _report("toy_rv", passed=True)

    toy = _toy_in_module(
        monkeypatch, tmp_path, "toy_rv", "tc_toy_rv", run_verification=run_verification
    )
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 0
    )
    out = capsys.readouterr().out.splitlines()
    assert calls == [str(tmp_path)]
    assert f"verifying {toy.__name__} with tc_toy_rv.run_verification" in out
    assert "verification PASSED: 1 report(s) by tc_toy_rv.run_verification" in out
    manifest = tmp_path / "data/torchcell/toy_rv/preprocess/build_manifest.json"
    assert manifest.is_file()


def test_verify_writes_the_report_a_verify_build_entry_point_returns(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``verify_build(dataset_root, data_root)`` report is written to the store.

    The gap the 105-store array rebuild of 2026-10-09 left: a build writes a build
    manifest and nothing else, so ``preprocess/verification_report.json`` -- which each
    dataset's own ``--data`` test reads -- was absent. The runner writes it in a family
    run; here the builder does.
    """
    seen: list[tuple[str, str | None]] = []

    def verify_build(
        dataset_root: str, data_root: str | None = None
    ) -> VerificationReport:
        seen.append((dataset_root, data_root))
        return _report("toy_vb", passed=True)

    toy = _toy_in_module(
        monkeypatch, tmp_path, "toy_vb", "tc_toy_vb", verify_build=verify_build
    )
    root = tmp_path / "data" / "torchcell" / "toy_vb"
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 0
    )
    assert seen == [(str(root), str(tmp_path))]
    report = json.loads((root / "preprocess" / "verification_report.json").read_text())
    assert report["dataset_name"] == "toy_vb"
    assert [result["name"] for result in report["results"]] == ["structural"]
    assert all(result["passed"] for result in report["results"])


def test_verify_retires_the_manifest_when_the_verification_raises(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A raising verifier exits 1 and the store stops reading fresh.

    The manifest is RENAMED to ``build_manifest.json.unverified.<stamp>``, never
    deleted, so the store reads ``no_manifest`` to ``--list-stale`` and the live
    rebuild's preflight refuses it instead of serializing unverified records.
    """
    import torchcell.knowledge_graphs.dataset_adapter_map as adapter_map

    def run_verification(data_root: str | None = None) -> VerificationReport:
        raise RuntimeError("raw mirror absent")

    toy = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_raise",
        "tc_toy_raise",
        run_verification=run_verification,
    )
    monkeypatch.setattr(
        adapter_map, "build_adapter_map", lambda include_private=False: {toy: object}
    )
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 1
    )
    captured = capsys.readouterr()
    assert (
        "ERROR: verification of "
        f"{toy.__name__} raised RuntimeError: raw mirror absent" in captured.err
    )
    preprocess = tmp_path / "data" / "torchcell" / "toy_raise" / "preprocess"
    assert not (preprocess / "build_manifest.json").exists()
    retired = [p.name for p in preprocess.iterdir() if ".unverified." in p.name]
    assert len(retired) == 1
    assert retired[0].startswith("build_manifest.json.unverified.")
    assert f"ERROR: build manifest retired to {preprocess / retired[0]}" in captured.err
    (status,) = m.mapped_store_status(str(tmp_path), include_private=False)
    assert (status.state, status.needs_rebuild) == ("no_manifest", True)


def test_verify_retires_the_manifest_when_a_report_does_not_pass(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A FAILED report is treated as a failed build: same retirement, exit 1.

    A store whose L0-L4 gate fails is not a store the knowledge graph may be built
    from, and the preflight reads manifests, not reports -- so the manifest is where
    the refusal has to land.
    """

    def run_verification(data_root: str | None = None) -> VerificationReport:
        return _report("toy_fail", passed=False)

    toy = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_fail",
        "tc_toy_fail",
        run_verification=run_verification,
    )
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 1
    )
    captured = capsys.readouterr()
    assert "ERROR: verification FAILED for ['toy_fail']" in captured.err
    preprocess = tmp_path / "data" / "torchcell" / "toy_fail" / "preprocess"
    assert not (preprocess / "build_manifest.json").exists()


def test_verify_reports_a_dataset_with_no_per_dataset_entry_point(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No entry point is said out loud and leaves the manifest: the family runner covers it.

    38 of the 123 mapped datasets are verified only by a family runner, which reads
    every store in its family and so is not a per-dataset build step.
    """
    toy = _toy_in_module(monkeypatch, tmp_path, "toy_none", "tc_toy_none")
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 0
    )
    out = capsys.readouterr().out
    assert "NO PER-DATASET VERIFIER -- " in out
    assert "tc_toy_none declares no run_verification()/verify_build()" in out
    assert (
        tmp_path / "data/torchcell/toy_none/preprocess/build_manifest.json"
    ).is_file()


def test_a_build_without_verify_runs_no_verification(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag is opt-in: the array script passes it, a bare build is unchanged."""
    calls: list[str] = []

    def run_verification(data_root: str | None = None) -> VerificationReport:
        calls.append("called")
        return _report("toy_off", passed=True)

    toy = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_off",
        "tc_toy_off",
        run_verification=run_verification,
    )
    assert m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path)]) == 0
    assert calls == []
    assert "verifying" not in capsys.readouterr().out


def test_resolve_verifier_prefers_a_bioproduction_registry_over_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry entry wins: it adds the family L4 the loader's own gate does not.

    Resolution is registry, then ``run_verification`` with no required parameter, then
    ``verify_build`` whose only required parameter is ``dataset_root``.
    """
    from torchcell.verification import runners

    def run_verification(data_root: str | None = None) -> VerificationReport:
        return _report("unused", passed=True)

    toy = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_reg",
        "tc_toy_reg",
        run_verification=run_verification,
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runners,
        "PRODUCT_TITER_DATASETS",
        {"toy_reg": {"root": "data/torchcell/toy_reg", "verify": object()}},
    )

    def verify_bacterial_dataset(name: str, data_root: str) -> VerificationReport:
        calls.append((name, data_root))
        return _report("toy_reg", passed=True)

    monkeypatch.setattr(runners, "verify_bacterial_dataset", verify_bacterial_dataset)
    kind, run = m.resolve_verifier(toy)
    assert kind == "runners.verify_bacterial_dataset('toy_reg')"
    assert [report.dataset_name for report in run("/dr")] == ["toy_reg"]
    assert calls == [("toy_reg", "/dr")]


def test_resolve_verifier_refuses_an_entry_point_it_cannot_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``family`` / ``arm`` / ``name`` argument makes an entry point unresolvable.

    Those modules build several stores from one reader set, and nothing here can choose
    which one a given class wants; the registries name that choice, which is rule 1.
    """

    def verify_build(
        dataset_root: str, data_root: str | None = None, *, family: str
    ) -> VerificationReport:
        return _report("multi", passed=True)

    def run_verification(name: str, data_root: str | None = None) -> VerificationReport:
        return _report("multi", passed=True)

    needs_family = _toy_in_module(
        monkeypatch, tmp_path, "toy_fam", "tc_toy_fam", verify_build=verify_build
    )
    needs_name = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_name",
        "tc_toy_name",
        run_verification=run_verification,
    )
    for toy in (needs_family, needs_name):
        with pytest.raises(LookupError, match="no run_verification"):
            m.resolve_verifier(toy)


def test_retire_manifest_is_a_rename_and_refuses_a_taken_stamp(tmp_path: Path) -> None:
    """Nothing is deleted, and a second retirement under one stamp is refused."""
    preprocess = tmp_path / "store" / "preprocess"
    preprocess.mkdir(parents=True)
    assert m.retire_manifest(str(tmp_path / "store")) is None
    (preprocess / "build_manifest.json").write_text('{"a": 1}')
    dest = m.retire_manifest(str(tmp_path / "store"), stamp="20261009-000000")
    assert dest == str(preprocess / "build_manifest.json.unverified.20261009-000000")
    assert Path(dest).read_text() == '{"a": 1}'
    (preprocess / "build_manifest.json").write_text('{"a": 2}')
    with pytest.raises(FileExistsError, match="refusing to retire over it"):
        m.retire_manifest(str(tmp_path / "store"), stamp="20261009-000000")


def test_the_array_script_builds_with_retire_existing_and_verify() -> None:
    """``--verify`` is ON where the rebuilds actually happen, per array task.

    The flag is opt-in on the CLI and passed by the script that rebuilds the fleet, so
    the 105-store rebuild that produced no verification report cannot recur.
    """
    script = (
        Path(__file__).resolve().parents[3]
        / "database"
        / "slurm"
        / "scripts"
        / "gilahyper_build_dataset_lmdbs_array.slurm"
    ).read_text(encoding="utf-8")
    assert (
        "python -m torchcell.database.build_dataset_lmdb \\\n"
        '    --dataset "$DATASET_CLASS" --io-workers "$SLURM_CPUS_PER_TASK" '
        "--retire-existing \\\n"
        "    --verify" in script
    )


def test_reports_of_flattens_a_tuple_a_dict_and_a_single_report() -> None:
    """Three shapes the entry points return; anything that is not a report is dropped.

    ``run_verification`` returns one report in most modules, a tuple of them where one
    module builds several arms, and a dict keyed by arm in one. ``--verify`` judges
    ``passed`` over all of them, so they are flattened to one list.
    """
    one = _report("one", passed=True)
    two = _report("two", passed=False)
    assert [r.dataset_name for r in m._reports_of(one)] == ["one"]
    assert [r.dataset_name for r in m._reports_of((one, two))] == ["one", "two"]
    assert [r.dataset_name for r in m._reports_of({"a": one, "b": two})] == [
        "one",
        "two",
    ]
    assert m._reports_of(None) == []


def test_verify_retires_the_manifest_when_the_entry_point_returns_no_report(
    tmp_path: Path,
    cli: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A verifier that returns nothing verified nothing: the store stops reading fresh.

    An entry point that printed its findings and returned ``None`` would otherwise pass
    the gate with no report on disk, which is the state the array rebuild left.
    """

    def run_verification(data_root: str | None = None) -> None:
        return None

    toy = _toy_in_module(
        monkeypatch,
        tmp_path,
        "toy_empty",
        "tc_toy_empty",
        run_verification=run_verification,
    )
    assert (
        m.main(["--dataset", toy.__name__, "--data-root", str(tmp_path), "--verify"])
        == 1
    )
    captured = capsys.readouterr()
    assert (
        "ERROR: tc_toy_empty.run_verification returned no verification report for "
        f"{toy.__name__}" in captured.err
    )
    preprocess = tmp_path / "data" / "torchcell" / "toy_empty" / "preprocess"
    assert not (preprocess / "build_manifest.json").exists()

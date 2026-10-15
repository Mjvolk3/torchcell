# tests/torchcell/knowledge_graphs/test_generate_adapter.py
# [[tests.torchcell.knowledge_graphs.test_generate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_generate_adapter.py
"""``generate_adapter``: the flags it takes off argv before hydra, the index-to-slot
resolution, and the one-dataset instantiation with the genomes injected by parameter
name. ``main`` itself runs hydra, wandb, BioCypher and the sink, as the serial build
does; its tested contract is the marker protocol pinned in ``test_generation_array``
(a directory is complete only with ``DONE``) and the slurm script pins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.knowledge_graphs._kg_build_fakes import (
    FakeAdapter,
    FakeDataset,
    reset_instances,
)
from torchcell.data.experiment_dataset import Organism
from torchcell.knowledge_graphs import generate_adapter as gen
from torchcell.knowledge_graphs.generation_array import AdapterSlot


def test_value_flags_leave_argv_before_hydra_parses_it() -> None:
    argv = [
        "prog",
        "--config-name",
        "kg_uncapped",
        "--adapter-index",
        "65",
        "--out-dir=/o",
        "a=b",
    ]
    assert gen.take_value_flag(argv, "--adapter-index") == "65"
    assert gen.take_value_flag(argv, "--out-dir") == "/o"
    assert argv == ["prog", "--config-name", "kg_uncapped", "a=b"]
    assert gen.take_value_flag(argv, "--adapter-index") is None
    with pytest.raises(ValueError, match="--out-dir needs a value"):
        gen.take_value_flag(["prog", "--out-dir"], "--out-dir")
    with pytest.raises(ValueError, match="given 2 times"):
        gen.take_value_flag(["--out-dir", "a", "--out-dir", "b"], "--out-dir")


def test_the_worker_count_is_the_tier_cpus_the_array_script_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert gen.get_num_workers() == 32
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    with pytest.raises(RuntimeError, match="SLURM_CPUS_PER_TASK is not set"):
        gen.get_num_workers()


class FakeYeast(FakeDataset):
    """A yeast loader naming ``genome`` and ``scerevisiae_graph``."""

    n_records = 2
    instances: list[FakeDataset] = []

    def __init__(
        self,
        root: str = "data/torchcell/yeast",
        io_workers: int = 1,
        genome: Any = None,
        scerevisiae_graph: Any = None,
    ) -> None:
        """Record every kwarg the task injects."""
        super().__init__(
            root,
            io_workers=io_workers,
            genome=genome,
            scerevisiae_graph=scerevisiae_graph,
        )


class FakeYeastAdapter(FakeAdapter):
    """Adapter for ``FakeYeast``."""

    instances: list[FakeAdapter] = []


class FakeKeio(FakeDataset):
    """A BW25113 loader naming ``ecoli_genome``."""

    REFERENCE_STRAIN = "BW25113"
    organism = Organism.ecoli
    n_records = 3
    instances: list[FakeDataset] = []

    def __init__(
        self,
        root: str = "data/torchcell/keio",
        io_workers: int = 1,
        ecoli_genome: Any = None,
    ) -> None:
        """Record every kwarg the task injects."""
        super().__init__(root, io_workers=io_workers, ecoli_genome=ecoli_genome)


class FakeKeioAdapter(FakeAdapter):
    """Adapter for ``FakeKeio``."""

    instances: list[FakeAdapter] = []


@pytest.fixture
def fake_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Two mapped datasets with staged LMDB directories and fake genomes."""
    reset_instances(FakeYeast, FakeYeastAdapter, FakeKeio, FakeKeioAdapter)
    mapping = {FakeYeast: FakeYeastAdapter, FakeKeio: FakeKeioAdapter}
    import torchcell.knowledge_graphs.dataset_adapter_map as dam

    monkeypatch.setattr(
        dam, "build_adapter_map", lambda include_private=False, organisms=None: mapping
    )
    for name in ("yeast", "keio"):
        (tmp_path / "data" / "torchcell" / name / "processed" / "lmdb").mkdir(
            parents=True
        )
    genomes: list[Any] = []

    class FakeGenome:
        def __init__(self, genome_root: str, overwrite: bool) -> None:
            genomes.append((genome_root, overwrite))

    class FakeGraph:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeInjector:
        def __init__(self, data_root: str) -> None:
            self.data_root = data_root

        def genome_kwargs(self, dataset_class: type) -> dict[str, Any]:
            if dataset_class is FakeKeio:
                return {"ecoli_genome": ("ecoli", "BW25113", self.data_root)}
            return {}

    monkeypatch.setattr(gen, "SCerevisiaeGenome", FakeGenome)
    monkeypatch.setattr(gen, "SCerevisiaeGraph", FakeGraph)
    monkeypatch.setattr(gen, "BacterialGenomeInjector", FakeInjector)
    return tmp_path


def test_resolve_slot_reads_the_order_at_the_index(fake_map: Path) -> None:
    slot = gen.resolve_slot(1)
    assert (slot.index, slot.dataset_class, slot.adapter_class) == (
        1,
        "FakeKeio",
        "FakeKeioAdapter",
    )
    with pytest.raises(IndexError, match="outside the build order of 2"):
        gen.resolve_slot(2)


def test_instantiate_builds_the_one_dataset_with_its_genomes_injected_by_name(
    fake_map: Path,
) -> None:
    yeast, adapter_class = gen.instantiate(
        gen.resolve_slot(0), str(fake_map), io_workers=16
    )
    assert adapter_class is FakeYeastAdapter
    (built,) = FakeYeast.instances
    assert built is yeast
    assert built.kwargs["io_workers"] == 16
    assert type(built.kwargs["genome"]).__name__ == "FakeGenome"
    assert built.kwargs["scerevisiae_graph"].kwargs["genome"] is built.kwargs["genome"]
    assert built.kwargs["scerevisiae_graph"].kwargs["sgd_root"] == str(
        fake_map / "data/sgd/genome"
    )
    keio, _ = gen.instantiate(gen.resolve_slot(1), str(fake_map), io_workers=4)
    assert keio.kwargs == {
        "io_workers": 4,
        "ecoli_genome": ("ecoli", "BW25113", str(fake_map)),
    }
    assert FakeKeio.instances == [keio]


def test_a_missing_lmdb_is_an_error_for_the_one_dataset_asked_for(
    fake_map: Path,
) -> None:
    import shutil

    shutil.rmtree(fake_map / "data" / "torchcell" / "keio")
    with pytest.raises(FileNotFoundError, match="FakeKeio: no LMDB at"):
        gen.instantiate(
            AdapterSlot(
                index=1, dataset_class="FakeKeio", adapter_class="FakeKeioAdapter"
            ),
            str(fake_map),
            1,
        )

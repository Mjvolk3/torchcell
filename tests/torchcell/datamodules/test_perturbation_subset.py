# tests/torchcell/datamodules/test_perturbation_subset.py
# [[tests.torchcell.datamodules.test_perturbation_subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodules/test_perturbation_subset.py
"""``PerturbationSubsetDataModule`` on an in-memory stand-in for ``CellDataModule``.

The stand-in holds 20 records split train = 0..15 (16), val = {16, 17}, test = {18, 19},
so the split ratios are 0.8 / 0.1 / 0.1. Its ``index_details`` files the train records by
perturbation count as 1: 0..7, 2: 8..11, 3: 12..15; val as 1: {16}, 2: {17}; test as
2: {18}, 3: {19} (no singles, so test exercises the "no count-1" branch). Record ``i``
is ``Data(x=[[i]])`` so a batch's ``x`` column reads back the record indices.

Requesting ``size=15`` gives targets ``int(15 * 0.8) = 12``, ``int(1.5) = 1``,
``int(1.5) = 1``; the shortfall ``15 - 14 = 1`` goes to train, so train = 13. Train takes
its 8 singles first, then needs 5 from levels {2, 3}: ``5 // 2 = 2`` per level, then
``1 // 2 -> max(1, 0) = 1`` from level 2. Under ``random.seed(42)`` the three draws are
``random.sample([8, 9, 10, 11], 2) = [8, 11]``, ``random.sample([12, 13, 14, 15], 2) =
[14, 12]``, ``random.sample([9, 10], 1) = [9]``, so train = [0..7, 8, 9, 11, 12, 14],
val = [16] (its single), test = [18] (one draw from level 2 with ``1 // 2 -> 1`` per level,
and the loop breaks once the remaining size reaches 0).
"""

import json
import os.path as osp
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch_geometric.loader.dataloader import Collater

from torchcell.datamodules import (
    DataModuleIndex,
    DataModuleIndexDetails,
    DatasetSplit,
    IndexSplit,
)
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.loader.dense_padding_data_loader import DensePaddingDataLoader
from torchcell.sequence import GeneSet

N_RECORDS = 20
TRAIN = list(range(16))
VAL = [16, 17]
TEST = [18, 19]


def _split(indices: list[int]) -> IndexSplit:
    return IndexSplit(indices=indices, count=len(indices))


class _FakeDataset:
    """20 records; ``is_any_perturbed_gene_index`` maps two genes to record indices."""

    def __init__(self) -> None:
        self.is_any_perturbed_gene_index: dict[str, list[int]] = {
            "YAL001C": [0, 1, 16],
            "YAL002W": [18],
        }

    def __len__(self) -> int:
        return N_RECORDS

    def __getitem__(self, idx: int) -> Data:
        return Data(x=torch.tensor([[float(idx)]]))


class _FakeCellDataModule:
    """The attributes ``PerturbationSubsetDataModule`` reads from its parent module."""

    def __init__(self, cache_dir: str, single_level: bool = False) -> None:
        self.dataset = _FakeDataset()
        self.cache_dir = cache_dir
        self.index = DataModuleIndex(train=TRAIN, val=VAL, test=TEST)
        if single_level:
            self.index_details = DataModuleIndexDetails(
                methods=["perturbation_count_index"],
                train=DatasetSplit(perturbation_count_index={1: _split(TRAIN)}),
                val=DatasetSplit(perturbation_count_index={1: _split([16])}),
                test=DatasetSplit(perturbation_count_index={1: _split(TEST)}),
            )
            return
        self.index_details = DataModuleIndexDetails(
            methods=[
                "perturbation_count_index",
                "phenotype_label_index",
                "dataset_name_index",
            ],
            train=DatasetSplit(
                perturbation_count_index={
                    1: _split(list(range(8))),
                    2: _split([8, 9, 10, 11]),
                    3: _split([12, 13, 14, 15]),
                },
                phenotype_label_index={
                    "fitness": _split(list(range(0, 16, 2))),
                    "gene_interaction": _split(list(range(1, 16, 2))),
                },
            ),
            val=DatasetSplit(
                perturbation_count_index={1: _split([16]), 2: _split([17])},
                phenotype_label_index={"fitness": _split([16, 17])},
            ),
            test=DatasetSplit(
                perturbation_count_index={2: _split([18]), 3: _split([19])},
                phenotype_label_index={"fitness": _split([18, 19])},
            ),
        )


def _module(tmp_path: Path, **kwargs: Any) -> PerturbationSubsetDataModule:
    kwargs.setdefault("size", 15)
    return PerturbationSubsetDataModule(_FakeCellDataModule(str(tmp_path)), **kwargs)


EXPECTED_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]


def test_subset_of_15_is_13_1_1_with_the_seeded_draws(tmp_path: Path) -> None:
    """Targets 12 + 1 + 1 with the remainder to train; the level draws under seed 42."""
    dm = _module(tmp_path)
    assert dm.size == 15
    assert dm.subset_dir == osp.join(str(tmp_path), "perturbation_subset_1.5e01")
    index = dm.index
    assert index.train == EXPECTED_TRAIN
    assert index.val == [16]
    assert index.test == [18]


def test_index_details_intersect_every_method_with_the_subset(tmp_path: Path) -> None:
    """Per-method groupings are the parent's groupings cut down to the selected records.

    Train keeps 8 singles, level 2 {8, 9, 11}, level 3 {12, 14}; fitness (even records)
    {0, 2, 4, 6, 8, 12, 14}, gene_interaction (odd) {1, 3, 5, 7, 9, 11}. A method the
    parent leaves as ``None`` (``dataset_name_index``) stays ``None``; an empty
    intersection is an ``IndexSplit`` with count 0, not a dropped key.
    """
    details = _module(tmp_path).index_details
    assert details.methods == [
        "perturbation_count_index",
        "phenotype_label_index",
        "dataset_name_index",
    ]
    assert details.train.perturbation_count_index == {
        1: _split(list(range(8))),
        2: _split([8, 9, 11]),
        3: _split([12, 14]),
    }
    assert details.train.phenotype_label_index == {
        "fitness": _split([0, 2, 4, 6, 8, 12, 14]),
        "gene_interaction": _split([1, 3, 5, 7, 9, 11]),
    }
    assert details.train.dataset_name_index is None
    assert details.val.perturbation_count_index == {1: _split([16]), 2: _split([])}
    assert details.test.perturbation_count_index == {2: _split([18]), 3: _split([])}
    assert details.test.phenotype_label_index == {"fitness": _split([18])}


def test_index_is_saved_as_json_and_reloaded_by_the_next_instance(
    tmp_path: Path,
) -> None:
    """The two cache files hold ``model_dump()`` of the index and details verbatim."""
    dm = _module(tmp_path)
    index, details = dm.index, dm.index_details
    index_file = osp.join(dm.subset_dir, "index_1.5e01_seed_42.json")
    details_file = osp.join(dm.subset_dir, "index_details_1.5e01_seed_42.json")
    assert dm._cached_files_exist()
    with open(index_file) as f:
        assert json.load(f) == {"train": EXPECTED_TRAIN, "val": [16], "test": [18]}
    with open(details_file) as f:
        stored = json.load(f)
    # JSON stringifies the int perturbation-count keys; pydantic turns them back.
    assert stored["train"]["perturbation_count_index"]["2"] == {
        "indices": [8, 9, 11],
        "count": 3,
    }
    assert DataModuleIndexDetails(**stored) == details

    again = _module(tmp_path)
    assert again._cached_files_exist()  # same dir, same size, same seed
    assert again.index == index
    assert again.index_details == details


def test_cached_files_are_loaded_instead_of_recomputed(tmp_path: Path) -> None:
    """Hand-written cache files win: the module reports their content, not a fresh draw."""
    subset_dir = tmp_path / "perturbation_subset_3e00"
    subset_dir.mkdir()
    (subset_dir / "index_3e00_seed_42.json").write_text(
        json.dumps({"train": [1, 5], "val": [7], "test": []})
    )
    (subset_dir / "index_details_3e00_seed_42.json").write_text(
        json.dumps(
            {
                "methods": ["perturbation_count_index"],
                "train": {
                    "perturbation_count_index": {"1": {"indices": [1, 5], "count": 2}}
                },
                "val": {
                    "perturbation_count_index": {"1": {"indices": [7], "count": 1}}
                },
                "test": {"perturbation_count_index": {}},
            }
        )
    )
    dm = _module(tmp_path, size=3)
    assert dm.index == DataModuleIndex(train=[1, 5], val=[7], test=[])
    assert dm.index_details.train.perturbation_count_index == {1: _split([1, 5])}
    assert dm.index_details.test.perturbation_count_index == {}


def test_size_above_the_dataset_or_gene_subset_is_rejected(tmp_path: Path) -> None:
    """21 > 20 records; with the two-gene subset the ceiling is its 4 touched records."""
    with pytest.raises(
        ValueError, match=r"Requested subset size 21 exceeds maximum possible 20 "
    ):
        _module(tmp_path, size=21)
    genes = {"tag": GeneSet(["YAL001C", "YAL002W"])}
    with pytest.raises(
        ValueError, match=r"Requested subset size 5 exceeds maximum possible 4 "
    ):
        _module(tmp_path, size=5, gene_subsets=genes)
    dm = _module(tmp_path, size=4, gene_subsets=genes)
    assert dm.subset_tag == "tag"
    assert dm.gene_subset == GeneSet(["YAL001C", "YAL002W"])
    assert dm.subset_dir == osp.join(str(tmp_path), "perturbation_subset_4e00_tag")


def test_gene_subset_shortfall_shrinks_size_and_the_cache_name(tmp_path: Path) -> None:
    """Finding: a shortfall rewrites ``size`` and the cache file is named after the new size.

    The gene subset touches {0, 1, 16, 18}. ``size=4`` targets ``int(3.2) = 3``,
    ``int(0.4) = 0``, ``0``, remainder to train, so train wants 4. Its valid singles are
    {0, 1}; levels 2 and 3 have no valid record, so the loop exhausts and the module
    selects 2. ``self.size`` becomes 2 and the files are written as ``index_2e00_...``
    inside ``perturbation_subset_4e00_tag``, so the next construction with ``size=4``
    misses the cache and recomputes (``perturbation_subset.py:302, 351``).
    """
    genes = {"tag": GeneSet(["YAL001C", "YAL002W"])}
    dm = _module(tmp_path, size=4, gene_subsets=genes)
    index = dm.index
    assert index == DataModuleIndex(train=[0, 1], val=[], test=[])
    assert dm.size == 2
    assert sorted(p.name for p in Path(dm.subset_dir).iterdir()) == [
        "index_2e00_seed_42.json",
        "index_details_2e00_seed_42.json",
    ]
    assert dm._cached_files_exist()
    fresh = _module(tmp_path, size=4, gene_subsets=genes)
    assert fresh.size == 4
    assert not fresh._cached_files_exist()


def test_split_with_only_singles_stops_at_its_singles(tmp_path: Path) -> None:
    """A split whose parent has only count-1 records cannot be topped up from other levels.

    Every split is filed under count 1 alone. ``size=20`` targets 16 / 2 / 2. Val has one
    single, so after taking it the remaining 1 finds no other level and the split is left
    at [16]; test takes both of its singles. 19 of 20 are selected and ``size`` becomes 19.
    """
    dm = PerturbationSubsetDataModule(
        _FakeCellDataModule(str(tmp_path), single_level=True), size=20
    )
    index = dm.index
    assert index.train == TRAIN
    assert index.val == [16]
    assert index.test == TEST
    assert dm.size == 19


def test_setup_builds_subsets_over_the_selected_indices(tmp_path: Path) -> None:
    """Each ``Subset`` wraps the shared dataset with exactly the split's indices."""
    dm = _module(tmp_path)
    dm.setup()
    assert dm.train_dataset.indices == EXPECTED_TRAIN
    assert dm.val_dataset.indices == [16]
    assert dm.test_dataset.indices == [18]
    assert dm.train_dataset.dataset is dm.dataset
    assert dm.train_dataset[8].x.tolist() == [[8.0]]
    assert len(dm.train_dataset) == 13


def test_train_loader_yields_two_batches_of_7_and_6_in_index_order(
    tmp_path: Path,
) -> None:
    """Unshuffled, batch 7: ``x`` reads [0..6] then [7, 8, 9, 11, 12, 14]."""
    dm = _module(tmp_path, batch_size=7, train_shuffle=False)
    dm.setup()
    loader = dm.train_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.num_workers == 0
    assert loader.batch_size == 7
    batches = [b.x.squeeze(-1).tolist() for b in loader]
    assert batches == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 11.0, 12.0, 14.0],
    ]


def test_val_test_all_and_cell_module_loaders_cover_their_datasets(
    tmp_path: Path,
) -> None:
    """Val uses ``val_batch_size``; test is [18]; all is the 20 records; the parent's test is [18, 19]."""
    dm = _module(tmp_path, batch_size=7, val_batch_size=3)
    dm.setup()
    assert dm.val_dataloader().batch_size == 3
    assert dm.test_dataloader().batch_size == 7
    assert [b.x.squeeze(-1).tolist() for b in dm.val_dataloader()] == [[16.0]]
    assert [b.x.squeeze(-1).tolist() for b in dm.test_dataloader()] == [[18.0]]
    everything = torch.cat([b.x.squeeze(-1) for b in dm.all_dataloader()])
    assert everything.tolist() == [float(i) for i in range(N_RECORDS)]
    parent_test = list(dm.test_cell_module_dataloader())
    assert len(parent_test) == 1
    assert parent_test[0].x.squeeze(-1).tolist() == [18.0, 19.0]


def test_collate_fn_is_stored_but_discarded_by_the_pyg_loader(tmp_path: Path) -> None:
    """Finding: a caller's ``collate_fn`` never reaches the batches.

    ``_get_dataloader`` forwards it (``perturbation_subset.py:427-428``) but PyG's
    ``DataLoader.__init__`` pops ``collate_fn`` from its kwargs and installs its own
    ``Collater``, so the loader collates with PyG and ``my_collate`` is never called. The
    ``follow_batch`` list, by contrast, does reach the collater.
    """
    calls: list[int] = []

    def my_collate(items: list[Any]) -> list[Any]:
        calls.append(len(items))
        return items

    dm = _module(tmp_path, collate_fn=my_collate, follow_batch=["x"], batch_size=7)
    dm.setup()
    loader = dm.train_dataloader()
    assert dm.collate_fn is my_collate
    assert isinstance(loader, TorchDataLoader)
    assert isinstance(loader.collate_fn, Collater)
    assert loader.collate_fn is not my_collate
    assert loader.collate_fn.follow_batch == ["x"]
    first = next(iter(loader))
    assert calls == []
    assert first.x_batch.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert _module(tmp_path).follow_batch == ["x", "x_pert"]


def test_dense_selects_the_dense_padding_loader_but_needs_workers(
    tmp_path: Path,
) -> None:
    """Finding: ``dense=True`` with ``num_workers=0`` is unconstructible.

    The dense branch passes ``prefetch_factor`` unconditionally
    (``perturbation_subset.py:411``) and torch rejects it at ``num_workers=0``, unlike the
    guarded plain branch. With one worker the loader is a ``DensePaddingDataLoader``
    carrying the follow_batch list, spawn context and prefetch factor; it is not iterated
    here because that would spawn a worker process.
    """
    dm = _module(tmp_path, dense=True)
    dm.setup()
    with pytest.raises(
        ValueError, match="prefetch_factor option could only be specified"
    ):
        dm.train_dataloader()
    worker_dm = _module(tmp_path, dense=True, num_workers=1, prefetch_factor=3)
    worker_dm.setup()
    loader = worker_dm.test_dataloader()
    assert isinstance(loader, DensePaddingDataLoader)
    assert loader.num_workers == 1
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True
    assert loader.follow_batch == ["x", "x_pert"]
    assert loader.collator.follow_batch == ["x", "x_pert"]
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_prefetch_wraps_the_loader_in_a_prefetch_loader(tmp_path: Path) -> None:
    """``prefetch=True`` returns a ``PrefetchLoader`` around the plain loader."""
    dm = _module(tmp_path, prefetch=True, batch_size=7)
    dm.setup()
    loader = dm.train_dataloader()
    assert isinstance(loader, PrefetchLoader)
    assert isinstance(loader.loader, DataLoader)
    assert loader.loader.batch_size == 7
    assert loader.loader.dataset.indices == EXPECTED_TRAIN

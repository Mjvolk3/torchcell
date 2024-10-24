# torchcell/datamodules/perturbation_subset
# [[torchcell.datamodules.perturbation_subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/perturbation_subset
# Test file: tests/torchcell/datamodules/test_perturbation_subset.py

import os
import os.path as osp
import json
import random
from typing import Optional
import lightning as L
from torch.utils.data import Subset
from torchcell.utils import format_scientific_notation
from torchcell.datamodules import (
    IndexSplit,
    DatasetSplit,
    DataModuleIndex,
    DataModuleIndexDetails,
)
from torch_geometric.loader import DataLoader, PrefetchLoader
import torch
from torch_geometric.loader import DenseDataLoader
# from torchcell.loader.dense_padding_data_loader import DensePaddingDataLoader


class PerturbationSubsetDataModule(L.LightningDataModule):
    def __init__(
        self,
        cell_data_module,
        size: int,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch: bool = False,
        seed: int = 42,
        # dense: bool = False,
    ):
        super().__init__()
        self.cell_data_module = cell_data_module
        self.dataset = cell_data_module.dataset
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.seed = seed
        # self.dense = dense

        self.cache_dir = self.cell_data_module.cache_dir
        self.subset_dir = osp.join(
            self.cache_dir, f"perturbation_subset_{format_scientific_notation(size)}"
        )
        os.makedirs(self.subset_dir, exist_ok=True)

        random.seed(self.seed)
        self._index = None
        self._index_details = None

    @property
    def index(self) -> DataModuleIndex:
        if self._index is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index

    @property
    def index_details(self) -> DataModuleIndexDetails:
        if self._index_details is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index_details

    def _load_or_compute_index(self):
        index_file = osp.join(
            self.subset_dir,
            f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        details_file = osp.join(
            self.subset_dir,
            f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )

        if osp.exists(index_file) and osp.exists(details_file):
            with open(index_file, "r") as f:
                self._index = DataModuleIndex(**json.load(f))
            with open(details_file, "r") as f:
                self._index_details = DataModuleIndexDetails(**json.load(f))
        else:
            self._create_subset()
            self._save_index()

    def _create_subset(self):
        cell_index_details = self.cell_data_module.index_details
        cell_index = self.cell_data_module.index

        # Calculate original split ratios
        total_samples = sum(
            len(getattr(cell_index, split)) for split in ["train", "val", "test"]
        )
        original_ratios = {
            split: len(getattr(cell_index, split)) / total_samples
            for split in ["train", "val", "test"]
        }

        # Calculate target sizes for each split
        target_sizes = {
            split: int(self.size * ratio) for split, ratio in original_ratios.items()
        }

        # Adjust for rounding errors to ensure we get exactly self.size samples
        difference = self.size - sum(target_sizes.values())
        target_sizes["train"] += difference  # Add any difference to the train set

        selected_indices = {split: [] for split in ["train", "val", "test"]}

        for split in ["train", "val", "test"]:
            pert_count_index = getattr(
                cell_index_details, split
            ).perturbation_count_index
            remaining_size = target_sizes[split]

            # First, select all perturbation count 1 data
            single_pert_indices = pert_count_index[1].indices
            selected_indices[split].extend(single_pert_indices[:remaining_size])
            remaining_size -= len(selected_indices[split])

            if remaining_size > 0:
                # Equally sample from other perturbation levels
                other_pert_levels = [
                    level for level in pert_count_index.keys() if level != 1
                ]
                while remaining_size > 0 and other_pert_levels:
                    samples_per_level = max(1, remaining_size // len(other_pert_levels))
                    for level in other_pert_levels:
                        available_indices = set(pert_count_index[level].indices) - set(
                            selected_indices[split]
                        )
                        sampled = random.sample(
                            list(available_indices),
                            min(samples_per_level, len(available_indices)),
                        )
                        selected_indices[split].extend(sampled)
                        remaining_size -= len(sampled)
                        if remaining_size <= 0:
                            break
                    other_pert_levels = [
                        level
                        for level in other_pert_levels
                        if set(pert_count_index[level].indices)
                        - set(selected_indices[split])
                    ]

        self._index = DataModuleIndex(
            train=sorted(selected_indices["train"]),
            val=sorted(selected_indices["val"]),
            test=sorted(selected_indices["test"]),
        )
        self._create_index_details()

        # Verify total size
        total_selected = sum(len(indices) for indices in selected_indices.values())
        assert (
            total_selected == self.size
        ), f"Expected {self.size} samples, but got {total_selected}"

    def _create_index_details(self):
        cell_index_details = self.cell_data_module.index_details
        methods = cell_index_details.methods

        self._index_details = DataModuleIndexDetails(
            methods=methods,
            train=self._create_dataset_split(
                self._index.train, cell_index_details.train, methods
            ),
            val=self._create_dataset_split(
                self._index.val, cell_index_details.val, methods
            ),
            test=self._create_dataset_split(
                self._index.test, cell_index_details.test, methods
            ),
        )

    def _create_dataset_split(
        self, indices: list[int], cell_split: DatasetSplit, methods: list[str]
    ) -> DatasetSplit:
        dataset_split = DatasetSplit()
        for method in methods:
            split_data = {}
            method_data = getattr(cell_split, method)
            if method_data is not None:
                for key, index_split in method_data.items():
                    if method == "perturbation_count_index":
                        key = int(
                            key
                        )  # Convert key to int for perturbation_count_index
                    intersect = sorted(list(set(indices) & set(index_split.indices)))
                    split_data[key] = IndexSplit(
                        indices=intersect, count=len(intersect)
                    )
                setattr(dataset_split, method, split_data)
        return dataset_split

    def _save_index(self):
        with open(
            osp.join(
                self.subset_dir,
                f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
            ),
            "w",
        ) as f:
            json.dump(self._index.dict(), f, indent=2)
        with open(
            osp.join(
                self.subset_dir,
                f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
            ),
            "w",
        ) as f:
            json.dump(self._index_details.dict(), f, indent=2)

    def _cached_files_exist(self):
        index_file = osp.join(
            self.subset_dir,
            f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        details_file = osp.join(
            self.subset_dir,
            f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        return osp.exists(index_file) and osp.exists(details_file)

    def setup(self, stage: Optional[str] = None):
        print("Setting up PerturbationSubsetDataModule...")

        if (
            self._index is None
            or self._index_details is None
            or not self._cached_files_exist()
        ):
            self._load_or_compute_index()

        print("Creating subset datasets...")
        self.train_dataset = Subset(self.dataset, self.index.train)
        self.val_dataset = Subset(self.dataset, self.index.val)
        self.test_dataset = Subset(self.dataset, self.index.test)
        print("Setup complete.")

    def _get_dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
        )
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self):
        return self._get_dataloader(self.dataset)

    def test_cell_module_dataloader(self):
        return self._get_dataloader(
            Subset(self.dataset, self.cell_data_module.index.test)
        )

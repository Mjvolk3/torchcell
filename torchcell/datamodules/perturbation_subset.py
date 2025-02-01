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

from torchcell.loader.dense_padding_data_loader import DensePaddingDataLoader
from torchcell.sequence import GeneSet
from torchcell.datamodules import DataModuleIndex
from tqdm import tqdm


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
        dense: bool = False,
        gene_subsets: Optional[dict[str, GeneSet]] = None,
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
        self.dense = dense

        # Use the first key–value pair from the gene_subsets dict (if provided)
        if gene_subsets is not None and len(gene_subsets) > 0:
            self.subset_tag, self.gene_subset = list(gene_subsets.items())[0]
        else:
            self.subset_tag, self.gene_subset = "", None

        self.cache_dir = self.cell_data_module.cache_dir
        self.subset_dir = osp.join(
            self.cache_dir,
            f"perturbation_subset_{format_scientific_notation(size)}_{self.subset_tag}",
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
        tqdm.write("Starting _create_subset()")
        # Get base indices and details.
        cell_index_details = self.cell_data_module.index_details
        cell_index = self.cell_data_module.index

        total_samples = sum(
            len(getattr(cell_index, split)) for split in ["train", "val", "test"]
        )
        tqdm.write(f"Total samples in base module: {total_samples}")
        original_ratios = {
            split: len(getattr(cell_index, split)) / total_samples
            for split in ["train", "val", "test"]
        }
        tqdm.write(f"Original split ratios: {original_ratios}")

        target_sizes = {
            split: int(self.size * ratio) for split, ratio in original_ratios.items()
        }
        difference = self.size - sum(target_sizes.values())
        target_sizes["train"] += difference
        tqdm.write(f"Target sizes per split: {target_sizes}")

        # If a gene subset is provided, compute valid indices.
        valid_indices = None
        if self.gene_subset is not None:
            valid_indices = set()
            dataset = self.cell_data_module.dataset
            tqdm.write("Computing valid indices based on gene subset")
            for gene in tqdm(self.gene_subset, desc="Processing gene subset"):
                if gene in dataset.is_any_perturbed_gene_index:
                    valid_indices.update(dataset.is_any_perturbed_gene_index[gene])
            tqdm.write(f"Total valid indices: {len(valid_indices)}")

        selected_indices = {split: [] for split in ["train", "val", "test"]}

        for split in ["train", "val", "test"]:
            tqdm.write(f"Processing split: {split}")
            pert_count_index = getattr(
                cell_index_details, split
            ).perturbation_count_index
            remaining_size = target_sizes[split]
            tqdm.write(f"Remaining size for {split}: {remaining_size}")

            # First, sample from perturbation count 1 indices.
            single_pert_indices = pert_count_index[1].indices
            if valid_indices is not None:
                single_pert_indices = [
                    i for i in single_pert_indices if i in valid_indices
                ]
            num_single = len(single_pert_indices)
            tqdm.write(f"Found {num_single} single-perturbation indices for {split}")
            sampled_single = single_pert_indices[:remaining_size]
            selected_indices[split].extend(sampled_single)
            remaining_size -= len(sampled_single)
            tqdm.write(
                f"After sampling count=1, remaining size for {split}: {remaining_size}"
            )

            # Then, sample from other perturbation levels if needed.
            if remaining_size > 0:
                other_levels = [
                    level for level in pert_count_index.keys() if level != 1
                ]
                tqdm.write(f"Sampling from other levels for {split}: {other_levels}")
                while remaining_size > 0 and other_levels:
                    samples_per_level = max(1, remaining_size // len(other_levels))
                    tqdm.write(f"Samples per level for {split}: {samples_per_level}")
                    for level in tqdm(
                        other_levels, desc=f"Sampling levels for {split}"
                    ):
                        available = set(pert_count_index[level].indices) - set(
                            selected_indices[split]
                        )
                        if valid_indices is not None:
                            available = available.intersection(valid_indices)
                        if available:
                            sampled = random.sample(
                                list(available), min(samples_per_level, len(available))
                            )
                            selected_indices[split].extend(sampled)
                            remaining_size -= len(sampled)
                            tqdm.write(
                                f"Level {level}: sampled {len(sampled)} indices; remaining {remaining_size}"
                            )
                            if remaining_size <= 0:
                                break
                    other_levels = [
                        level
                        for level in other_levels
                        if len(
                            set(pert_count_index[level].indices)
                            - set(selected_indices[split])
                        )
                        > 0
                    ]
                    tqdm.write(f"Updated other levels for {split}: {other_levels}")

        self._index = DataModuleIndex(
            train=sorted(selected_indices["train"]),
            val=sorted(selected_indices["val"]),
            test=sorted(selected_indices["test"]),
        )
        self._create_index_details()  # Assuming you have an implementation for this
        total_selected = sum(len(indices) for indices in selected_indices.values())
        tqdm.write(f"Total selected indices: {total_selected}")
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
        index_path = osp.join(
            self.subset_dir,
            f"index_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        details_path = osp.join(
            self.subset_dir,
            f"index_details_{format_scientific_notation(self.size)}_seed_{self.seed}.json",
        )
        with open(index_path, "w") as f:
            json.dump(self._index.model_dump(), f, indent=2)
        with open(details_path, "w") as f:
            json.dump(self._index_details.model_dump(), f, indent=2)

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
        if self.dense:
            # loader = DenseDataLoader(
            #     dataset,
            #     batch_size=self.batch_size,
            #     shuffle=shuffle,
            #     num_workers=self.num_workers,
            #     pin_memory=self.pin_memory,
            #     # follow_batch=["x", "x_pert"],
            # )
            loader = DensePaddingDataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                follow_batch=["x", "x_pert"],
            )
        else:
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

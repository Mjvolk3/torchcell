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
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        seed: int = 42,
        dense: bool = False,
        gene_subsets: Optional[dict[str, GeneSet]] = None,
        follow_batch: Optional[list] = None,
        train_shuffle: bool = True,
        collate_fn: Optional[object] = None,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.cell_data_module = cell_data_module
        self.dataset = cell_data_module.dataset
        self.train_shuffle = train_shuffle

        # Use the first key–value pair from the gene_subsets dict (if provided)
        if gene_subsets is not None and len(gene_subsets) > 0:
            self.subset_tag, self.gene_subset = list(gene_subsets.items())[0]
        else:
            self.subset_tag, self.gene_subset = None, None

        # Check that requested size does not exceed maximum possible.
        self._set_size(size)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.dense = dense
        if follow_batch is None:
            self.follow_batch = ["x", "x_pert"]
        else:
            self.follow_batch = follow_batch
        self.collate_fn = collate_fn
        self.cache_dir = self.cell_data_module.cache_dir
        if self.subset_tag:
            self.subset_dir = osp.join(
                self.cache_dir,
                f"perturbation_subset_{format_scientific_notation(self.size)}_{self.subset_tag}",
            )
        else:
            self.subset_dir = osp.join(
                self.cache_dir,
                f"perturbation_subset_{format_scientific_notation(self.size)}",
            )
        os.makedirs(self.subset_dir, exist_ok=True)
        random.seed(self.seed)
        self._index = None
        self._index_details = None

    def _set_size(self, size: int):
        # Determine maximum possible subset size.
        if self.gene_subset is not None:
            valid = set()
            for gene in self.gene_subset:
                if gene in self.cell_data_module.dataset.is_any_perturbed_gene_index:
                    valid.update(
                        self.cell_data_module.dataset.is_any_perturbed_gene_index[gene]
                    )
            max_possible = len(valid)
        else:
            max_possible = len(self.dataset)
        if size > max_possible:
            raise ValueError(
                f"Requested subset size {size} exceeds maximum possible {max_possible} "
                f"for the given gene subset (or full dataset)."
            )
        self.size = size

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
            print("Loading cached index files...")
            with open(index_file, "r") as f:
                self._index = DataModuleIndex(**json.load(f))
            with open(details_file, "r") as f:
                self._index_details = DataModuleIndexDetails(**json.load(f))
        else:
            print("Computing subset index...")
            self._create_subset()
            self._save_index()

    def _create_subset(self):
        tqdm.write("Starting _create_subset()")
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

            # Print available perturbation counts for debugging
            tqdm.write(
                f"Available perturbation counts: {list(pert_count_index.keys())}"
            )

            # Try to get single perturbations first if they exist
            if 1 in pert_count_index:
                tqdm.write("Processing single perturbations (count=1)")
                single_pert_indices = pert_count_index[1].indices
                if valid_indices is not None:
                    single_pert_indices = [
                        i for i in single_pert_indices if i in valid_indices
                    ]
                num_single = len(single_pert_indices)
                tqdm.write(
                    f"Found {num_single} single-perturbation indices for {split}"
                )
                sampled_single = single_pert_indices[:remaining_size]
                selected_indices[split].extend(sampled_single)
                remaining_size -= len(sampled_single)
                tqdm.write(
                    f"After sampling count=1, remaining size for {split}: {remaining_size}"
                )
            else:
                tqdm.write(
                    f"No single-perturbation indices (count=1) found for {split}"
                )

            if remaining_size > 0:
                # Use all available perturbation counts
                available_levels = list(pert_count_index.keys())
                if 1 in available_levels:
                    available_levels.remove(
                        1
                    )  # Remove single perturbations if already processed

                tqdm.write(
                    f"Sampling from available levels for {split}: {available_levels}"
                )

                if not available_levels:
                    tqdm.write(
                        f"Warning: No other perturbation levels available for {split}"
                    )
                    continue

                while remaining_size > 0 and available_levels:
                    samples_per_level = max(1, remaining_size // len(available_levels))
                    tqdm.write(f"Samples per level for {split}: {samples_per_level}")

                    # Keep track of levels that still have available samples
                    levels_with_remaining_samples = []

                    for level in tqdm(
                        available_levels, desc=f"Sampling levels for {split}"
                    ):
                        # Get available indices for this level that haven't been selected yet
                        available = set(pert_count_index[level].indices) - set(
                            selected_indices[split]
                        )

                        # Apply gene subset filter if specified
                        if valid_indices is not None:
                            available = available.intersection(valid_indices)

                        if available:
                            # Sample from available indices
                            sampled = random.sample(
                                list(available), min(samples_per_level, len(available))
                            )
                            selected_indices[split].extend(sampled)
                            remaining_size -= len(sampled)
                            tqdm.write(
                                f"Level {level}: sampled {len(sampled)} indices; remaining {remaining_size}"
                            )

                            # Check if there are still available indices for this level
                            if len(available) > len(sampled):
                                levels_with_remaining_samples.append(level)

                            if remaining_size <= 0:
                                break

                    # Update available levels to only those that still have samples
                    available_levels = levels_with_remaining_samples
                    if not available_levels and remaining_size > 0:
                        tqdm.write(
                            f"Warning: Exhausted all available indices for {split}, but still need {remaining_size} more"
                        )
                        break

        self._index = DataModuleIndex(
            train=sorted(selected_indices["train"]),
            val=sorted(selected_indices["val"]),
            test=sorted(selected_indices["test"]),
        )
        self._create_index_details()

        total_selected = sum(len(indices) for indices in selected_indices.values())
        tqdm.write(f"Total selected indices: {total_selected}")

        # Check if we got fewer samples than requested
        if total_selected < self.size:
            tqdm.write(
                f"Warning: Could only select {total_selected} samples out of the requested {self.size}"
            )
            self.size = total_selected  # Update size to match actual selection

        assert (
            total_selected <= self.size
        ), f"Selected {total_selected} samples, which exceeds the requested {self.size}"

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
                        key = int(key)
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

    def _get_dataloader(self, dataset, shuffle=False, batch_size=None):
        # Use provided batch_size or fall back to self.batch_size
        if batch_size is None:
            batch_size = self.batch_size

        if self.dense:
            loader = DensePaddingDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                pin_memory=self.pin_memory,
                follow_batch=self.follow_batch,
                multiprocessing_context=("spawn" if self.num_workers > 0 else None),
                prefetch_factor=self.prefetch_factor,
            )
        else:
            dataloader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": self.num_workers,
                "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
                "pin_memory": self.pin_memory,
                "follow_batch": self.follow_batch,
                "multiprocessing_context": ("spawn" if self.num_workers > 0 else None),
            }

            # Add collate_fn if provided
            if self.collate_fn is not None:
                dataloader_kwargs["collate_fn"] = self.collate_fn

            loader = DataLoader(dataset, **dataloader_kwargs)
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self):
        return self._get_dataloader(self.dataset)

    def test_cell_module_dataloader(self):
        return self._get_dataloader(
            Subset(self.dataset, self.cell_data_module.index.test)
        )

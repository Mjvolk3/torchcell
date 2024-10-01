import os
import os.path as osp
import json
import random
from typing import Optional, Dict, List, Tuple
import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from tqdm import tqdm


class PerturbationSubsetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        size: int,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_dir: Optional[str] = None,
        original_cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_dir = cache_dir or osp.join(
            dataset.root, "perturbation_subset_cache"
        )
        self.original_cache_dir = original_cache_dir or osp.join(
            dataset.root, "data_module_cache"
        )
        self.seed = seed
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pert_count_index = None
        self._set_size(size)
        random.seed(self.seed)
        self.subset_info = None
        self.split_ratio = None

    def _calculate_min_size(self) -> int:
        return min(len(indices) for indices in self.pert_count_index.values())

    def _set_size(self, size: int):
        with open(
            osp.join(self.dataset.processed_dir, "perturbation_count_index.json"), "r"
        ) as f:
            self.pert_count_index = {int(k): v for k, v in json.load(f).items()}

        min_size = self._calculate_min_size()
        max_size = len(self.dataset)

        if not min_size <= size <= max_size:
            raise ValueError(
                f"Invalid subset size. Size must be between {min_size} (smallest perturbation level) and {max_size}. "
                f"Provided size: {size}"
            )

        self.size = size

    def setup(self, stage: Optional[str] = None):
        print("Setting up PerturbationSubsetDataModule...")
        # Always load the original split
        with open(osp.join(self.original_cache_dir, "cached_indices.json"), "r") as f:
            self.original_split = json.load(f)

        self.split_ratio = self._calculate_split_ratio()
        print(f"Original split ratio: {self.split_ratio}")

        cached_subset = self._load_cached_subset()
        if cached_subset:
            print("Loading cached subset...")
            (
                self.train_indices,
                self.val_indices,
                self.test_indices,
                self.selected_indices,
                self.subset_info,
            ) = cached_subset
        else:
            print("Creating new subset...")
            max_pert = max(self.pert_count_index.keys())

            pert_sizes = self._calculate_pert_sizes(max_pert)
            self.selected_indices = self._select_indices(pert_sizes)
            self.train_indices, self.val_indices, self.test_indices = (
                self._split_indices(self.selected_indices)
            )
            self._calculate_subset_info()
            self._save_subset_indices()

        print("Creating subset datasets...")
        self.train_dataset = Subset(self.dataset, self.train_indices)
        self.val_dataset = Subset(self.dataset, self.val_indices)
        self.test_dataset = Subset(self.dataset, self.test_indices)
        self.full_test_dataset = Subset(
            self.dataset, self.original_split["test_indices"]
        )
        print("Setup complete.")

    def _calculate_split_ratio(self) -> Tuple[float, float, float]:
        total = sum(len(indices) for indices in self.original_split.values())
        train_ratio = len(self.original_split["train_indices"]) / total
        val_ratio = len(self.original_split["val_indices"]) / total
        test_ratio = len(self.original_split["test_indices"]) / total
        return train_ratio, val_ratio, test_ratio

    def _calculate_pert_sizes(self, max_pert: int) -> Dict[int, int]:
        print(f"Calculating sizes for {max_pert} perturbation levels...")
        pert_sizes = {}
        remaining_size = self.size

        # First, allocate all instances of the smallest perturbation
        min_pert_level = min(self.pert_count_index.keys())
        min_pert_count = len(self.pert_count_index[min_pert_level])
        pert_sizes[min_pert_level] = min(min_pert_count, remaining_size)
        remaining_size -= pert_sizes[min_pert_level]

        if remaining_size > 0:
            # Distribute remaining size equally among other perturbation levels
            other_pert_levels = [
                i for i in range(1, max_pert + 1) if i != min_pert_level
            ]
            num_other_levels = len(other_pert_levels)

            for i in other_pert_levels:
                pert_size = min(
                    len(self.pert_count_index[i]), remaining_size // num_other_levels
                )
                pert_sizes[i] = pert_size
                remaining_size -= pert_size

        print(f"Perturbation sizes: {pert_sizes}")
        return pert_sizes

    def _select_indices(self, pert_sizes: Dict[int, int]) -> List[int]:
        print("Selecting indices...")
        selected = []
        for pert_level, size in pert_sizes.items():
            if pert_level == min(pert_sizes.keys()):
                # For the smallest perturbation level, select all available indices up to the size
                selected += self.pert_count_index[pert_level][:size]
            else:
                # For other perturbation levels, randomly sample
                selected += random.sample(self.pert_count_index[pert_level], size)
        return selected

    def _split_indices(
        self, indices: List[int]
    ) -> tuple[List[int], List[int], List[int]]:
        print("Splitting indices...")
        train = []
        val = []
        test = []
        for idx in tqdm(indices, desc="Splitting indices"):
            if idx in self.original_split["train_indices"]:
                train.append(idx)
            elif idx in self.original_split["val_indices"]:
                val.append(idx)
            elif idx in self.original_split["test_indices"]:
                test.append(idx)

        # Adjust split to match original ratio as closely as possible
        total = len(indices)
        train_target = int(self.split_ratio[0] * total)
        val_target = int(self.split_ratio[1] * total)
        test_target = total - train_target - val_target

        while len(train) > train_target:
            idx = train.pop()
            if len(val) < val_target:
                val.append(idx)
            else:
                test.append(idx)

        while len(val) > val_target:
            idx = val.pop()
            if len(train) < train_target:
                train.append(idx)
            else:
                test.append(idx)

        while len(test) > test_target:
            idx = test.pop()
            if len(train) < train_target:
                train.append(idx)
            else:
                val.append(idx)

        actual_ratio = (len(train) / total, len(val) / total, len(test) / total)
        print(f"Target split ratio: {self.split_ratio}")
        print(f"Actual split ratio: {actual_ratio}")

        return train, val, test

    def _calculate_pert_sizes(self, max_pert: int) -> Dict[int, int]:
        print(f"Calculating sizes for {max_pert} perturbation levels...")
        pert_sizes = {}
        remaining_size = self.size

        # First, allocate all instances of the smallest perturbation (level 1)
        min_pert_level = min(self.pert_count_index.keys())
        min_pert_count = len(self.pert_count_index[min_pert_level])
        pert_sizes[min_pert_level] = min(min_pert_count, remaining_size)
        remaining_size -= pert_sizes[min_pert_level]

        if remaining_size > 0:
            # Prepare to distribute equally across the remaining perturbation levels
            other_pert_levels = [
                i for i in range(2, max_pert + 1)
            ]  # Exclude the minimum perturbation level (1)

            # Allocate perturbations in equal amounts across the levels
            # Continue until we run out of remaining size or perturbations
            while remaining_size > 0 and other_pert_levels:
                max_possible_per_level = min(
                    len(self.pert_count_index[level]) for level in other_pert_levels
                )
                pert_per_level = min(
                    max_possible_per_level, remaining_size // len(other_pert_levels)
                )

                if pert_per_level == 0:
                    break

                for level in other_pert_levels:
                    pert_sizes[level] = pert_per_level
                    remaining_size -= pert_per_level

                # Remove levels that have been fully allocated
                other_pert_levels = [
                    level for level in other_pert_levels if remaining_size > 0
                ]

        print(f"Perturbation sizes: {pert_sizes}")
        return pert_sizes

    def _save_subset_indices(self):
        print("Saving subset indices...")
        subset_data = {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "selected_indices": self.selected_indices,
            "subset_info": self.subset_info,
            "seed": self.seed,
            "size": self.size,
        }
        with open(osp.join(self.cache_dir, "subset_indices.json"), "w") as f:
            json.dump(subset_data, f)

    def _load_cached_subset(
        self,
    ) -> Optional[tuple[List[int], List[int], List[int], List[int], Dict[str, int]]]:
        cache_file = osp.join(self.cache_dir, "subset_indices.json")
        if osp.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            if cached_data["seed"] == self.seed and cached_data["size"] == self.size:
                return (
                    cached_data["train_indices"],
                    cached_data["val_indices"],
                    cached_data["test_indices"],
                    cached_data["selected_indices"],
                    cached_data["subset_info"],
                )
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num,
        )

    def _calculate_subset_info(self):
        print("Calculating subset info...")
        pert_counts = {i: 0 for i in self.pert_count_index.keys()}

        for pert_level, indices in self.pert_count_index.items():
            pert_counts[pert_level] = len(set(self.selected_indices) & set(indices))

        self.subset_info = {
            "total_samples": self.size,
            "train_samples": len(self.train_indices),
            "val_samples": len(self.val_indices),
            "test_samples": len(self.test_indices),
            "pert_samples": pert_counts,
        }

    def get_subset_info(self) -> Dict[str, int]:
        if self.subset_info is None:
            self._calculate_subset_info()
        return self.subset_info

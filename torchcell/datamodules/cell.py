# torchcell/datamodules/cell.py
# [[torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/cell.py
# Test file: torchcell/datamodules/test_cell.py
import json
import os
from typing import List, Dict, Union, Optional, Set
import pandas as pd
import lightning as L
import torch
import random
from collections import defaultdict
import logging
import os.path as osp
from torch_geometric.loader import DataLoader, PrefetchLoader
from torchcell.datamodels import ModelStrict
from pydantic import BaseModel, model_validator, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class IndexSplit(ModelStrict):
    indices: List[int] = Field(..., description="Must be sorted in ascending order")
    count: int

    @model_validator(mode="before")
    @classmethod
    def check_sorted_indices(cls, values):
        indices = values.get("indices")
        if indices and not all(
            indices[i] <= indices[i + 1] for i in range(len(indices) - 1)
        ):
            raise ValueError("Indices must be sorted in ascending order")
        return values

    def __repr__(self):
        max_indices = 3
        indices_str = (
            f"[{', '.join(map(str, self.indices[:max_indices]))}"
            f"{', ...' if len(self.indices) > max_indices else ''}]"
        )
        return f"IndexSplit(indices={indices_str}, count={self.count})"


class DatasetSplit(BaseModel):
    phenotype_label_index: Optional[Dict[str, IndexSplit]] = None
    perturbation_count_index: Optional[Dict[int, IndexSplit]] = None
    dataset_name_index: Optional[Dict[str, IndexSplit]] = None


class DataModuleIndexDetails(ModelStrict):
    methods: List[str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit

    def df_summary(self):
        data = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)

        for split in ["train", "val", "test"]:
            split_data = getattr(self, split)
            for index_type, index_data in split_data.dict().items():
                if index_data is not None:
                    for key, index_split in index_data.items():
                        # Handle both IndexSplit objects and dictionaries
                        if isinstance(index_split, dict):
                            count = index_split.get("count")
                        else:
                            count = index_split.count

                        data[(index_type, key)][split] = count
                        totals[(index_type, key)] += count

        summary_data = []
        for (index_type, key), splits in data.items():
            total = totals[(index_type, key)]
            for split in ["train", "val", "test"]:
                count = splits[split]
                ratio = count / total if total > 0 else 0
                summary_data.append(
                    {
                        "split": split,
                        "index_type": index_type,
                        "key": key,
                        "count": count,
                        "ratio": ratio,
                        "total": total,
                    }
                )

        df = pd.DataFrame(summary_data)

        # Create a categorical column for 'split' with the desired order
        df["split"] = pd.Categorical(
            df["split"], categories=["train", "val", "test"], ordered=True
        )

        # Sort the DataFrame
        df = df.sort_values(["split", "index_type", "key"])

        df["ratio"] = df["ratio"].round(3)
        df = df.reset_index(drop=True)

        return df

    def __str__(self):
        df = self.df_summary()
        if df.empty:
            return "DataModuleIndexDetails(empty)"
        return df.to_string()


class DataModuleIndex(ModelStrict):
    train: List[int] = Field(..., description="Must be sorted in ascending order")
    val: List[int] = Field(..., description="Must be sorted in ascending order")
    test: List[int] = Field(..., description="Must be sorted in ascending order")

    @model_validator(mode="before")
    @classmethod
    def check_sorted_and_unique_indices(cls, values):
        for split in ["train", "val", "test"]:
            indices = values.get(split, [])
            if not all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1)):
                raise ValueError(f"{split} indices must be sorted in ascending order")

        all_indices = (
            values.get("train", []) + values.get("val", []) + values.get("test", [])
        )
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Indices in train, val, and test must not overlap")

        return values

    def __repr__(self):
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        return f"DataModuleIndex(train={train_str_index}, val={val_str_index}, test={test_str_index})"

    def __str__(self):
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        train_str = train_str_index + f" ({len(self.train)} indices)"
        val_str = val_str_index + f" ({len(self.val)} indices)"
        test_str = test_str_index + f" ({len(self.test)} indices)"
        return f"DataModuleIndex(train={train_str}, val={val_str}, test={test_str})"


class DatasetIndexSplit(ModelStrict):
    train: dict[Union[str, int], list[int]] = None
    val: dict[Union[str, int], list[int]] = None
    test: dict[Union[str, int], list[int]] = None


def overlap_dataset_index_split(
    dataset_index: dict[str | int, list[int]], data_module_index: DataModuleIndex
) -> DatasetIndexSplit:
    train_set = set(data_module_index.train)
    val_set = set(data_module_index.val)
    test_set = set(data_module_index.test)

    train_dict = {}
    val_dict = {}
    test_dict = {}

    for dataset_name, indices in dataset_index.items():
        train_indices = sorted(list(set(indices) & train_set))
        val_indices = sorted(list(set(indices) & val_set))
        test_indices = sorted(list(set(indices) & test_set))

        if train_indices:
            train_dict[dataset_name] = train_indices
        if val_indices:
            val_dict[dataset_name] = val_indices
        if test_indices:
            test_dict[dataset_name] = test_indices

    return DatasetIndexSplit(
        train=train_dict if train_dict else None,
        val=val_dict if val_dict else None,
        test=test_dict if test_dict else None,
    )


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        cache_dir: str = "cache",
        batch_size: int = 32,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch: bool = False,
        split_indices: Union[str, List[str], None] = None,
        follow_batch: Optional[list] = None,
        train_shuffle: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.train_shuffle = train_shuffle
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.split_indices = (
            split_indices
            if isinstance(split_indices, list)
            else [split_indices] if split_indices else []
        )
        self._index = None
        self._index_details = None
        if follow_batch is None:
            self.follow_batch = ["x", "x_pert"]
        else:
            self.follow_batch = follow_batch

        # Compute index during initialization
        self.index
        self.index_details

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
        os.makedirs(self.cache_dir, exist_ok=True)
        index_file = osp.join(self.cache_dir, f"index_seed_{self.random_seed}.json")
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}.json"
        )
        if osp.exists(index_file) and osp.exists(details_file):
            try:
                with open(index_file, "r") as f:
                    log.info(f"Loading index from {index_file}")
                    index_dict = json.load(f)
                    self._index = DataModuleIndex(**index_dict)
                with open(details_file, "r") as f:
                    log.info(f"Loading index details from {details_file}")
                    details_dict = json.load(f)
                    self._index_details = DataModuleIndexDetails(**details_dict)
            except Exception as e:
                print(f"Error loading index or details: {e}. Regenerating...")
                self._compute_and_save_index(index_file, details_file)
        else:
            self._compute_and_save_index(index_file, details_file)

    def _compute_and_save_index(self, index_file, details_file):
        log.info("Generating detailed index...")
        random.seed(self.random_seed)

        all_indices = set(range(len(self.dataset)))
        split_data = {
            "train": defaultdict(set),
            "val": defaultdict(set),
            "test": defaultdict(set),
        }

        # First, split each index independently
        for index_name in self.split_indices:
            original_index = getattr(self.dataset, index_name)
            for key, indices in original_index.items():
                indices = list(indices)
                random.shuffle(indices)
                num_samples = len(indices)
                num_train = int(self.train_ratio * num_samples)
                num_val = int(self.val_ratio * num_samples)

                split_data["train"][index_name].update(indices[:num_train])
                split_data["val"][index_name].update(
                    indices[num_train : num_train + num_val]
                )
                split_data["test"][index_name].update(indices[num_train + num_val :])

        # Then, create initial final splits
        final_splits = {
            "train": all_indices.intersection(
                *[split_data["train"][index] for index in self.split_indices]
            ),
            "val": all_indices.intersection(
                *[split_data["val"][index] for index in self.split_indices]
            ),
            "test": all_indices.intersection(
                *[split_data["test"][index] for index in self.split_indices]
            ),
        }

        # Sophisticated assignment of remaining indices
        remaining = all_indices - (
            final_splits["train"] | final_splits["val"] | final_splits["test"]
        )
        target_ratios = {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": 1 - self.train_ratio - self.val_ratio,
        }

        for index_name in self.split_indices:
            original_index = getattr(self.dataset, index_name)
            for key, indices in original_index.items():
                key_remaining = set(indices) & remaining
                if not key_remaining:
                    continue

                current_counts = {
                    split: len(set(indices) & final_splits[split])
                    for split in ["train", "val", "test"]
                }
                total_count = sum(current_counts.values()) + len(key_remaining)

                for idx in key_remaining:
                    target_counts = {
                        split: int(total_count * ratio)
                        for split, ratio in target_ratios.items()
                    }
                    best_split = min(
                        ["train", "val", "test"],
                        key=lambda x: (current_counts[x] - target_counts[x])
                        / target_counts[x],
                    )
                    final_splits[best_split].add(idx)
                    current_counts[best_split] += 1
                    remaining.remove(idx)

        # Create DataModuleIndexDetails object
        self._index_details = DataModuleIndexDetails(
            methods=self.split_indices,
            train=DatasetSplit(),
            val=DatasetSplit(),
            test=DatasetSplit(),
        )

        for split in ["train", "val", "test"]:
            for index_name in self.split_indices:
                original_index = getattr(self.dataset, index_name)
                split_data = {}
                for key, indices in original_index.items():
                    intersect = sorted(list(set(indices) & final_splits[split]))
                    split_data[key] = IndexSplit(
                        indices=intersect, count=len(intersect)
                    )
                setattr(getattr(self._index_details, split), index_name, split_data)

        # Create DataModuleIndex object
        self._index = DataModuleIndex(
            train=sorted(list(final_splits["train"])),
            val=sorted(list(final_splits["val"])),
            test=sorted(list(final_splits["test"])),
        )

        # Save the index and details separately
        with open(index_file, "w") as f:
            json.dump(self._index.dict(), f, indent=2)
        with open(details_file, "w") as f:
            json.dump(self._index_details.dict(), f, indent=2)

    def _cached_files_exist(self):
        index_file = osp.join(self.cache_dir, f"index_seed_{self.random_seed}.json")
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}.json"
        )
        return osp.exists(index_file) and osp.exists(details_file)

    def setup(self, stage=None):
        train_index = self.index.train
        val_index = self.index.val
        test_index = self.index.test

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_index)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_index)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_index)

    def _get_dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=self.pin_memory,
            follow_batch=self.follow_batch,
            timeout=10800,
            multiprocessing_context=(
                "spawn" if self.num_workers > 0 else None
            ),  # Add this
        )
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self):
        return self._get_dataloader(self.dataset)


if __name__ == "__main__":
    pass

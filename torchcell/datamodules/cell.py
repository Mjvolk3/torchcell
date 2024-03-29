# torchcell/datamodules/cell.py
# [[torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/cell.py
# Test file: torchcell/datamodules/test_cell.py

import json
import os
import lightning as L
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        cache_dir: str = "cache",
        batch_size: int = 32,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.train_epoch_size = int(
            len(self.dataset) * self.train_ratio / self.batch_size
        )

    def setup(self, stage=None):
        # Set the random seed for reproducibility
        torch.manual_seed(self.random_seed)

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check if cached indices exist
        cached_indices_file = os.path.join(self.cache_dir, "cached_indices.json")
        if os.path.exists(cached_indices_file):
            try:
                # Load cached indices from file
                with open(cached_indices_file, "r") as f:
                    log.info(f"Loading cached indices from {cached_indices_file}")
                    cached_data = json.load(f)
                    train_indices = cached_data["train_indices"]
                    val_indices = cached_data["val_indices"]
                    test_indices = cached_data["test_indices"]
                    phenotype_label_index = (
                        self.dataset.phenotype_label_index
                    )  # Assign the phenotype_label_index
            except json.decoder.JSONDecodeError:
                # If JSON decoding fails, regenerate the cached indices
                print(
                    "Cached indices JSON is corrupted. Regenerating cached indices..."
                )
                os.remove(cached_indices_file)
                self.setup(stage)
                return
        else:
            log.info("Generating indices for train, val, and test sets...")
            # Get the phenotype label index from the dataset
            phenotype_label_index = self.dataset.phenotype_label_index

            # Split the indices for each phenotype label into train, val, and test sets
            train_indices = []
            val_indices = []
            test_indices = []

            for label, indices in phenotype_label_index.items():
                num_samples = len(indices)
                num_train = int(self.train_ratio * num_samples)
                num_val = int(self.val_ratio * num_samples)

                # Shuffle the indices before subsetting
                shuffled_indices = torch.randperm(num_samples).tolist()
                label_indices = [indices[i] for i in shuffled_indices]

                label_train_indices = label_indices[:num_train]
                label_val_indices = label_indices[num_train : num_train + num_val]
                label_test_indices = label_indices[num_train + num_val :]

                train_indices.extend(label_train_indices)
                val_indices.extend(label_val_indices)
                test_indices.extend(label_test_indices)

            # Cache the computed indices
            cached_data = {
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
            }
            with open(cached_indices_file, "w") as f:
                json.dump(cached_data, f)

        # Create subset datasets for train, val, and test
        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_indices)

        # Create phenotype label indices for each split
        (self.train_phenotype_label_index, self.train_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(
                train_indices, phenotype_label_index
            )
        )
        (self.val_phenotype_label_index, self.val_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(val_indices, phenotype_label_index)
        )
        (self.test_phenotype_label_index, self.test_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(
                test_indices, phenotype_label_index
            )
        )

    def create_subset_phenotype_label_index(
        self, subset_indices, phenotype_label_index
    ):
        subset_phenotype_label_index = {}
        subset_phenotype_label_index_mapped = {}

        for label, indices in phenotype_label_index.items():
            subset_indices_set = set(subset_indices)
            subset_label_indices = [idx for idx in indices if idx in subset_indices_set]
            subset_phenotype_label_index[label] = subset_label_indices
            subset_phenotype_label_index_mapped[label] = [
                subset_indices.index(idx) for idx in subset_label_indices
            ]

        return subset_phenotype_label_index, subset_phenotype_label_index_mapped

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def all_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x", "x_pert"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )


if __name__ == "__main__":
    pass

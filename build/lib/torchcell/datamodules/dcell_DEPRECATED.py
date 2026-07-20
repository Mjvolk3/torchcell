"""Deprecated Lightning datamodule for the DCell dataset (no longer used)."""

# torchcell/datamodules/dcell.py
# [[torchcell.datamodules.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/dcell.py
# Test file: torchcell/datamodules/test_dcell.py

import lightning as L
import torch
from torch_geometric.loader import DataLoader


class DCellDataModule(L.LightningDataModule):
    """Deprecated datamodule splitting a DCell dataset into train/val/test loaders."""

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Store the dataset, loader options, and fixed train/val/test ratios."""
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_epoch_size = None
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.train_epoch_size = int(
            len(self.dataset) * self.train_ratio / self.batch_size
        )

    def setup(self, stage=None):
        """Randomly split the dataset into train, val, and test subsets."""
        # Split the dataset into train, val, and test sets
        num_train = int(self.train_ratio * len(self.dataset))
        num_val = int(self.val_ratio * len(self.dataset))
        num_test = len(self.dataset) - num_train - num_val

        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            torch.utils.data.random_split(self.dataset, [num_train, num_val, num_test])
        )

    def train_dataloader(self):
        """Return the shuffled training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def test_dataloader(self):
        """Return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"],
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )


if __name__ == "__main__":
    pass

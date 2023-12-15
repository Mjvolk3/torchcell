# torchcell/datamodules/cell.py
# [[torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/cell.py
# Test file: torchcell/datamodules/test_cell.py

import lightning as L
import torch
from torch_geometric.loader import DataLoader


class CellDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
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
        # Split the dataset into train, val, and test sets
        num_train = int(self.train_ratio * len(self.dataset))
        num_val = int(self.val_ratio * len(self.dataset))
        num_test = len(self.dataset) - num_train - num_val

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(self.dataset, [num_train, num_val, num_test])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"]
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"]
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            follow_batch=["x"]
            # follow_batch=["x", "x_pert", "x_one_hop_pert"],
        )


if __name__ == "__main__":
    pass

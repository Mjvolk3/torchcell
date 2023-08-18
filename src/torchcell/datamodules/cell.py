# src/torchcell/datamodules/cell.py
# [[src.torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datamodules/cell.py
# Test file: src/torchcell/datamodules/test_cell.py

import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader


class CellDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Split the dataset into train, val, and test sets
        num_train = int(0.8 * len(self.dataset))
        num_val = int(0.1 * len(self.dataset))
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    pass

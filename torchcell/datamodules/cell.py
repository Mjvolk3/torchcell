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
import os.path as osp
from torch_geometric.loader import PrefetchLoader
from torchcell.datamodels import ModelStrict
from typing import List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DataModuleIndex(ModelStrict):
    train: List[int]
    val: List[int]
    test: List[int]
    
    @property
    def train_ratio(self):
        return len(self.train) / (len(self.train) + len(self.val) + len(self.test))
    
    @property
    def val_ratio(self):
        return len(self.val) / (len(self.train) + len(self.val) + len(self.test))
    
    @property
    def test_ratio(self):
        return len(self.test) / (len(self.train) + len(self.val) + len(self.test))


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
    ):
        super().__init__()
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.train_epoch_size = int(
            len(self.dataset) * self.train_ratio / self.batch_size
        )
        self._index: DataModuleIndex = None

    @property
    def index(self) -> DataModuleIndex:
        if self._index is None:
            self._load_or_compute_index()
        return self._index

    def _load_or_compute_index(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        index_file = osp.join(self.cache_dir, "index.json")
        if osp.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    log.info(f"Loading index from {index_file}")
                    index_dict = json.load(f)
                    # Handle backward compatibility
                    if "train_indices" in index_dict:
                        index_dict = {
                            "train": index_dict["train_indices"],
                            "val": index_dict["val_indices"],
                            "test": index_dict["test_indices"],
                        }
                    self._index = DataModuleIndex(**index_dict)
            except Exception as e:
                print(f"Error loading index: {e}. Regenerating...")
                self._compute_and_save_index(index_file)
        else:
            self._compute_and_save_index(index_file)

    def _compute_and_save_index(self, index_file):
        log.info("Generating index for train, val, and test sets...")
        torch.manual_seed(self.random_seed)
        phenotype_label_index = self.dataset.phenotype_label_index

        train_index, val_index, test_index = [], [], []

        for label, indices in tqdm(phenotype_label_index.items()):
            num_samples = len(indices)
            num_train = int(self.train_ratio * num_samples)
            num_val = int(self.val_ratio * num_samples)

            shuffled_indices = torch.randperm(num_samples).tolist()
            label_indices = [indices[i] for i in shuffled_indices]

            train_index.extend(label_indices[:num_train])
            val_index.extend(label_indices[num_train : num_train + num_val])
            test_index.extend(label_indices[num_train + num_val :])

        self._index = DataModuleIndex(
            train=train_index, val=val_index, test=test_index
        )

        with open(index_file, "w") as f:
            json.dump(self._index.dict(), f)

    def setup(self, stage=None):
        train_index = self.index.train
        val_index = self.index.val
        test_index = self.index.test
        phenotype_label_index = self.dataset.phenotype_label_index

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_index)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_index)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_index)

        (self.train_phenotype_label_index, self.train_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(
                train_index, phenotype_label_index
            )
        )
        (self.val_phenotype_label_index, self.val_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(val_index, phenotype_label_index)
        )
        (self.test_phenotype_label_index, self.test_subset_phenotype_label_index) = (
            self.create_subset_phenotype_label_index(
                test_index, phenotype_label_index
            )
        )

    def create_subset_phenotype_label_index(
        self, subset_index, phenotype_label_index
    ):
        subset_phenotype_label_index = {}
        subset_phenotype_label_index_mapped = {}

        log.info("Creating subset phenotype label index...")
        for label, indices in tqdm(phenotype_label_index.items()):
            subset_index_set = set(subset_index)
            subset_label_indices = [idx for idx in indices if idx in subset_index_set]
            subset_phenotype_label_index[label] = subset_label_indices
            subset_phenotype_label_index_mapped[label] = [
                subset_index.index(idx) for idx in subset_label_indices
            ]

        return subset_phenotype_label_index, subset_phenotype_label_index_mapped

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

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         follow_batch=["x", "x_pert"],
    #         # follow_batch=["x", "x_pert", "x_one_hop_pert"],
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         follow_batch=["x", "x_pert"],
    #         # follow_batch=["x", "x_pert", "x_one_hop_pert"],
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         follow_batch=["x", "x_pert"],
    #         # follow_batch=["x", "x_pert", "x_one_hop_pert"],
    #     )

    # def all_dataloader(self):
    #     return DataLoader(
    #         self.dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         follow_batch=["x", "x_pert"],
    #         # follow_batch=["x", "x_pert", "x_one_hop_pert"],
    #     )


if __name__ == "__main__":
    pass

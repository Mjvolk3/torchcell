---
id: 7utg0wsflx2hsieybi6i39e
title: Cell
desc: ''
updated: 1711494690750
created: 1692324679287
---
- Describes data splitting and loading.

## 2024.03.26 - Uncertain if is is Feasible to Keep a Base Class

```python
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

        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            torch.utils.data.random_split(self.dataset, [num_train, num_val, num_test])
        )

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


if __name__ == "__main__":
    pass
```

## 2026.09.30 - Split integrity fixes (issue #517)

Six behaviors pinned as findings in Phase 11 are now refused or corrected (issue #517, `tests/torchcell/datamodules/test_cell.py`).

- **Balancing targets partition the key.** The reassignment of records the split indices disagree on sized the test target as `int(n * (1 - train - val))`; `1 - 0.8 - 0.1 = 0.09999999999999995`, so a 10-record key got a test target of 0 and `ZeroDivisionError`. Targets are now `int(0.8 n)`, `int(0.1 n)` and the remainder, the same partition the first-pass slicing uses. A key that still has a zero target (fewer than 10 records) and holds a record left to balance raises a `ValueError` naming the key, its size and the empty split. Ten is the smallest key that balances. Tests: `test_two_indices_balance_ten_record_keys_to_eight_one_one`, `test_two_indices_refuse_a_nine_record_key_with_a_disputed_record`, `test_multiply_keyed_records_are_reassigned_exactly_once` (20-record key now 16 / 2 / 2, was 17 / 2 / 1).
- **Orphans refused.** A record of the pool under no split-index key was placed in no split, silently. After pinning, any unplaced record now raises `ValueError` listing the first ten indices and the total. A record placed by a pin is not an orphan. Tests: `test_records_under_no_key_are_refused_with_their_indices`, `test_unkeyed_records_placed_by_a_pin_are_not_orphans`.
- **`split_indices` required in effect.** `None` (the default) or `[]` produced three empty splits. The default stays `None` because legacy callers omit the argument (`torchcell/trainers/cell.py`, experiments `002-dmi-tmi` query_explore 1 to 3 and traditional_ml_dataset(_it_speed), `smf-dmf-tmf-001` deep_set and traditional_ml_dataset, and the `DEPRECATED_costanzo_*` scripts); construction now raises `ValueError` before any cache is written. All 87 other call sites pass it. Test: `test_no_split_indices_is_refused_at_construction`.
- **Custom `collate_fn` honored.** PyG's `DataLoader` pops `collate_fn` and installs its own `Collater`, so experiment 006's `LazyCollater` never ran. With a `collate_fn` the module now builds `torch.utils.data.DataLoader` with it (same worker options, no `follow_batch`; the collate owns batching); without one the PyG loader is unchanged. Test: `test_custom_collate_fn_is_honored_through_a_torch_loader`.
- **Empty details print `"DataModuleIndexDetails(empty)"`.** `df_summary` builds its frame with explicit columns and numeric dtypes, so an empty details object gives an empty six-column frame instead of `KeyError('split')`. Test: `test_details_str_of_an_empty_summary_is_the_empty_form`.
- **`overlap_dataset_index_split` allows an empty split.** A split with no overlapping dataset is `{}` (the fields are plain required dicts); it used to pass `None`, which `DatasetIndexSplit` refused. `torchcell/viz/datamodules.py` already skips a falsy split. Test: `test_overlap_dataset_index_split_gives_an_empty_dict_for_an_empty_split`.

A split recomputed on a key with a disputed record can now differ from one cached before this change (the test target grew by one for keys whose size is a multiple of ten); cached index files load unchanged.

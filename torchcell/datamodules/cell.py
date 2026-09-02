"""Lightning data module and split-index models for cell datasets."""

# torchcell/datamodules/cell.py
# [[torchcell.datamodules.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodules/cell.py
# Test file: torchcell/datamodules/test_cell.py
import hashlib
import json
import logging
import os
import os.path as osp
import random
import resource
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

import lightning as L
import pandas as pd
import torch
import torch.multiprocessing
from pydantic import BaseModel, Field, model_validator
from torch_geometric.loader import DataLoader, PrefetchLoader

from torchcell.datamodels import ModelStrict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# FILE-DESCRIPTOR EXHAUSTION KILLS LONG RUNS, AND IT KILLS THEM SILENTLY LATE.
# Torch's default multiprocessing sharing strategy on Linux is `file_descriptor`: every
# shared tensor storage handed from a worker to the main process travels as its OWN open
# fd over a Unix socket. A PyG HeteroData batch is not one tensor, it is dozens (each edge
# store carries index/attr tensors), so a single in-flight batch costs dozens of fds, and
# num_workers x prefetch_factor batches are in flight at once, with the pin_memory thread
# holding more. Against the default RLIMIT_NOFILE the process eventually cannot receive
# another handle and torch raises, from deep inside the pin-memory loop:
#     RuntimeError: received 0 items of ancdata
# MEASURED: IGB cabbi job 2368337_4 (Q_point seed 2) died exactly this way at epoch 2,580
# after 21.5 h, and its sibling 2368339 (the incumbent resume) died at epoch ~12,230 with
# `DataLoader timed out after 10800 seconds`, which is the same exhaustion presenting as a
# hang rather than a raise. Both had run for a day first, so nothing surfaces in a canary.
#
# `file_system` passes storages by name through /dev/shm instead of by fd, so the fd count
# no longer scales with in-flight tensors. Its documented cost is that a HARD-killed
# process can leak shm files; that is the right trade against losing a multi-day run.
# The rlimit raise is belt and braces for the fds we still open (LMDB, sockets, logs).
#
# This is module scope on purpose: the strategy is process-global and must be set before
# ANY DataLoader is constructed, and every consumer of a cell dataset imports this module.
torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore[no-untyped-call]
_NOFILE_SOFT, _NOFILE_HARD = resource.getrlimit(resource.RLIMIT_NOFILE)
if _NOFILE_SOFT < _NOFILE_HARD:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_NOFILE_HARD, _NOFILE_HARD))


def _worker_init(worker_id: int) -> None:
    """Apply the fd-sharing policy inside a dataloader worker.

    Setting it in the parent alone is NOT enough, and that is a property of torch rather
    than a precaution: ``torch.multiprocessing.reductions.reduce_storage`` calls
    ``get_sharing_strategy()`` in whichever process is PICKLING, and the process pickling
    a batch is the worker. Workers here start under a ``spawn`` context, so they get a
    fresh interpreter that does not inherit the parent's setting and does not necessarily
    import this module. Without this hook the parent would receive by name while every
    worker kept sending by fd, which is the exact configuration that failed.

    Args:
        worker_id: Dataloader-assigned worker index. Unused; the policy is process-wide.
    """
    torch.multiprocessing.set_sharing_strategy("file_system")  # type: ignore[no-untyped-call]
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class IndexSplit(ModelStrict):
    """A sorted list of indices together with its element count."""

    indices: list[int] = Field(..., description="Must be sorted in ascending order")
    count: int

    @model_validator(mode="before")
    @classmethod
    def check_sorted_indices(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that the indices list is sorted in ascending order."""
        indices = values.get("indices")
        if indices and not all(
            indices[i] <= indices[i + 1] for i in range(len(indices) - 1)
        ):
            raise ValueError("Indices must be sorted in ascending order")
        return values

    def __repr__(self) -> str:
        """Return a truncated representation showing the first few indices."""
        max_indices = 3
        indices_str = (
            f"[{', '.join(map(str, self.indices[:max_indices]))}"
            f"{', ...' if len(self.indices) > max_indices else ''}]"
        )
        return f"IndexSplit(indices={indices_str}, count={self.count})"


class DatasetSplit(BaseModel):
    """Per-split index groupings keyed by phenotype, perturbation count, or dataset."""

    phenotype_label_index: dict[str, IndexSplit] | None = None
    perturbation_count_index: dict[int, IndexSplit] | None = None
    dataset_name_index: dict[str, IndexSplit] | None = None


class DataModuleIndexDetails(ModelStrict):
    """Detailed per-split index breakdown with summary reporting."""

    methods: list[str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit

    def df_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing counts and ratios per split and index key."""
        data: defaultdict[tuple[str, Any], defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        totals: defaultdict[tuple[str, Any], int] = defaultdict(int)

        for split in ["train", "val", "test"]:
            split_data = getattr(self, split)
            for index_type, index_data in split_data.dict().items():
                if index_data is not None:
                    for key, index_split in index_data.items():
                        # Handle both IndexSplit objects and dictionaries
                        if isinstance(index_split, dict):
                            count = cast(int, index_split.get("count"))
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

    def __str__(self) -> str:
        """Return the summary DataFrame rendered as a string."""
        df = self.df_summary()
        if df.empty:
            return "DataModuleIndexDetails(empty)"
        return df.to_string()


class DataModuleIndex(ModelStrict):
    """Train/val/test index lists that are sorted and mutually non-overlapping."""

    train: list[int] = Field(..., description="Must be sorted in ascending order")
    val: list[int] = Field(..., description="Must be sorted in ascending order")
    test: list[int] = Field(..., description="Must be sorted in ascending order")

    @model_validator(mode="before")
    @classmethod
    def check_sorted_and_unique_indices(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate each split is sorted and the splits do not overlap."""
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

    def __repr__(self) -> str:
        """Return a truncated representation of the train/val/test indices."""
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        return f"DataModuleIndex(train={train_str_index}, val={val_str_index}, test={test_str_index})"

    def __str__(self) -> str:
        """Return a truncated representation with per-split index counts."""
        max_indices = 3
        train_str_index = f"[{', '.join(map(str, self.train[:max_indices]))}{', ...' if len(self.train) > max_indices else ''}]"
        val_str_index = f"[{', '.join(map(str, self.val[:max_indices]))}{', ...' if len(self.val) > max_indices else ''}]"
        test_str_index = f"[{', '.join(map(str, self.test[:max_indices]))}{', ...' if len(self.test) > max_indices else ''}]"
        train_str = train_str_index + f" ({len(self.train)} indices)"
        val_str = val_str_index + f" ({len(self.val)} indices)"
        test_str = test_str_index + f" ({len(self.test)} indices)"
        return f"DataModuleIndex(train={train_str}, val={val_str}, test={test_str})"


class DatasetIndexSplit(ModelStrict):
    """Per-dataset index lists grouped by train/val/test split."""

    train: dict[str | int, list[int]] = None  # type: ignore[assignment]  # pydantic field: keep declared type + None default (changing either alters validation/runtime)
    val: dict[str | int, list[int]] = None  # type: ignore[assignment]  # pydantic field: keep declared type + None default (changing either alters validation/runtime)
    test: dict[str | int, list[int]] = None  # type: ignore[assignment]  # pydantic field: keep declared type + None default (changing either alters validation/runtime)


def overlap_dataset_index_split(
    dataset_index: dict[str | int, list[int]], data_module_index: DataModuleIndex
) -> DatasetIndexSplit:
    """Intersect each dataset's indices with the train/val/test split indices."""
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
        train=train_dict if train_dict else None,  # type: ignore[arg-type]  # field accepts None default (see DatasetIndexSplit); type kept as-is
        val=val_dict if val_dict else None,  # type: ignore[arg-type]  # field accepts None default (see DatasetIndexSplit); type kept as-is
        test=test_dict if test_dict else None,  # type: ignore[arg-type]  # field accepts None default (see DatasetIndexSplit); type kept as-is
    )


class CellDataModule(L.LightningDataModule):
    """Lightning data module that splits a cell dataset and builds dataloaders."""

    def __init__(
        self,
        dataset: Any,  # dynamic cell dataset duck-typed by split-index attrs
        cache_dir: str = "cache",
        batch_size: int = 32,
        random_seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        split_indices: str | list[str] | None = None,
        follow_batch: list[str] | None = None,
        train_shuffle: bool = True,
        collate_fn: object | None = None,
        val_batch_size: int | None = None,
        pinned_test_indices: Iterable[int] | None = None,
    ) -> None:
        """Store dataloader/split configuration and compute the split indices.

        ``pinned_test_indices`` forces those record indices into the TEST split, overriding
        the random assignment. It reproduces an EXTERNAL split inside ours -- e.g. Merzbacher
        2025's betaxanthin test ORFs, so their published numbers and ours are computed on the
        same genes. The remaining records are split train/val/test by ``random_seed`` as
        usual, so sweeping the seed still re-rolls train/val while the pinned comparison set
        stays fixed.
        """
        super().__init__()
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.train_shuffle = train_shuffle
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.pinned_test_indices: set[int] = (
            set(pinned_test_indices) if pinned_test_indices is not None else set()
        )
        self.split_indices = (
            split_indices
            if isinstance(split_indices, list)
            else [split_indices]
            if split_indices
            else []
        )
        self._index: DataModuleIndex | None = None
        self._index_details: DataModuleIndexDetails | None = None
        if follow_batch is None:
            self.follow_batch = ["x", "x_pert"]
        else:
            self.follow_batch = follow_batch
        self.collate_fn = collate_fn

        # Compute index during initialization
        self.index
        self.index_details

    @property
    def index(self) -> DataModuleIndex:
        """Return the train/val/test split index, computing or loading it if needed."""
        if self._index is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index  # type: ignore[return-value]  # _load_or_compute_index() always populates _index

    @property
    def index_details(self) -> DataModuleIndexDetails:
        """Return the detailed split-index breakdown, computing or loading if needed."""
        if self._index_details is None or not self._cached_files_exist():
            self._load_or_compute_index()
        return self._index_details  # type: ignore[return-value]  # _load_or_compute_index() always populates _index_details

    def _pin_tag(self) -> str:
        """Cache-key suffix identifying the pinned test set, empty when there is none.

        The cached index is keyed by seed alone, so WITHOUT this a pinned run would silently
        load an UNPINNED `index_seed_42.json` left by an earlier run and train on the wrong
        split -- with no error, and with the pinned genes back in train. The tag hashes the
        pinned index set so any change to it (or removing the pin) selects a different file.
        """
        if not self.pinned_test_indices:
            return ""
        payload = ",".join(map(str, sorted(self.pinned_test_indices))).encode()
        return f"_pin{len(self.pinned_test_indices)}-{hashlib.sha256(payload).hexdigest()[:8]}"

    def _load_or_compute_index(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        tag = self._pin_tag()
        index_file = osp.join(
            self.cache_dir, f"index_seed_{self.random_seed}{tag}.json"
        )
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}{tag}.json"
        )
        if osp.exists(index_file) and osp.exists(details_file):
            try:
                with open(index_file) as f:
                    log.info(f"Loading index from {index_file}")
                    index_dict = json.load(f)
                    self._index = DataModuleIndex(**index_dict)
                with open(details_file) as f:
                    log.info(f"Loading index details from {details_file}")
                    details_dict = json.load(f)
                    self._index_details = DataModuleIndexDetails(**details_dict)
            except Exception as e:
                print(f"Error loading index or details: {e}. Regenerating...")
                self._compute_and_save_index(index_file, details_file)
        else:
            self._compute_and_save_index(index_file, details_file)

    def _compute_and_save_index(self, index_file: str, details_file: str) -> None:
        log.info("Generating detailed index...")
        random.seed(self.random_seed)

        all_indices = set(range(len(self.dataset)))
        split_data: dict[str, defaultdict[str, set[int]]] = {
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

        # A record may appear under SEVERAL keys of the same split index, and those keys
        # are shuffled independently -- so it can be drawn into train under one key and
        # into val under another, landing in two of the intersections above. That is a
        # train/val/test leak, and `DataModuleIndex` rightly refuses it.
        #
        # It is not hypothetical: under deletion-keyed aggregation one genotype group
        # holds records with DIFFERENT perturbation counts (a betaxanthin record carries
        # its 4-gene cassette + the deletion = 5, a beta-carotene record 4, a metabolome
        # record 1), so `perturbation_count_index` files that single record under three
        # keys. The same happens in `phenotype_label_index` for any multi-modality
        # genotype.
        #
        # Conflicts are pulled OUT of every split and returned to `remaining`, so the
        # ratio-balancing assignment below places each conflicted record exactly once.
        # When no record is multiply-keyed this is a no-op and the split is unchanged.
        conflicted = (
            (final_splits["train"] & final_splits["val"])
            | (final_splits["train"] & final_splits["test"])
            | (final_splits["val"] & final_splits["test"])
        )
        if conflicted:
            log.info(
                f"{len(conflicted)} records were assigned to more than one split by "
                "independently-shuffled index keys; reassigning them exactly once."
            )
            for split in ("train", "val", "test"):
                final_splits[split] -= conflicted

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
                        key=lambda x: (
                            (current_counts[x] - target_counts[x]) / target_counts[x]
                        ),
                    )
                    final_splits[best_split].add(idx)
                    current_counts[best_split] += 1
                    remaining.remove(idx)

        # PINNED TEST SET, applied last so it overrides every ratio-driven assignment above.
        # Reproduces an EXTERNAL split inside ours: the pinned records go to test and are
        # removed from train/val, and everything else keeps its seed-driven assignment. That
        # is what makes a head-to-head comparison possible -- their metric and ours are then
        # computed on the same genes -- while sweeping `random_seed` still re-rolls train/val
        # around a comparison set that never moves.
        #
        # Test therefore ends up LARGER than `1 - train_ratio - val_ratio`, by design; the
        # ratios govern the remaining pool, not the pinned block.
        if self.pinned_test_indices:
            pinned = self.pinned_test_indices & all_indices
            missing = len(self.pinned_test_indices) - len(pinned)
            final_splits["train"] -= pinned
            final_splits["val"] -= pinned
            final_splits["test"] |= pinned
            log.info(
                f"Pinned {len(pinned)} records into test "
                f"({missing} requested indices are outside the dataset). "
                f"Splits: train={len(final_splits['train'])} "
                f"val={len(final_splits['val'])} test={len(final_splits['test'])}"
            )

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
                split_index_data: dict[str | int, IndexSplit] = {}
                for key, indices in original_index.items():
                    intersect = sorted(list(set(indices) & final_splits[split]))
                    split_index_data[key] = IndexSplit(
                        indices=intersect, count=len(intersect)
                    )
                setattr(
                    getattr(self._index_details, split), index_name, split_index_data
                )

        # Create DataModuleIndex object
        self._index = DataModuleIndex(
            train=sorted(list(final_splits["train"])),
            val=sorted(list(final_splits["val"])),
            test=sorted(list(final_splits["test"])),
        )

        # Save the index and details separately
        with open(index_file, "w") as f:
            json.dump(self._index.model_dump(), f, indent=2)
        with open(details_file, "w") as f:
            json.dump(self._index_details.model_dump(), f, indent=2)

    def _cached_files_exist(self) -> bool:
        tag = self._pin_tag()
        index_file = osp.join(
            self.cache_dir, f"index_seed_{self.random_seed}{tag}.json"
        )
        details_file = osp.join(
            self.cache_dir, f"index_details_seed_{self.random_seed}{tag}.json"
        )
        return osp.exists(index_file) and osp.exists(details_file)

    def setup(self, stage: str | None = None) -> None:
        """Build train/val/test Subset datasets from the computed split indices.

        IDEMPOTENT BY DESIGN. Lightning calls ``setup()`` again inside ``trainer.fit()``
        even when the caller already invoked it, so an unconditional rebuild silently
        discarded any post-setup narrowing of ``Subset.indices`` (e.g. the
        ``require_modalities`` filter in experiments/019, which logged "4074 -> 1161 rows"
        while every epoch still ran 128 batches = 4074/32). Rebuilding only when the
        subsets do not yet exist makes the caller's filter survive the second call.
        """
        if (
            getattr(self, "train_dataset", None) is not None
            and getattr(self, "val_dataset", None) is not None
            and getattr(self, "test_dataset", None) is not None
        ):
            return

        self.train_dataset = torch.utils.data.Subset(self.dataset, self.index.train)
        self.val_dataset = torch.utils.data.Subset(self.dataset, self.index.val)
        self.test_dataset = torch.utils.data.Subset(self.dataset, self.index.test)

    def _get_dataloader(
        self,
        dataset: Any,  # dynamic dataset/Subset passed through to the loaders
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> DataLoader | PrefetchLoader:
        # Use provided batch_size or fall back to self.batch_size
        if batch_size is None:
            batch_size = self.batch_size

        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers
            if self.num_workers > 0
            else False,
            "pin_memory": self.pin_memory,
            "follow_batch": self.follow_batch,
            # THE SAME GUARD AS `prefetch_factor` BELOW, and for the same reason -- this one
            # was missed when that fix landed (commit 2cca6d83), so num_workers=0 was still
            # unconstructible afterwards. `timeout` is a WORKER-QUEUE timeout, so at
            # num_workers=0 torch asserts outright:
            #   AssertionError: _SingleProcessDataLoaderIter requires timeout == 0
            # raised from `iter(dataloader)`, i.e. during sanity-check, AFTER the dataset and
            # embeddings have been loaded. A trial therefore burns its full startup cost and
            # then dies, which is why 119 of them could fail in a job that still exited
            # COMPLETED 0:0.
            "timeout": (10800 if self.num_workers > 0 else 0),
            "multiprocessing_context": ("spawn" if self.num_workers > 0 else None),
            # Torch REJECTS a non-None prefetch_factor at num_workers=0 ("could only be
            # specified in multiprocessing"), so this needs the same guard its two
            # neighbours already carry. Without it num_workers=0 is not merely slow, it is
            # unconstructible -- every DataLoader raises ValueError before the first batch.
            #
            # That is not hypothetical: on 2026-07-28 Delta job 20556837 ran NUM_WORKERS=0
            # and all 119 optuna trials died here, ~15s apiece, while the job still exited
            # COMPLETED 0:0. num_workers=0 is worth supporting because on a parallel
            # filesystem the spawn path is the expensive one -- each worker re-imports the
            # whole stack off /work/hdd.
            "prefetch_factor": (self.prefetch_factor if self.num_workers > 0 else None),
            # See `_worker_init`: the sharing strategy is read in the SENDING process, so
            # the workers need it too. Unguarded, unlike its three neighbours above: torch
            # ACCEPTS a worker_init_fn at num_workers=0 (verified) and simply never calls
            # it, so a guard here would be a conditional that can never change behavior.
            "worker_init_fn": _worker_init,
        }

        # Add collate_fn if provided
        if self.collate_fn is not None:
            dataloader_kwargs["collate_fn"] = self.collate_fn

        loader = DataLoader(dataset, **dataloader_kwargs)
        if self.prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        return loader

    def train_dataloader(self) -> DataLoader | PrefetchLoader:
        """Return a dataloader over the training subset."""
        return self._get_dataloader(self.train_dataset, shuffle=self.train_shuffle)

    def val_dataloader(self) -> DataLoader | PrefetchLoader:
        """Return a dataloader over the validation subset."""
        return self._get_dataloader(self.val_dataset, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader | PrefetchLoader:
        """Return a dataloader over the test subset."""
        return self._get_dataloader(self.test_dataset)

    def all_dataloader(self) -> DataLoader | PrefetchLoader:
        """Return a dataloader over the entire dataset."""
        return self._get_dataloader(self.dataset)


if __name__ == "__main__":
    pass

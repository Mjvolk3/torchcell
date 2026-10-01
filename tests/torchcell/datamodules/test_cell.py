# tests/torchcell/datamodules/test_cell.py
# [[tests.torchcell.datamodules.test_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodules/test_cell.py
"""Tests for CellDataModule's split construction, focused on the PINNED TEST SET.

The pin reproduces an external gene-level split inside ours (Merzbacher 2025's betaxanthin
test ORFs) so their published numbers and ours are computed on the same genes. Everything
about that comparison rests on two properties that are invisible at runtime -- pinned records
really are in test, and really are absent from train/val -- so they are asserted here rather
than trusted. A pin that silently failed would produce a *better-looking* score, because the
comparison genes would be back in training.

The second half (from "Split arithmetic, exactly") pins the split itself on hand-built
indices. ``_KeyedDataset(n, **indices)`` exposes ``__len__`` and whichever split indices
it is given. Expected values come from the stdlib RNG the module seeds
(``random.seed(42)``): one shuffle of ``list(range(10))`` gives
[7, 3, 2, 8, 5, 6, 9, 4, 0, 1], and a key of n records is sliced
``int(0.8 n) / int(0.1 n) / rest``, so ten records give train [2..9], val [0], test [1].
Balancing targets partition a key exactly, test taking the remainder, so a 20-record
key targets 16 / 2 / 2 and a 10-record key 8 / 1 / 1 (the old test target
``int(n * 0.09999999999999995)`` was 1 and 0 respectively).
Cache-file tags are ``_pin{n}-`` / ``_sub{n}-`` plus the first eight hex digits of the
sha256 of the payload spelled out in each test.
"""

import json
import os.path as osp
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.loader import PrefetchLoader
from torch_geometric.loader.dataloader import Collater

from torchcell.datamodules.cell import (
    CellDataModule,
    DataModuleIndex,
    DataModuleIndexDetails,
    DatasetSplit,
    IndexSplit,
    overlap_dataset_index_split,
)


class _FakeDataset:
    """Minimal stand-in exposing what CellDataModule's split computation reads.

    `phenotype_label_index` maps a label to the record indices carrying it, mirroring
    `Neo4jCellDataset`. Two labels are used so the intersection logic is exercised rather
    than short-circuited.
    """

    def __init__(self, n: int = 200) -> None:
        self._n = n
        self.phenotype_label_index: dict[str, list[int]] = {
            "a": list(range(0, n, 2)),
            "b": list(range(1, n, 2)),
        }

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Any:
        return idx


def _build(tmp_path: Any, pinned: set[int] | None, seed: int = 42) -> CellDataModule:
    return CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(tmp_path / "cache"),
        split_indices=["phenotype_label_index"],
        random_seed=seed,
        pinned_test_indices=pinned,
    )


def test_no_pin_splits_by_ratio(tmp_path: Any) -> None:
    """Without a pin the split is the ordinary seeded 80/10/10."""
    dm = _build(tmp_path, pinned=None)
    idx = dm.index
    assert len(idx.train) + len(idx.val) + len(idx.test) == 200
    assert set(idx.train) & set(idx.val) == set()
    assert set(idx.train) & set(idx.test) == set()
    assert set(idx.val) & set(idx.test) == set()
    # ~80/10/10, allowing for the ratio-balancing assignment of leftovers
    assert 150 <= len(idx.train) <= 170


def test_pinned_indices_land_in_test_and_nowhere_else(tmp_path: Any) -> None:
    """THE load-bearing property: every pinned record is in test, and in no other split.

    Chosen to include records that the unpinned split puts in TRAIN, so the test would fail
    if the pin were applied before the ratio assignment and then overwritten.
    """
    unpinned = _build(tmp_path / "u", pinned=None).index
    pinned = set(unpinned.train[:20]) | set(unpinned.val[:5]) | set(unpinned.test[:5])

    dm = _build(tmp_path / "p", pinned=pinned)
    idx = dm.index
    assert pinned <= set(idx.test), "pinned records are missing from test"
    assert not (pinned & set(idx.train)), "pinned records leaked into train"
    assert not (pinned & set(idx.val)), "pinned records leaked into val"
    # nothing is lost or duplicated by the pin
    assert len(idx.train) + len(idx.val) + len(idx.test) == 200
    assert len(set(idx.train) | set(idx.val) | set(idx.test)) == 200


def test_pin_enlarges_test_and_shrinks_train_val(tmp_path: Any) -> None:
    """Test ends up LARGER than the nominal ratio -- by design, and worth pinning down.

    The ratios govern the remaining pool, not the pinned block, so a reader comparing test
    sizes across runs should expect this rather than suspect a bug.
    """
    unpinned = _build(tmp_path / "u", pinned=None).index
    pinned = set(unpinned.train[:40])
    idx = _build(tmp_path / "p", pinned=pinned).index
    assert len(idx.test) > len(unpinned.test)
    assert len(idx.train) < len(unpinned.train)


def test_pin_changes_the_cache_key(tmp_path: Any) -> None:
    """A pinned run must NOT load an unpinned cached index.

    The cache is keyed by seed, so without a pin-dependent tag the pinned run would silently
    reuse `index_seed_42.json` from an earlier unpinned run -- no error, and the pinned genes
    back in train. This is the failure mode that would quietly invalidate the comparison.
    """
    cache = tmp_path / "shared"
    unpinned = CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(cache),
        split_indices=["phenotype_label_index"],
        random_seed=42,
    )
    pinned_set = set(unpinned.index.train[:20])
    assert osp.exists(osp.join(str(cache), "index_seed_42.json"))

    dm = CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(cache),
        split_indices=["phenotype_label_index"],
        random_seed=42,
        pinned_test_indices=pinned_set,
    )
    assert pinned_set <= set(dm.index.test), (
        "pinned run reused the unpinned cached index"
    )
    # the two indices coexist under different names
    files = sorted(
        f for f in __import__("os").listdir(str(cache)) if f.startswith("index_seed")
    )
    assert len(files) == 2, f"expected two distinct cached indices, got {files}"


def test_pin_cache_is_reused_across_instances(tmp_path: Any) -> None:
    """The same pin must hit the same cache file, so a requeued job does not re-split."""
    cache = tmp_path / "c"
    pinned = {1, 3, 5, 7, 9}
    first = CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(cache),
        split_indices=["phenotype_label_index"],
        random_seed=42,
        pinned_test_indices=pinned,
    ).index
    second = CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(cache),
        split_indices=["phenotype_label_index"],
        random_seed=42,
        pinned_test_indices=pinned,
    ).index
    assert first.train == second.train
    assert first.test == second.test


def test_pinned_indices_outside_the_dataset_are_ignored_not_fatal(
    tmp_path: Any,
) -> None:
    """Genes absent from a stale build must not crash or silently corrupt the split.

    The Cachera LMDB currently predates the shared name resolver (issue #195) and lacks a
    handful of the requested genes; the run reports the shortfall and proceeds on the ones
    it has.
    """
    idx = _build(tmp_path, pinned={5, 7, 9999, 10000}).index
    assert {5, 7} <= set(idx.test)
    assert 9999 not in set(idx.test)
    assert len(idx.train) + len(idx.val) + len(idx.test) == 200


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_pin_is_invariant_to_seed_while_train_val_reroll(
    tmp_path: Any, seed: int
) -> None:
    """Sweeping the seed must re-roll train/val while the comparison set stays fixed.

    That is what lets the confirm stage average over 5 seeds without ever moving the genes
    the external comparison is computed on.
    """
    pinned = {2, 4, 6, 8, 10, 12}
    idx = _build(tmp_path / f"s{seed}", pinned=pinned, seed=seed).index
    assert pinned <= set(idx.test)
    assert not (pinned & set(idx.train))


# ---------------------------------------------------------------------------
# num_workers=0 -- the setting that was unconstructible until 2026-07-28
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
def test_dataloaders_construct_at_any_worker_count(
    tmp_path: Any, num_workers: int
) -> None:
    """Every dataloader must build at num_workers=0 as well as >0.

    torch rejects a non-None `prefetch_factor` when num_workers=0, so passing it
    unconditionally made zero-worker mode raise ValueError before the first batch --
    silently, from the caller's view, since the failure surfaces inside Lightning's
    sanity check rather than at datamodule construction.

    This is not a theoretical setting. On a parallel filesystem the spawn path is the
    expensive one (each worker re-imports the stack), so num_workers=0 is the lever for
    diagnosing slow-start jobs, and it must actually work when reached for.
    """
    dm = CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(tmp_path / "cache"),
        split_indices=["phenotype_label_index"],
        random_seed=42,
        num_workers=num_workers,
    )
    dm.setup()
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        loader = getattr(dm, name)()
        assert loader.num_workers == num_workers, name
        expected = 2 if num_workers > 0 else None
        assert loader.prefetch_factor == expected, name


def _build_full_pin(
    tmp_path: Any, pinned_splits: dict[str, set[int]] | None, seed: int = 42
) -> CellDataModule:
    return CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(tmp_path / "cache"),
        split_indices=["phenotype_label_index"],
        random_seed=seed,
        pinned_split_indices=pinned_splits,
    )


def test_full_pin_places_every_record_in_its_named_split(tmp_path: Any) -> None:
    """pinned_split_indices forces records into train/val/test exactly as named."""
    pinned = {"train": {0, 2, 4}, "val": {1, 3}, "test": {5, 7, 9}}
    dm = _build_full_pin(tmp_path, pinned)
    idx = dm.index
    assert pinned["train"] <= set(idx.train)
    assert pinned["val"] <= set(idx.val)
    assert pinned["test"] <= set(idx.test)
    # and absent from every other split
    assert not pinned["train"] & (set(idx.val) | set(idx.test))
    assert not pinned["val"] & (set(idx.train) | set(idx.test))
    assert not pinned["test"] & (set(idx.train) | set(idx.val))
    assert len(idx.train) + len(idx.val) + len(idx.test) == 200


def test_full_pin_overrides_seed_assignment(tmp_path: Any) -> None:
    """A record the seed puts in train stays in val when pinned there."""
    baseline = _build_full_pin(tmp_path / "a", None)
    seed_train = set(baseline.index.train)
    moved = set(list(seed_train)[:4])
    dm = _build_full_pin(tmp_path / "b", {"val": moved})
    assert moved <= set(dm.index.val)
    assert not moved & set(dm.index.train)


def test_full_pin_overlap_refused(tmp_path: Any) -> None:
    """The same record pinned into two splits is a construction-time error."""
    with pytest.raises(AssertionError, match="overlap"):
        _build_full_pin(tmp_path, {"train": {1, 2}, "test": {2, 3}})


def test_full_pin_changes_cache_tag(tmp_path: Any) -> None:
    """A pinned run never reuses the unpinned cache file (and vice versa)."""
    dm_plain = _build_full_pin(tmp_path, None)
    dm_pinned = _build_full_pin(tmp_path, {"val": {1, 2, 3}})
    assert dm_plain._pin_tag() != dm_pinned._pin_tag()
    cache = tmp_path / "cache"
    assert (cache / "index_seed_42.json").exists()
    assert any("_pin3-" in p.name for p in cache.iterdir())


def test_full_pin_out_of_range_indices_ignored(tmp_path: Any) -> None:
    """Indices outside the dataset are dropped, not crashed on."""
    dm = _build_full_pin(tmp_path, {"test": {0, 1, 5000}})
    idx = dm.index
    assert {0, 1} <= set(idx.test)
    assert 5000 not in set(idx.train) | set(idx.val) | set(idx.test)


def _build_subset(
    tmp_path: Any,
    subset: set[int] | None,
    pinned_splits: dict[str, set[int]] | None = None,
    seed: int = 42,
) -> CellDataModule:
    return CellDataModule(
        dataset=_FakeDataset(),
        cache_dir=str(tmp_path / "cache"),
        split_indices=["phenotype_label_index"],
        random_seed=seed,
        index_subset=subset,
        pinned_split_indices=pinned_splits,
    )


def test_subset_restricts_the_pool_and_nothing_else_appears(tmp_path: Any) -> None:
    """THE load-bearing property: no record outside the subset reaches any split.

    Pinning assigns but never excludes, so a subset is the only way to say "train on the
    376,732 triples of a 13,525,071-record build". A subset that silently failed would
    train on 36x the intended data while every log line still named the arm.
    """
    subset = set(range(0, 60))
    idx = _build_subset(tmp_path, subset).index
    placed = set(idx.train) | set(idx.val) | set(idx.test)
    assert placed == subset
    assert len(idx.train) + len(idx.val) + len(idx.test) == 60


def test_subset_still_splits_by_ratio_within_the_subset(tmp_path: Any) -> None:
    """The 80/10/10 is drawn on the subset, not inherited from the full pool."""
    idx = _build_subset(tmp_path, set(range(0, 100))).index
    assert 75 <= len(idx.train) <= 85
    assert len(idx.val) > 0 and len(idx.test) > 0


def test_subset_changes_the_cache_key(tmp_path: Any) -> None:
    """An S0 arm must not load the full-pool cached index left by an earlier run.

    Separate from the pin tag because the 025 ladder varies the two independently: the
    split regime stays fixed while the training pool grows S0 -> S2 -> S3.
    """
    full = _build_subset(tmp_path, None)
    sub = _build_subset(tmp_path, set(range(0, 60)))
    assert full._subset_tag() == ""
    assert sub._subset_tag().startswith("_sub60-")
    cache = tmp_path / "cache"
    assert (cache / "index_seed_42.json").exists()
    assert any("_sub60-" in p.name for p in cache.iterdir())
    assert set(sub.index.train) | set(sub.index.val) | set(sub.index.test) == set(
        range(0, 60)
    )


def test_two_subsets_of_equal_size_get_different_cache_keys(tmp_path: Any) -> None:
    """The tag hashes the membership, not just the count."""
    a = _build_subset(tmp_path, set(range(0, 60)))
    b = _build_subset(tmp_path, set(range(60, 120)))
    assert a._subset_tag() != b._subset_tag()


def test_subset_composes_with_pinned_splits(tmp_path: Any) -> None:
    """The 025 replication arm: subset = the triples, pinned splits = 010's assignment."""
    subset = set(range(0, 60))
    pinned = {
        "train": set(range(0, 40)),
        "val": set(range(40, 50)),
        "test": set(range(50, 60)),
    }
    idx = _build_subset(tmp_path, subset, pinned_splits=pinned).index
    assert set(idx.train) == pinned["train"]
    assert set(idx.val) == pinned["val"]
    assert set(idx.test) == pinned["test"]


def test_pinned_records_outside_the_subset_are_not_placed(tmp_path: Any) -> None:
    """A pin cannot smuggle a record back into a split the subset excluded.

    This is the leak that would matter in 025: the R and Q splits are pinned over all
    376,732 triples, so an arm that subsets to a smaller pool must drop the rest rather
    than honor the pin.
    """
    idx = _build_subset(
        tmp_path, set(range(0, 30)), pinned_splits={"test": {5, 6, 100, 101}}
    ).index
    placed = set(idx.train) | set(idx.val) | set(idx.test)
    assert {5, 6} <= set(idx.test)
    assert not ({100, 101} & placed)
    assert placed == set(range(0, 30))


def test_empty_subset_raises(tmp_path: Any) -> None:
    """An empty subset is a config mistake, not a request to train on nothing."""
    with pytest.raises(AssertionError, match="index_subset is empty"):
        _build_subset(tmp_path, set())


def test_subset_outside_the_dataset_raises(tmp_path: Any) -> None:
    """Unlike a pin, an out-of-range subset index is fatal.

    A pin naming a missing record is a benign no-op, but a subset naming one means the
    index artifact was built against a DIFFERENT dataset than the one being trained on,
    and the arm would then train on whichever indices happened to overlap.
    """
    with pytest.raises(AssertionError, match="outside the"):
        _build_subset(tmp_path, {1, 2, 5000})


# ---------------------------------------------------------------------------
# Split arithmetic, exactly
# ---------------------------------------------------------------------------


class _KeyedDataset:
    """``n`` records plus the split indices passed as keyword arguments."""

    def __init__(self, n: int, **indices: dict[Any, list[int]]) -> None:
        self._n = n
        for name, index in indices.items():
            setattr(self, name, index)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        return idx


class _UnreadableDataset(_KeyedDataset):
    """Raises on ``len``: proves a split was loaded from the cache, not recomputed."""

    def __len__(self) -> int:
        raise AssertionError("the split was recomputed instead of loaded from cache")


def _ten(tmp_path: Path, **kwargs: Any) -> CellDataModule:
    """Ten records, one key ``a`` over all of them, seed 42."""
    return CellDataModule(
        dataset=_KeyedDataset(10, phenotype_label_index={"a": list(range(10))}),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=42,
        **kwargs,
    )


def test_seed_42_shuffle_sliced_eight_one_one(tmp_path: Path) -> None:
    """Ten records under one key: the seeded shuffle [7, 3, 2, 8, 5, 6, 9, 4, 0, 1] is
    sliced 8 / 1 / 1, so train [2..9], val [0], test [1]. A string ``split_indices`` is
    wrapped into a one-element list.
    """
    rng = random.Random(42)
    order = list(range(10))
    rng.shuffle(order)
    assert order == [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
    dm = _ten(tmp_path)
    assert dm.split_indices == ["phenotype_label_index"]
    assert dm.index.model_dump() == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1],
    }


def _keyed_zero_to_nine(tmp_path: Path, n: int, **kwargs: Any) -> CellDataModule:
    """``n`` records of which only 0 to 9 sit under a key, seed 42."""
    return CellDataModule(
        dataset=_KeyedDataset(n, phenotype_label_index={"a": list(range(10))}),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=42,
        **kwargs,
    )


def test_records_under_no_key_are_refused_with_their_indices(tmp_path: Path) -> None:
    """Records 10 and 11 exist (``len`` is 12) but sit under no key, so the key-driven
    assignment never reaches them; dropping them silently would be a data loss, so the
    construction refuses and names them. Past ten orphans the list is cut after the
    first ten and the total is given. No cache file is written for a refused split.
    """
    with pytest.raises(ValueError) as excinfo:
        _keyed_zero_to_nine(tmp_path, 12)
    assert str(excinfo.value) == (
        "2 record(s) sit under no key of ['phenotype_label_index'] and would land in "
        "no split: [10, 11]"
    )
    with pytest.raises(ValueError) as excinfo:
        _keyed_zero_to_nine(tmp_path, 25)
    assert str(excinfo.value) == (
        "15 record(s) sit under no key of ['phenotype_label_index'] and would land in "
        "no split: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ... (15 in all)]"
    )
    assert list((tmp_path / "cache").iterdir()) == []


def test_unkeyed_records_placed_by_a_pin_are_not_orphans(tmp_path: Path) -> None:
    """The orphan check runs after pinning: records 10 and 11, under no key but pinned
    to test, are placed, and the keyed ten split 8 / 1 / 1 as usual around them.
    """
    index = _keyed_zero_to_nine(tmp_path, 12, pinned_test_indices=[10, 11]).index
    assert index.model_dump() == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1, 10, 11],
    }


@pytest.mark.parametrize("split_indices", [None, []])
def test_no_split_indices_is_refused_at_construction(
    tmp_path: Path, split_indices: list[str] | None
) -> None:
    """Records are placed only through split-index keys, so with none (the ``None``
    default, or an empty list) every record would land in no split and the module would
    train on nothing. Construction refuses before any cache is written. The argument keeps
    its ``None`` default because legacy callers (experiments 002, smf-dmf-tmf-001, the
    DEPRECATED_costanzo scripts, ``torchcell/trainers/cell.py``) omit it.
    """
    with pytest.raises(ValueError) as excinfo:
        CellDataModule(
            dataset=_KeyedDataset(10),
            cache_dir=str(tmp_path / "cache"),
            random_seed=42,
            split_indices=split_indices,
        )
    assert str(excinfo.value) == (
        "split_indices is required: name at least one split index of the dataset "
        "(e.g. 'phenotype_label_index'); without one every record lands in no split"
    )
    assert not (tmp_path / "cache").exists()


def test_multiply_keyed_records_are_reassigned_exactly_once(tmp_path: Path) -> None:
    """Twenty records, each under BOTH keys ``a`` and ``b`` of one index.

    Seed 42 shuffles ``a`` to [19, 5, 14, 4, 9, 13, 15, 18, 6, 12, 17, 10, 1, 11, 2, 16,
    7, 8, 0, 3] (train = first 16, val {7, 8}, test {0, 3}) and then ``b`` to [14, 8, 3,
    11, 10, 16, 1, 12, 5, 18, 2, 0, 4, 9, 15, 7, 13, 19, 6, 17] (val {13, 19}, test
    {6, 17}). The per-split unions make {0, 3, 6, 7, 8, 13, 17, 19} conflicted; they are
    pulled out, leaving train = the 12 records both keys put in train. Balancing under key
    ``a`` (total 20, targets train 16, val 2, test the remainder 2) visits the eight in
    ascending order and picks the split with the most negative (count - target) / target,
    first in train/val/test order on a tie: 0 -> val (-1 ties test), 3 -> test (-1),
    6 -> val (-0.5 ties test), 7 -> test (-0.5), then 8, 13, 17, 19 -> train. Final:
    train 16, val [0, 6], test [3, 7], exactly the 16 / 2 / 2 targets.
    """
    rng = random.Random(42)
    a, b = list(range(20)), list(range(20))
    rng.shuffle(a)
    rng.shuffle(b)
    assert (a[16:18], a[18:], b[16:18], b[18:]) == ([7, 8], [0, 3], [13, 19], [6, 17])
    dm = CellDataModule(
        dataset=_KeyedDataset(
            20, phenotype_label_index={"a": list(range(20)), "b": list(range(20))}
        ),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=42,
    )
    assert dm.index.model_dump() == {
        "train": [1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "val": [0, 6],
        "test": [3, 7],
    }
    details = dm.index_details.model_dump()
    assert details["train"]["phenotype_label_index"]["a"]["count"] == 16
    assert details["val"]["phenotype_label_index"]["b"] == {
        "indices": [0, 6],
        "count": 2,
    }


def _two_indices(tmp_path: Path, n: int) -> CellDataModule:
    """``2n`` records: phenotype keys ``a`` = 0..n-1 and ``b`` = n..2n-1, perturbation
    count keys 1 = evens and 2 = odds, so the two indices disagree on some records.
    """
    return CellDataModule(
        dataset=_KeyedDataset(
            2 * n,
            phenotype_label_index={"a": list(range(n)), "b": list(range(n, 2 * n))},
            perturbation_count_index={
                1: list(range(0, 2 * n, 2)),
                2: list(range(1, 2 * n, 2)),
            },
        ),
        cache_dir=str(tmp_path / f"cache{n}"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        random_seed=42,
    )


def test_two_indices_balance_ten_record_keys_to_eight_one_one(tmp_path: Path) -> None:
    """When two split indices disagree, the disputed records are balanced under the first
    index's keys toward targets that partition the key exactly, test taking the
    remainder: a 10-record key targets 8 / 1 / 1 (the old ``int(10 * 0.0999...) = 0``
    test target divided by zero). Ten is the smallest key that balances, since at nine the
    val target ``int(0.9)`` is 0. The seeded result puts both phenotype keys at exactly
    8 / 1 / 1 and the three splits partition all 20 records.
    """
    index = _two_indices(tmp_path, 10).index
    assert index.model_dump() == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18],
        "val": [0, 16],
        "test": [1, 19],
    }
    assert sorted(index.train + index.val + index.test) == list(range(20))


def test_two_indices_refuse_a_nine_record_key_with_a_disputed_record(
    tmp_path: Path,
) -> None:
    """A 9-record key targets 7 / 0 / 2; a zero target cannot be balanced toward, so the
    construction refuses with the key, its size and the empty split named.
    """
    with pytest.raises(ValueError) as excinfo:
        _two_indices(tmp_path, 9)
    assert str(excinfo.value) == (
        "Key 'a' of phenotype_label_index has 1 record(s) left to balance, but its 9 "
        "records give a target of 0 for ['val']; balancing needs at least 10 records "
        "per such key"
    )


def test_subset_skips_a_key_with_no_members_and_reports_it_empty(
    tmp_path: Path,
) -> None:
    """``index_subset`` = records 0 to 9 empties key ``b`` (records 10 to 19), which the
    first pass skips (line 438); key ``a`` splits exactly as the plain ten-record case,
    and the details still list ``b`` with an empty ``IndexSplit`` in every split. The tag
    is ``_sub10-`` + sha256("0,1,2,3,4,5,6,7,8,9")[:8] = ``f4972c7d``.
    """
    dm = CellDataModule(
        dataset=_KeyedDataset(
            20, phenotype_label_index={"a": list(range(10)), "b": list(range(10, 20))}
        ),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=42,
        index_subset=range(10),
    )
    assert dm.index.model_dump() == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1],
    }
    empty = {"indices": [], "count": 0}
    dumped = dm.index_details.model_dump()
    assert [
        dumped[split]["phenotype_label_index"]["b"]
        for split in ("train", "val", "test")
    ] == [empty, empty, empty]
    assert dumped["test"]["phenotype_label_index"]["a"] == {"indices": [1], "count": 1}
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [
        "index_details_seed_42_sub10-f4972c7d.json",
        "index_seed_42_sub10-f4972c7d.json",
    ]


def test_cache_file_names_carry_both_tags_in_order(tmp_path: Path) -> None:
    """Seed 7, pinned test {1, 3}, pinned val {5}, subset {0, 1, 2}.

    Pin payload "1,3|train:|val:5|test:" (n = 2 + 1 = 3), sha256 prefix ``e0d22c21``;
    subset payload "0,1,2" (n = 3), prefix ``c0be322c``. The pin tag comes first.
    """
    dm = CellDataModule(
        dataset=_KeyedDataset(12, phenotype_label_index={"a": list(range(10))}),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=7,
        pinned_test_indices=[3, 1],
        pinned_split_indices={"val": [5]},
        index_subset=[0, 1, 2],
    )
    assert dm._pin_tag() == "_pin3-e0d22c21"
    assert dm._subset_tag() == "_sub3-c0be322c"
    cache = str(tmp_path / "cache")
    assert dm._cache_files() == (
        osp.join(cache, "index_seed_7_pin3-e0d22c21_sub3-c0be322c.json"),
        osp.join(cache, "index_details_seed_7_pin3-e0d22c21_sub3-c0be322c.json"),
    )


def test_cached_files_hold_the_index_and_details_dumps(tmp_path: Path) -> None:
    """The two JSON files are ``model_dump()`` of the index and of the details."""
    dm = _ten(tmp_path)
    cache = tmp_path / "cache"
    assert json.loads((cache / "index_seed_42.json").read_text()) == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1],
    }
    details = json.loads((cache / "index_details_seed_42.json").read_text())
    assert details["methods"] == ["phenotype_label_index"]
    assert details["val"]["phenotype_label_index"] == {
        "a": {"indices": [0], "count": 1}
    }
    assert details == dm.index_details.model_dump()


def test_existing_cache_is_loaded_without_touching_the_dataset(tmp_path: Path) -> None:
    """A second module on the same cache reads the files: its dataset raises on ``len``,
    so any recomputation would fail, and the loaded split equals the first.
    """
    first = _ten(tmp_path)
    second = CellDataModule(
        dataset=_UnreadableDataset(10, phenotype_label_index={"a": list(range(10))}),
        cache_dir=str(tmp_path / "cache"),
        split_indices="phenotype_label_index",
        random_seed=42,
    )
    assert second.index == first.index
    assert second.index_details == first.index_details


def test_corrupt_cache_is_reported_and_regenerated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable index file prints the JSON error and recomputes (lines 400 to 402);
    the seeded recomputation rewrites the same split.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "index_seed_42.json").write_text("")
    (cache / "index_details_seed_42.json").write_text("{}")
    dm = _ten(tmp_path)
    assert capsys.readouterr().out == (
        "Error loading index or details: Expecting value: line 1 column 1 (char 0). "
        "Regenerating...\n"
    )
    assert json.loads((cache / "index_seed_42.json").read_text()) == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1],
    }
    assert dm.index.test == [1]


def test_index_property_recomputes_when_the_cache_file_disappears(
    tmp_path: Path,
) -> None:
    """``index`` re-checks the cache on every access (line 328): deleting the files after
    construction makes the next access rewrite them, with the same seeded split.
    """
    dm = _ten(tmp_path)
    cache = tmp_path / "cache"
    (cache / "index_seed_42.json").unlink()
    (cache / "index_details_seed_42.json").unlink()
    assert dm.index.model_dump() == {
        "train": [2, 3, 4, 5, 6, 7, 8, 9],
        "val": [0],
        "test": [1],
    }
    assert sorted(p.name for p in cache.iterdir()) == [
        "index_details_seed_42.json",
        "index_seed_42.json",
    ]


def test_unknown_pinned_split_name_is_refused(tmp_path: Path) -> None:
    """Only train/val/test may be pinned; the message names the unknown key set."""
    with pytest.raises(AssertionError) as excinfo:
        _ten(tmp_path, pinned_split_indices={"dev": [1]})
    assert str(excinfo.value) == "pinned_split_indices has unknown splits: {'dev'}"


def test_setup_builds_subsets_once_and_keeps_a_later_narrowing(tmp_path: Path) -> None:
    """``setup`` wraps the three index lists in ``Subset``s; a second call is a no-op, so
    a caller's post-setup narrowing of ``train_dataset.indices`` survives Lightning's
    second ``setup``.
    """
    dm = _ten(tmp_path)
    dm.setup()
    assert list(dm.train_dataset.indices) == [2, 3, 4, 5, 6, 7, 8, 9]
    assert list(dm.val_dataset.indices) == [0]
    assert list(dm.test_dataset.indices) == [1]
    dm.train_dataset.indices = [2, 3]
    dm.setup("fit")
    assert dm.train_dataset.indices == [2, 3]


def test_dataloader_options_at_two_workers(tmp_path: Path) -> None:
    """With two workers: timeout 10800 s, a spawn context, persistent workers, prefetch 2;
    train shuffles (``RandomSampler``), val and test do not; val uses ``val_batch_size``;
    the collater carries the custom ``follow_batch``; ``all_dataloader`` spans all 10
    records.
    """
    dm = _ten(
        tmp_path, num_workers=2, batch_size=3, val_batch_size=5, follow_batch=["x_gene"]
    )
    dm.setup()
    train, val, test, full = (
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
        dm.all_dataloader(),
    )
    for loader in (train, val, test, full):
        assert loader.timeout == 10800
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert loader.persistent_workers is True
        assert loader.pin_memory is False
        assert isinstance(loader.collate_fn, Collater)
        assert loader.collate_fn.follow_batch == ["x_gene"]
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(val.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert (train.batch_size, val.batch_size, test.batch_size) == (3, 5, 3)
    assert isinstance(full.dataset, _KeyedDataset)
    assert len(full.dataset) == 10


def test_dataloader_defaults_at_zero_workers(tmp_path: Path) -> None:
    """At zero workers: timeout 0, no multiprocessing context, persistence off even though
    ``persistent_workers`` defaults to True; the default ``follow_batch`` is
    ["x", "x_pert"]; ``train_shuffle=False`` gives train a ``SequentialSampler``.
    """
    dm = _ten(tmp_path, train_shuffle=False)
    dm.setup()
    loader = dm.train_dataloader()
    assert loader.timeout == 0
    assert loader.multiprocessing_context is None
    assert loader.persistent_workers is False
    assert isinstance(loader.collate_fn, Collater)
    assert loader.collate_fn.follow_batch == ["x", "x_pert"]
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.batch_size == 32


def test_custom_collate_fn_is_honored_through_a_torch_loader(tmp_path: Path) -> None:
    """PyG's ``DataLoader`` pops any ``collate_fn`` and installs its own ``Collater``, so
    a caller's collate (experiment 006 passes a ``LazyCollater``) is honored by building
    the plain torch ``DataLoader`` instead; it receives each batch's items and its return
    value is the batch. The worker options still apply, and without a ``collate_fn`` the
    PyG loader and its ``Collater`` are kept.
    """
    calls: list[list[int]] = []

    def my_collate(items: list[int]) -> tuple[str, list[int]]:
        calls.append(items)
        return ("batch", items)

    dm = _ten(tmp_path, collate_fn=my_collate, batch_size=5, train_shuffle=False)
    dm.setup()
    loader = dm.train_dataloader()
    assert type(loader) is torch.utils.data.DataLoader
    assert loader.collate_fn is my_collate
    assert (loader.batch_size, loader.timeout) == (5, 0)
    assert list(loader) == [("batch", [2, 3, 4, 5, 6]), ("batch", [7, 8, 9])]
    assert calls == [[2, 3, 4, 5, 6], [7, 8, 9]]
    plain = _ten(tmp_path)
    plain.setup()
    assert type(plain.train_dataloader().collate_fn) is Collater


def test_prefetch_wraps_the_loader_on_the_cpu_when_cuda_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``prefetch=True`` returns a ``PrefetchLoader`` over the plain loader, on the CPU
    when ``torch.cuda.is_available()`` is False.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dm = _ten(tmp_path, prefetch=True, batch_size=4)
    dm.setup()
    wrapped = dm.test_dataloader()
    assert isinstance(wrapped, PrefetchLoader)
    assert wrapped.device_helper.device == torch.device("cpu")
    assert wrapped.device_helper.is_gpu is False
    assert wrapped.loader.batch_size == 4
    assert list(wrapped.loader.dataset.indices) == [1]


# ---------------------------------------------------------------------------
# The pydantic index records
# ---------------------------------------------------------------------------


def test_index_split_refuses_unsorted_indices_and_truncates_its_repr() -> None:
    """The validator message, and a repr that shows three indices then an ellipsis."""
    with pytest.raises(ValueError, match="Indices must be sorted in ascending order"):
        IndexSplit(indices=[2, 1], count=2)
    assert repr(IndexSplit(indices=[1, 2, 3, 4], count=4)) == (
        "IndexSplit(indices=[1, 2, 3, ...], count=4)"
    )
    assert repr(IndexSplit(indices=[1], count=1)) == "IndexSplit(indices=[1], count=1)"


def test_data_module_index_validators_and_string_forms() -> None:
    """Unsorted val and an overlap are refused with their own messages; repr truncates,
    str adds the per-split counts.
    """
    with pytest.raises(
        ValueError, match="val indices must be sorted in ascending order"
    ):
        DataModuleIndex(train=[0], val=[2, 1], test=[])
    with pytest.raises(
        ValueError, match="Indices in train, val, and test must not overlap"
    ):
        DataModuleIndex(train=[0, 1], val=[1], test=[])
    index = DataModuleIndex(train=[0, 1, 2, 3], val=[4], test=[])
    assert repr(index) == "DataModuleIndex(train=[0, 1, 2, ...], val=[4], test=[])"
    assert str(index) == (
        "DataModuleIndex(train=[0, 1, 2, ...] (4 indices), val=[4] (1 indices), "
        "test=[] (0 indices))"
    )


def _details() -> DataModuleIndexDetails:
    """Key ``a``: 3 train, 1 val, 0 test; key ``b``: 1 train, 0 val, 2 test."""
    return DataModuleIndexDetails(
        methods=["phenotype_label_index"],
        train=DatasetSplit(
            phenotype_label_index={
                "a": IndexSplit(indices=[0, 1, 2], count=3),
                "b": IndexSplit(indices=[5], count=1),
            }
        ),
        val=DatasetSplit(phenotype_label_index={"a": IndexSplit(indices=[3], count=1)}),
        test=DatasetSplit(
            phenotype_label_index={"b": IndexSplit(indices=[6, 7], count=2)}
        ),
    )


def test_df_summary_counts_ratios_and_totals_per_key() -> None:
    """Totals: ``a`` 3 + 1 + 0 = 4, ``b`` 1 + 0 + 2 = 3. Ratios rounded to 3 places:
    a 0.75 / 0.25 / 0, b 1/3 = 0.333 / 0 / 2/3 = 0.667. A key absent from a split counts
    0. Rows sorted by split (train, val, test), then index type, then key.
    """
    records = _details().df_summary().to_dict(orient="records")
    assert records == [
        {"split": "train", "index_type": "phenotype_label_index", "key": "a",
         "count": 3, "ratio": 0.75, "total": 4},
        {"split": "train", "index_type": "phenotype_label_index", "key": "b",
         "count": 1, "ratio": 0.333, "total": 3},
        {"split": "val", "index_type": "phenotype_label_index", "key": "a",
         "count": 1, "ratio": 0.25, "total": 4},
        {"split": "val", "index_type": "phenotype_label_index", "key": "b",
         "count": 0, "ratio": 0.0, "total": 3},
        {"split": "test", "index_type": "phenotype_label_index", "key": "a",
         "count": 0, "ratio": 0.0, "total": 4},
        {"split": "test", "index_type": "phenotype_label_index", "key": "b",
         "count": 2, "ratio": 0.667, "total": 3},
    ]  # fmt: skip
    assert str(_details()) == _details().df_summary().to_string()


def test_details_str_of_an_empty_summary_is_the_empty_form() -> None:
    """With no index data ``df_summary`` returns an empty frame that still carries its
    six columns, so ``__str__`` prints ``"DataModuleIndexDetails(empty)"``.
    """
    empty = DataModuleIndexDetails(
        methods=[], train=DatasetSplit(), val=DatasetSplit(), test=DatasetSplit()
    )
    df = empty.df_summary()
    assert list(df.columns) == ["split", "index_type", "key", "count", "ratio", "total"]
    assert len(df) == 0
    assert str(empty) == "DataModuleIndexDetails(empty)"


def test_overlap_dataset_index_split_intersects_each_dataset_with_each_split() -> None:
    """Dataset ``x`` = {0, 1, 5}, ``y`` = {9}, ``3`` = {4, 6}; train {0, 1, 4}, val {5},
    test {6}: x -> train [0, 1], val [5]; 3 -> train [4], test [6]; y is in no split and
    is left out of all three dicts.
    """
    split = overlap_dataset_index_split(
        {"x": [5, 1, 0], "y": [9], 3: [6, 4]},
        DataModuleIndex(train=[0, 1, 4], val=[5], test=[6]),
    )
    assert split.model_dump() == {
        "train": {"x": [0, 1], 3: [4]},
        "val": {"x": [5]},
        "test": {3: [6]},
    }


def test_overlap_dataset_index_split_gives_an_empty_dict_for_an_empty_split() -> None:
    """A split that holds none of the datasets is an empty dict (the experiment 003
    plotting scripts that call this skip a falsy split): dataset ``x`` = {0, 1} over
    train {0}, val {1}, test {} gives train {x: [0]}, val {x: [1]}, test {}; with no
    datasets at all every split is empty.
    """
    split = overlap_dataset_index_split(
        {"x": [0, 1]}, DataModuleIndex(train=[0], val=[1], test=[])
    )
    assert split.model_dump() == {"train": {"x": [0]}, "val": {"x": [1]}, "test": {}}
    assert overlap_dataset_index_split(
        {}, DataModuleIndex(train=[0], val=[1], test=[2])
    ).model_dump() == {"train": {}, "val": {}, "test": {}}

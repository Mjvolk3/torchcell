# tests/torchcell/datamodules/test_cell.py
"""Tests for CellDataModule's split construction, focused on the PINNED TEST SET.

The pin reproduces an external gene-level split inside ours (Merzbacher 2025's betaxanthin
test ORFs) so their published numbers and ours are computed on the same genes. Everything
about that comparison rests on two properties that are invisible at runtime -- pinned records
really are in test, and really are absent from train/val -- so they are asserted here rather
than trusted. A pin that silently failed would produce a *better-looking* score, because the
comparison genes would be back in training.
"""

import os.path as osp
from typing import Any

import pytest

from torchcell.datamodules.cell import CellDataModule


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

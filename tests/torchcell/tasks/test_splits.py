# tests/torchcell/tasks/test_splits.py
# [[tests.torchcell.tasks.test_splits]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/tasks/test_splits.py
"""Pure tests for the benchmark split functions in ``torchcell.tasks.splits``.

The small hand-checked case is ten members in seven groups: ``a`` twice, ``c`` three
times and ``b``, ``d``, ``e``, ``f``, ``g`` once each. With the default fractions the
train boundary is ``0.8 * 10 = 8`` members and the validation boundary ``0.9 * 10 = 9``.
The sorted groups are dealt in the order of ``default_rng(seed).permutation(7)``:

- seed 0 deals ``[2, 4, 3, 6, 5, 0, 1]`` = c, e, d, g, f, a, b. Before ``a`` the
  running count is 7 (< 8), so ``a`` goes to train and the count becomes 9; ``b`` then
  starts at 9, past the validation boundary, so it is test. Train 9, val 0, test 1.
- seed 1 deals ``[5, 0, 1, 4, 2, 6, 3]`` = f, a, b, e, c, g, d. ``c`` starts at 5
  (train, count 8), ``g`` starts at 8 (val, count 9), ``d`` starts at 9 (test).
  Train 8, val 1, test 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchcell.tasks.splits import (
    FRACTIONS,
    PARTS,
    assign_groups,
    block_groups,
    pair_novelty,
    pair_parts,
    part_sizes,
    random_split,
)

GROUPS = ["a", "a", "b", "c", "c", "c", "d", "e", "f", "g"]


def test_constants() -> None:
    assert PARTS == ("train", "val", "test")
    assert FRACTIONS == (0.8, 0.1, 0.1)


def test_assign_groups_seed0_exact() -> None:
    out = assign_groups(GROUPS, seed=0)
    expected = ["train", "train", "test"] + ["train"] * 7
    assert out.tolist() == expected
    assert part_sizes(out.tolist()) == {"train": 9, "val": 0, "test": 1}


def test_assign_groups_seed1_exact() -> None:
    out = assign_groups(GROUPS, seed=1)
    expected = ["train"] * 6 + ["test", "train", "train", "val"]
    assert out.tolist() == expected
    assert part_sizes(out.tolist()) == {"train": 8, "val": 1, "test": 1}


def test_assign_groups_is_deterministic_for_a_seed() -> None:
    first = assign_groups(GROUPS, seed=7)
    second = assign_groups(GROUPS, seed=7)
    np.testing.assert_array_equal(first, second)


def test_assign_groups_different_seed_changes_assignment() -> None:
    assert assign_groups(GROUPS, seed=0).tolist() != assign_groups(GROUPS, 1).tolist()


def test_assign_groups_never_divides_a_group() -> None:
    groups = [f"g{i % 37}" for i in range(500)]
    for seed in range(5):
        out = assign_groups(groups, seed=seed)
        parts_of_group = {
            group: set(out[np.asarray(groups) == group].tolist())
            for group in set(groups)
        }
        assert {len(parts) for parts in parts_of_group.values()} == {1}
        assert len(parts_of_group) == 37


def test_assign_groups_singletons_hit_exact_fractions() -> None:
    keys = [f"k{i:02d}" for i in range(20)]
    out = assign_groups(keys, seed=3)
    assert part_sizes(out.tolist()) == {"train": 16, "val": 2, "test": 2}


def test_assign_groups_custom_fractions() -> None:
    keys = [f"k{i}" for i in range(10)]
    out = assign_groups(keys, seed=0, fractions=(0.5, 0.0, 0.5))
    assert part_sizes(out.tolist()) == {"train": 5, "val": 0, "test": 5}


def test_assign_groups_fractions_must_sum_to_one() -> None:
    with pytest.raises(
        ValueError, match=r"^fractions must sum to 1, got \(0\.8, 0\.1, 0\.2\)$"
    ):
        assign_groups(GROUPS, seed=0, fractions=(0.8, 0.1, 0.2))


def test_random_split_equals_assign_groups_on_keys() -> None:
    keys = ["x", "y", "x", "z", "w", "y"]
    out = random_split(keys, seed=4)
    np.testing.assert_array_equal(out, assign_groups(keys, seed=4))
    assert out[0] == out[2]
    assert out[1] == out[5]


def test_block_groups_exact() -> None:
    # Stable ranks: 10 (index 1) -> 0, 10 (index 3) -> 1, 30 -> 2, 50 -> 3, 70 -> 4,
    # 90 -> 5; block = rank * 3 // 6.
    out = block_groups([50, 10, 30, 10, 90, 70], n_blocks=3)
    assert out.tolist() == [
        "block001",
        "block000",
        "block001",
        "block000",
        "block002",
        "block002",
    ]


def test_block_groups_uneven_count() -> None:
    # Ranks 0..4 for five members in two blocks: rank * 2 // 5 = 0, 0, 0, 1, 1.
    out = block_groups([4, 3, 2, 1, 0], n_blocks=2)
    assert out.tolist() == ["block001", "block001", "block000", "block000", "block000"]


def test_block_groups_one_block() -> None:
    assert block_groups([3, 1, 2], n_blocks=1).tolist() == ["block000"] * 3


def test_block_groups_rejects_zero_blocks() -> None:
    with pytest.raises(ValueError, match=r"^n_blocks must be at least 1$"):
        block_groups([1, 2, 3], n_blocks=0)


# Every ordered combination of the two genes' parts.
TRUTH_TABLE = [
    ("train", "train", "train", 0),
    ("train", "val", "val", 1),
    ("train", "test", "test", 1),
    ("val", "train", "val", 1),
    ("val", "val", "val", 2),
    ("val", "test", "test", 2),
    ("test", "train", "test", 1),
    ("test", "val", "test", 2),
    ("test", "test", "test", 2),
]


def test_pair_parts_truth_table() -> None:
    a = [row[0] for row in TRUTH_TABLE]
    b = [row[1] for row in TRUTH_TABLE]
    assert pair_parts(a, b).tolist() == [row[2] for row in TRUTH_TABLE]


def test_pair_novelty_truth_table() -> None:
    a = [row[0] for row in TRUTH_TABLE]
    b = [row[1] for row in TRUTH_TABLE]
    out = pair_novelty(a, b)
    assert out.dtype == np.int64
    assert out.tolist() == [row[3] for row in TRUTH_TABLE]


def test_part_sizes_exact() -> None:
    parts = ["train", "test", "train", "val", "train", "test"]
    assert part_sizes(parts) == {"train": 3, "val": 1, "test": 2}


def test_part_sizes_ignores_other_labels_and_lists_every_part() -> None:
    assert part_sizes(["train", "other"]) == {"train": 1, "val": 0, "test": 0}

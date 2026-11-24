# torchcell/tasks/splits.py
# [[torchcell.tasks.splits]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/tasks/splits.py
"""Split functions for benchmark tasks: random, grouped, positional and gene-pair.

Every function is deterministic in its inputs and its seed, and assigns whole groups:
a gene, a sequence-similarity cluster or a chromosome block is never divided between
train, validation and test. The parts are the strings ``"train"``, ``"val"`` and
``"test"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

TRAIN, VAL, TEST = "train", "val", "test"
PARTS = (TRAIN, VAL, TEST)
FRACTIONS = (0.8, 0.1, 0.1)

StrArray = npt.NDArray[np.str_]


def assign_groups(
    groups: Sequence[str], seed: int, fractions: tuple[float, float, float] = FRACTIONS
) -> StrArray:
    """Assign every member to a part so that no group is divided.

    The distinct groups are sorted, shuffled with ``seed`` and dealt in that order:
    to train until train holds its fraction of the members, then to validation until
    train plus validation hold theirs, then to test. A group that crosses a boundary
    stays whole on the earlier side, so a part can exceed its fraction by at most one
    group.

    Args:
        groups: The group of each member, one entry per member.
        seed: Seed of the shuffle.
        fractions: Target fractions of members in train, validation and test.

    Returns:
        The part of each member, aligned with ``groups``.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    labels = np.asarray(groups, dtype=str)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    total = len(labels)
    train_end = fractions[0] * total
    val_end = (fractions[0] + fractions[1]) * total
    part_of_group = np.empty(len(unique), dtype="<U5")
    filled = 0
    for index in order:
        if filled < train_end:
            part_of_group[index] = TRAIN
        elif filled < val_end:
            part_of_group[index] = VAL
        else:
            part_of_group[index] = TEST
        filled += int(counts[index])
    return part_of_group[inverse]


def random_split(
    keys: Sequence[str], seed: int, fractions: tuple[float, float, float] = FRACTIONS
) -> StrArray:
    """Random split in which members that share a key stay together."""
    return assign_groups(keys, seed, fractions)


def block_groups(positions: Sequence[int], n_blocks: int) -> StrArray:
    """Group members into ``n_blocks`` contiguous blocks of equal count by position.

    Members are ranked by position (ties by input order) and the ranking is cut into
    ``n_blocks`` runs, so a block is a contiguous stretch of the chromosome.
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    ranks = np.argsort(np.argsort(np.asarray(positions), kind="stable"), kind="stable")
    block = ranks * n_blocks // max(len(ranks), 1)
    return np.array([f"block{b:03d}" for b in block], dtype=str)


def pair_parts(part_a: Sequence[str], part_b: Sequence[str]) -> StrArray:
    """Part of each gene pair from the parts of its two genes.

    A pair is test when either gene is a test gene, validation when neither is a test
    gene and either is a validation gene, and train when both are train genes. No
    training pair therefore holds a validation or test gene, and no validation pair
    holds a test gene.
    """
    a = np.asarray(part_a, dtype=str)
    b = np.asarray(part_b, dtype=str)
    out = np.full(len(a), TRAIN, dtype="<U5")
    out[(a == VAL) | (b == VAL)] = VAL
    out[(a == TEST) | (b == TEST)] = TEST
    return out


def pair_novelty(part_a: Sequence[str], part_b: Sequence[str]) -> npt.NDArray[np.int64]:
    """Number of genes of each pair (0, 1 or 2) that are not training genes."""
    a = np.asarray(part_a, dtype=str)
    b = np.asarray(part_b, dtype=str)
    novel: npt.NDArray[np.int64] = (a != TRAIN).astype(np.int64) + (b != TRAIN).astype(
        np.int64
    )
    return novel


def part_sizes(parts: Sequence[str]) -> dict[str, int]:
    """Number of members in each part."""
    array = np.asarray(parts, dtype=str)
    return {part: int((array == part).sum()) for part in PARTS}

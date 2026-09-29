# tests/torchcell/knowledge_graphs/test_subset.py
# [[tests.torchcell.knowledge_graphs.test_subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_subset.py
"""Prefilter and subsample: ``RecordFilter``, ``load_gene_sets``, ``select_indices``,
``subset_dataset``.

Fixture for ``select_indices``: a real LMDB under ``tmp_path`` with twelve records keyed
``b"0"`` .. ``b"11"``, each a pickled ``{"experiment": {"genotype": {"perturbations":
[{"systematic_gene_name": g}, ...]}}}`` with gene sets 0:{A} 1:{B} 2:{A,B} 3:{C} 4:{A,C}
5:{B,C} 6:{A,B,C} 7:{D} 8:{A,D} 9:{B,D} 10:{A,B} 11:{C,D}. Twelve records matter: LMDB
iterates keys bytewise (``"10"`` before ``"2"``), so a filter matching records 2 and 10
returns ``[2, 10]`` only because the indices are sorted numerically after decoding.

``subset_dataset`` is exercised on a ten-record indexable stand-in whose ``__getitem__``
returns ``("view", indices)``. Seeded samples, computed with ``random.Random(seed)
.sample(pool, size)`` and sorted: pool range(10), seed 42, size 3 -> [0, 1, 4]; pool
[4, 1, 3], seed 42, size 2 -> [3, 4].

The relative-path branch of ``load_gene_sets`` resolves against the installed package,
so it is checked on the committed ``knowledge_graphs/conf/gene_sets/sameith_double_pairs
.txt``: 72 pairs, the first ``YBL054W YER088C``, equal to reading the same file by its
absolute path.
"""

from __future__ import annotations

import os.path as osp
import pickle
import re
from pathlib import Path
from typing import Any

import lmdb
import pytest
from pydantic import ValidationError

import torchcell
from torchcell.knowledge_graphs.subset import (
    RecordFilter,
    _perturbation_summary,
    load_gene_sets,
    select_indices,
    subset_dataset,
)

GENE_SETS: dict[int, list[str]] = {
    0: ["A"],
    1: ["B"],
    2: ["A", "B"],
    3: ["C"],
    4: ["A", "C"],
    5: ["B", "C"],
    6: ["A", "B", "C"],
    7: ["D"],
    8: ["A", "D"],
    9: ["B", "D"],
    10: ["A", "B"],
    11: ["C", "D"],
}


def _record(genes: list[str]) -> dict[str, Any]:
    return {
        "experiment": {
            "genotype": {"perturbations": [{"systematic_gene_name": g} for g in genes]}
        }
    }


@pytest.fixture
def lmdb_path(tmp_path: Path) -> str:
    """Twelve pickled records under integer string keys."""
    path = str(tmp_path / "lmdb")
    env = lmdb.open(path, map_size=2**24)
    with env.begin(write=True) as txn:
        for index, genes in GENE_SETS.items():
            txn.put(str(index).encode(), pickle.dumps(_record(genes)))
    env.close()
    return path


# -------------------------------------------------------------------- RecordFilter


def test_an_all_unset_filter_is_rejected() -> None:
    """No criterion, or an empty ``gene_sets`` list, fails validation with the reason."""
    message = re.escape(
        "RecordFilter has no criteria set; it would match nothing. "
        "Set gene_sets/gene_sets_file, any_genes, all_genes, or n_perturbations."
    )
    with pytest.raises(ValidationError, match=message):
        RecordFilter()
    with pytest.raises(ValidationError, match=message):
        RecordFilter(gene_sets=[])


def test_matches_applies_every_set_criterion_as_an_and() -> None:
    """gene_sets compares as frozensets (order-free); any/all/n_perturbations each
    narrow; two criteria must both hold.
    """
    by_set = RecordFilter(gene_sets=[["B", "A"], ["C"]])
    assert by_set.matches(frozenset({"A", "B"}), 2) is True
    assert by_set.matches(frozenset({"C"}), 1) is True
    assert by_set.matches(frozenset({"A"}), 1) is False
    assert by_set.matches(frozenset({"A", "B", "C"}), 3) is False

    any_d = RecordFilter(any_genes=["D"])
    assert any_d.matches(frozenset({"A", "D"}), 2) is True
    assert any_d.matches(frozenset({"A", "B"}), 2) is False

    all_ab = RecordFilter(all_genes=["A", "B"])
    assert all_ab.matches(frozenset({"A", "B", "C"}), 3) is True
    assert all_ab.matches(frozenset({"A", "C"}), 2) is False

    singles = RecordFilter(n_perturbations=[1])
    assert singles.matches(frozenset({"A"}), 1) is True
    assert singles.matches(frozenset({"A", "B"}), 2) is False

    both = RecordFilter(any_genes=["D"], n_perturbations=[2])
    assert both.matches(frozenset({"A", "D"}), 2) is True
    assert both.matches(frozenset({"D"}), 1) is False
    assert both.matches(frozenset({"A", "B"}), 2) is False


def test_perturbation_summary_reads_names_and_count() -> None:
    """The raw record's systematic names as a frozenset, with the perturbation count."""
    assert _perturbation_summary(_record(["B", "A"])) == (frozenset({"A", "B"}), 2)
    assert _perturbation_summary(_record([])) == (frozenset(), 0)


# ------------------------------------------------------------------ load_gene_sets


def test_load_gene_sets_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    """One set per non-empty line, whitespace-split, with ``#`` comments stripped."""
    path = tmp_path / "sets.txt"
    path.write_text(
        "# Generated by a script\nYAL001C YAL002W   # a trailing comment\n\n"
        "   \nYBR001C\n#\n"
    )
    assert load_gene_sets(str(path)) == [["YAL001C", "YAL002W"], ["YBR001C"]]


def test_relative_path_resolves_against_the_torchcell_package() -> None:
    """``knowledge_graphs/conf/gene_sets/sameith_double_pairs.txt`` reads the committed
    72 Sameith pairs, the same list as the absolute path gives.
    """
    relative = "knowledge_graphs/conf/gene_sets/sameith_double_pairs.txt"
    sets = load_gene_sets(relative)
    assert len(sets) == 72
    assert sets[0] == ["YBL054W", "YER088C"]
    assert sets == load_gene_sets(
        osp.join(osp.dirname(osp.abspath(torchcell.__file__)), relative)
    )


def test_gene_sets_file_is_folded_after_the_inline_sets(tmp_path: Path) -> None:
    """The file's sets are appended to ``gene_sets`` and the file alone is a criterion."""
    path = tmp_path / "sets.txt"
    path.write_text("P Q\nR\n")
    merged = RecordFilter(gene_sets=[["X", "Y"]], gene_sets_file=str(path))
    assert merged.gene_sets == [["X", "Y"], ["P", "Q"], ["R"]]
    assert merged.gene_sets_file == str(path)
    alone = RecordFilter(gene_sets_file=str(path))
    assert alone.gene_sets == [["P", "Q"], ["R"]]
    assert alone.matches(frozenset({"Q", "P"}), 2) is True


# ------------------------------------------------------------------ select_indices


def test_select_indices_returns_matching_keys_in_numeric_order(lmdb_path: str) -> None:
    """{A,B} matches records 2 and 10 as [2, 10] (bytewise key order would give
    [10, 2]); singles are [0, 1, 3, 7]; D-containing pairs are [8, 9, 11]; A and B
    together are [2, 6, 10]; an unmatched set gives [].
    """
    assert select_indices(lmdb_path, RecordFilter(gene_sets=[["B", "A"]])) == [2, 10]
    assert select_indices(lmdb_path, RecordFilter(n_perturbations=[1])) == [0, 1, 3, 7]
    assert select_indices(
        lmdb_path, RecordFilter(any_genes=["D"], n_perturbations=[2])
    ) == [8, 9, 11]
    assert select_indices(lmdb_path, RecordFilter(all_genes=["A", "B"])) == [2, 6, 10]
    assert select_indices(lmdb_path, RecordFilter(gene_sets=[["Z"]])) == []


# ------------------------------------------------------------------ subset_dataset


class _Indexable:
    """Ten records; indexing with a list returns a tagged view."""

    def __init__(self, n: int = 10) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, indices: list[int]) -> tuple[str, list[int]]:
        return ("view", list(indices))


def test_subset_dataset_returns_the_dataset_itself_when_nothing_narrows() -> None:
    """``size`` None and no prefilter is the real build: the same object comes back."""
    dataset = _Indexable()
    assert subset_dataset(dataset, None) is dataset
    assert subset_dataset(dataset, None, 7, None) is dataset


def test_subset_dataset_takes_the_whole_pool_when_size_covers_it() -> None:
    """A prefiltered pool is indexed sorted; ``size`` >= pool takes all of it, and
    ``size`` == len(dataset) with no prefilter indexes every record (a view, not the
    dataset).
    """
    dataset = _Indexable()
    assert subset_dataset(dataset, None, 42, [4, 1, 3]) == ("view", [1, 3, 4])
    assert subset_dataset(dataset, 5, 42, [4, 1, 3]) == ("view", [1, 3, 4])
    assert subset_dataset(dataset, 3, 42, [4, 1, 3]) == ("view", [1, 3, 4])
    assert subset_dataset(dataset, 10) == ("view", list(range(10)))


def test_subset_dataset_samples_with_the_seed_and_sorts() -> None:
    """range(10) at seed 42, size 3 -> [0, 1, 4] (also the default seed); pool
    [4, 1, 3] at seed 42, size 2 -> [3, 4].
    """
    dataset = _Indexable()
    assert subset_dataset(dataset, 3, 42) == ("view", [0, 1, 4])
    assert subset_dataset(dataset, 3) == ("view", [0, 1, 4])
    assert subset_dataset(dataset, 2, 42, [4, 1, 3]) == ("view", [3, 4])

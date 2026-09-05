"""Tests for the Neo4jCellDataset.

Focused on `__getstate__`, which decides what a spawned DataLoader worker receives. On
the 13.5M-record 025 build the bulk index caches are 0.62 GB of a 0.65 GB pickle, and 56
worker copies of them OOM-killed a 250 GB allocation (job 1597) during worker spawn. The
properties that would silently make the caches reappear in the pickle are the risk, so
the attribute list and the drop are asserted here rather than trusted.
"""

import pickle
from typing import Any

import pandas as pd

from torchcell.data.neo4j_cell import Neo4jCellDataset


def _stub(**overrides: Any) -> Neo4jCellDataset:
    """A dataset instance carrying only the attributes `__getstate__` reads.

    Built with `object.__new__` on purpose: `__init__` opens a real build, and
    `__getstate__` is a pure transform of `__dict__` that needs none of it.
    """
    dataset = object.__new__(Neo4jCellDataset)
    dataset.__dict__.update(
        {
            "env": "an open lmdb environment",
            "root": "/some/build",
            "phenotype_labels": ["gene_interaction"],
            "_phenotype_label_index": {"gene_interaction": [0, 1, 2]},
            "_dataset_name_index": {"kuzmin2020": [0, 1]},
            "_perturbation_count_index": {3: [0, 1, 2]},
            "_is_any_perturbed_gene_index": {"YAL001C": [0]},
            "_is_any_deletion_gene_index_cache": {"YAL001C": [0]},
            "_label_df": pd.DataFrame(
                {"index": [0, 1], "gene_interaction": [0.1, 0.2]}
            ),
            **overrides,
        }
    )
    return dataset


def test_getstate_drops_the_lmdb_environment() -> None:
    """The pre-existing contract: an open LMDB env cannot be pickled."""
    assert _stub().__getstate__()["env"] is None


def test_getstate_drops_every_bulk_cache() -> None:
    """None of the six bulk caches may travel to a worker."""
    state = _stub().__getstate__()
    for name in Neo4jCellDataset._WORKER_DROPPED_CACHES:
        assert state[name] is None, f"{name} would be pickled to every worker"


def test_getstate_keeps_everything_else() -> None:
    """The drop is targeted: attributes a worker needs are untouched.

    `get()` reads the LMDB record, `cell_graph`, and `phenotype_info`; dropping more than
    the caches would break item construction in the worker rather than in the parent,
    which is the harder place to see it.
    """
    state = _stub(
        cell_graph="hetero-data", _phenotype_info=["GeneInteraction"]
    ).__getstate__()
    assert state["cell_graph"] == "hetero-data"
    assert state["_phenotype_info"] == ["GeneInteraction"]
    assert state["root"] == "/some/build"
    assert state["phenotype_labels"] == ["gene_interaction"]


def test_getstate_does_not_mutate_the_live_dataset() -> None:
    """Pickling must not empty the caches in the PARENT process.

    `__getstate__` copies `__dict__` before clearing. Clearing in place would work once
    and then quietly strip the parent, whose `CellDataModule` reads these indices.
    """
    dataset = _stub()
    dataset.__getstate__()
    assert dataset._phenotype_label_index == {"gene_interaction": [0, 1, 2]}
    assert dataset.env == "an open lmdb environment"


def test_pickle_round_trip_is_small_and_restores() -> None:
    """End to end: the caches are absent from the bytes, and the rest survives."""
    dataset = _stub(cell_graph="hetero-data")
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.cell_graph == "hetero-data"
    assert restored.env is None
    assert restored._label_df is None
    assert restored._perturbation_count_index is None


def test_dropped_cache_names_are_real_attributes() -> None:
    """Guards against a rename silently un-dropping a cache.

    A typo in `_WORKER_DROPPED_CACHES` is invisible: `__getstate__` skips a name that is
    not in `__dict__`, so the real attribute keeps travelling and the pickle quietly
    grows back to its old size.
    """
    live = _stub().__dict__
    for name in Neo4jCellDataset._WORKER_DROPPED_CACHES:
        assert name in live, f"{name} is not an attribute of Neo4jCellDataset"

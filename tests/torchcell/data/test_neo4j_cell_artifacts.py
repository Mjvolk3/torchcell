# tests/torchcell/data/test_neo4j_cell_artifacts.py
# [[tests.torchcell.data.test_neo4j_cell_artifacts]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_cell_artifacts.py
"""``Neo4jCellDataset.refs_of`` and ``Neo4jCellDataset.materialize``, no Neo4j.

The processed LMDB is written by hand (the aggregator's JSON-array shape, as in
test_neo4j_cell_hermetic_build.py) with the ``"artifact fitness"`` test family of
test_neo4j_query_raw_artifacts.py registered in the cell module's
``EXPERIMENT_TYPE_MAP``:

* entry 0 aggregates two records, refs [A] and [A2, B] (A2 is A's file);
* entry 1 is one record with no ref.

``refs_of(0)`` is [A, B] (A2 collapses onto A), ``refs_of(1)`` is [], and neither consults
an artifact source. ``materialize`` resolves through ``DATA_ROOT``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
import pytest

from tests.torchcell.artifacts._fakes import write_tier
from tests.torchcell.data.test_neo4j_query_raw_artifacts import (
    A2,
    A,
    B,
    RefFitnessExperiment,
    _plain_record,
    _ref_record,
)
from torchcell.data import neo4j_cell as nc
from torchcell.data.graph_processor import Perturbation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels import schema as s
from torchcell.sequence import GeneSet

# The package re-exports the function ``resolve`` under the submodule's name.
artifact_resolve = importlib.import_module("torchcell.artifacts.resolve")
GENES = GeneSet(["YAL001C", "YAL002W", "YAL003W"])
ENTRIES = [
    [_ref_record("YAL001C", ref=A), _ref_record("YAL002W", ref=A2, refs=[B])],
    [_plain_record("YAL003W")],
]


def _payload(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        [
            {
                "experiment": r["experiment"].model_dump(mode="json"),
                "experiment_reference": r["experiment_reference"].model_dump(
                    mode="json"
                ),
            }
            for r in records
        ]
    ).encode()


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Neo4jCellDataset]:
    """The dataset over a hand-written processed build holding ``ENTRIES``."""
    monkeypatch.setattr(
        nc,
        "EXPERIMENT_TYPE_MAP",
        {**s.EXPERIMENT_TYPE_MAP, "artifact fitness": RefFitnessExperiment},
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    env = lmdb.open(str(processed / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for idx, records in enumerate(ENTRIES):
            txn.put(str(idx).encode(), _payload(records))
    env.close()
    (processed / "experiment_types.json").write_text(
        json.dumps(["artifact fitness", "fitness"])
    )
    ds = Neo4jCellDataset(
        root=str(tmp_path), gene_set=GENES, graph_processor=Perturbation()
    )
    yield ds
    ds.close_lmdb()


def test_refs_of_lists_an_entrys_distinct_refs_without_resolving(
    dataset: Neo4jCellDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[A, B] for the two-record entry, [] for the plain one; the resolver is never
    reached (it raises if it is).
    """

    def no_io(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("refs_of reached the resolver")

    monkeypatch.setattr(artifact_resolve, "resolve", no_io)
    monkeypatch.setattr(nc, "materialize_artifact", no_io)
    assert dataset.refs_of(0) == [A, B]
    assert dataset.refs_of(1) == []


def test_refs_of_a_missing_entry_raises(dataset: Neo4jCellDataset) -> None:
    """Entry 2 does not exist."""
    with pytest.raises(IndexError, match="no processed entry at index 2"):
        dataset.refs_of(2)


def test_materialize_resolves_against_data_root(
    dataset: Neo4jCellDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``DATA_ROOT`` holding the objects tier, ``materialize(B)`` is the tier's
    ``b.npy`` with B's bytes.
    """
    data_root = tmp_path / "data_root"
    write_tier(data_root, "objects", "set-a", {"a.npy": b"alpha", "b.npy": b"beta"})
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    path = dataset.materialize(B)
    assert path == data_root / "torchcell-objects" / "set-a" / "b.npy"
    assert path.read_bytes() == b"beta"

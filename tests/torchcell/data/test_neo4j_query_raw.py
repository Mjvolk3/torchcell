# tests/torchcell/data/test_neo4j_query_raw.py
# [[tests.torchcell.data.test_neo4j_query_raw]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_neo4j_query_raw.py
"""``Neo4jQueryRaw`` over an LMDB written by hand under ``tmp_path``, no Neo4j.

``__attrs_post_init__`` runs the query only when
``<root>/raw/lmdb/data.mdb`` is absent, so the fixture writes that store first and the
constructor opens it read-only without touching a driver. ``process()`` is reached by
replacing ``fetch_data`` on the class with an iterator over dict records (the class is
attrs-slotted, so the method cannot be set on an instance).

Three fitness records, one per key ``data_<i>``:

* 0: ``toy_a``, YAL001C deleted, fitness 0.9.
* 1: ``toy_a``, YAL002W and YAL003W deleted, fitness 0.4.
* 2: ``toy_b``, YAL003W deleted, fitness 0.7.

Derived by hand: ``len`` 3; the reference index groups by dataset name since the
reference differs only there, ``toy_a`` -> [0, 1], ``toy_b`` -> [2]; the gene set is the
three deleted genes sorted; the value ``process()`` stores under ``data_0`` is exactly
the JSON written out literally in ``EXPECTED_RECORD_0``.

2026.09.30 (Phase 17): ``fetch_data`` against a recording fake ``GraphDatabase`` (the
module attribute is replaced, so no driver is ever opened) and a fake
``releases.list_databases`` that serves one online database ``torchcell`` with the
aliases ``latest`` and ``pinned``, so the real ``resolve_database`` runs its alias
pass-through with ``probe=False``. The expected call sequence is driver(uri, auth), then
``session(database=<resolved>, fetch_size=1000)``, ``run(query, **cypher_kwargs)``, the
session exit, then ``driver.close()``; the version is the instance's own when set and
``TORCHCELL_KG_VERSION`` otherwise. A consumer that stops after the first record still
closes the driver (the close is in a ``finally``). A construction whose query returns no
records raises ``EmptyQueryResultError`` before any store file exists, so a retry runs
the query again (both retired Findings of issue #541). ``parallel_hash_computation`` returns ``(idx, sha256(json.dumps(ref,
sort_keys=True)))``, recomputed here with ``hashlib``; ``_get_record`` on a missing key
raises ``Record not found for key: data_9``; and a cached reference index is returned
without rewriting a deleted JSON file.
"""

import hashlib
import json
import multiprocessing
import pickle
import re
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
import pytest

import torchcell.data.neo4j_query_raw as neo4j_query_raw
import torchcell.knowledge_graphs.releases as releases
from torchcell.data.neo4j_query_raw import (
    Neo4jQueryRaw,
    compute_experiment_reference_index,
    compute_experiment_reference_index_parallel,
    parallel_hash_computation,
    partition_queries,
)
from torchcell.datamodels.interned_constant import constant_id
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
)

ENVIRONMENT = Environment(media=Media(name="YPD", state="solid", is_synthetic=False))
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
URI = "bolt://example.invalid:7687"
QUERY = "MATCH (e:Experiment)<-[:ExperimentReferenceOf]-(ref) RETURN e, ref"


def _record(name: str, genes: list[str], fitness: float) -> dict[str, Any]:
    perturbations: list[Any] = [
        KanMxDeletionPerturbation(systematic_gene_name=g, perturbed_gene_name=g)
        for g in genes
    ]
    return {
        "experiment": FitnessExperiment(
            dataset_name=name,
            genotype=Genotype(perturbations=perturbations),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=name,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


RECORDS = [
    _record("toy_a", ["YAL001C"], 0.9),
    _record("toy_a", ["YAL002W", "YAL003W"], 0.4),
    _record("toy_b", ["YAL003W"], 0.7),
]

_ENV_JSON: dict[str, Any] = {
    "provenance_gaps": [],
    "media": {
        "name": "YPD",
        "state": "solid",
        "is_synthetic": False,
        "base_medium": None,
        "components": [],
        "dropouts": [],
        "provenance": [],
    },
    "temperature": None,
    "perturbations": [],
    "aerobicity": "aerobic",
    "duration_hours": None,
    "duration_generations": None,
}


def _fitness_json(fitness: float) -> dict[str, Any]:
    return {
        "provenance_gaps": [],
        "graph_level": "global",
        "label_name": "fitness",
        "label_statistic_name": "fitness_se",
        "fitness": fitness,
        "fitness_se": None,
        "fitness_std": None,
        "n_samples": None,
        "fitness_uncertainty": None,
        "fitness_uncertainty_type": None,
        "sample_unit": None,
        "screen_id": None,
    }


EXPECTED_RECORD_0 = {
    "experiment": {
        "experiment_type": "fitness",
        "dataset_name": "toy_a",
        "genotype": {
            "perturbations": [
                {
                    "systematic_gene_name": "YAL001C",
                    "perturbed_gene_name": "YAL001C",
                    "provenance": "engineered",
                    "state": "absent",
                    "mechanism_so_id": "SO:0000159",
                    "mechanism_so_name": "deletion",
                    "description": "Deletion via KanMX or NatMX gene replacement",
                    "perturbation_type": "kanmx_deletion",
                    "deletion_description": "Deletion via KanMX gene replacement.",
                    "deletion_type": "KanMX",
                }
            ]
        },
        "environment": _ENV_JSON,
        "phenotype": _fitness_json(0.9),
    },
    "experiment_reference": {
        "experiment_reference_type": "fitness",
        "dataset_name": "toy_a",
        "genome_reference": {
            "species": "Saccharomyces cerevisiae",
            "strain": "S288C",
            "ploidy": "haploid",
        },
        "environment_reference": _ENV_JSON,
        "phenotype_reference": _fitness_json(1.0),
    },
}


def _serialize(record: dict[str, Any]) -> bytes:
    return json.dumps(record, default=lambda o: o.model_dump()).encode()


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    """``<root>/raw/lmdb`` holding RECORDS under ``data_0..2``; the env is closed."""
    lmdb_dir = tmp_path / "raw" / "lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=10**8)
    with env.begin(write=True) as txn:
        for i, record in enumerate(RECORDS):
            txn.put(f"data_{i}".encode(), _serialize(record))
    env.close()
    return tmp_path


@pytest.fixture
def view(store_root: Path) -> Iterator[Neo4jQueryRaw]:
    """A ``Neo4jQueryRaw`` opened over the pre-written store; closed at teardown."""
    raw = Neo4jQueryRaw(
        uri=URI, username="u", password="p", root_dir=str(store_root), query=QUERY
    )
    yield raw
    raw.close_lmdb()


def test_constructor_skips_the_query_when_data_mdb_exists_and_opens_read_only(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """Paths derive from root_dir; the env is open on the LMDB dir; repr names uri, root, query."""
    assert view.raw_dir == str(store_root / "raw")
    assert view.lmdb_dir == str(store_root / "raw" / "lmdb")
    assert view.env.path() == str(store_root / "raw" / "lmdb")
    assert view.env.flags()["readonly"] is True
    assert repr(view) == (
        f"Neo4jQueryRaw(uri={URI}, root_dir={store_root}, query={QUERY})"
    )
    assert sorted(p.name for p in (store_root / "raw").iterdir()) == ["lmdb"]


def test_len_counts_entries_and_closes_the_environment(view: Neo4jQueryRaw) -> None:
    """``__len__`` reads the entry count and then closes the environment (before the fix
    it returned inside the transaction and the close never ran). A slice after ``len``
    reopens it for the threaded reads.
    """
    assert len(view) == 3
    assert view.env is None
    assert view[0:2] == [RECORDS[0], RECORDS[1]]


def test_getitem_by_int_slice_and_list_rebuilds_the_pydantic_records(
    view: Neo4jQueryRaw,
) -> None:
    """Each access equals the record written; a missing key is IndexError, a str TypeError."""
    assert view[0] == RECORDS[0]
    assert type(view[0]["experiment"]) is FitnessExperiment
    assert type(view[0]["experiment_reference"]) is FitnessExperimentReference
    assert view[0:3:2] == [RECORDS[0], RECORDS[2]]
    assert view[[2, 0]] == [RECORDS[2], RECORDS[0]]
    with pytest.raises(IndexError, match="Record not found at index: 5"):
        view[5]
    with pytest.raises(TypeError, match=r"Invalid index type: <class 'str'>"):
        view["data_0"]  # type: ignore[index]  # the error path under test


def test_experiment_reference_index_streams_the_store_and_persists_json(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """Groups by reference: toy_a -> [0, 1], toy_b -> [2]; the JSON file is the model dumps
    and is read back in preference to recomputing.
    """
    index = view.experiment_reference_index
    assert [(e.reference.dataset_name, e.member_indices) for e in index] == [
        ("toy_a", [0, 1]),
        ("toy_b", [2]),
    ]
    assert index[0].reference == RECORDS[0]["experiment_reference"]
    assert view.env is None
    path = store_root / "raw" / "experiment_reference_index.json"
    assert json.loads(path.read_text(encoding="utf-8")) == [
        e.model_dump() for e in index
    ]
    path.write_text(
        json.dumps(
            [{"reference": index[1].reference.model_dump(), "member_indices": [2]}]
        ),
        encoding="utf-8",
    )
    assert [
        (e.reference.dataset_name, e.member_indices)
        for e in view.experiment_reference_index
    ] == [("toy_b", [2])]


def test_phenotype_label_index_groups_records_by_label_name_and_persists_json(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """All three records are ``FitnessPhenotype`` with ``label_name`` "fitness", so the
    index is ``{"fitness": [0, 1, 2]}`` and is written to ``phenotype_label_index.json``
    (before the fix it read ``phenotype.label`` and raised ``AttributeError``). A file
    already on disk is read back in preference to recomputing.
    """
    assert view.phenotype_label_index == {"fitness": [0, 1, 2]}
    path = store_root / "raw" / "phenotype_label_index.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"fitness": [0, 1, 2]}
    path.write_text('{"fitness": [2]}', encoding="utf-8")
    assert view.phenotype_label_index == {"fitness": [2]}


def test_gene_set_is_computed_from_perturbations_and_the_setter_writes_sorted_json(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """The getter computes without writing; the setter writes ``indent=0`` JSON that the
    getter then reads back; an empty value is refused.
    """
    gene_set_path = store_root / "raw" / "gene_set.json"
    assert list(view.gene_set) == ["YAL001C", "YAL002W", "YAL003W"]
    assert view.env is None
    assert not gene_set_path.exists()
    view.gene_set = view.compute_gene_set()
    assert gene_set_path.read_text(encoding="utf-8") == (
        '[\n"YAL001C",\n"YAL002W",\n"YAL003W"\n]'
    )
    gene_set_path.write_text('["YZZ999W"]', encoding="utf-8")
    assert list(view.gene_set) == ["YZZ999W"]
    with pytest.raises(
        ValueError, match="Cannot set an empty or None value for gene_set"
    ):
        view.gene_set = view.gene_set.__class__()


def test_extract_systematic_gene_names_lists_every_perturbation_in_order() -> None:
    """Two deletions give two names in genotype order."""
    genotype = RECORDS[1]["experiment"].genotype.model_dump()
    assert Neo4jQueryRaw.extract_systematic_gene_names(genotype) == [
        "YAL002W",
        "YAL003W",
    ]


def test_close_lmdb_is_idempotent_and_a_closed_view_pickles_and_reopens(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """Attrs' slotted ``__getstate__`` is the field dict with ``env`` None; the copy counts
    the same store (``len`` opens and closes it) and reopens the same LMDB directory on its
    first record read.
    """
    view.close_lmdb()
    view.close_lmdb()
    assert view.env is None
    assert view.__getstate__() == {
        "uri": URI,
        "username": "u",
        "password": "p",
        "root_dir": str(store_root),
        "query": QUERY,
        "io_workers": None,
        "num_workers": None,
        "_experiment_reference_index": None,
        "_phenotype_label_index": None,
        "lmdb_dir": str(store_root / "raw" / "lmdb"),
        "raw_dir": str(store_root / "raw"),
        "env": None,
        "_gene_set": None,
        "cypher_kwargs": {},
        "version": None,
        "record_observers": [],
        "raw_stage_ran": False,
        "fetch_workers": 0,
        "partition_prefix_length": 1,
        "_stream": neo4j_query_raw._StreamState(),
    }
    copy = pickle.loads(pickle.dumps(view))
    assert copy.__getstate__()["env"] is None
    assert len(copy) == 3
    assert copy.__getstate__()["env"] is None
    assert copy[2] == RECORDS[2]
    assert copy.env.path() == str(store_root / "raw" / "lmdb")
    copy.close_lmdb()


def test_write_to_lmdb_puts_one_key_in_a_writable_environment(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """``_init_lmdb(readonly=False)`` reopens writable; the put is visible in the next txn."""
    view._init_lmdb(readonly=False)
    assert view.env.flags()["readonly"] is False
    view.write_to_lmdb(b"data_3", b"x")
    with view.env.begin() as txn:
        assert txn.get(b"data_3") == b"x"
    assert len(view) == 4


def test_process_accepts_both_record_shapes_and_builds_the_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A property-shape record (``e_serialized``) and a node-shape record (``e`` with
    ``serialized_data``) both land as ``data_<i>``; the reference index and gene set are
    written beside the store. The bytes ``process()`` wrote under ``data_0`` are exactly
    ``json.dumps(EXPECTED_RECORD_0)``, the literal JSON spelled out above.
    """
    property_shape = {
        "e_serialized": json.dumps(RECORDS[0]["experiment"].model_dump()),
        "ref_serialized": json.dumps(RECORDS[0]["experiment_reference"].model_dump()),
    }
    node_shape = {
        "e": {"serialized_data": json.dumps(RECORDS[2]["experiment"].model_dump())},
        "ref": {
            "serialized_data": json.dumps(
                RECORDS[2]["experiment_reference"].model_dump()
            )
        },
    }
    monkeypatch.setattr(
        Neo4jQueryRaw, "fetch_data", lambda self: iter([property_shape, node_shape])
    )
    raw = Neo4jQueryRaw(
        uri=URI, username="u", password="p", root_dir=str(tmp_path), query=QUERY
    )
    assert len(raw) == 2
    assert raw[0] == RECORDS[0]
    assert raw[1] == RECORDS[2]
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == [
        "experiment_reference_index.json",
        "gene_set.json",
        "lmdb",
    ]
    assert (tmp_path / "raw" / "gene_set.json").read_text(encoding="utf-8") == (
        '[\n"YAL001C",\n"YAL003W"\n]'
    )
    stored_index = json.loads(
        (tmp_path / "raw" / "experiment_reference_index.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        (e["reference"]["dataset_name"], e["member_indices"]) for e in stored_index
    ] == [("toy_a", [0]), ("toy_b", [1])]
    raw.close_lmdb()
    env = lmdb.open(str(tmp_path / "raw" / "lmdb"), readonly=True)
    with env.begin() as txn:
        stored = txn.get(b"data_0")
    env.close()
    assert stored == json.dumps(EXPECTED_RECORD_0).encode()


def test_compute_experiment_reference_index_groups_records_by_reference_hash() -> None:
    """Sequential path: toy_a -> [0, 1], toy_b -> [2], references taken from the first member."""
    index = compute_experiment_reference_index(RECORDS)
    assert [(e.reference.dataset_name, e.member_indices) for e in index] == [
        ("toy_a", [0, 1]),
        ("toy_b", [2]),
    ]
    assert index[1].reference == RECORDS[2]["experiment_reference"]


def test_parallel_and_sequential_paths_give_the_same_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both paths read ``experiment_reference``, so ``num_workers=1``, ``num_workers=None``
    and ``compute_experiment_reference_index_parallel`` agree exactly: toy_a -> [0, 1],
    toy_b -> [2] (before the fix the parallel path read ``"reference"`` and raised
    ``KeyError`` on these records). ``mp.cpu_count`` is patched to 1 so the parallel
    helper forks one worker, not one per core.
    """
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 1)
    sequential = compute_experiment_reference_index(RECORDS)
    parallel = compute_experiment_reference_index(RECORDS, num_workers=1)
    helper = compute_experiment_reference_index_parallel(RECORDS)
    expected = [("toy_a", [0, 1]), ("toy_b", [2])]
    for index in (sequential, parallel, helper):
        assert [(e.reference.dataset_name, e.member_indices) for e in index] == expected
    assert [e.model_dump() for e in parallel] == [e.model_dump() for e in sequential]
    assert [e.model_dump() for e in helper] == [e.model_dump() for e in sequential]


def test_parallel_hash_computation_hashes_the_sorted_reference_json() -> None:
    """The worker returns its index untouched and the sha256 of the reference dump with
    sorted keys; a record that carries only ``reference`` (the dataset item key) is a
    ``KeyError`` on ``experiment_reference``.
    """
    reference = RECORDS[2]["experiment_reference"]
    expected = hashlib.sha256(
        json.dumps(reference.model_dump(), sort_keys=True).encode()
    ).hexdigest()
    assert parallel_hash_computation((7, RECORDS[2])) == (7, expected)
    assert parallel_hash_computation((0, RECORDS[0]))[1] != expected
    with pytest.raises(KeyError, match="experiment_reference"):
        parallel_hash_computation((0, {"reference": reference}))


def test_get_record_by_key_refuses_a_missing_key(view: Neo4jQueryRaw) -> None:
    """The slice reader's per-key helper names the missing key; a present key rebuilds."""
    view._init_lmdb()
    assert view._get_record(b"data_1") == RECORDS[1]
    with pytest.raises(IndexError, match="Record not found for key: data_9"):
        view._get_record(b"data_9")


def test_cached_reference_index_is_returned_without_rewriting_a_deleted_file(
    store_root: Path, view: Neo4jQueryRaw
) -> None:
    """Once computed, the index is held on the instance: deleting the JSON file and
    reading again returns the same groups and does not recreate the file.
    """
    first = view.experiment_reference_index
    path = store_root / "raw" / "experiment_reference_index.json"
    path.unlink()
    again = view.experiment_reference_index
    assert [e.model_dump() for e in again] == [e.model_dump() for e in first]
    assert [e.member_indices for e in again] == [[0, 1], [2]]
    assert not path.exists()


class _FakeNeo4j:
    """Records every driver, session and run call; ``run`` returns ``records``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[tuple[Any, ...]] = []
        self.listings: list[tuple[Any, ...]] = []

    def driver(self, uri: str, auth: tuple[str, str]) -> "_FakeNeo4j":
        self.calls.append(("driver", uri, auth))
        return self

    def session(self, **kwargs: Any) -> "_FakeNeo4j":
        self.calls.append(("session", kwargs))
        return self

    def __enter__(self) -> "_FakeNeo4j":
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append(("session_exit",))

    def run(self, query: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        self.calls.append(("run", query, kwargs))
        return iter(self.records)

    def close(self) -> None:
        self.calls.append(("close",))


_SERVED = [
    releases.ServedDatabase(
        name="torchcell", aliases=["latest", "pinned"], default=True, status="online"
    )
]


def _property_shape(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "e_serialized": json.dumps(record["experiment"].model_dump()),
        "ref_serialized": json.dumps(record["experiment_reference"].model_dump()),
    }


@pytest.fixture
def fake_neo4j(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeNeo4j]:
    """Replace the module's ``GraphDatabase`` and the served-database listing."""
    fake = _FakeNeo4j([_property_shape(RECORDS[0]), _property_shape(RECORDS[2])])

    def list_databases(
        uri: str, user: str, password: str, *, probe: bool = True
    ) -> list[releases.ServedDatabase]:
        fake.listings.append((uri, user, password, probe))
        return _SERVED

    monkeypatch.setattr(neo4j_query_raw, "GraphDatabase", fake)
    monkeypatch.setattr(releases, "list_databases", list_databases)
    yield fake


def test_fetch_data_resolves_the_env_version_and_closes_the_driver_after_the_last_record(
    view: Neo4jQueryRaw, fake_neo4j: _FakeNeo4j, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no instance version, ``TORCHCELL_KG_VERSION=pinned`` is resolved (an alias of
    the served database, so it passes through) and opened with ``fetch_size=1000``; the
    query runs with no parameters; both records are yielded in order; the session exits
    and the driver closes exactly once, after the last record.
    """
    monkeypatch.setenv("TORCHCELL_KG_VERSION", "pinned")
    records = list(view.fetch_data())
    assert records == fake_neo4j.records
    assert fake_neo4j.listings == [(URI, "u", "p", False)]
    assert fake_neo4j.calls == [
        ("driver", URI, ("u", "p")),
        ("session", {"database": "pinned", "fetch_size": 1000}),
        ("run", QUERY, {}),
        ("session_exit",),
        ("close",),
    ]


def test_a_query_built_store_uses_the_instance_version_and_cypher_parameters(
    tmp_path: Path, fake_neo4j: _FakeNeo4j, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh root runs the real ``fetch_data``: the instance version ``torchcell`` (a
    database name) wins over ``TORCHCELL_KG_VERSION``, ``cypher_kwargs`` reach ``run``
    as keyword parameters, and the two property-shape rows are stored as ``data_0`` and
    ``data_1``.
    """
    monkeypatch.setenv("TORCHCELL_KG_VERSION", "latest")
    raw = Neo4jQueryRaw(
        uri=URI,
        username="u",
        password="p",
        root_dir=str(tmp_path),
        query=QUERY,
        cypher_kwargs={"gene_set": ["YAL001C", "YAL003W"]},
        version="torchcell",
    )
    assert fake_neo4j.listings == [(URI, "u", "p", False)]
    assert fake_neo4j.calls == [
        ("driver", URI, ("u", "p")),
        ("session", {"database": "torchcell", "fetch_size": 1000}),
        ("run", QUERY, {"gene_set": ["YAL001C", "YAL003W"]}),
        ("session_exit",),
        ("close",),
    ]
    assert len(raw) == 2
    assert raw[0:2] == [RECORDS[0], RECORDS[2]]
    raw.close_lmdb()


def test_a_consumer_that_stops_early_still_closes_the_driver(
    view: Neo4jQueryRaw, fake_neo4j: _FakeNeo4j
) -> None:
    """Closing the generator after one record exits the session, then closes the driver.

    Contract (issue #541): ``driver.close()`` sits in a ``finally`` around the session,
    so an early-stopping consumer gets the same call sequence as a full read, ending in
    exactly one ``close`` after ``session_exit``.
    """
    records = view.fetch_data()
    assert isinstance(records, types.GeneratorType)
    assert next(records) == fake_neo4j.records[0]
    records.close()
    assert [call[0] for call in fake_neo4j.calls] == [
        "driver",
        "session",
        "run",
        "session_exit",
        "close",
    ]


def test_a_query_with_no_records_writes_no_store_and_a_retry_reruns_the_query(
    tmp_path: Path, fake_neo4j: _FakeNeo4j
) -> None:
    """Zero records raise ``EmptyQueryResultError`` before any store file is written.

    Contract (issue #541): the first record is read before the LMDB store is opened, so
    an empty result leaves only the empty ``raw/lmdb`` directory (no ``data.mdb``, no
    reference index, no gene set) and the driver is closed. A second construction on the
    same root therefore runs the query again (a second driver/session/run/close sequence)
    and, now that the query returns two rows, stores them as ``data_0`` and ``data_1``.
    """
    fake_neo4j.records = []
    lmdb_dir = tmp_path / "raw" / "lmdb"
    with pytest.raises(neo4j_query_raw.EmptyQueryResultError) as excinfo:
        Neo4jQueryRaw(
            uri=URI,
            username="u",
            password="p",
            root_dir=str(tmp_path),
            query=QUERY,
            version="latest",
        )
    assert str(excinfo.value) == (
        f"the query returned no records; no store was written at {lmdb_dir}. "
        f"Query: {QUERY}"
    )
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == ["lmdb"]
    assert list(lmdb_dir.iterdir()) == []
    one_query = [
        ("driver", URI, ("u", "p")),
        ("session", {"database": "latest", "fetch_size": 1000}),
        ("run", QUERY, {}),
        ("session_exit",),
        ("close",),
    ]
    assert fake_neo4j.calls == one_query

    fake_neo4j.records = [_property_shape(RECORDS[0]), _property_shape(RECORDS[2])]
    raw = Neo4jQueryRaw(
        uri=URI,
        username="u",
        password="p",
        root_dir=str(tmp_path),
        query=QUERY,
        version="latest",
    )
    assert fake_neo4j.calls == one_query + one_query
    assert len(raw) == 2
    assert raw[0:2] == [RECORDS[0], RECORDS[2]]
    raw.close_lmdb()


def test_a_query_that_fails_midway_leaves_no_store_and_a_retry_rebuilds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query that yields one record then raises leaves no store at the target.

    Contract (issue #541 review): the store is written in ``raw/lmdb.partial`` and moved
    into place only when complete, so after the ``ConnectionError`` the target
    ``raw/lmdb`` is empty, the staging directory is gone, and no index or gene set was
    written. A retry on the same root runs the whole query again and stores all three
    records (before the fix it found a one-record ``data.mdb``, ran no query and
    reported ``len`` 1 of 3).
    """
    rows = [_property_shape(record) for record in RECORDS]
    calls: list[str] = []

    def failing(self: Neo4jQueryRaw) -> Iterator[dict[str, Any]]:
        calls.append("failing")
        yield rows[0]
        raise ConnectionError("connection reset after one record")

    def complete(self: Neo4jQueryRaw) -> Iterator[dict[str, Any]]:
        calls.append("complete")
        yield from rows

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_data", failing)
    with pytest.raises(ConnectionError, match=r"^connection reset after one record$"):
        Neo4jQueryRaw(
            uri=URI, username="u", password="p", root_dir=str(tmp_path), query=QUERY
        )
    raw_dir = tmp_path / "raw"
    assert sorted(p.name for p in raw_dir.iterdir()) == ["lmdb"]
    assert list((raw_dir / "lmdb").iterdir()) == []

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_data", complete)
    raw = Neo4jQueryRaw(
        uri=URI, username="u", password="p", root_dir=str(tmp_path), query=QUERY
    )
    assert calls == ["failing", "complete"]
    assert len(raw) == 3
    assert raw[0:3] == RECORDS
    assert sorted(p.name for p in raw_dir.iterdir()) == [
        "experiment_reference_index.json",
        "gene_set.json",
        "lmdb",
    ]
    raw.close_lmdb()


def test_a_leftover_staging_store_is_refused_before_the_query_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``raw/lmdb.partial`` left by a killed build is refused by name, never reused.

    The query is not run, and the leftover directory and its contents stay as found.
    """
    staging = tmp_path / "raw" / "lmdb.partial"
    staging.mkdir(parents=True)
    (staging / "data.mdb").write_bytes(b"partial")
    calls: list[str] = []

    def fetch(self: Neo4jQueryRaw) -> Iterator[dict[str, Any]]:
        calls.append("fetch")
        yield from []

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_data", fetch)
    with pytest.raises(neo4j_query_raw.StaleStagingStoreError) as excinfo:
        Neo4jQueryRaw(
            uri=URI, username="u", password="p", root_dir=str(tmp_path), query=QUERY
        )
    assert str(excinfo.value) == (
        f"{tmp_path / 'raw' / 'lmdb'}.partial is left from an interrupted build and "
        "may hold a partial store; inspect and remove it, then construct again"
    )
    assert calls == []
    assert (staging / "data.mdb").read_bytes() == b"partial"


# ------------------------------------------------------------------------------------
# 2026.10.01 (phase 19): partition rendering, the fetch-worker task and fetch_constants.
#
# ``partition_queries`` is pinned on a hand-written two-block query by whole-string
# equality of selected partitions; ``_render_partition`` runs in THIS process with the
# module's ``_PARTITION_WORKER`` set to the ``view`` fixture and ``fetch_query``
# replaced on the class (attrs-slotted) by a recorder serving the property-shape rows
# of RECORDS; ``fetch_constants`` runs against the recording ``_FakeNeo4j`` above.
# ------------------------------------------------------------------------------------

TWO_BLOCKS = (
    "MATCH (e:Experiment) WHERE e.d = 'A'{partition} RETURN e ORDER BY e.id\n"
    "union all\n"
    "MATCH (e:Experiment) WHERE e.d = 'B'{partition} RETURN e ORDER BY e.id"
)


def test_partition_queries_renders_each_prefix_and_guard_by_whole_string() -> None:
    """Blocks split at a lowercase ``union all`` too; each block yields 16 hex prefixes
    in ascending order then the guard. The block text keeps its surrounding newline
    (the split point is the ``union all`` token only).
    """
    parts = partition_queries(TWO_BLOCKS, 1)
    assert len(parts) == 34
    assert [(b, p) for b, p, _ in parts] == [
        (b, p) for b in (0, 1) for p in [*"0123456789abcdef", ""]
    ]
    assert parts[0] == (
        0,
        "0",
        "MATCH (e:Experiment) WHERE e.d = 'A' AND e.id STARTS WITH '0' RETURN e "
        "ORDER BY e.id\n",
    )
    assert parts[16] == (
        0,
        "",
        "MATCH (e:Experiment) WHERE e.d = 'A' AND NOT e.id =~ '[0-9a-f]{1}.*' "
        "RETURN e ORDER BY e.id\n",
    )
    assert parts[32] == (
        1,
        "f",
        "\nMATCH (e:Experiment) WHERE e.d = 'B' AND e.id STARTS WITH 'f' RETURN e "
        "ORDER BY e.id",
    )
    assert parts[33][2] == (
        "\nMATCH (e:Experiment) WHERE e.d = 'B' AND NOT e.id =~ '[0-9a-f]{1}.*' "
        "RETURN e ORDER BY e.id"
    )


def test_partition_queries_with_two_character_prefixes() -> None:
    """prefix_length 2: 256 prefixes '00' ... 'ff' then a guard of length 2, per block;
    four partitions pinned by whole-string equality.
    """
    parts = partition_queries(TWO_BLOCKS, 2)
    assert len(parts) == 2 * 257
    assert [p for _, p, _ in parts[:3]] == ["00", "01", "02"]
    assert [p for _, p, _ in parts[254:258]] == ["fe", "ff", "", "00"]
    assert parts[0xA7] == (
        0,
        "a7",
        "MATCH (e:Experiment) WHERE e.d = 'A' AND e.id STARTS WITH 'a7' RETURN e "
        "ORDER BY e.id\n",
    )
    assert parts[256] == (
        0,
        "",
        "MATCH (e:Experiment) WHERE e.d = 'A' AND NOT e.id =~ '[0-9a-f]{2}.*' "
        "RETURN e ORDER BY e.id\n",
    )
    assert parts[257] == (
        1,
        "00",
        "\nMATCH (e:Experiment) WHERE e.d = 'B' AND e.id STARTS WITH '00' RETURN e "
        "ORDER BY e.id",
    )
    assert parts[513] == (
        1,
        "",
        "\nMATCH (e:Experiment) WHERE e.d = 'B' AND NOT e.id =~ '[0-9a-f]{2}.*' "
        "RETURN e ORDER BY e.id",
    )


@pytest.mark.parametrize(
    ("query", "prefix_length", "message"),
    [
        (TWO_BLOCKS, 0, "prefix_length must be at least 1, got 0"),
        (
            "MATCH (e) WHERE x{partition}{partition} RETURN e ORDER BY e.id",
            1,
            "block 0 of the query carries the partition marker '{partition}' 2 times; "
            "the partitioned raw stage needs it exactly once per UNION ALL block",
        ),
        (
            "MATCH (e) WHERE x{partition} RETURN e ORDER BY e.id UNION ALL "
            "MATCH (e) WHERE y RETURN e ORDER BY e.id",
            1,
            "block 1 of the query carries the partition marker '{partition}' 0 times; "
            "the partitioned raw stage needs it exactly once per UNION ALL block",
        ),
        (
            "MATCH (e) WHERE x{partition} RETURN e ORDER BY e.id UNION "
            "MATCH (e) WHERE y RETURN e ORDER BY e.id",
            1,
            "block 0 contains a UNION that is not UNION ALL",
        ),
        (
            "MATCH (e) WHERE x{partition} RETURN e ORDER BY e.name",
            1,
            "block 0 is not ordered by e.id",
        ),
    ],
)
def test_partition_queries_refusals(
    query: str, prefix_length: int, message: str
) -> None:
    """Each malformed query is refused with its exact message."""
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        partition_queries(query, prefix_length)


def _index_hash(record: dict[str, Any]) -> str:
    """sha256 of the reference dump with sorted keys (``parallel_hash_computation``)."""
    dump = record["experiment_reference"].model_dump()
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest()


class _Prepare:
    """A split observer whose ``prepare`` is (dataset name, gene count)."""

    def prepare(self, record: dict[str, Any]) -> tuple[str, int]:
        experiment = record["experiment"]
        return experiment["dataset_name"], len(experiment["genotype"]["perturbations"])

    def accept(self, index: int, prepared: Any) -> None:
        pass

    def __call__(self, index: int, record: dict[str, Any]) -> None:
        self.accept(index, self.prepare(record))


@pytest.fixture
def partition_worker(
    view: Neo4jQueryRaw, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], list[int]]:
    """``view`` as the module's partition worker; records the queries and batch sizes."""
    queries: list[str] = []
    batches: list[int] = []
    original = Neo4jQueryRaw._render_batch

    def fetch_query(self: Neo4jQueryRaw, query: str) -> Iterator[Any]:
        queries.append(query)
        yield from (_property_shape(r) for r in RECORDS)

    def render_batch(
        self: Neo4jQueryRaw, batch: Any, constants: Any, observe: Any
    ) -> Any:
        batches.append(len(batch))
        assert observe == "prepare"
        return original(self, batch, constants, observe)

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_query", fetch_query)
    monkeypatch.setattr(Neo4jQueryRaw, "_render_batch", render_batch)
    monkeypatch.setattr(neo4j_query_raw, "_PARTITION_WORKER", view)
    monkeypatch.setattr(neo4j_query_raw, "_PARTITION_CONSTANTS", {})
    return queries, batches


@pytest.mark.parametrize(("process_batch", "sizes"), [(2, [2, 1]), (1000, [3])])
def test_render_partition_renders_every_record_in_order_in_batches(
    partition_worker: tuple[list[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
    process_batch: int,
    sizes: list[int],
) -> None:
    """Three served records become three rows: the LMDB value byte-equal to the
    stored serialization ``json.dumps(record, default=model_dump)`` (the first one is
    EXPECTED_RECORD_0), the reference index hash, the perturbed genes in order, and no
    payload without observers. ``PROCESS_BATCH`` 2 renders [2, 1] records per batch, the
    default 1000 one batch of 3, with identical rows.
    """
    queries, batches = partition_worker
    monkeypatch.setattr(neo4j_query_raw, "PROCESS_BATCH", process_batch)
    rows = neo4j_query_raw._render_partition("Q AND e.id STARTS WITH 'a'")
    assert queries == ["Q AND e.id STARTS WITH 'a'"]
    assert batches == sizes
    assert [row[0] for row in rows] == [_serialize(r).decode() for r in RECORDS]
    assert json.loads(rows[0][0]) == EXPECTED_RECORD_0
    assert [row[1] for row in rows] == [_index_hash(r) for r in RECORDS]
    assert rows[0][1] == rows[1][1] != rows[2][1]
    assert [row[2] for row in rows] == [
        ("YAL001C",),
        ("YAL002W", "YAL003W"),
        ("YAL003W",),
    ]
    assert [row[3] for row in rows] == [None, None, None]


def test_render_partition_carries_each_observers_prepare_result(
    view: Neo4jQueryRaw, partition_worker: tuple[list[str], list[int]]
) -> None:
    """With a split observer the payload is the tuple of its ``prepare`` results:
    ('toy_a', 1), ('toy_a', 2), ('toy_b', 1).
    """
    view.record_observers = [_Prepare()]
    rows = neo4j_query_raw._render_partition("Q")
    assert [row[3] for row in rows] == [
        (("toy_a", 1),),
        (("toy_a", 2),),
        (("toy_b", 1),),
    ]


def test_render_partition_with_no_records_renders_nothing(
    partition_worker: tuple[list[str], list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty partition returns [] without rendering a batch."""
    _, batches = partition_worker

    def nothing(self: Neo4jQueryRaw, query: str) -> Iterator[Any]:
        yield from ()

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_query", nothing)
    assert neo4j_query_raw._render_partition("Q") == []
    assert batches == []


def test_render_partition_without_a_worker_is_an_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch worker started before the stage set ``_PARTITION_WORKER`` fails loudly."""
    monkeypatch.setattr(neo4j_query_raw, "_PARTITION_WORKER", None)
    with pytest.raises(
        AssertionError, match=re.escape("fetch worker started without a raw stage")
    ):
        neo4j_query_raw._render_partition("Q")


CONSTANT_A = '{"media":"YPD"}'
CONSTANT_B = '{"temperature":30}'
CONSTANT_QUERY = (
    "UNWIND $ids AS ref MATCH (n:InternedConstant {id: ref}) "
    "RETURN ref, n.serialized_data AS payload"
)


def test_fetch_constants_runs_one_unwind_and_parses_verified_payloads(
    view: Neo4jQueryRaw, fake_neo4j: _FakeNeo4j, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact Cypher text, ``ids`` passed as the parameter list, a session with no
    ``fetch_size``, the driver closed after the session; each payload is parsed after
    its sha256 matched the id it was fetched by.
    """
    monkeypatch.setenv("TORCHCELL_KG_VERSION", "pinned")
    ids = [constant_id(CONSTANT_A), constant_id(CONSTANT_B)]
    assert ids[0] == hashlib.sha256(CONSTANT_A.encode()).hexdigest()
    fake_neo4j.records = [
        {"ref": ids[1], "payload": CONSTANT_B},
        {"ref": ids[0], "payload": CONSTANT_A},
    ]
    found = view.fetch_constants(ids)
    assert found == {ids[0]: {"media": "YPD"}, ids[1]: {"temperature": 30}}
    assert fake_neo4j.calls == [
        ("driver", URI, ("u", "p")),
        ("session", {"database": "pinned"}),
        ("run", CONSTANT_QUERY, {"ids": ids}),
        ("session_exit",),
        ("close",),
    ]


def test_fetch_constants_refuses_ids_the_store_does_not_serve(
    view: Neo4jQueryRaw, fake_neo4j: _FakeNeo4j, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four ids asked, one served: KeyError naming the 3 missing, sorted, after the
    driver closed.
    """
    monkeypatch.setenv("TORCHCELL_KG_VERSION", "pinned")
    served = constant_id(CONSTANT_A)
    fake_neo4j.records = [{"ref": served, "payload": CONSTANT_A}]
    missing = ["c" * 64, "a" * 64, "b" * 64]
    message = (
        f"3 interned constants are missing from the store, first {sorted(missing)[:3]}"
    )
    with pytest.raises(KeyError, match=re.escape(message)):
        view.fetch_constants([served, *missing])
    assert fake_neo4j.calls[-1] == ("close",)


def test_fetch_constants_on_a_corrupt_payload_raises_without_closing_the_driver(
    view: Neo4jQueryRaw, fake_neo4j: _FakeNeo4j, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: a payload that does not hash to its id raises from ``verified_constant``
    inside the session (neo4j_query_raw.py:450-452) and ``driver.close()`` (:453) is not
    in a ``finally`` (``fetch_query`` closes in one, :428-429), so the driver is left
    open: the recorded calls end at the session exit with no ``close``. The error aborts
    the build either way. Pinned until the close moves into a ``finally``.
    """
    monkeypatch.setenv("TORCHCELL_KG_VERSION", "pinned")
    ref = constant_id(CONSTANT_A)
    fake_neo4j.records = [{"ref": ref, "payload": CONSTANT_B}]
    message = (
        f"interned constant {ref} holds a payload hashing to {constant_id(CONSTANT_B)}; "
        "the store is corrupt or was written by a different serializer"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        view.fetch_constants([ref])
    assert fake_neo4j.calls == [
        ("driver", URI, ("u", "p")),
        ("session", {"database": "pinned"}),
        ("run", CONSTANT_QUERY, {"ids": [ref]}),
        ("session_exit",),
    ]


def _pointered(record: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Property-shape row whose environment and phenotype are ``$ref`` pointers.

    Each pointer is ``{"$ref": id, "kind": field}`` with id = sha256 of the field's
    ``json.dumps`` (the layout ``split_experiment_dump`` writes above its size
    threshold); returns the row and the {id: payload} store it needs.
    """
    dump = record["experiment"].model_dump()
    store = {}
    for name in ("environment", "phenotype"):
        payload = json.dumps(dump[name])
        ref = constant_id(payload)
        store[ref] = payload
        dump[name] = {"$ref": ref, "kind": name}
    row = {
        "e_serialized": json.dumps(dump),
        "ref_serialized": json.dumps(record["experiment_reference"].model_dump()),
    }
    return row, store


@pytest.fixture
def pointer_server(
    view: Neo4jQueryRaw, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[list[str]], dict[str, str]]:
    """Partitions "A" = records [0, 1], "B" = [2], "C" = [0] in pointer layout; every
    ``fetch_constants`` call's id list is recorded and served from the store.
    """
    rows, store = [], {}
    for record in RECORDS:
        row, part = _pointered(record)
        rows.append(row)
        store.update(part)
    partitions = {"A": [rows[0], rows[1]], "B": [rows[2]], "C": [rows[0]]}
    fetched: list[list[str]] = []

    def fetch_query(self: Neo4jQueryRaw, query: str) -> Iterator[Any]:
        yield from partitions[query]

    def fetch_constants(self: Neo4jQueryRaw, refs: list[str]) -> dict[str, Any]:
        fetched.append(list(refs))
        return {r: json.loads(store[r]) for r in refs}

    monkeypatch.setattr(Neo4jQueryRaw, "fetch_query", fetch_query)
    monkeypatch.setattr(Neo4jQueryRaw, "fetch_constants", fetch_constants)
    monkeypatch.setattr(neo4j_query_raw, "_PARTITION_WORKER", view)
    monkeypatch.setattr(neo4j_query_raw, "_PARTITION_CONSTANTS", {})
    return fetched, store


def test_partition_constants_persist_and_only_unseen_ids_are_fetched(
    pointer_server: tuple[list[list[str]], dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker keeps ``_PARTITION_CONSTANTS`` across partitions. Ids: E (the shared
    environment) and one phenotype id per record, P0, P1, P2. With ``PROCESS_BATCH`` 2:

    * partition A (records 0, 1, one batch) fetches sorted({E, P0, P1}), each id once
      though E appears in both records;
    * partition B (record 2) fetches [P2] only, E being held from A;
    * partition C (record 0 again) fetches nothing.

    The rendered values are byte-identical to the inline serialization of each record
    (pointers resolved), so the pointer layout is invisible downstream.
    """
    fetched, store = pointer_server
    monkeypatch.setattr(neo4j_query_raw, "PROCESS_BATCH", 2)
    env = constant_id(json.dumps(RECORDS[0]["experiment"].environment.model_dump()))
    pheno = [
        constant_id(json.dumps(r["experiment"].phenotype.model_dump())) for r in RECORDS
    ]
    assert sorted(store) == sorted({env, *pheno})
    rows_a = neo4j_query_raw._render_partition("A")
    rows_b = neo4j_query_raw._render_partition("B")
    rows_c = neo4j_query_raw._render_partition("C")
    assert fetched == [sorted([env, pheno[0], pheno[1]]), [pheno[2]]]
    assert sorted(neo4j_query_raw._PARTITION_CONSTANTS) == sorted(store)
    assert [r[0] for r in rows_a] == [_serialize(RECORDS[i]).decode() for i in (0, 1)]
    assert [r[0] for r in rows_b] == [_serialize(RECORDS[2]).decode()]
    assert [r[0] for r in rows_c] == [_serialize(RECORDS[0]).decode()]


def test_full_caches_are_emptied_between_batches(
    view: Neo4jQueryRaw,
    pointer_server: tuple[list[list[str]], dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CONSTANT_CACHE_MAX``: a stream cache holding that many entries is cleared at the
    start of the next batch. Counting ``Environment`` validations over partition A with
    ``PROCESS_BATCH`` 1: at the default cap the shared environment is validated once for
    both records; at cap 1 the cache (1 entry after batch 1) is cleared before batch 2
    and the environment is validated again, 2 in all. Rows are identical either way.
    """
    real = Environment  # the schema class the module validates with
    built: list[int] = []

    def counting(**kwargs: Any) -> Any:
        built.append(1)
        return real(**kwargs)

    monkeypatch.setattr(neo4j_query_raw, "Environment", counting)
    monkeypatch.setattr(neo4j_query_raw, "PROCESS_BATCH", 1)
    rows_default = neo4j_query_raw._render_partition("A")
    assert len(built) == 1
    view._stream.environments.clear()
    view._stream.references.clear()
    built.clear()
    monkeypatch.setattr(neo4j_query_raw, "CONSTANT_CACHE_MAX", 1)
    rows_capped = neo4j_query_raw._render_partition("A")
    assert len(built) == 2
    assert rows_capped == rows_default

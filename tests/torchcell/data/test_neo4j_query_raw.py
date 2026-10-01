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
)
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

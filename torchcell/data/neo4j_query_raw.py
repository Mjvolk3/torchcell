# torchcell/data/neo4j_query_raw
# [[torchcell.data.neo4j_query_raw]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_query_raw
# Test file: tests/torchcell/data/test_neo4j_query_raw.py
"""Run a Cypher query against Neo4j and cache the raw experiment records in LMDB."""

import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import shutil
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from typing import Any, cast

import lmdb
from attrs import define, field
from neo4j import GraphDatabase
from tqdm import tqdm

from torchcell.data import ExperimentReferenceIndex, compute_sha256_hash
from torchcell.datamodels.interned_constant import (
    INTERNED_CONSTANT_NEO4J_LABEL,
    POINTER_KEY,
    collect_pointers,
    resolve_pointers,
    verified_constant,
)
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    Environment,
)
from torchcell.sequence import GeneSet

PROCESS_BATCH = 1000
"""Records resolved and written per LMDB transaction in ``Neo4jQueryRaw.process``."""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class StaleStagingStoreError(RuntimeError):
    """A staging store from an interrupted build sits beside the target store.

    A build that was killed outright (no Python cleanup ran) leaves ``raw/lmdb.partial``
    behind. It may hold any prefix of the query, so it is never reused or silently
    replaced: the caller inspects it and removes it before the query runs again.
    """


class EmptyQueryResultError(ValueError):
    """The Cypher query returned no records, so no store is written.

    Refused before the LMDB store is created: an empty ``data.mdb`` would make the next
    construction on the same root skip the query and reuse the empty store.
    """


def parallel_hash_computation(data: tuple[int, Any]) -> tuple[int, str]:
    """
    Function to compute the hash for a single dataset item.
    Returns a tuple of the original index in the dataset and the computed hash.
    """
    idx, data_item = data
    # Raw-query records carry the reference under "experiment_reference" (the key the
    # sequential branch of compute_experiment_reference_index reads), not "reference"
    # (the ExperimentDataset item key).
    return idx, compute_sha256_hash(
        json.dumps((data_item["experiment_reference"].model_dump()), sort_keys=True)
    )


def compute_experiment_reference_index_parallel(
    dataset: Sequence[Any],
) -> list[ExperimentReferenceIndex]:
    """Compute reference indices by hashing each reference across worker processes."""
    num_workers = mp.cpu_count()  # Or set manually to a preferred number

    # Use ProcessPoolExecutor to compute hashes in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare dataset with original indices for parallel processing
        indexed_dataset = list(enumerate(dataset))
        # Execute parallel computation
        results = list(executor.map(parallel_hash_computation, indexed_dataset))

    # Sort results by original indices to maintain dataset order
    sorted_results = sorted(results, key=lambda x: x[0])

    # Extract hashes in their original order
    reference_hashes = [result[1] for result in sorted_results]

    # Continue with the aggregation logic as before
    unique_hashes_to_indices: dict[str, list[int]] = {}
    for idx, hash_val in enumerate(reference_hashes):
        if hash_val not in unique_hashes_to_indices:
            unique_hashes_to_indices[hash_val] = []
        unique_hashes_to_indices[hash_val].append(idx)

    reference_indices_list = []
    for hash_val, indices in unique_hashes_to_indices.items():
        reference_obj = dataset[indices[0]]["experiment_reference"].model_dump()
        exp_ref_index = ExperimentReferenceIndex(
            reference=reference_obj, member_indices=indices
        )
        reference_indices_list.append(exp_ref_index)

    return reference_indices_list


def compute_experiment_reference_index(
    dataset: Sequence[Any], num_workers: int | None = None
) -> list[ExperimentReferenceIndex]:
    """Group records by reference hash into boolean masks, sequentially or in parallel."""
    if num_workers is None or num_workers <= 0:
        # Sequential version
        log.info("Computing experiment reference index sequentially")
        reference_hashes = [
            compute_sha256_hash(
                json.dumps(data["experiment_reference"].model_dump(), sort_keys=True)
            )
            for data in dataset
        ]
    else:
        log.info("Computing experiment reference index in parallel")
        # Parallel version
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Prepare dataset with original indices for parallel processing
            indexed_dataset = list(enumerate(dataset))
            # Execute parallel computation
            results = list(executor.map(parallel_hash_computation, indexed_dataset))
            # Sort results by original indices to maintain dataset order
            sorted_results = sorted(results, key=lambda x: x[0])
            # Extract hashes in their original order
            reference_hashes = [result[1] for result in sorted_results]

    # Common aggregation logic for both versions
    unique_hashes_to_indices: dict[str, list[int]] = {}
    for idx, hash_val in enumerate(reference_hashes):
        if hash_val not in unique_hashes_to_indices:
            unique_hashes_to_indices[hash_val] = []
        unique_hashes_to_indices[hash_val].append(idx)

    reference_indices_list = []
    for hash_val, indices in unique_hashes_to_indices.items():
        reference_obj = dataset[indices[0]]["experiment_reference"].model_dump()
        exp_ref_index = ExperimentReferenceIndex(
            reference=reference_obj, member_indices=indices
        )
        reference_indices_list.append(exp_ref_index)

    return reference_indices_list


CONSTANT_CACHE_MAX = 200_000
"""Entries each validated-constant cache holds before it is emptied.

The caches only save work (a hit and a miss write the same bytes), so emptying one
bounds memory on a dataset whose references or environments are all distinct without
changing any output."""


def _dumps(obj: Any) -> str:
    """The raw stage's serializer: ``json.dumps`` with pydantic models dumped."""
    return json.dumps(obj, default=lambda o: o.model_dump())


RecordObserver = Callable[[int, dict[str, Any]], None]
"""Called by ``Neo4jQueryRaw.process`` once per written record, in record order, with
the record index and a dict equal to ``json.loads`` of the value written under
``data_<index>``. Sub-dicts of a cached environment or reference are SHARED between
records, so an observer must not mutate the dict."""


@define
class _CachedEnvironment:
    """A validated environment, the JSON it contributes to a record, and that parsed."""

    model: Environment
    fragment: str
    parsed: dict[str, Any]


@define
class _CachedReference:
    """A validated experiment reference, its stored JSON parsed and not, its hash."""

    model: Any
    fragment: str
    parsed: dict[str, Any]
    index_hash: str


@define
class _StreamState:
    """What ``process`` carries across batches instead of re-reading the LMDB.

    ``environments`` is keyed by ``("ref", <constant id>)`` on the pointer layout or
    ``("json", <environment JSON>)`` on the inline one; ``references`` by the raw
    ``ref_serialized`` string. ``reference_members`` maps each reference-index hash to
    the record indices that carry it, and ``gene_set`` collects every perturbed gene,
    both read off the exact dicts that were serialized into the LMDB.
    """

    environments: dict[tuple[str, str], _CachedEnvironment] = field(factory=dict)
    references: dict[str, _CachedReference] = field(factory=dict)
    reference_members: dict[str, list[int]] = field(factory=dict)
    gene_set: GeneSet = field(factory=GeneSet)


@define
class Neo4jQueryRaw:
    """LMDB-cached, indexable view over the results of a raw Neo4j Cypher query."""

    uri: str
    username: str
    password: str
    root_dir: str
    query: str
    io_workers: int | None = None
    num_workers: int | None = None
    _experiment_reference_index: list[ExperimentReferenceIndex] | None = field(
        init=False, default=None, repr=False
    )
    _phenotype_label_index: dict[str, list[int]] | None = field(
        init=False, default=None, repr=False
    )
    lmdb_dir: str = field(init=False, default=None)
    raw_dir: str = field(init=False, default=None)
    env: Any = field(init=False, default=None)
    _gene_set: GeneSet | None = field(init=False, default=None)
    cypher_kwargs: dict[str, str | int | float | list[Any]] = field(factory=dict)
    # The knowledge-graph version to query: ``latest`` (default, from
    # TORCHCELL_KG_VERSION), ``pinned``, a release id, or a ``major.minor`` version;
    # resolved to a database name at fetch time by ``releases.resolve_database``.
    version: str | None = None
    # Called once per record as ``process`` writes it (see ``RecordObserver``), so a
    # later stage can compute what it would otherwise re-read the LMDB for.
    record_observers: list[RecordObserver] = field(factory=list)
    # True once ``process`` has written the LMDB in this instance, i.e. the observers
    # have seen every record; False when the LMDB already existed on disk.
    raw_stage_ran: bool = field(init=False, default=False)
    _stream: _StreamState = field(init=False, factory=_StreamState, repr=False)

    def __attrs_post_init__(self) -> None:
        """Set up raw/LMDB paths, run the query on first use, and open the LMDB env."""
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.lmdb_dir, exist_ok=True)

        if not os.path.exists(osp.join(self.lmdb_dir, "data.mdb")):
            self.process()
            self.close_lmdb()

        # Initialize LMDB environment
        self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12), readonly=True)

    def close_lmdb(self) -> None:
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _connect(self) -> tuple[Any, str]:
        """Return ``(driver, database)`` for the configured KG version."""
        from torchcell.database.connection import neo4j_connection_settings
        from torchcell.knowledge_graphs.releases import resolve_database

        version = self.version or neo4j_connection_settings().version
        database = resolve_database(version, self.uri, self.username, self.password)
        log.info("Connecting to Neo4j (%s -> %s)", version, database)
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return driver, database

    def fetch_data(self) -> Iterator[Any]:
        """Open a Neo4j session, run the query, and yield each result record."""
        driver, database = self._connect()
        # The close is in ``finally`` so a consumer that stops early (closing the
        # generator) still closes the driver, after the session exits.
        try:
            # 1000 is default
            with driver.session(database=database, fetch_size=1000) as session:
                log.info("Running query...")
                result = session.run(self.query, **self.cypher_kwargs)
                log.info("Query executed, about to process results...")
                yield from result
            log.info("All records processed.")
        finally:
            driver.close()

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        """Fetch interned constants by id and verify each payload against its id.

        Every id must resolve: a pointer the store cannot serve is a broken build,
        so a missing one raises instead of leaving a ``$ref`` in the record.
        """
        driver, database = self._connect()
        found: dict[str, Any] = {}
        with driver.session(database=database) as session:
            result = session.run(
                f"UNWIND $ids AS ref MATCH (n:{INTERNED_CONSTANT_NEO4J_LABEL} {{id: ref}}) "
                "RETURN ref, n.serialized_data AS payload",
                ids=refs,
            )
            for record in result:
                found[record["ref"]] = verified_constant(
                    record["ref"], record["payload"]
                )
        driver.close()
        missing = sorted(set(refs) - found.keys())
        if missing:
            raise KeyError(
                f"{len(missing)} interned constants are missing from the store, "
                f"first {missing[:3]}"
            )
        return found

    def _init_lmdb(self, readonly: bool = True, readahead: bool = False) -> None:
        """Initialize the LMDB environment.

        ``readahead`` is for whole-store cursor walks: on a cold 025 processed LMDB a
        cursor walk read 2,630 MB/s with it and 213 MB/s without
        (experiments/tcdb-002-build-speed/results/query_build_cost_probe.csv). Random
        single-record access keeps it off.
        """
        if self.env is not None:
            self.close_lmdb()
        self.env = lmdb.open(
            self.lmdb_dir,
            map_size=int(1e12),
            readonly=readonly,
            lock=not readonly,
            readahead=readahead,
            meminit=False,
        )

    def write_to_lmdb(self, key: bytes, value: bytes) -> None:
        """Write a single key/value pair to LMDB inside a write transaction."""
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    @staticmethod
    def _environment_key(environment: dict[str, Any]) -> tuple[str, str]:
        """Cache key of a record's environment: its pointer id, else its JSON.

        A pointer (``{"$ref": id}``) is keyed by its id, which is the sha256 of the
        payload; an inline environment by its JSON. Either key determines the input to
        validation exactly, so a hit returns what a fresh validation would.
        """
        ref = environment.get(POINTER_KEY)
        return ("ref", ref) if ref is not None else ("json", json.dumps(environment))

    def _environment(
        self,
        key: tuple[str, str],
        environment: dict[str, Any],
        constants: dict[str, Any],
    ) -> _CachedEnvironment:
        """Return the validated environment for a record, building it once per key.

        The input to validation is what ``resolve_pointers`` made of the environment
        before: a pointer becomes its fetched payload as is, an inline environment
        has any pointers nested in it resolved.
        """
        cache = self._stream.environments
        hit = cache.get(key)
        if hit is None:
            source = (
                constants[key[1]]
                if key[0] == "ref"
                else resolve_pointers(environment, constants)
            )
            model = Environment(**source)
            fragment = _dumps(model)
            hit = _CachedEnvironment(
                model=model, fragment=fragment, parsed=json.loads(fragment)
            )
            cache[key] = hit
        return hit

    def _reference(
        self,
        ref_serialized: str,
        ref_node_data: dict[str, Any],
        constants: dict[str, Any],
    ) -> _CachedReference:
        """Return the validated reference for a raw ``ref_serialized`` string.

        ``index_hash`` is the hash ``experiment_reference_index`` computes off the STORED
        reference dict (``json.loads`` of the written fragment, keys sorted).
        """
        cache = self._stream.references
        hit = cache.get(ref_serialized)
        if hit is None:
            resolved = resolve_pointers(ref_node_data, constants)
            model = EXPERIMENT_REFERENCE_TYPE_MAP[
                resolved["experiment_reference_type"]
            ](**resolved)
            fragment = _dumps(model)
            parsed = json.loads(fragment)
            hit = _CachedReference(
                model=model,
                fragment=fragment,
                parsed=parsed,
                index_hash=compute_sha256_hash(json.dumps(parsed, sort_keys=True)),
            )
            cache[ref_serialized] = hit
        return hit

    @staticmethod
    def _experiment_json(
        experiment: Any, environment: _CachedEnvironment
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """``_dumps(experiment)`` with the environment's cached JSON spliced in.

        ``json.dumps`` of a str-keyed dict is ``{`` + ``"k": v`` joined by ``, `` +
        ``}``, so dumping every field but the environment and inserting the cached
        fragment at the environment's position reproduces the whole-record bytes. The
        field order is the model's (Experiment classes declare no computed or excluded
        fields, which the length check enforces). Returns the JSON, the dump of the
        other fields (whose ``genotype`` the gene set is read from), and each field's
        JSON fragment.
        """
        fields = type(experiment).model_fields
        dump = experiment.model_dump(exclude={"environment"})
        if len(dump) != len(fields) - 1:
            raise ValueError(
                f"{type(experiment).__name__} dumps fields {sorted(dump)}, not its "
                f"declared fields {list(fields)} without the environment"
            )
        fragments = {
            name: environment.fragment if name == "environment" else _dumps(dump[name])
            for name in fields
        }
        parts = [f"{json.dumps(name)}: {frag}" for name, frag in fragments.items()]
        return "{" + ", ".join(parts) + "}", dump, fragments

    def _write_batch(
        self, batch: list[tuple[int, dict[str, Any], str]], constants: dict[str, Any]
    ) -> None:
        """Resolve a batch's pointers (one fetch for its unseen ids) and write it.

        Each item is ``(index, experiment blob, ref_serialized)``. Only the
        experiment is validated per record: its environment and the reference come
        from caches of validated models (``_environment``, ``_reference``), and the
        written value is byte-identical to
        ``json.dumps({"experiment": e, "experiment_reference": r}, default=model_dump)``
        (tests/torchcell/data/test_neo4j_query_raw_single_pass.py). The reference-index
        hash and the gene set are accumulated in ``self._stream`` as each record is
        written, which is what lets ``process`` skip both re-reads of the LMDB.
        """
        # Caches are emptied only between batches, so every reference this batch
        # finds cached below stays cached until the batch is written.
        for cache in (self._stream.environments, self._stream.references):
            if len(cache) >= CONSTANT_CACHE_MAX:
                cache.clear()
        refs: set[str] = set()
        new_references: dict[str, dict[str, Any]] = {}
        environment_keys: list[tuple[str, str]] = []
        for _, e_node_data, ref_serialized in batch:
            # Walk the record for pointers, but an environment only on a cache miss:
            # a cached key is content that was walked (and validated) already.
            for name, value in e_node_data.items():
                if name != "environment":
                    collect_pointers(value, refs)
            environment_key = self._environment_key(e_node_data["environment"])
            environment_keys.append(environment_key)
            if environment_key[0] == "ref":
                refs.add(environment_key[1])
            elif environment_key not in self._stream.environments:
                collect_pointers(e_node_data["environment"], refs)
            if (
                ref_serialized not in self._stream.references
                and ref_serialized not in new_references
            ):
                ref_node_data = json.loads(ref_serialized)
                collect_pointers(ref_node_data, refs)
                new_references[ref_serialized] = ref_node_data
        unseen = sorted(refs - constants.keys())
        if unseen:
            constants.update(self.fetch_constants(unseen))
        stream = self._stream
        with self.env.begin(write=True) as txn:
            for (i, e_node_data, ref_serialized), environment_key in zip(
                batch, environment_keys, strict=True
            ):
                environment = self._environment(
                    environment_key, e_node_data["environment"], constants
                )
                experiment = EXPERIMENT_TYPE_MAP[e_node_data["experiment_type"]](
                    dataset_name=e_node_data["dataset_name"],
                    genotype=resolve_pointers(e_node_data["genotype"], constants),
                    environment=environment.model,
                    phenotype=resolve_pointers(e_node_data["phenotype"], constants),
                )
                reference = (
                    stream.references[ref_serialized]
                    if ref_serialized in stream.references
                    else self._reference(
                        ref_serialized, new_references[ref_serialized], constants
                    )
                )
                experiment_json, dump, fragments = self._experiment_json(
                    experiment, environment
                )
                data_json = (
                    '{"experiment": '
                    + experiment_json
                    + ', "experiment_reference": '
                    + reference.fragment
                    + "}"
                )
                txn.put(f"data_{i}".encode(), data_json.encode())
                stream.reference_members.setdefault(reference.index_hash, []).append(i)
                for gene_name in self.extract_systematic_gene_names(dump["genotype"]):
                    stream.gene_set.add(gene_name)
                if self.record_observers:
                    # json.loads of the concatenation is the composition of the parts'
                    # parses, so this dict equals json.loads(data_json).
                    record = {
                        "experiment": {
                            name: environment.parsed
                            if name == "environment"
                            else json.loads(frag)
                            for name, frag in fragments.items()
                        },
                        "experiment_reference": reference.parsed,
                    }
                    for observer in self.record_observers:
                        observer(i, record)

    def _reference_index_from_groups(
        self, groups: list[list[int]]
    ) -> list[ExperimentReferenceIndex]:
        """Build the index from reference-hash groups in LMDB cursor order, and save it.

        ``groups`` (one list of record indices per reference hash) must be ordered,
        and each list ordered, as a cursor walk of the LMDB meets them: keys are
        ``data_<i>``, so cursor order is the lexicographic order of the key string,
        not of ``i``. The stored reference is the re-validated reference of each
        group's first member in that order, which is what
        ``self[i]["experiment_reference"]`` returns; it is read here in one
        transaction without validating the experiment half.
        """
        self._init_lmdb(readonly=True)
        index: list[ExperimentReferenceIndex] = []
        with self.env.begin() as txn:
            for indices in groups:
                stored = json.loads(txn.get(f"data_{indices[0]}".encode()).decode())
                ref = stored["experiment_reference"]
                reference = EXPERIMENT_REFERENCE_TYPE_MAP[
                    ref["experiment_reference_type"]
                ](**ref)
                index.append(
                    ExperimentReferenceIndex(
                        reference=reference.model_dump(), member_indices=sorted(indices)
                    )
                )
        self._experiment_reference_index = index
        with open(osp.join(self.raw_dir, "experiment_reference_index.json"), "w") as f:
            json.dump([eri.model_dump() for eri in self._experiment_reference_index], f)
        self.close_lmdb()
        return self._experiment_reference_index

    def process(self) -> None:
        """Stream query results into LMDB and write the reference index and gene set.

        The store is written in a staging directory beside the target
        (``raw/lmdb.partial``) and moved into place with ``os.replace`` only once every
        record is written and the environment is closed, so the target never holds a
        partial store that a later construction would reuse without querying. A query
        that returns no records raises ``EmptyQueryResultError`` before any store
        exists. A failure while writing closes the environment, removes the staging
        directory this call created, and re-raises; a staging directory found at the
        start (left by a build that was killed outright) raises
        ``StaleStagingStoreError`` before the query runs.

        Experiment blobs carry ``{"$ref": <id>}`` pointers where the build interned a
        large sub-object (torchcell/datamodels/interned_constant.py); each batch fetches
        the ids it has not seen, verifies them, and splices them back before the
        record is written, so the LMDB holds the same inlined records it always did.

        The reference index and gene set are computed while streaming and written to
        the files their properties read, so neither property re-reads the LMDB. Both
        files are the ones the two streaming passes wrote before
        (tests/torchcell/data/test_neo4j_query_raw_single_pass.py).
        """
        staging_dir = self.lmdb_dir + ".partial"
        if osp.exists(staging_dir):
            raise StaleStagingStoreError(
                f"{staging_dir} is left from an interrupted build and may hold a "
                "partial store; inspect and remove it, then construct again"
            )
        log.info("Processing data...")
        records = self.fetch_data()
        first = next(records, None)
        if first is None:
            raise EmptyQueryResultError(
                f"the query returned no records; no store was written at "
                f"{self.lmdb_dir}. Query: {self.query}"
            )
        final_dir = self.lmdb_dir
        os.makedirs(staging_dir)
        self.lmdb_dir = staging_dir
        try:
            self._init_lmdb(readonly=False)
            n_records = self._write_records(chain([first], records))
            self.close_lmdb()
        except BaseException:
            self.close_lmdb()
            shutil.rmtree(staging_dir)
            raise
        finally:
            self.lmdb_dir = final_dir
        os.replace(staging_dir, final_dir)
        log.info(f"Total records processed: {n_records}")

        # Order the groups as the streaming property's cursor walk meets them: keys
        # are data_<i>, so cursor order is lexicographic in str(i).
        groups = [
            sorted(members, key=str)
            for members in self._stream.reference_members.values()
        ]
        groups.sort(key=lambda members: str(members[0]))
        self._reference_index_from_groups(groups)
        self.gene_set = self._stream.gene_set
        self._stream = _StreamState()
        self.raw_stage_ran = True

    def _write_records(self, records: Iterator[Any]) -> int:
        """Write each query record as ``data_<i>``; return the number written."""
        i = -1
        constants: dict[str, Any] = {}
        self._stream = _StreamState()
        batch: list[tuple[int, dict[str, Any], str]] = []
        for i, record in tqdm(enumerate(records)):
            # Two record shapes, by what the query RETURNs. Property shape
            # (e_serialized/ref_serialized strings) is REQUIRED at scale: returning
            # whole nodes makes the driver register every hydrated Node in the
            # result's Graph cache (neo4j/graph/__init__.py, graph._nodes) for the
            # life of the result, retaining 13.6 KB/record -- measured on the 025
            # build, which held 29 GB of heap at 2.2M records and projects past
            # machine RAM at 44M. Returning the serialized_data property instead
            # measures 3 B/record. Node shape (RETURN e, ref) stays supported for
            # the existing experiment queries, which are historical records.
            if "e_serialized" in record.keys():
                e_node_data = json.loads(record["e_serialized"])
            else:
                e_node_data = json.loads(record["e"]["serialized_data"])
            if "ref_serialized" in record.keys():
                ref_serialized = record["ref_serialized"]
            else:
                ref_serialized = record["ref"]["serialized_data"]
            batch.append((i, e_node_data, ref_serialized))
            if len(batch) >= PROCESS_BATCH:
                self._write_batch(batch, constants)
                batch = []
        if batch:
            self._write_batch(batch, constants)

        log.info(f"Interned constants resolved: {len(constants)}")
        return i + 1

    def __getitem__(self, index: int | slice | list[int]) -> Any:
        """Return the record(s) for an int, slice, or list of indices."""
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):  # New case for a list of indices
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(self, index: int) -> dict[str, Any]:
        self._init_lmdb()
        data_key = f"data_{index}".encode()

        with self.env.begin() as txn:
            data_json = txn.get(data_key)

            if data_json is None:
                raise IndexError(f"Record not found at index: {index}")

            data_dict = json.loads(data_json.decode())
            experiment_class = EXPERIMENT_TYPE_MAP[
                data_dict["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                data_dict["experiment_reference"]["experiment_reference_type"]
            ]

            experiment = experiment_class(**data_dict["experiment"])
            experiment_reference = experiment_reference_class(
                **data_dict["experiment_reference"]
            )

            return {
                "experiment": experiment,
                "experiment_reference": experiment_reference,
            }

    def _get_record(self, key: bytes) -> dict[str, Any]:
        with self.env.begin() as txn:
            data_json = txn.get(key)
            if data_json is None:
                raise IndexError(f"Record not found for key: {key.decode()}")
            data_dict = json.loads(data_json.decode())

            experiment_class = EXPERIMENT_TYPE_MAP[
                data_dict["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                data_dict["experiment_reference"]["experiment_reference_type"]
            ]

            experiment = experiment_class(**data_dict["experiment"])
            experiment_reference = experiment_reference_class(
                **data_dict["experiment_reference"]
            )

            return {
                "experiment": experiment,
                "experiment_reference": experiment_reference,
            }

    def _get_records_by_slice(self, slice_obj: slice) -> list[dict[str, Any]]:
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"data_{i}".encode() for i in range(start, stop, step)]
        # ``len`` closes the environment; the threaded reads below need it open.
        self._init_lmdb()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.io_workers
        ) as executor:
            records = list(executor.map(self._get_record, data_keys))

        return records

    def __len__(self) -> int:
        """Return the number of cached records in the LMDB store."""
        if self.env is None:
            self._init_lmdb()
        with self.env.begin() as txn:
            entries = cast(int, txn.stat()["entries"])
        self.close_lmdb()
        return entries

    @staticmethod
    def extract_systematic_gene_names(genotype: dict[str, Any]) -> list[str]:
        """Return the systematic gene names of all perturbations in a genotype."""
        gene_names: list[str] = []
        for perturbation in cast(list[dict[str, Any]], genotype.get("perturbations")):
            gene_name = cast(str, perturbation.get("systematic_gene_name"))
            gene_names.append(gene_name)
        return gene_names

    @property
    def experiment_reference_index(self) -> list[ExperimentReferenceIndex]:
        """Return the cached reference index, computing and persisting it if missing."""
        index_file_path = osp.join(self.raw_dir, "experiment_reference_index.json")

        if osp.exists(index_file_path):
            with open(index_file_path) as file:
                data = json.load(file)
            # Deserialize each dict in the list to an ExperimentReferenceIndex object
            self._experiment_reference_index = [
                ExperimentReferenceIndex.from_stored(item) for item in data
            ]
        elif self._experiment_reference_index is None:
            # Stream the hash pass off LMDB instead of materializing the dataset.
            # The previous `[self[i] for i in range(len(self))]` built every record
            # as pydantic objects in one list -- at 43.8M records that is hundreds
            # of GB, and it OOM-killed build 1559 six minutes after a clean 18 h
            # fetch. Hashing the STORED reference dict is exact, not approximate:
            # process() serialized it from model_dump(), so json.loads returns
            # byte-identical structure to what the old path re-dumped and hashed.
            log.info("Computing experiment reference index (streaming)...")
            self._init_lmdb(readonly=True, readahead=True)
            hash_to_indices: dict[str, list[int]] = {}
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key, value in tqdm(cursor):
                    # Cursor order is lexicographic (data_0, data_1, data_10, ...);
                    # parse the true index from the key rather than counting.
                    idx = int(key.decode().split("_")[1])
                    ref_dict = json.loads(value.decode("utf-8"))["experiment_reference"]
                    hash_val = compute_sha256_hash(json.dumps(ref_dict, sort_keys=True))
                    hash_to_indices.setdefault(hash_val, []).append(idx)
            self.close_lmdb()
            self._experiment_reference_index = self._reference_index_from_groups(
                list(hash_to_indices.values())
            )

        self.close_lmdb()
        return self._experiment_reference_index

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        """Return a mapping of phenotype label to the record indices having it."""
        print("Computing phenotype label index...")
        # Fetch all phenotype labels
        phenotype_labels: list[tuple[int, str]] = []
        for i in range(len(self)):
            record: dict[str, Any] = self[i]
            phenotype_labels.append((i, record["experiment"].phenotype.label_name))

        # Initialize the phenotype label index dictionary
        phenotype_label_index: dict[str, list[int]] = {}

        # Populate the index lists with indices
        for i, label in phenotype_labels:
            if label not in phenotype_label_index:
                phenotype_label_index[label] = []
            phenotype_label_index[label].append(i)

        return phenotype_label_index

    @property
    def phenotype_label_index(self) -> dict[str, list[int]]:
        """Return the cached phenotype-label index, computing and persisting if missing."""
        if osp.exists(osp.join(self.raw_dir, "phenotype_label_index.json")):
            with open(osp.join(self.raw_dir, "phenotype_label_index.json")) as file:
                self._phenotype_label_index = json.load(file)
        else:
            self._phenotype_label_index = self.compute_phenotype_label_index()
            with open(
                osp.join(self.raw_dir, "phenotype_label_index.json"), "w"
            ) as file:
                json.dump(self._phenotype_label_index, file)
        return self._phenotype_label_index

    def compute_gene_set(self) -> GeneSet:
        """Return the GeneSet of all perturbed genes found across cached records."""
        gene_set = GeneSet()
        self._init_lmdb(readahead=True)

        with self.env.begin() as txn:
            cursor = txn.cursor()
            log.info("Computing gene set...")
            for key, value in tqdm(cursor):
                # Corrected line: use json.loads for JSON strings
                deserialized_data = json.loads(
                    value.decode("utf-8")
                )  # Assuming value is a bytes object
                experiment = deserialized_data["experiment"]

                extracted_gene_names = self.extract_systematic_gene_names(
                    experiment["genotype"]
                )
                for gene_name in extracted_gene_names:
                    gene_set.add(gene_name)

        self.close_lmdb()
        return gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self) -> GeneSet:
        """Return the GeneSet, loading from JSON if cached else computing it."""
        if osp.exists(osp.join(self.raw_dir, "gene_set.json")):
            with open(osp.join(self.raw_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        else:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value: GeneSet) -> None:
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.raw_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self) -> str:
        """Return a string with the query's URI, root directory, and Cypher text."""
        return f"Neo4jQueryRaw(uri={self.uri}, root_dir={self.root_dir}, query={self.query})"


##########################33

# Example usage
if __name__ == "__main__":
    from hashlib import sha256

    from dotenv import load_dotenv

    from torchcell.sequence import GeneSet

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    # TODO change to process and io workers
    neo4j_db = Neo4jQueryRaw(
        uri="bolt://localhost:7687",  # Include the database name here
        # uri="bolt://gilahyper.zapto.org:7687",  # Include the database name here
        username="neo4j",
        password="torchcell",
        root_dir=osp.join(DATA_ROOT, "data/torchcell/neo4j_query_test"),
        query="""
            MATCH (e:Experiment)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
            RETURN e, ref
            LIMIT 10;
        """,
        io_workers=10,
        num_workers=10,
        cypher_kwargs={"gene_set": list(genome.gene_set)},
    )
    neo4j_db[0]
    neo4j_db[0:2]
    # neo4j_db.phenotype_label_index.keys()

    duplicate_check: dict[str, list[int]] = {}
    for i in tqdm(range(len(neo4j_db))):
        perturbations = neo4j_db[i]["experiment"].genotype.perturbations
        sorted_gene_names = sorted(
            [pert.systematic_gene_name for pert in perturbations]
        )
        hash_key = sha256(str(sorted_gene_names).encode()).hexdigest()

        if hash_key not in duplicate_check:
            duplicate_check[hash_key] = []
        duplicate_check[hash_key].append(i)

    # Save the duplicate_check dictionary to a file for inspection
    with open("duplicate_check.json", "w") as file:
        json.dump(duplicate_check, file, indent=2)

    print("Duplicate check complete. Results saved to duplicate_check.json.")
    print([(k, v) for k, v in duplicate_check.items() if len(v) > 1])

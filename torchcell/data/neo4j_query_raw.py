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
import re
import shutil
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import chain, product
from typing import Any, Literal, Protocol, cast, runtime_checkable

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


@runtime_checkable
class SplitRecordObserver(Protocol):
    """A ``RecordObserver`` split into a per-record map and an in-order reduce.

    ``prepare`` is a pure function of one record (it may run in a fetch worker
    process, so it must not depend on state the parent changes after the workers
    fork); ``accept`` folds its result in, called in record order in the parent.
    ``observer(index, record)`` must equal ``observer.accept(index,
    observer.prepare(record))``. The partitioned raw stage (``fetch_workers > 0``)
    takes only observers of this kind, so that the per-record work runs in the workers.
    """

    def prepare(self, record: dict[str, Any]) -> Any:
        """Compute what ``accept`` needs from one record."""
        ...

    def accept(self, index: int, prepared: Any) -> None:
        """Fold one record's prepared value in, in record order."""
        ...

    def __call__(self, index: int, record: dict[str, Any]) -> None:
        """``accept(index, prepare(record))``."""
        ...


PARTITION_MARKER = "{partition}"
"""Placeholder a query carries once per ``UNION ALL`` block, inside the block's WHERE
right after its dataset filter, where the partitioned raw stage puts its ``e.id``
prefix filter. The single-session path removes it."""

PARTITIONS_IN_FLIGHT_PER_WORKER = 2
"""Partitions submitted but not yet consumed, per fetch worker; with the partition
size this bounds what the partitioned raw stage holds in memory at once."""

_HEX = "0123456789abcdef"

_Row = tuple[str, str, tuple[str, ...], Any]
"""One rendered record: the LMDB value, its reference-index hash, its perturbed
genes, and its observer payload (the record dict, or each observer's ``prepare``
result, or ``None`` with no observers)."""


def single_session_query(query: str) -> str:
    """The query as one session runs it: every partition marker removed."""
    return query.replace(PARTITION_MARKER, "")


def partition_queries(query: str, prefix_length: int) -> list[tuple[int, str, str]]:
    """Split a marked query into ``(block, prefix, query)`` partitions, in output order.

    The query is split at ``UNION ALL`` into blocks, each of which must carry the
    marker exactly once and end in ``ORDER BY e.id`` (its experiment node bound to
    ``e``). Each block yields one partition per lowercase hex prefix of
    ``prefix_length`` characters, in ascending order, with the marker replaced by
    ``AND e.id STARTS WITH '<prefix>'``, then one GUARD partition (prefix ``""``)
    selecting the block's records whose id does NOT start with a hex prefix; the
    raw stage requires the guard to return nothing.

    Ordering. A single session returns the blocks in query order (``UNION ALL``
    concatenates), each block sorted by ``e.id``. Experiment ids are lowercase hex
    sha256 digests (``CellAdapter._experiment_node``: the sha256 of the inlined
    record), so every id starts with one of the prefixes, the prefixes are disjoint,
    and since ``0 < ... < 9 < a < ... < f`` in string order, concatenating the prefix
    partitions in ascending order, each sorted by ``e.id``, IS the block sorted by
    ``e.id``. The guard turns the hex assumption into a checked one.
    """
    if prefix_length < 1:
        raise ValueError(f"prefix_length must be at least 1, got {prefix_length}")
    blocks = re.split(r"\bUNION\s+ALL\b", query, flags=re.IGNORECASE)
    partitions: list[tuple[int, str, str]] = []
    prefixes = ["".join(p) for p in product(_HEX, repeat=prefix_length)]
    for b, block in enumerate(blocks):
        if block.count(PARTITION_MARKER) != 1:
            raise ValueError(
                f"block {b} of the query carries the partition marker "
                f"{PARTITION_MARKER!r} {block.count(PARTITION_MARKER)} times; the "
                "partitioned raw stage needs it exactly once per UNION ALL block"
            )
        if re.search(r"\bUNION\b", block, flags=re.IGNORECASE):
            raise ValueError(f"block {b} contains a UNION that is not UNION ALL")
        if not re.search(r"ORDER\s+BY\s+e\.id\b", block, flags=re.IGNORECASE):
            raise ValueError(f"block {b} is not ordered by e.id")
        for prefix in prefixes:
            partitions.append(
                (
                    b,
                    prefix,
                    block.replace(
                        PARTITION_MARKER, f" AND e.id STARTS WITH '{prefix}'"
                    ),
                )
            )
        guard = f" AND NOT e.id =~ '[0-9a-f]{{{prefix_length}}}.*'"
        partitions.append((b, "", block.replace(PARTITION_MARKER, guard)))
    return partitions


_PARTITION_WORKER: "Neo4jQueryRaw | None" = None
"""The raw stage a forked fetch worker runs partitions for (set before the fork)."""

_PARTITION_CONSTANTS: dict[str, Any] = {}
"""A fetch worker's resolved interned constants, kept across its partitions."""


def _render_partition(query: str) -> list[_Row]:
    """Fetch-worker task: run one partition query and render its records in order."""
    raw = _PARTITION_WORKER
    assert raw is not None, "fetch worker started without a raw stage"
    rows: list[_Row] = []
    batch: list[tuple[int, dict[str, Any], str]] = []
    for position, record in enumerate(raw.fetch_query(query)):
        batch.append((position, *raw._record_parts(record)))
        if len(batch) >= PROCESS_BATCH:
            rows.extend(raw._render_batch(batch, _PARTITION_CONSTANTS, "prepare"))
            batch = []
    if batch:
        rows.extend(raw._render_batch(batch, _PARTITION_CONSTANTS, "prepare"))
    return rows


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
    # 0: one session runs the whole query (markers removed). N > 0: the query must
    # carry PARTITION_MARKER once per UNION ALL block; it is split into e.id-prefix
    # partitions (partition_queries) rendered by N forked worker processes, each with
    # its own driver, and written by this process in single-session order.
    fetch_workers: int = 0
    # Hex characters per partition prefix: 1 gives 16 partitions per block, 2 gives
    # 256. In-flight memory is about PARTITIONS_IN_FLIGHT_PER_WORKER * fetch_workers
    # partitions of rendered records, so a multi-million-record block wants 2.
    partition_prefix_length: int = 1
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

    def fetch_query(self, query: str) -> Iterator[Any]:
        """Open a Neo4j session (and driver), run ``query``, and yield each record.

        The one place a query reaches the server: the single-session path runs the
        whole query through it, and each fetch worker runs its partitions through
        it, every call on its own driver.
        """
        driver, database = self._connect()
        # The close is in ``finally`` so a consumer that stops early (closing the
        # generator) still closes the driver, after the session exits.
        try:
            # 1000 is default
            with driver.session(database=database, fetch_size=1000) as session:
                log.info("Running query...")
                result = session.run(query, **self.cypher_kwargs)
                log.info("Query executed, about to process results...")
                yield from result
            log.info("All records processed.")
        finally:
            driver.close()

    def fetch_data(self) -> Iterator[Any]:
        """Run the whole query in one session (partition markers removed)."""
        yield from self.fetch_query(single_session_query(self.query))

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

    @staticmethod
    def _record_parts(record: Any) -> tuple[dict[str, Any], str]:
        """``(experiment blob, ref_serialized)`` of one result record.

        Two record shapes, by what the query RETURNs. Property shape
        (e_serialized/ref_serialized strings) is REQUIRED at scale: returning
        whole nodes makes the driver register every hydrated Node in the
        result's Graph cache (neo4j/graph/__init__.py, graph._nodes) for the
        life of the result, retaining 13.6 KB/record -- measured on the 025
        build, which held 29 GB of heap at 2.2M records and projects past
        machine RAM at 44M. Returning the serialized_data property instead
        measures 3 B/record. Node shape (RETURN e, ref) stays supported for
        the existing experiment queries, which are historical records.
        """
        if "e_serialized" in record.keys():
            e_node_data = json.loads(record["e_serialized"])
        else:
            e_node_data = json.loads(record["e"]["serialized_data"])
        if "ref_serialized" in record.keys():
            ref_serialized = record["ref_serialized"]
        else:
            ref_serialized = record["ref"]["serialized_data"]
        return e_node_data, ref_serialized

    def _render_batch(
        self,
        batch: list[tuple[int, dict[str, Any], str]],
        constants: dict[str, Any],
        observe: Literal["record", "prepare"],
    ) -> list[_Row]:
        """Resolve a batch's pointers (one fetch for its unseen ids) and render it.

        Each item is ``(index, experiment blob, ref_serialized)``. Only the
        experiment is validated per record: its environment and the reference come
        from caches of validated models (``_environment``, ``_reference``), and the
        rendered value is byte-identical to
        ``json.dumps({"experiment": e, "experiment_reference": r}, default=model_dump)``
        (tests/torchcell/data/test_neo4j_query_raw_single_pass.py). Each row also
        carries the reference-index hash, the perturbed genes, and, with observers,
        the record dict (``observe="record"``) or each observer's ``prepare`` of it
        (``"prepare"``, run in a fetch worker). Nothing is written; ``_commit_rows``
        writes rows in record order.
        """
        # Caches are emptied only between batches, so every reference this batch
        # finds cached below stays cached until the batch is rendered.
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
        observers = self.record_observers
        rows: list[_Row] = []
        for (_, e_node_data, ref_serialized), environment_key in zip(
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
            genes = tuple(self.extract_systematic_gene_names(dump["genotype"]))
            payload: Any = None
            if observers:
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
                payload = (
                    record
                    if observe == "record"
                    else tuple(
                        cast(SplitRecordObserver, o).prepare(record) for o in observers
                    )
                )
            rows.append((data_json, reference.index_hash, genes, payload))
        return rows

    def _commit_rows(
        self, first_index: int, rows: list[_Row], observe: Literal["record", "prepare"]
    ) -> None:
        """Write rendered rows from ``data_<first_index>`` on, in record order.

        Each row is also folded into the reference index, the gene set and the
        observers.
        """
        stream = self._stream
        observers = self.record_observers
        with self.env.begin(write=True) as txn:
            for offset, (data_json, index_hash, genes, payload) in enumerate(rows):
                i = first_index + offset
                txn.put(f"data_{i}".encode(), data_json.encode())
                stream.reference_members.setdefault(index_hash, []).append(i)
                for gene_name in genes:
                    stream.gene_set.add(gene_name)
                if not observers:
                    continue
                if observe == "record":
                    for observer in observers:
                        observer(i, payload)
                else:
                    for observer, prepared in zip(observers, payload, strict=True):
                        cast(SplitRecordObserver, observer).accept(i, prepared)

    def _write_batch(
        self, batch: list[tuple[int, dict[str, Any], str]], constants: dict[str, Any]
    ) -> None:
        """Render a batch of consecutive records and write it (single-session path)."""
        rows = self._render_batch(batch, constants, "record")
        self._commit_rows(batch[0][0], rows, "record")

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

    def _process_single_session(self, records: Iterator[Any]) -> int:
        """Write the one session's ``records`` batch by batch; return the count."""
        i = -1
        constants: dict[str, Any] = {}
        batch: list[tuple[int, dict[str, Any], str]] = []
        for i, record in tqdm(enumerate(records)):
            batch.append((i, *self._record_parts(record)))
            if len(batch) >= PROCESS_BATCH:
                self._write_batch(batch, constants)
                batch = []
        if batch:
            self._write_batch(batch, constants)
        log.info(f"Interned constants resolved: {len(constants)}")
        return i + 1

    def _process_partitioned(self) -> int:
        """Render ``partition_queries`` in ``fetch_workers`` forked processes; write
        them here in partition order, so the LMDB is the single session's.

        The workers fork from this process with its (empty) caches and observers, each
        running ``_render_partition``: its own driver per partition query, its own
        validated-constant caches and resolved constants (each payload verified
        against its id by ``fetch_constants``), and each observer's ``prepare``. This
        process consumes partitions strictly in submission order, assigns
        ``data_<i>`` sequentially, and keeps at most
        ``PARTITIONS_IN_FLIGHT_PER_WORKER * fetch_workers`` partitions submitted and
        unconsumed. Observers must be ``SplitRecordObserver``s.
        """
        global _PARTITION_WORKER
        plain = [
            o for o in self.record_observers if not isinstance(o, SplitRecordObserver)
        ]
        if plain:
            raise TypeError(
                f"fetch_workers={self.fetch_workers} needs SplitRecordObserver "
                f"observers (prepare/accept); got {plain}"
            )
        partitions = partition_queries(self.query, self.partition_prefix_length)
        in_flight = PARTITIONS_IN_FLIGHT_PER_WORKER * self.fetch_workers
        # Fork with no LMDB environment open; reopen it for writing once the workers
        # exist (the fork start method launches every worker at the first submit).
        self.close_lmdb()
        _PARTITION_WORKER = self
        pending: deque[tuple[int, str, Future[list[_Row]]]] = deque()
        next_partition = 0
        n_records = 0
        with ProcessPoolExecutor(
            max_workers=self.fetch_workers, mp_context=mp.get_context("fork")
        ) as pool:
            while next_partition < len(partitions) and len(pending) < in_flight:
                block, prefix, query = partitions[next_partition]
                pending.append((block, prefix, pool.submit(_render_partition, query)))
                next_partition += 1
            _PARTITION_WORKER = None
            self._init_lmdb(readonly=False)
            with tqdm(total=len(partitions), desc="partitions") as progress:
                while pending:
                    block, prefix, future = pending.popleft()
                    rows = future.result()
                    if next_partition < len(partitions):
                        b, p, query = partitions[next_partition]
                        pending.append((b, p, pool.submit(_render_partition, query)))
                        next_partition += 1
                    if prefix == "" and rows:
                        raise ValueError(
                            f"block {block}: {len(rows)} records have an e.id that "
                            "does not start with a lowercase hex prefix, so the "
                            "prefix partitions would have dropped them"
                        )
                    for start in range(0, len(rows), PROCESS_BATCH):
                        chunk = rows[start : start + PROCESS_BATCH]
                        self._commit_rows(n_records, chunk, "prepare")
                        n_records += len(chunk)
                    progress.update(1)
        return n_records

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
        self._stream = _StreamState()
        records: Iterator[Any] | None = None
        if self.fetch_workers == 0:
            # One session: the first record decides whether a store is written at all.
            # The partitioned path learns the count only once every partition is in,
            # and then removes its staging store the same way.
            records = self.fetch_data()
            first = next(records, None)
            if first is None:
                raise EmptyQueryResultError(
                    f"the query returned no records; no store was written at "
                    f"{self.lmdb_dir}. Query: {self.query}"
                )
            records = chain([first], records)
        final_dir = self.lmdb_dir
        os.makedirs(staging_dir)
        self.lmdb_dir = staging_dir
        try:
            if records is None:
                n_records = self._process_partitioned()
            else:
                self._init_lmdb(readonly=False)
                n_records = self._process_single_session(records)
            self.close_lmdb()
            if n_records == 0:
                raise EmptyQueryResultError(
                    f"the query returned no records; no store was written at "
                    f"{final_dir}. Query: {self.query}"
                )
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

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
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, cast

import lmdb
from attrs import define, field
from neo4j import GraphDatabase
from tqdm import tqdm

from torchcell.data import ExperimentReferenceIndex, compute_sha256_hash
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.lmdb_map_size import BUILD_LMDB_MAP_SIZE
from torchcell.sequence import GeneSet

RAW_BATCH_RECORDS = 2000
"""Records per worker task and per LMDB write transaction in the raw stage.

At about 23 KB per serialized record a batch is about 46 MB, and with two batches in
flight per worker the raw stage holds under 2 GB of pending rows at ten workers.
"""


def _serialize_raw_batch(
    batch: list[tuple[int, str, str]],
) -> list[tuple[bytes, bytes]]:
    """Validate ``(index, experiment json, reference json)`` rows into LMDB rows.

    Runs in a worker process. Constructing the pydantic experiment and reference is
    the raw stage's CPU cost and its schema check; the driver thread only hands over
    the two strings each record carries.
    """
    rows: list[tuple[bytes, bytes]] = []
    for i, e_json, ref_json in batch:
        e_node_data = json.loads(e_json)
        experiment_class = EXPERIMENT_TYPE_MAP[e_node_data["experiment_type"]]
        experiment = experiment_class(
            dataset_name=e_node_data["dataset_name"],
            genotype=e_node_data["genotype"],
            environment=e_node_data["environment"],
            phenotype=e_node_data["phenotype"],
        )
        ref_node_data = json.loads(ref_json)
        experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
            ref_node_data["experiment_reference_type"]
        ]
        experiment_reference = experiment_reference_class(**ref_node_data)
        data_dict = {
            "experiment": experiment,
            "experiment_reference": experiment_reference,
        }
        data_json = json.dumps(data_dict, default=lambda o: o.model_dump())
        rows.append((f"data_{i}".encode(), data_json.encode()))
    return rows


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parallel_hash_computation(data: tuple[int, Any]) -> tuple[int, str]:
    """
    Function to compute the hash for a single dataset item.
    Returns a tuple of the original index in the dataset and the computed hash.
    """
    idx, data_item = data
    return idx, compute_sha256_hash(
        json.dumps((data_item["reference"].model_dump()), sort_keys=True)
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
        reference_obj = dataset[indices[0]]["reference"].model_dump()
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

    def __attrs_post_init__(self) -> None:
        """Set up raw/LMDB paths, run the query on first use, and open the LMDB env."""
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.lmdb_dir, exist_ok=True)

        # Resume marker, as the later stages have (neo4j_cell.py): the raw LMDB is
        # created before the first record is written, so a build killed mid-query leaves
        # a partial store whose existence alone must never count as completion. The
        # 030 build (slurm 2681) was cancelled 1.5M records into a 44M-record query; had
        # it been restarted on the old rule, the dataset would have been silently
        # truncated to those 1.5M.
        marker = osp.join(self.raw_dir, "STAGE_COMPLETE")
        if not osp.exists(marker):
            if osp.exists(osp.join(self.lmdb_dir, "data.mdb")):
                raise RuntimeError(
                    f"{self.lmdb_dir} holds a raw LMDB but {marker} is absent: the query "
                    "that wrote it did not finish (or predates the marker). Move the raw "
                    "directory aside and rebuild; a partial raw store is not resumable."
                )
            self._init_lmdb(readonly=False)
            self.process()
            self.close_lmdb()
            with open(marker, "w") as f:
                f.write("")

        # Initialize LMDB environment
        self.env = lmdb.open(self.lmdb_dir, map_size=BUILD_LMDB_MAP_SIZE, readonly=True)

    def close_lmdb(self) -> None:
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def fetch_data(self) -> Iterator[Any]:
        """Open a Neo4j session, run the query, and yield each result record."""
        from torchcell.database.connection import neo4j_connection_settings
        from torchcell.knowledge_graphs.releases import resolve_database

        version = self.version or neo4j_connection_settings().version
        database = resolve_database(version, self.uri, self.username, self.password)
        log.info(
            "Connecting to Neo4j (%s -> %s) and executing query...", version, database
        )
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        # 1000 is default
        with driver.session(database=database, fetch_size=1000) as session:
            log.info("Running query...")
            result = session.run(self.query, **self.cypher_kwargs)
            log.info("Query executed, about to process results...")
            yield from result
        log.info("All records processed.")
        driver.close()

    def _init_lmdb(self, readonly: bool = True) -> None:
        """Initialize the LMDB environment."""
        if self.env is not None:
            self.close_lmdb()
        self.env = lmdb.open(
            self.lmdb_dir,
            map_size=BUILD_LMDB_MAP_SIZE,
            readonly=readonly,
            lock=not readonly,
            readahead=False,
            meminit=False,
        )

    def write_to_lmdb(self, key: bytes, value: bytes) -> None:
        """Write a single key/value pair to LMDB inside a write transaction."""
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def _write_rows(self, rows: list[tuple[bytes, bytes]]) -> None:
        """Commit a batch of rows in ONE write transaction."""
        with self.env.begin(write=True) as txn:
            for key, value in rows:
                txn.put(key, value)

    def process(self) -> None:
        """Stream query results into LMDB and build the reference and gene-set indices.

        The driver thread pulls records and hands batches of their two serialized
        strings to ``num_workers`` processes, which validate them into pydantic models
        and return LMDB rows; each returned batch is committed in one write
        transaction, in query order. The previous loop validated and committed one
        record per transaction on the driver thread: 700 records/s at half a core
        (slurm 2687), most of it waiting on the per-commit flush, 19 h for 44M records.
        """
        log.info("Processing data...")
        # Two record shapes, by what the query RETURNs. Property shape
        # (e_serialized/ref_serialized strings) is REQUIRED at scale: returning whole
        # nodes makes the driver register every hydrated Node in the result's Graph
        # cache (neo4j/graph/__init__.py, graph._nodes) for the life of the result,
        # retaining 13.6 KB/record -- measured on the 025 build, which held 29 GB of
        # heap at 2.2M records and projects past machine RAM at 44M. Returning the
        # serialized_data property instead measures 3 B/record. Node shape (RETURN e,
        # ref) stays supported for the existing experiment queries.
        if self.num_workers is None:
            raise ValueError("num_workers must be set to run the raw stage")
        n_workers: int = self.num_workers
        n_written = 0
        pending: deque[Future[list[tuple[bytes, bytes]]]] = deque()
        max_in_flight = 2 * n_workers
        progress = tqdm(unit="rec", unit_scale=True)

        def drain_one() -> None:
            nonlocal n_written
            rows = pending.popleft().result()
            self._write_rows(rows)
            n_written += len(rows)
            progress.update(len(rows))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            batch: list[tuple[int, str, str]] = []
            for i, record in enumerate(self.fetch_data()):
                if "e_serialized" in record.keys():
                    e_json, ref_json = record["e_serialized"], record["ref_serialized"]
                else:
                    e_json = record["e"]["serialized_data"]
                    ref_json = record["ref"]["serialized_data"]
                batch.append((i, e_json, ref_json))
                if len(batch) == RAW_BATCH_RECORDS:
                    pending.append(pool.submit(_serialize_raw_batch, batch))
                    batch = []
                    if len(pending) >= max_in_flight:
                        drain_one()
            if batch:
                pending.append(pool.submit(_serialize_raw_batch, batch))
            while pending:
                drain_one()
        progress.close()
        log.info(f"Total records processed: {n_written}")

        self.experiment_reference_index
        self.gene_set = self.compute_gene_set()

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
            return cast(int, txn.stat()["entries"])
        self.close_lmdb()

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
            self._init_lmdb(readonly=True)
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
            self._experiment_reference_index = [
                ExperimentReferenceIndex(
                    reference=self[indices[0]]["experiment_reference"].model_dump(),
                    member_indices=sorted(indices),
                )
                for indices in hash_to_indices.values()
            ]
            # Serialize each ExperimentReferenceIndex object to dict and save the list of dicts
            with open(index_file_path, "w") as file:
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index], file
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
            phenotype_labels.append((i, record["experiment"].phenotype.label))

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
        if self.env is None:
            self._init_lmdb()

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

# torchcell/neo4j_fitness_query
# [[torchcell.neo4j_fitness_query]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/neo4j_fitness_query
# Test file: tests/torchcell/test_neo4j_fitness_query.py

import attrs
import lmdb
from neo4j import GraphDatabase
import os
from tqdm import tqdm
from attrs import define, field
import os.path as osp
import concurrent.futures
from typing import Union
from torchcell.datamodels.schema import FitnessExperiment, FitnessExperimentReference
import json
from torchcell.data import ExperimentReferenceIndex, ReferenceIndex
from torchcell.data import (
    ExperimentReferenceIndex,
    serialize_for_hashing,
    compute_sha256_hash,
)
import multiprocessing as mp


def compute_experiment_reference_index(dataset) -> list[ExperimentReferenceIndex]:
    # Hashes for each reference
    print("Computing experiment_reference_index hashes...")
    # reference_hashes = [
    #     compute_sha256_hash(serialize_for_hashing(data["reference"]))
    #     for data in tqdm(dataset)
    # ]
    reference_hashes = [
        compute_sha256_hash(
            serialize_for_hashing(
                {"experiment": data[0].model_dump(), "reference": data[1].model_dump()}[
                    "reference"
                ]
            )
        )
        for data in tqdm(dataset)
    ]

    # Identify unique hashes
    unique_hashes = set(reference_hashes)

    # Initialize ExperimentReferenceIndex list
    reference_indices = []

    print("Finding unique references...")
    for unique_hash in tqdm(unique_hashes):
        # Create a boolean list where True indicates the presence of the unique reference
        index_list = [ref_hash == unique_hash for ref_hash in reference_hashes]

        # Find the corresponding reference object for the unique hash
        ref_index = reference_hashes.index(unique_hash)
        unique_ref = dataset[ref_index][1].model_dump()

        # Create ExperimentReferenceIndex object
        exp_ref_index = ExperimentReferenceIndex(reference=unique_ref, index=index_list)
        reference_indices.append(exp_ref_index)

    return reference_indices


@define
class Neo4jQueryRaw:
    uri: str
    username: str
    password: str
    root_dir: str
    query: str
    max_workers: int = None
    num_workers: int = None
    _experiment_reference_index: ExperimentReferenceIndex = field(
        init=False, default=None, repr=False
    )
    _phenotype_label_index: dict = field(init=False, default=None, repr=False)
    lmdb_dir: str = field(init=False, default=None)
    raw_dir: str = field(init=False, default=None)
    env: str = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        # Initialize LMDB environment
        self.env = lmdb.open(self.lmdb_dir, map_size=int(1e6))
        if len(self) == 0:
            self.process()

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def fetch_data(self):
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        with driver.session(database="torchcell") as session:
            result = session.run(self.query)
            for record in result:
                yield record

        driver.close()

    def _init_lmdb(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.raw_dir, "lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def write_to_lmdb(self, key: bytes, value: bytes):
        with self.env.begin(write=True) as txn:
            txn.put(key, value)

    def process(self):
        for i, record in tqdm(enumerate(self.fetch_data())):
            # Extract the serialized data from the 'e' node
            e_node_data = json.loads(record["e"]["serialized_data"])

            # Create an instance of the FitnessExperiment model
            experiment = FitnessExperiment(
                genotype=e_node_data["genotype"],
                environment=e_node_data["environment"],
                phenotype=e_node_data["phenotype"],
            )

            # Extract the serialized data from the 'ref' node
            ref_node_data = json.loads(record["ref"]["serialized_data"])

            # Create an instance of the FitnessExperimentReference model
            reference = FitnessExperimentReference(**ref_node_data)

            # Serialize the experiment and reference objects to JSON
            experiment_json = experiment.model_dump_json()
            reference_json = reference.model_dump_json()

            # Generate keys for the experiment and reference
            experiment_key = f"experiment_{i}".encode()
            reference_key = f"reference_{i}".encode()

            # Write the serialized objects to LMDB
            self.write_to_lmdb(experiment_key, experiment_json.encode())
            self.write_to_lmdb(reference_key, reference_json.encode())

        self.experiment_reference_index

    def __getitem__(self, index: Union[int, slice, list]):
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):  # New case for a list of indices
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()["entries"]
        self.close_lmdb()

    def _get_record_by_index(self, index: int):
        self._init_lmdb()
        experiment_key = f"experiment_{index}".encode()
        reference_key = f"reference_{index}".encode()

        with self.env.begin() as txn:
            experiment_json = txn.get(experiment_key)
            reference_json = txn.get(reference_key)

            if experiment_json is None or reference_json is None:
                raise IndexError(f"Record not found at index: {index}")

            experiment = FitnessExperiment(**json.loads(experiment_json.decode()))
            reference = FitnessExperimentReference(
                **json.loads(reference_json.decode())
            )

            return experiment, reference

    def _get_records_by_slice(self, slice_obj: slice):
        start, stop, step = slice_obj.indices(len(self))
        experiment_keys = [f"experiment_{i}".encode() for i in range(start, stop, step)]
        reference_keys = [f"reference_{i}".encode() for i in range(start, stop, step)]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            experiments = list(executor.map(self._get_experiment, experiment_keys))
            references = list(executor.map(self._get_reference, reference_keys))

        return list(zip(experiments, references))

    def _get_experiment(self, key: bytes):
        with self.env.begin() as txn:
            experiment_json = txn.get(key)
            if experiment_json is None:
                raise IndexError(f"Experiment not found for key: {key.decode()}")
            experiment = FitnessExperiment(**json.loads(experiment_json.decode()))
            return experiment

    def _get_reference(self, key: bytes):
        with self.env.begin() as txn:
            reference_json = txn.get(key)
            if reference_json is None:
                raise IndexError(f"Reference not found for key: {key.decode()}")
            reference = FitnessExperimentReference(
                **json.loads(reference_json.decode())
            )
            return reference

    @property
    def experiment_reference_index(self):
        index_file_path = osp.join(self.raw_dir, "experiment_reference_index.json")

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
            # Assuming ReferenceIndex can be constructed from a list of dictionaries
            self._experiment_reference_index = [
                ExperimentReferenceIndex(**item) for item in data
            ]
        elif self._experiment_reference_index is None:
            if self.num_workers is not None:
                # Create a pool of worker processes
                with mp.Pool(processes=self.num_workers) as pool:
                    # Split the dataset into chunks for parallel processing
                    chunk_size = len(self) // self.num_workers
                    chunks = [
                        self[i : min(i + chunk_size, len(self))]
                        for i in range(0, len(self), chunk_size)
                    ]

                    # Apply the compute_experiment_reference_index function to each chunk in parallel
                    results = pool.starmap(
                        compute_experiment_reference_index,
                        [(chunk,) for chunk in chunks],
                    )

                # Flatten the results into a single list
                self._experiment_reference_index = [
                    item for sublist in results for item in sublist
                ]
            else:
                # Fall back to the original method without parallelization
                self._experiment_reference_index = compute_experiment_reference_index(
                    self
                )

            with open(index_file_path, "w") as file:
                # Convert each ExperimentReferenceIndex object to dict and save the list of dicts
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index],
                    file,
                    indent=4,
                )

        self.close_lmdb()
        return self._experiment_reference_index

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        print("Computing phenotype label index...")
        # Fetch all phenotype labels
        phenotype_labels = [
            (i, record[0].phenotype.label) for i, record in enumerate(self)
        ]

        # Initialize the phenotype label index dictionary
        phenotype_label_index = {}

        # Populate the index lists with indices
        for i, label in phenotype_labels:
            if label not in phenotype_label_index:
                phenotype_label_index[label] = []
            phenotype_label_index[label].append(i)

        return phenotype_label_index

    @property
    def phenotype_label_index(self) -> dict[str, list[bool]]:
        if osp.exists(osp.join(self.raw_dir, "phenotype_label_index.json")):
            with open(
                osp.join(self.raw_dir, "phenotype_label_index.json"), "r"
            ) as file:
                self._phenotype_label_index = json.load(file)
        else:
            self._phenotype_label_index = self.compute_phenotype_label_index()
            with open(
                osp.join(self.raw_dir, "phenotype_label_index.json"), "w"
            ) as file:
                json.dump(self._phenotype_label_index, file, indent=4)
        return self._phenotype_label_index

    def __repr__(self):
        return f"Neo4jQueryRaw(uri={self.uri}, root_dir={self.root_dir}, query={self.query})"


# Example usage
if __name__ == "__main__":
    neo4j_db = Neo4jQueryRaw(
        uri="bolt://localhost:7687",  # Include the database name here
        username="neo4j",
        password="torchcell",
        root_dir="data/torchcell/dmf-2024_03_13",
        query="""
        MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
        WITH e, g, COLLECT(p) AS perturbations
        WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
        WITH DISTINCT e
        MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
        MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
        WHERE phen.graph_level = 'global' 
        AND (phen.label = 'smf' OR phen.label = 'dmf')
        AND phen.fitness_std < 0.05
        RETURN e, ref
        LIMIT 10;
        """,
        max_workers=4,
    )
    neo4j_db[0]
    neo4j_db[0:2]
    neo4j_db.phenotype_label_index

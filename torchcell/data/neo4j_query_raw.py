# torchcell/data/neo4j_query_raw
# [[torchcell.data.neo4j_query_raw]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_query_raw
# Test file: tests/torchcell/data/test_neo4j_query_raw.py



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
from torchcell.data import ExperimentReferenceIndex, compute_sha256_hash
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torchcell.sequence import GeneSet
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parallel_hash_computation(data):
    """
    Function to compute the hash for a single dataset item.
    Returns a tuple of the original index in the dataset and the computed hash.
    """
    idx, data_item = data
    return idx, compute_sha256_hash(
        json.dumps((data_item["reference"].model_dump()), sort_keys=True)
    )


def compute_experiment_reference_index_parallel(
    dataset,
) -> list[ExperimentReferenceIndex]:
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
    unique_hashes_to_indices = {}
    for idx, hash_val in enumerate(reference_hashes):
        if hash_val not in unique_hashes_to_indices:
            unique_hashes_to_indices[hash_val] = []
        unique_hashes_to_indices[hash_val].append(idx)

    reference_indices_list = []
    for hash_val, indices in unique_hashes_to_indices.items():
        index_list = [False] * len(dataset)
        for idx in indices:
            index_list[idx] = True
        reference_obj = dataset[indices[0]]["reference"].model_dump()
        exp_ref_index = ExperimentReferenceIndex(
            reference=reference_obj, index=index_list
        )
        reference_indices_list.append(exp_ref_index)

    return reference_indices_list


def compute_experiment_reference_index(
    dataset, num_workers=None
) -> list[ExperimentReferenceIndex]:
    if num_workers is None or num_workers <= 0:
        # Sequential version
        log.info("Computing experiment reference index sequentially")
        reference_hashes = [
            compute_sha256_hash(
                json.dumps(data["reference"].model_dump(), sort_keys=True)
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
    unique_hashes_to_indices = {}
    for idx, hash_val in enumerate(reference_hashes):
        if hash_val not in unique_hashes_to_indices:
            unique_hashes_to_indices[hash_val] = []
        unique_hashes_to_indices[hash_val].append(idx)

    reference_indices_list = []
    for hash_val, indices in unique_hashes_to_indices.items():
        index_list = [False] * len(dataset)
        for idx in indices:
            index_list[idx] = True
        reference_obj = dataset[indices[0]]["reference"].model_dump()
        exp_ref_index = ExperimentReferenceIndex(
            reference=reference_obj, index=index_list
        )
        reference_indices_list.append(exp_ref_index)

    return reference_indices_list


@define
class Neo4jQueryRaw:
    uri: str
    username: str
    password: str
    root_dir: str
    query: str
    io_workers: int = None
    num_workers: int = None
    _experiment_reference_index: ExperimentReferenceIndex = field(
        init=False, default=None, repr=False
    )
    _phenotype_label_index: dict = field(init=False, default=None, repr=False)
    lmdb_dir: str = field(init=False, default=None)
    raw_dir: str = field(init=False, default=None)
    env: str = field(init=False, default=None)
    _gene_set: str = field(init=False, default=None)
    cypher_kwargs: dict[str, Union[str, int, float, list]] = field(factory=dict)

    def __attrs_post_init__(self):
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        # Initialize LMDB environment
        self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12), readonly=True)
        if len(self) == 0:
            self.process()

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def fetch_data(self):
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        with driver.session(database="torchcell") as session:
            log.info("Fetching data from Neo4j...")
            result = session.run(self.query, **self.cypher_kwargs)
            log.info("Data fetched successfully.")
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
        log_batch_size = int(1e10)

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

            # Create a dictionary with experiment and reference objects
            data_dict = {"experiment": experiment, "reference": reference}

            # Serialize the dictionary to JSON
            data_json = json.dumps(data_dict, default=lambda o: o.model_dump())

            # Generate a key for the data
            data_key = f"data_{i}".encode()

            # Write the serialized dictionary to LMDB
            self.write_to_lmdb(data_key, data_json.encode())

            # Log progress every log_batch_size records
            # if (i + 1) % log_batch_size == 0:
            #     log.info(f"Processed {i + 1} records")

        log.info(f"Total records processed: {i + 1}")
        

        self.experiment_reference_index
        self.gene_set = self.compute_gene_set()

    def __getitem__(self, index: Union[int, slice, list]):
        if isinstance(index, int):
            return self._get_record_by_index(index)
        elif isinstance(index, slice):
            return self._get_records_by_slice(index)
        elif isinstance(index, list):  # New case for a list of indices
            return [self._get_record_by_index(idx) for idx in index]
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def _get_record_by_index(self, index: int):
        self._init_lmdb()
        data_key = f"data_{index}".encode()

        with self.env.begin() as txn:
            data_json = txn.get(data_key)

            if data_json is None:
                raise IndexError(f"Record not found at index: {index}")

            data_dict = json.loads(data_json.decode())
            experiment = FitnessExperiment(**data_dict["experiment"])
            reference = FitnessExperimentReference(**data_dict["reference"])

            return {"experiment": experiment, "reference": reference}

    def _get_record(self, key: bytes):
        with self.env.begin() as txn:
            data_json = txn.get(key)
            if data_json is None:
                raise IndexError(f"Record not found for key: {key.decode()}")
            data_dict = json.loads(data_json.decode())
            experiment = FitnessExperiment(**data_dict["experiment"])
            reference = FitnessExperimentReference(**data_dict["reference"])
            return {"experiment": experiment, "reference": reference}

    def _get_records_by_slice(self, slice_obj: slice):
        start, stop, step = slice_obj.indices(len(self))
        data_keys = [f"data_{i}".encode() for i in range(start, stop, step)]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.io_workers
        ) as executor:
            records = list(executor.map(self._get_record, data_keys))

        return records

    def __len__(self):
        if self.env is None:
            self._init_lmdb()
        with self.env.begin() as txn:
            return txn.stat()["entries"]
        self.close_lmdb()

    @staticmethod
    def extract_systematic_gene_names(genotype):
        gene_names = []
        for perturbation in genotype.get("perturbations"):
            gene_name = perturbation.get("systematic_gene_name")
            gene_names.append(gene_name)
        return gene_names

    @property
    def experiment_reference_index(self) -> list[ExperimentReferenceIndex]:
        index_file_path = osp.join(self.raw_dir, "experiment_reference_index.json")

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
            # Deserialize each dict in the list to an ExperimentReferenceIndex object
            self._experiment_reference_index = [
                ExperimentReferenceIndex(**item) for item in data
            ]
        elif self._experiment_reference_index is None:
            self._experiment_reference_index = compute_experiment_reference_index(
                [self[i] for i in range(len(self))]
            )
            # Serialize each ExperimentReferenceIndex object to dict and save the list of dicts
            with open(index_file_path, "w") as file:
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index], file
                )

        self.close_lmdb()
        return self._experiment_reference_index

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        print("Computing phenotype label index...")
        # Fetch all phenotype labels
        phenotype_labels = [
            (i, record["experiment"].phenotype.label) for i, record in enumerate(self)
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
                json.dump(self._phenotype_label_index, file)
        return self._phenotype_label_index

    def compute_gene_set(self):
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
    def gene_set(self):
        if osp.exists(osp.join(self.raw_dir, "gene_set.json")):
            with open(osp.join(self.raw_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        else:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.raw_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def __repr__(self):
        return f"Neo4jQueryRaw(uri={self.uri}, root_dir={self.root_dir}, query={self.query})"


##########################33

# Example usage
if __name__ == "__main__":
    from torchcell.sequence import GeneSet
    from dotenv import load_dotenv
    from hashlib import sha256

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    # TODO change to process and io workers
    neo4j_db = Neo4jQueryRaw(
        uri="bolt://localhost:7687",  # Include the database name here
        username="neo4j",
        password="torchcell",
        root_dir="data/torchcell/neo4j_query_test",
        query="""
            MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
            RETURN e, ref
        """,
        io_workers=10,
        num_workers=10,
        cypher_kwargs={"gene_set": list(genome.gene_set)},
    )
    neo4j_db[0]
    neo4j_db[0:2]
    # neo4j_db.phenotype_label_index.keys()

    duplicate_check = {}
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

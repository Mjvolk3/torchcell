"""Abstract LMDB-backed experiment dataset and reference-index utilities."""

# torchcell/dataset/experiment_dataset
# [[torchcell.dataset.experiment_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/dataset/experiment_dataset
# Test file: tests/torchcell/dataset/test_experiment_dataset.py

import json
import logging
import os.path as osp
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps

import lmdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from torchcell.data import ExperimentReferenceIndex, compute_sha256_hash
from torchcell.datamodels import (
    Experiment,
    ExperimentReference,
    ExperimentReferenceType,
    Publication,
)
from torchcell.loader import CpuExperimentLoaderMultiprocessing
from torchcell.sequence import GeneSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def process_reference_batch(batch):
    """Compute SHA-256 reference hashes for each item in a batch."""
    # This function will process a batch of data to compute reference hashes.
    # Adjust the implementation based on your data structure.
    reference_hashes = []
    for data in batch:
        reference_hash = compute_sha256_hash(serialize_for_hashing(data["reference"]))
        reference_hashes.append(reference_hash)
    return reference_hashes


def _compute_reference_hash_parallel(data):
    """Return the SHA-256 hash of a single item's reference."""
    # Parallel processing function for computing reference hash
    reference = data["reference"]
    return compute_sha256_hash(serialize_for_hashing(reference))


# return reference_indices
# TODO FitnessExperimentReference Will need to generalize away from fitness
def serialize_for_hashing(obj):
    """Return a deterministic JSON string for hashing a reference or object."""
    if isinstance(obj, ExperimentReferenceType):
        # Convert FitnessExperimentReference to a dictionary
        obj_dict = obj.model_dump()
        # Sort the dictionary keys for consistent serialization
        sorted_dict = dict(sorted(obj_dict.items()))
        return json.dumps(sorted_dict)
    else:
        return json.dumps(obj, sort_keys=True)


def compute_experiment_reference_index_sequential(
    dataset: Dataset,
) -> list[ExperimentReferenceIndex]:
    """Build reference indices by hashing every item sequentially."""
    # Hashes for each reference
    print("Computing experiment_reference_index hashes...")
    reference_hashes = [
        compute_sha256_hash(serialize_for_hashing(data["reference"]))
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
        unique_ref = dataset[ref_index]["reference"]

        # Create ExperimentReferenceIndex object
        exp_ref_index = ExperimentReferenceIndex(reference=unique_ref, index=index_list)
        reference_indices.append(exp_ref_index)

    return reference_indices


def compute_experiment_reference_index_parallel(
    dataset, batch_size=int(1e4), io_workers=1
) -> list[ExperimentReferenceIndex]:
    """Build reference indices by hashing items via a multiprocessing loader."""
    print("Computing experiment_reference_index hashes in parallel...")
    data_loader = CpuExperimentLoaderMultiprocessing(
        dataset, batch_size=batch_size, num_workers=io_workers
    )

    reference_hashes = []
    reference_data = []  # Add this line to store full reference data
    for batch in tqdm(data_loader, total=len(data_loader)):
        for data in batch:
            reference_hash = compute_sha256_hash(
                serialize_for_hashing(data["reference"])
            )
            reference_hashes.append(reference_hash)
            reference_data.append(data["reference"])  # Store full reference data

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
        unique_ref = reference_data[ref_index]  # Use stored reference data

        # Create ExperimentReferenceIndex object
        exp_ref_index = ExperimentReferenceIndex(reference=unique_ref, index=index_list)
        reference_indices.append(exp_ref_index)

    return reference_indices


def post_process(func):
    """Decorate ``process`` to rebuild the gene set/reference index and validate."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Execute the original process method
        result = func(self, *args, **kwargs)

        # Perform the original post-processing tasks
        self.gene_set = self.compute_gene_set()
        self.experiment_reference_index

        # Additional validation step
        total_index_coverage = torch.zeros(len(self), dtype=torch.bool)
        for eri in self.experiment_reference_index:
            index_tensor = torch.tensor(eri.index, dtype=torch.bool)
            total_index_coverage |= index_tensor

        assert torch.all(total_index_coverage).item() is True, (
            "Each item in the dataset must be covered exactly once."
        )

        return result

    return wrapper


class ExperimentDataset(Dataset, ABC):
    """Abstract PyG dataset storing experiment items in an LMDB store."""

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        skip_process_file_exist: bool = False,
    ):
        """Set up paths/state and trigger PyG download/process via super().__init__."""
        self.io_workers = io_workers
        self.preprocess_dir = osp.join(root, "preprocess")
        # TODO This is part of our custom Dataset to speed things up but should be removed when using pure pyg
        self.skip_process_file_exist = skip_process_file_exist
        self.env = None
        self._length = None
        self._gene_set = None
        self._df = None
        self._experiment_reference_index = None

        # Automatically set the name based on the class name
        self.name = self.__class__.__name__

        super().__init__(root, transform, pre_transform)

    @property
    @abstractmethod
    def experiment_class(self) -> type[Experiment]:
        """Return the Experiment subclass produced by this dataset."""
        ...

    @property
    @abstractmethod
    def reference_class(self) -> type[ExperimentReference]:
        """Return the ExperimentReference subclass produced by this dataset."""
        ...

    @property
    @abstractmethod
    def raw_file_names(self) -> list[str]:
        """Return the raw file names expected under the raw directory."""
        ...

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed artifact name (the LMDB directory)."""
        return "lmdb"

    @post_process
    @abstractmethod
    def process(self):
        """Build the processed LMDB store from raw data (implemented by subclasses)."""
        raise NotImplementedError

    @abstractmethod
    def download(self):
        """Download raw data into the raw directory (implemented by subclasses)."""
        raise NotImplementedError

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def close_lmdb(self):
        """Close the LMDB environment if it is open."""
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def df(self):
        """Return the preprocessed data CSV as a DataFrame, if present."""
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    @abstractmethod
    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame:
        """Clean/normalize the raw DataFrame (implemented by subclasses)."""
        ...

    @abstractmethod
    def create_experiment(self):
        """Construct experiment records from preprocessed data (subclasses)."""
        ...

    def len(self) -> int:
        """Return the number of entries stored in the LMDB database."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader io_workers > 0
        self.close_lmdb()

        return length

    def get(self, idx):
        """Return one item, or a list of items for list/array/boolean indices."""
        if self.env is None:
            self._init_db()

        # Handling boolean index arrays or numpy arrays
        if isinstance(idx, (list, np.ndarray)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if idx.dtype == np.bool_:
                idx = np.where(idx)[0]

            # If idx is a list/array of indices, return a list of data objects
            return [self.get_single_item(i) for i in idx]
        else:
            # Single item retrieval
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        """Return the deserialized item at ``idx``, or None if absent."""
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    @staticmethod
    def extract_systematic_gene_names(genotype):
        """Return the systematic gene names of all perturbations in a genotype."""
        gene_names = []
        for perturbation in genotype.get("perturbations"):
            gene_name = perturbation.get("systematic_gene_name")
            gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        """Compute the dataset gene set sequentially or in parallel."""
        if self.io_workers > 0:
            log.info("Computing gene set in parallel...")
            return self.compute_gene_set_parallel(io_workers=self.io_workers)
        else:
            log.info("Computing gene set sequentially...")
            return self.compute_gene_set_sequential()

    def compute_gene_set_sequential(self):
        """Compute the gene set by scanning every LMDB entry in order."""
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            log.info("Computing gene set...")
            for key, value in tqdm(cursor):
                deserialized_data = pickle.loads(value)
                experiment = deserialized_data["experiment"]

                extracted_gene_names = self.extract_systematic_gene_names(
                    experiment["genotype"]
                )
                for gene_name in extracted_gene_names:
                    gene_set.add(gene_name)

        self.close_lmdb()
        return gene_set

    def compute_gene_set_parallel(
        self, batch_size: int = int(1e4), io_workers: int = 1
    ):
        """Compute the gene set using a multiprocessing batch loader."""
        gene_set = GeneSet()

        log.info("Computing gene set in parallel...")
        data_loader = CpuExperimentLoaderMultiprocessing(
            self, batch_size=batch_size, num_workers=io_workers
        )
        for batch in tqdm(data_loader, total=len(data_loader)):
            gene_names_batch = set()
            for data in batch:
                gene_names = self.extract_systematic_gene_names(
                    data["experiment"]["genotype"]
                )
                gene_names_batch.update(gene_names)
            gene_set.update(gene_names_batch)

        return gene_set

    @property
    def experiment_reference_index(self):
        """Load or compute (and cache) the experiment reference index."""
        index_file_path = osp.join(
            self.preprocess_dir, "experiment_reference_index.json"
        )

        if osp.exists(index_file_path):
            with open(index_file_path) as file:
                data = json.load(file)
                self._experiment_reference_index = [
                    ExperimentReferenceIndex(**item) for item in data
                ]
        elif self._experiment_reference_index is None:
            if self.io_workers > 1:
                log.info("Computing experiment reference index in parallel...")
                self._experiment_reference_index = (
                    compute_experiment_reference_index_parallel(
                        dataset=self, io_workers=self.io_workers
                    )
                )
            else:
                log.info("Computing experiment reference index sequentially...")
                self._experiment_reference_index = (
                    compute_experiment_reference_index_sequential(dataset=self)
                )

            with open(index_file_path, "w") as file:
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index],
                    file,
                    indent=4,
                )

        self.close_lmdb()
        return self._experiment_reference_index

    @property
    def gene_set(self):
        """Return the gene set, loading the cached JSON or computing it."""
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        else:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        """Persist the sorted gene set to JSON and cache it in memory."""
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def transform_item(self, item):
        """Rebuild typed experiment/reference/publication objects from a raw item."""
        experiment_data = item["experiment"]
        reference_data = item["reference"]
        publication_data = item["publication"]
        experiment = self.experiment_class(**experiment_data)
        reference = self.reference_class(**reference_data)
        reference = self.reference_class(**reference_data)
        publication = Publication(**publication_data)
        return {
            "experiment": experiment,
            "reference": reference,
            "publication": publication,
        }

    def __repr__(self):
        """Return ``ClassName(<length>)``."""
        return f"{self.__class__.__name__}({len(self)})"

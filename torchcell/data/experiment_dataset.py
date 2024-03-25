# torchcell/dataset/experiment_dataset
# [[torchcell.dataset.experiment_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/dataset/experiment_dataset
# Test file: tests/torchcell/dataset/test_experiment_dataset.py

import multiprocessing as mp
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import json
import logging
import os.path as osp
import pickle
from collections.abc import Callable
import numpy as np
import lmdb
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Dataset
from torchcell.datamodels import BaseExperiment, ExperimentReference
from torchcell.sequence import GeneSet
from abc import ABC, abstractmethod
from functools import wraps
import torch
from torchcell.datamodels import FitnessExperimentReference
from torchcell.data import ExperimentReferenceIndex, compute_sha256_hash
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from torchcell.loader import CpuExperimentLoaderMultiprocessing
import multiprocessing as mp
import pickle
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def process_reference_batch(batch):
    # This function will process a batch of data to compute reference hashes.
    # Adjust the implementation based on your data structure.
    reference_hashes = []
    for data in batch:
        reference_hash = compute_sha256_hash(serialize_for_hashing(data["reference"]))
        reference_hashes.append(reference_hash)
    return reference_hashes


def _compute_reference_hash_parallel(data):
    # Parallel processing function for computing reference hash
    reference = data["reference"]
    return compute_sha256_hash(serialize_for_hashing(reference))


# return reference_indices
# TODO FitnessExperimentReference Will need to generalize away from fitness
def serialize_for_hashing(obj):
    if isinstance(obj, FitnessExperimentReference):
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
    # Hashes for each reference
    print("Computing experiment_reference_index hashes in parallel...")
    data_loader = CpuExperimentLoaderMultiprocessing(
        dataset, batch_size=batch_size, num_workers=io_workers
    )

    reference_hashes = []
    for batch in tqdm(data_loader, total=len(data_loader)):
        for data in batch:
            reference_hash = compute_sha256_hash(
                serialize_for_hashing(data["reference"])
            )
            reference_hashes.append(reference_hash)

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


def post_process(func):
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

        assert (
            torch.all(total_index_coverage).item() is True
        ), "Each item in the dataset must be covered exactly once."

        return result

    return wrapper


class ExperimentDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        skip_process_file_exist: bool = False,
    ):
        self.io_workers = io_workers
        self.preprocess_dir = osp.join(root, "preprocess")
        # TODO This is part of our custom Dataset to speed things up but should be removed when using pure pyg
        self.skip_process_file_exist = skip_process_file_exist
        self.env = None
        self._length = None
        self._gene_set = None
        self._df = None
        self._experiment_reference_index = None
        super().__init__(root, transform, pre_transform)

    @property
    @abstractmethod
    def experiment_class(self) -> BaseExperiment: ...

    @property
    @abstractmethod
    def reference_class(self) -> ExperimentReference: ...

    @property
    @abstractmethod
    def raw_file_names(self) -> list[str]: ...

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    @post_process
    @abstractmethod
    def process(self):
        raise NotImplementedError

    @abstractmethod
    def download(self):
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
        if self.env is not None:
            self.env.close()
            self.env = None

    @property
    def df(self):
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    @abstractmethod
    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def create_experiment(self): ...

    def len(self) -> int:
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader io_workers > 0
        self.close_lmdb()

        return length

    def get(self, idx):
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
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    @staticmethod
    def extract_systematic_gene_names(genotype):
        gene_names = []
        for perturbation in genotype.get("perturbations"):
            gene_name = perturbation.get("systematic_gene_name")
            gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        if self.io_workers > 0:
            log.info("Computing gene set in parallel...")
            return self.compute_gene_set_parallel(io_workers=self.io_workers)
        else:
            log.info("Computing gene set sequentially...")
            return self.compute_gene_set_sequential()

    def compute_gene_set_sequential(self):
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
        index_file_path = osp.join(
            self.preprocess_dir, "experiment_reference_index.json"
        )

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
                self._experiment_reference_index = [
                    ExperimentReferenceIndex(**item) for item in data
                ]
        elif self._experiment_reference_index is None:
            if self.io_workers > 0:
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
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        else:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    # @property
    # def experiment_reference_index(self):
    #     index_file_path = osp.join(
    #         self.preprocess_dir, "experiment_reference_index.json"
    #     )

    #     if osp.exists(index_file_path):
    #         with open(index_file_path, "r") as file:
    #             data = json.load(file)
    #             # Assuming ReferenceIndex can be constructed from a list of dictionaries
    #             self._experiment_reference_index = [
    #                 ExperimentReferenceIndex(**item) for item in data
    #             ]
    #     elif self._experiment_reference_index is None:
    #         self._experiment_reference_index = compute_experiment_reference_index(self)
    #         with open(index_file_path, "w") as file:
    #             # Convert each ExperimentReferenceIndex object to dict and save the list of dicts
    #             json.dump(
    #                 [eri.model_dump() for eri in self._experiment_reference_index],
    #                 file,
    #                 indent=4,
    #             )

    #     self.close_lmdb()
    #     return self._experiment_reference_index

    @property
    @abstractmethod
    def experiment_class(self): ...

    @property
    @abstractmethod
    def reference_class(self): ...

    def transform_item(self, item):
        experiment_data = item["experiment"]
        reference_data = item["reference"]
        experiment = self.experiment_class(**experiment_data)
        reference = self.reference_class(**reference_data)
        return {"experiment": experiment, "reference": reference}

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"

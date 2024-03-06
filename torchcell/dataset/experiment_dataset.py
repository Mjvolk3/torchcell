# torchcell/dataset/experiment_dataset
# [[torchcell.dataset.experiment_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/dataset/experiment_dataset
# Test file: tests/torchcell/dataset/test_experiment_dataset.py


import json
import logging
import os
import os.path as osp
import pickle
import shutil
import zipfile
from collections.abc import Callable
import numpy as np
import lmdb
import pandas as pd
from torch_geometric.data import download_url
from tqdm import tqdm
from torch_geometric.data import Dataset
from torchcell.dataset import compute_experiment_reference_index
from torchcell.data import ExperimentReferenceIndex
from torchcell.datamodels import (
    BaseEnvironment,
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaDampPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    BaseExperiment,
    ExperimentReference,
)
from torchcell.sequence import GeneSet
from multiprocessing import Process, Queue
from typing import Iterable
from abc import ABC, abstractmethod
from functools import wraps

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def post_process(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.gene_set = self.compute_gene_set()
        self.experiment_reference_index
        return result

    return wrapper


class ExperimentDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
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

        # Must be closed for dataloader num_workers > 0
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

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        # elif self._gene_set is None:
        #     raise ValueError(
        #         "gene_set not written during process. "
        #         "Please call compute_gene_set at the end of process."
        #     )
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

    @property
    def experiment_reference_index(self):
        index_file_path = osp.join(
            self.preprocess_dir, "experiment_reference_index.json"
        )

        if osp.exists(index_file_path):
            with open(index_file_path, "r") as file:
                data = json.load(file)
                # Assuming ReferenceIndex can be constructed from a list of dictionaries
                self._experiment_reference_index = [
                    ExperimentReferenceIndex(**item) for item in data
                ]
        elif self._experiment_reference_index is None:
            self._experiment_reference_index = compute_experiment_reference_index(self)
            with open(index_file_path, "w") as file:
                # Convert each ExperimentReferenceIndex object to dict and save the list of dicts
                json.dump(
                    [eri.model_dump() for eri in self._experiment_reference_index],
                    file,
                    indent=4,
                )

        self.close_lmdb()
        return self._experiment_reference_index

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

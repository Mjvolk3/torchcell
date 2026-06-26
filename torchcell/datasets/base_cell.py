# torchcell/datasets/base_cell.py
# [[torchcell.datasets.base_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/base_cell.py
# Test file: tests/torchcell/datasets/test_base_cell.py
"""Base cell dataset built from Neo4j queries and stored in LMDB."""

import json
import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable

import lmdb
import numpy as np
import pandas as pd
import torch
from attrs import define, field
from neo4j import GraphDatabase
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.sequence import GeneSet

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@define
class BaseQuery:
    """Cypher query bound to a Neo4j connection, yielding records on demand."""

    query: str
    uri: str = field(default="neo4j://localhost:7687")
    username: str = field(default="neo4j")
    password: str = field(default="neo4j")

    def __attrs_post_init__(self):
        """Open the Neo4j driver using the configured URI and credentials."""
        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )

    def get_data(self):
        """Run the query in a session and yield each result record, then close."""
        with self.driver.session() as session:
            result = session.run(self.query)
            yield from result
        self.driver.close()


class BaseCellDataset(Dataset):
    """Marker base class for cell datasets."""

    pass


class Cell(Dataset):
    """Cell dataset that processes raw query output into an LMDB-backed store."""

    def __init__(
        self,
        root: str = "data/torchcell/dmf_costanzo2016",
        subset_n: int = None,
        preprocess: dict | None = None,
        skip_process_file_exist: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        """Set up preprocessing config, validate it against any existing config, and process.

        Args:
            root: Dataset root directory.
            subset_n: Optional number of rows to randomly subsample.
            preprocess: Preprocessing configuration dict.
            skip_process_file_exist: Whether to skip the processed-file existence check.
            transform: Optional runtime transform.
            pre_transform: Optional transform applied during processing.
        """
        self.subset_n = subset_n
        self._skip_process_file_exist = skip_process_file_exist
        # TODO consider moving to a well defined Dataset class
        self.preprocess = preprocess
        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")
        self._length = None
        self._gene_set = None
        self._df = None
        # Check for existing preprocess config
        existing_config = self.load_preprocess_config()
        if existing_config is not None:
            if existing_config != self.preprocess:
                raise ValueError(
                    "New preprocess does not match existing config."
                    "Delete the processed and process dir for a new Dataset."
                    "Or define a new root."
                )
        self.env = None
        self._experiment_reference_index = None
        super().__init__(root, transform, pre_transform)
        # This was here before - not sure if it has something to do with gpu
        # self.env = None

    @property
    def skip_process_file_exist(self):
        """Return whether the processed-file existence check is skipped."""
        return self._skip_process_file_exist

    @property
    def raw_file_names(self) -> list[str]:
        """Return the expected raw file name."""
        return "dummy.txt"

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed LMDB file name."""
        return "data.lmdb"

    def download(self):
        """Download raw data (placeholder; query-based download not implemented)."""
        # TODO Run query to download data
        pass

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
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
        """Return the preprocessed dataframe, loading it from CSV if present."""
        if osp.exists(osp.join(self.preprocess_dir, "data.csv")):
            self._df = pd.read_csv(osp.join(self.preprocess_dir, "data.csv"))
        return self._df

    def process(self):
        """Read raw files, preprocess, optionally subsample, and write experiments to LMDB."""
        os.makedirs(self.preprocess_dir, exist_ok=True)
        self._length = None
        # Initialize an empty DataFrame to hold all raw data
        df = pd.DataFrame()

        # Read and concatenate all raw files
        print("Reading and Concatenating Raw Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)

            # Reading data using Pandas; limit rows for demonstration
            df_temp = pd.read_csv(file_path, sep="\t")

            # Concatenating data frames
            df = pd.concat([df, df_temp], ignore_index=True)
        # Functions for data filtering... duplicates selection,
        df = self.preprocess_raw(df, self.preprocess)
        self.save_preprocess_config(self.preprocess)

        # Subset
        if self.subset_n is not None:
            df = df.sample(n=self.subset_n, random_state=42).reset_index(drop=True)

        # Save preprocssed df - mainly for quick stats
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        print("Processing DMF Files...")

        # Initialize LMDB environment
        env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            map_size=int(1e12),  # Adjust map_size as needed
        )

        with env.begin(write=True) as txn:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                experiment, reference = self.create_experiment(row)

                # Serialize the Pydantic objects
                serialized_data = pickle.dumps(
                    {"experiment": experiment, "reference": reference}
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()
        self.gene_set = self.compute_gene_set()

    @staticmethod
    def create_experiment(row):
        """Build an (experiment, reference) pair from a dataframe row (override in subclass)."""
        # return experiment, reference
        pass

    def preprocess_raw(self, df: pd.DataFrame, preprocess: dict | None = None):
        """Apply dataset-specific filtering to the raw dataframe (override in subclass)."""
        print("Preprocess on raw data...")
        # return df
        pass

    # New method to save preprocess configuration to a JSON file
    def save_preprocess_config(self, preprocess):
        """Write the preprocessing configuration to a JSON file in the preprocess dir."""
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "preprocess_config.json"), "w") as f:
            json.dump(preprocess, f)

    def load_preprocess_config(self):
        """Load the saved preprocessing config from JSON, or None if absent."""
        config_path = osp.join(self.preprocess_dir, "preprocess_config.json")

        if osp.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            return config
        else:
            return None

    def len(self) -> int:
        """Return the number of entries stored in the LMDB database."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

    def get(self, idx):
        """Retrieve one item or a list of items for the given index, mask, or array."""
        if self.env is None:
            self._init_db()

        # Handling boolean index tensors or numpy arrays
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            if isinstance(idx, list):
                idx = np.array(idx)
            if isinstance(idx, np.ndarray) and idx.dtype == np.bool:
                idx = np.where(idx)[0]
            elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                idx = idx.nonzero(as_tuple=False).squeeze(1)

        if isinstance(idx, (np.ndarray, list, torch.Tensor)):
            # If idx is a list/array/tensor of indices, return a list of data objects
            return [self.get_single_item(i.item()) for i in idx]
        else:
            # Single item retrieval
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        """Deserialize and return the stored record at the given index, or None."""
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None

            deserialized_data = pickle.loads(serialized_data)
            return deserialized_data

    @staticmethod
    def extract_systematic_gene_names(genotypes):
        """Return the systematic gene names from each genotype's perturbation."""
        gene_names = []
        for genotype in genotypes:
            if hasattr(genotype, "perturbation") and hasattr(
                genotype.perturbation, "systematic_gene_name"
            ):
                gene_name = genotype.perturbation.systematic_gene_name
                gene_names.append(gene_name)
        return gene_names

    def compute_gene_set(self):
        """Scan all LMDB records and build the GeneSet of perturbed gene names."""
        gene_set = GeneSet()
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            cursor = txn.cursor()
            print("Computing gene set...")
            for key, value in tqdm(cursor):
                deserialized_data = pickle.loads(value)
                experiment = deserialized_data["experiment"]

                extracted_gene_names = self.extract_systematic_gene_names(
                    experiment.genotype
                )
                for gene_name in extracted_gene_names:
                    gene_set.add(gene_name)

        self.close_lmdb()
        return gene_set

    # Reading from JSON and setting it to self._gene_set
    @property
    def gene_set(self):
        """Return the GeneSet, loading from gene_set.json if it exists."""
        if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
            with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                self._gene_set = GeneSet(json.load(f))
        elif self._gene_set is None:
            raise ValueError(
                "gene_set not written during process. "
                "Please call compute_gene_set in process."
            )
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
        """Return the experiment-to-reference index (not yet implemented; returns None)."""
        # TODO implement
        self._experiment_reference_index = None
        return self._experiment_reference_index

    def __repr__(self):
        """Return a string with the class name and number of items."""
        return f"{self.__class__.__name__}({len(self)})"


if __name__ == "__main__":
    pass
